#!/usr/bin/python3
import absl.logging
import argparse
from datetime import datetime
import logging
import os
import psutil
import socket

# fixes logging not showing up in tensorflow

from cptv import CPTVReader
import numpy as np
import json

from config.config import Config
from config.thermalconfig import ThermalConfig
from .cptvrecorder import CPTVRecorder
from .headerinfo import HeaderInfo
from ml_tools.logs import init_logging
from ml_tools import tools
from .motiondetector import MotionDetector
from .piclassifier import PiClassifier
from service import SnapshotService
from .cameras import lepton3

SOCKET_NAME = "/var/run/lepton-frames"
VOSPI_DATA_SIZE = 160
TELEMETRY_PACKET_COUNT = 4


class NeuralInterpreter:
    def __init__(self, model_name):
        from openvino.inference_engine import IENetwork, IECore

        # device = "CPU"
        device = "MYRIAD"
        model_xml = model_name + ".xml"
        model_bin = os.path.splitext(model_xml)[0] + ".bin"
        ie = IECore()
        net = IENetwork(model=model_xml, weights=model_bin)
        self.input_blob = next(iter(net.inputs))
        self.out_blob = next(iter(net.outputs))
        net.batch_size = 1
        self.exec_net = ie.load_network(network=net, device_name=device)
        self.load_json(model_name)

    def classify_frame_with_novelty(self, input_x):
        input_x = np.array([[input_x]])
        input_x = input_x.reshape((1, 48, 1, 5, 48))
        res = self.exec_net.infer(inputs={self.input_blob: input_x})
        res = res[self.out_blob]
        return res[0][0], res[0][1], None

    def load_json(self, filename):
        """ Loads model and parameters from file. """
        stats = json.load(open(filename + ".txt", "r"))

        self.MODEL_NAME = stats["name"]
        self.MODEL_DESCRIPTION = stats["description"]
        self.labels = stats["labels"]
        self.eval_score = stats["score"]
        self.params = stats["hyperparams"]


class LiteInterpreter:
    def __init__(self, model_name):
        import tensorflow as tf

        self.interpreter = tf.lite.Interpreter(model_path=model_name + ".tflite")

        self.interpreter.allocate_tensors()
        input_details = self.interpreter.get_input_details()
        tensors = self.interpreter.get_tensor_details()

        self.in_values = {}
        for detail in input_details:
            self.in_values[detail["name"]] = detail["index"]

        output_details = self.interpreter.get_output_details()
        self.out_values = {}
        for detail in output_details:

            self.out_values[detail["name"]] = detail["index"]

        self.load_json(model_name)

        self.state_out = self.out_values["state_out"]
        self.novelty = self.out_values["novelty"]
        self.prediction = self.out_values["prediction"]

    def run(self, input_x, state_in=None):
        input_x = input_x[np.newaxis, np.newaxis, :]
        self.interpreter.set_tensor(self.in_values["X"], input_x)
        if state_in is not None:
            self.interpreter.set_tensor(self.in_values["state_in"], state_in)

        self.interpreter.invoke()

    def classify_frame_with_novelty(self, input_x, state_in=None):
        self.run(input_x, state_in)
        pred = self.interpreter.get_tensor(self.out_values["prediction"])[0]
        nov = self.interpreter.get_tensor(self.out_values["novelty"])
        state = self.interpreter.get_tensor(self.out_values["state_out"])
        return pred, nov, state

    def load_json(self, filename):
        stats = json.load(open(filename + ".txt", "r"))

        self.MODEL_NAME = stats["name"]
        self.MODEL_DESCRIPTION = stats["description"]
        self.labels = stats["labels"]
        self.eval_score = stats["score"]
        self.params = stats["hyperparams"]


# TODO abstract interpreter class


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cptv", help="a CPTV file to send", default=None)
    parser.add_argument("-c", "--config-file", help="Path to config file to use")
    parser.add_argument(
        "--thermal-config-file", help="Path to pi-config file (config.toml) to use"
    )

    args = parser.parse_args()
    return args


def get_classifier(config):
    model_name, model_type = os.path.splitext(config.classify.model)
    if model_type == ".tflite":
        return LiteInterpreter(model_name)
    elif model_type == ".xml":
        return NeuralInterpreter(model_name)
    else:
        return get_full_classifier(config)


def get_full_classifier(config):
    from ml_tools.model import Model

    """
    Returns a classifier object, which is created on demand.
    This means if the ClipClassifier is copied to a new process a new Classifier instance will be created.
    """
    t0 = datetime.now()
    logging.info("classifier loading")
    classifier = Model(
        train_config=config.train,
        session=tools.get_session(disable_gpu=not config.use_gpu),
    )
    classifier.load(config.classify.model)
    logging.info("classifier loaded ({})".format(datetime.now() - t0))

    return classifier


# Links to socket and continuously waits for 1 connection
def main():
    logging.root.removeHandler(absl.logging._absl_handler)
    absl.logging._warn_preinit_stderr = False
    init_logging()
    args = parse_args()

    config = Config.load_from_file(args.config_file)
    thermal_config = ThermalConfig.load_from_file(args.thermal_config_file)

    if args.cptv:
        return parse_cptv(args.cptv, config, thermal_config)

    try:
        os.unlink(SOCKET_NAME)
    except OSError:
        if os.path.exists(SOCKET_NAME):
            raise

    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.bind(SOCKET_NAME)
    sock.listen(1)
    while True:
        logging.info("waiting for a connection")
        connection, client_address = sock.accept()
        logging.info("connection from %s", client_address)
        try:
            handle_connection(connection, config, thermal_config)
        finally:
            # Clean up the connection
            connection.close()


def parse_cptv(cptv_file, config, thermal_config):
    with open(cptv_file, "rb") as f:
        reader = CPTVReader(f)

        headers = HeaderInfo(
            res_x=reader.x_resolution,
            res_y=reader.y_resolution,
            fps=9,
            brand="",
            model="",
            frame_size=reader.x_resolution * reader.y_resolution * 2,
            pixel_bits=16,
        )
        processor = get_processor(config, thermal_config, headers)
        for frame in reader:
            processor.process_frame(frame)

        processor.disconnected()


def get_processor(config, thermal_config, headers):
    if thermal_config.motion.run_classifier:
        classifier = get_classifier(config)
        return PiClassifier(config, thermal_config, classifier, headers)

    return MotionDetector(
        thermal_config,
        config.tracking.motion.dynamic_thresh,
        CPTVRecorder(thermal_config, headers),
        headers,
    )


def handle_headers(connection):
    headers = ""
    line = ""
    while True:
        data = connection.recv(1).decode()

        line += data
        if data == "\n":
            if line.strip() == "":
                break

            headers += line
            line = ""
    return HeaderInfo.parse_header(headers)


def handle_connection(connection, config, thermal_config):
    headers = handle_headers(connection)
    logging.debug("parsed camera headers", headers)
    processor = get_processor(config, thermal_config, headers)
    service = SnapshotService(processor)

    raw_frame = lepton3.Lepton3(headers)

    while True:
        data = connection.recv(
            headers.frame_size + raw_frame.get_telemetry_size(), socket.MSG_WAITALL
        )
        if not data:
            logging.info("disconnected from camera")
            processor.disconnected()
            service.quit()
            return

        frame = raw_frame.parse(data)

        t_max = np.amax(frame.pix)
        t_min = np.amin(frame.pix)
        if t_max > 10000 or t_min == 0:
            logging.warning(
                "received frame has odd values skipping thermal frame max {} thermal frame min {} cpu % {} memory % {}".format(
                    t_max, t_min, psutil.cpu_percent(), psutil.virtual_memory()[2]
                )
            )
            # this frame has bad data probably from lack of CPU
            processor.skip_frame()
            continue
        processor.process_frame(frame)

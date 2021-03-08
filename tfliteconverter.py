import argparse
import os
import numpy as np

import shutil
import tensorflow as tf
from config.config import Config
from ml_tools.dataset import dataset_db_path
import pickle
from ml_tools.model import Model
from model_crnn import ModelCRNN_HQ, Model_CNN

MODEL_DIR = "../cptv-download/train/checkpoints"
MODEL_NAME = "training-most-recent.sav"
SAVED_DIR = "saved_model"
LITE_MODEL_NAME = "converted_model.tflite"

input_map = {"state_in:0": 0, "X:0": 1}

# some reason neural compute stick doesn't like the other frozen model
def optimizer_model(args):
    save_eval_model(args)
    loaded_graph = tf.Graph()
    with tf.Session(graph=loaded_graph) as sess:
        # import tensorflow as tf
        from tensorflow.python.framework import graph_io

        saver = tf.compat.v1.train.import_meta_graph(
            os.path.join(args.model_dir, "eval-model") + ".meta", clear_devices=True
        )
        saver.restore(sess, os.path.join(args.model_dir, "eval-model"))
        frozen = tf.graph_util.convert_variables_to_constants(
            sess, sess.graph_def, ["state_out", "prediction"]
        )
        graph_io.write_graph(frozen, "./", "inference_graph.pb", as_text=False)


# this removes training attributes from the model and resaves and sets
# frame count to 1 as is needed for tflite
def save_eval_model(args):
    config = Config.load_from_file()
    datasets_filename = dataset_db_path(config)
    with open(datasets_filename, "rb") as f:
        dsets = pickle.load(f)

    labels = ["hedgehog", "false-positive", "possum", "rodent", "bird"]

    # this needs to be the same as the source model class
    model = ModelCRNN_HQ(
        labels=len(labels),
        train_config=config.train,
        training=True,
        tflite=True,
        **config.train.hyper_params,
    )
    model.saver = tf.compat.v1.train.Saver(max_to_keep=1000)
    model.restore_params(os.path.join(args.model_dir, args.model_name))

    model.save(os.path.join(args.model_dir, "eval-model"))
    model.setup_summary_writers("convert")


def freeze_model(args):
    print("freezing: ", os.path.join(args.model_dir, args.model_name))

    save_eval_model(args)
    loaded_graph = tf.Graph()
    with tf.compat.v1.Session(graph=loaded_graph) as sess:

        saver = tf.compat.v1.train.import_meta_graph(
            os.path.join(args.model_dir, "eval-model") + ".meta", clear_devices=True
        )
        saver.restore(sess, os.path.join(args.model_dir, "eval-model"))

        try:
            from tensorflow.compat.v1.saved_model import simple_save
        except (ModuleNotFoundError, ImportError):
            from tensorflow.saved_model import simple_save

        in_names = ["X:0", "state_in:0"]
        out_names = ["state_out:0", "prediction:0", "novelty:0"]
        inputs = {}
        outputs = {}
        for name in in_names:
            inputs[name] = loaded_graph.get_tensor_by_name(name)
        inputs["X:0"].set_shape([1, 1, 5, 48, 48])

        for name in out_names:
            outputs[name] = loaded_graph.get_tensor_by_name(name)

        # complains if directory isn't empty
        if os.path.exists(os.path.join(args.model_dir, SAVED_DIR)):
            shutil.rmtree(os.path.join(args.model_dir, SAVED_DIR))

        simple_save(sess, os.path.join(args.model_dir, SAVED_DIR), inputs, outputs)


def run_model(args):
    interpreter = tf.lite.Interpreter(
        model_path=os.path.join(args.model_dir, args.tflite_name)
    )
    interpreter.allocate_tensors()
    input_details = interpreter.get_tensor_details()

    in_values = {}
    for detail in input_details:
        in_values[detail["name"]] = detail["index"]
    # print(in_values)
    output_details = interpreter.get_output_details()
    input_shape = input_details[in_values["input_1"]]["shape"]

    input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)
    interpreter.set_tensor(in_values["input_1"], input_data)
    interpreter.invoke()
    print("model pass 1")
    out_values = {}
    for detail in output_details:
        out_values[detail["name"]] = interpreter.get_tensor(detail["index"])
    print(out_values)
    print("pred", out_values["Identity"])
    input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)
    interpreter.set_tensor(in_values["input_1"], input_data)
    interpreter.set_tensor(in_values["state_in"], out_values["state_out"])
    interpreter.invoke()
    print("model pass 2")
    print("pred", out_values["prediction"])


def representative_dataset_gen():
    config = Config.load_from_file()

    dataset_filename = os.path.join(config.tracks_folder, "datasets.dat")
    datasets = pickle.load(open(dataset_filename, "rb"))
    train, validation, test = datasets
    num_calibration_steps = 1000
    for i in range(num_calibration_steps):
        X, y = train.next_batch(1)
        feed_dict = get_feed_dict(X)

        yield feed_dict


def get_feed_dict(X, state_in=None):
    """
    Returns a feed dictionary for TensorFlow placeholders.
    :param X: The examples to classify
    :param state_in: (optional) states from previous classification.  Used to maintain internal state across runs
    :return:
    """
    result = [None] * len(input_map)
    if state_in is None:
        result[input_map["state_in:0"][0]] = np.float32(np.zeros((1, 512, 2)))
    else:
        result[input_map["state_in:0"][0]] = np.float32(state_in)
    result[input_map["X:0"][0]] = np.float32(X[:, 0:1])

    return result


def convert_model(args):
    print("converting to tflite: ", os.path.join(args.model_dir, SAVED_DIR))

    model = tf.keras.models.load_model(args.model_dir)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    open(os.path.join(args.model_dir, args.tflite_name), "wb").write(tflite_model)
    return
    model = tf.saved_model.load(
        export_dir=os.path.join(args.model_dir, SAVED_DIR), tags=None
    )
    concrete_func = model.signatures[tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY]

    inputs = concrete_func.inputs[:2]
    # gets input shape and the order of the inputs as it seems to be random
    for i, val in enumerate(inputs):
        input_map[val.name] = (i, val.shape)
    concrete_func.inputs[input_map["X:0"][0]].set_shape([1, 1, 5, 48, 48])

    converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])

    # be good to get this going but throws error
    #         return _tensorflow_wrap_interpreter_wrapper.InterpreterWrapper_AllocateTensors(self)
    # RuntimeError: tensorflow/lite/kernels/maximum_minimum.cc:54 op_context.input1->type != op_context.input2->type (9 != 1)Node number 15 (MINIMUM) failed to prepare
    # converter.optimizations = [tf.lite.Optimize.DEFAULT]
    # converter.representative_dataset = representative_dataset_gen

    converter.post_training_quantize = True
    converter.experimental_new_converter = True
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
    tflite_model = converter.convert()
    open(os.path.join(args.model_dir, args.tflite_name), "wb").write(tflite_model)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-f", "--freeze", action="store_true", help="freeze saved model to .pb format"
    )
    parser.add_argument(
        "-c", "--convert", action="store_true", help="Convert frozen model to tflite"
    )
    parser.add_argument(
        "-r",
        "--run",
        action="store_true",
        help="Test converted model with random data using tflite interpreter",
    )
    parser.add_argument(
        "--model_dir",
        default=MODEL_DIR,
        help="Directory where meta data of the model you want to convert is stored",
    )
    parser.add_argument(
        "--model_name",
        default=MODEL_NAME,
        help="Name of the model to convert <name>.sav",
    )
    parser.add_argument(
        "--tflite_name",
        default=LITE_MODEL_NAME,
        help="Name to save converted tflite model under, also used to run model",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.freeze:
        # optimizer_model(args)
        freeze_model(args)
    if args.convert:
        convert_model(args)
    if args.run:
        run_model(args)


if __name__ == "__main__":
    main()

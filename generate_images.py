from multiprocessing import Process, Queue

from PIL import Image
import numpy as np
import pickle
from dateutil.parser import parse
import argparse
import logging
import os
import sys
from config.config import Config
from datetime import datetime

from ml_tools.newmodel import NewModel
from ml_tools.trackdatabase import TrackDatabase


def save_job(queue, dataset, folder, labels_dir):

    i = 0
    while True:
        i += 1
        track = queue.get()
        if track == "DONE":
            break
        else:
            save_track(dataset, track, folder, labels_dir)
        if i % 50 == 0:
            logging.debug("%s jobs left", queue.qsize())


def normalize(data, new_max=1):
    max = np.amax(data)
    min = np.amin(data[data > 0])
    data -= min
    data = data / (max - min) * new_max
    return data


def save_track(dataset, track, folder, labels_dir, ext="png"):
    track_data = dataset.fetch_track(track, original=True, preprocess=False)
    clip_dir = os.path.join(folder, str(track.clip_id))
    if not os.path.isdir(clip_dir):
        logging.debug("Creating %s", clip_dir)
        os.mkdir(clip_dir)

    for i, thermal in enumerate(track_data):
        normed = normalize(thermal, new_max=255)
        img = Image.fromarray(np.uint8(normed))
        filename = "{}-{}.{}".format(track.clip_id, i + track.start_frame, ext)
        logging.debug("Saving %s", os.path.join(clip_dir, filename))

        img.save(os.path.join(clip_dir, filename))
    save_metadata(track, labels_dir)


def save_metadata(track, folder):
    meta_dir = os.path.join(folder, track.label)
    if not os.path.isdir(meta_dir):
        logging.debug("Creating %s", meta_dir)
        os.mkdir(meta_dir)
    filename = "{}-{}.txt".format(track.clip_id, track.track_id)
    fullpath = os.path.join(meta_dir, filename)
    with open(fullpath, "w") as f:
        f.write(track.toJSON())


def save_all(dataset, worker_threads, folder):
    if not os.path.isdir(folder):
        logging.debug("Creating %s", folder)
        os.mkdir(folder)
    labels_dir = os.path.join(folder + "-labels")
    if not os.path.isdir(labels_dir):
        logging.debug("Creating %s", labels_dir)
        os.mkdir(labels_dir)
    job_queue = Queue()
    processes = []
    for i in range(max(1, worker_threads)):
        p = Process(
            target=save_job,
            args=(job_queue, dataset, folder, labels_dir),
        )
        processes.append(p)
        p.start()

    for track in dataset.tracks:
        job_queue.put(track)

    logging.debug("Processing %d", len(dataset.tracks))
    for i in range(len(processes)):
        job_queue.put("DONE")
    for process in processes:
        process.join()


def load_dataset(dataset_file):
    datasets = pickle.load(open(dataset_file, "rb"))
    return datasets


def load_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-f", "--base-folder", help="Base folder to save, train test and val"
    )

    parser.add_argument("-c", "--config-file", help="Path to config file to use")
    args = parser.parse_args()
    return args


def init_logging(timestamps=False):
    """Set up logging for use by various classifier pipeline scripts.

    Logs will go to stderr.
    """

    fmt = "%(levelname)7s %(message)s"
    if timestamps:
        fmt = "%(asctime)s " + fmt
    logging.basicConfig(
        stream=sys.stderr, level=logging.DEBUG, format=fmt, datefmt="%Y-%m-%d %H:%M:%S"
    )


args = load_args()
init_logging()
config = Config.load_from_file(args.config_file)
datasets_filename = os.path.join(config.tracks_folder, "datasets.dat")
datasets = load_dataset(datasets_filename)
base_dir = "."
if args.base_folder:
    base_dir = args.base_folder
if not os.path.isdir(base_dir):
    logging.debug("Creating %s", base_dir)
    os.mkdir(base_dir)
for dataset in datasets:
    save_all(dataset, config.worker_threads, os.path.join(base_dir, dataset.name))

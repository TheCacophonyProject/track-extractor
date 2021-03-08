import pickle
import math
import logging
import tensorflow.keras as keras
import numpy as np
import multiprocessing
import time
from ml_tools import tools
from ml_tools.preprocess import preprocess_movement, preprocess_frame

FRAMES_PER_SECOND = 9


class GeneartorParams:
    def __init__(self, output_dim, params):
        self.augment = params.get("augment", False)
        self.use_dots = params.get("use_dots", False)
        self.use_movement = params.get("use_movement")
        self.model_preprocess = params.get("model_preprocess")
        self.channel = params.get("channel")
        self.square_width = params.get("square_width", 5)
        self.output_dim = output_dim


class DataGenerator(keras.utils.Sequence):
    "Generates data for Keras"

    def __init__(
        self,
        dataset,
        labels,
        output_dim,
        **params,
    ):
        self.params = GeneartorParams(output_dim, params)
        self.use_previous_epoch = None
        self.labels = labels
        self.dataset = dataset
        self.preload = params.get("preload", True)
        self.epochs = params.get("epochs", 10)
        self.samples = None
        self.keep_epoch = params.get("keep_epoch")
        self.randomize_epoch = params.get("randomize_epoch", True)
        self.cap_at = params.get("cap_at")
        self.cap_samples = params.get("cap_samples", True)
        self.label_cap = params.get("label_cap")
        self.shuffle = params.get("shuffle", True)
        self.batch_size = params.get("batch_size", 16)

        self.cur_epoch = 0
        self.loaded_epochs = 0
        self.epoch_data = []
        self.epoch_stats = []
        if self.preload:
            self.load_queue = multiprocessing.Queue()
        if self.preload:
            self.preloader_queue = multiprocessing.Queue(params.get("buffer_size", 128))

        # load epoch
        self.load_next_epoch()
        self.epoch_stats.append({})
        self.epoch_data.append(([None] * len(self), [None] * len(self)))

        if self.preload:
            self.preloader_threads = [
                multiprocessing.Process(
                    target=preloader,
                    args=(
                        self.preloader_queue,
                        self.load_queue,
                        self.labels,
                        self.dataset,
                        self.params,
                    ),
                )
                for _ in range(params.get("load_threads", 2))
            ]
            for thread in self.preloader_threads:
                thread.start()
        logging.info(
            "datagen for %s shuffle %s cap %s",
            self.dataset.name,
            self.shuffle,
            self.cap_samples,
        )

    def stop_load(self):
        if not self.preload:
            return
        for thread in self.preloader_threads:
            if hasattr(thread, "terminate"):
                thread.terminate()
            else:
                thread.exit()
        self.preloader_queue = None

    def get_epoch_predictions(self, epoch=-1):
        if self.keep_epoch:
            return self.epoch_data[epoch][1]
        return None

    def __len__(self):
        "Denotes the number of batches per epoch"

        return int(math.ceil(len(self.samples) / self.batch_size))

    def __getitem__(self, index):
        "Generate one batch of data"
        # Generate indexes of the batch
        logging.debug("%s requsting index %s", self.dataset.name, index)
        if self.keep_epoch and self.use_previous_epoch is not None:
            X = self.epoch_data[self.use_previous_epoch][0][
                index * self.batch_size : (index + 1) * self.batch_size
            ]

            y = self.epoch_data[self.use_previous_epoch][1][
                index * self.batch_size : (index + 1) * self.batch_size
            ]
        else:
            if index == 0 and self.epoch_data[self.cur_epoch][0][0] is not None:
                # when tensorflow uses model.fit it requests index 0 twice
                X = self.epoch_data[self.cur_epoch][0][index]
                y = self.epoch_data[self.cur_epoch][1][index]
            elif self.preload:
                X, y, y_original = self.preloader_queue.get()
            else:
                X, y, y_original = self.loadbatch(index)

            if self.epoch_data[self.cur_epoch][0][index] is None:
                epoch_stats = self.epoch_stats[self.cur_epoch]
                out_y = np.argmax(y, axis=1)
                indices, counts = np.unique(out_y, return_counts=True)
                for lbl in y_original:
                    epoch_stats.setdefault(lbl, 0)
                    epoch_stats[lbl] += 1
                # for i, label_index in enumerate(indices):
                #     label = self.labels[label_index]
                #     count = counts[i]
                #     epoch_stats.setdefault(label, 0)
                #     epoch_stats[label] += count
        # always keep a copy of epoch data
        if index == 0 or (self.keep_epoch and self.use_previous_epoch is None):
            self.epoch_data[self.cur_epoch][0][index] = X
            self.epoch_data[self.cur_epoch][1][index] = y

        # can start loading next epoch of training before validation
        # if (index + 1) == len(self):
        #     self.load_next_epoch(True)
        return X, y

    def resuse_previous_epoch(self):
        """
        This is used in hyper training to speeed it up just load one epoch into memory
        and shuffle each time
        """
        if self.use_previous_epoch is None:
            # [batch, segment, height, width, rgb]
            X = [item for batch in self.epoch_data[self.cur_epoch][0] for item in batch]
            y = [item for batch in self.epoch_data[self.cur_epoch][1] for item in batch]
        else:
            # [segment, height, width, rgb]

            X = self.epoch_data[self.use_previous_epoch][0]
            y = self.epoch_data[self.use_previous_epoch][1]
        if self.shuffle:
            X = np.asarray(X)
            y = np.asarray(y)
            indices = np.arange(len(X))
            np.random.shuffle(indices)
            X = X[indices]
            y = y[indices]
        self.epoch_data[self.cur_epoch] = None
        self.epoch_data[0] = (X, y)
        self.use_previous_epoch = 0
        logging.info("Reusing previous epoch data for epoch %s", self.cur_epoch)
        self.stop_load()

    def load_next_epoch(self, reuse=False):
        if self.loaded_epochs >= self.epochs:
            return
        if self.randomize_epoch is False and reuse:
            return self.resuse_previous_epoch()

        else:
            self.samples = self.dataset.epoch_samples(
                cap_samples=self.cap_samples,
                replace=False,
                random=self.randomize_epoch,
                cap_at=self.cap_at,
                label_cap=self.label_cap,
            )
            self.samples = [sample.id for sample in self.samples]

            if self.shuffle:
                np.random.shuffle(self.samples)
        if self.preload:
            for index in range(len(self)):
                samples = self.samples[
                    index * self.batch_size : (index + 1) * self.batch_size
                ]
                pickled_samples = pickle.dumps((self.loaded_epochs + 1, samples))
                self.load_queue.put(pickled_samples)
        self.loaded_epochs += 1

    def on_epoch_end(self):
        "Updates indexes after each epoch"
        self.load_next_epoch(reuse=True)
        batches = len(self)
        if not self.keep_epoch:
            # zero last epoch
            self.epoch_data[-1] = None

        last_stats = self.epoch_stats[-1]
        self.epoch_stats.append({})
        self.epoch_data.append(([None] * batches, [None] * batches))

        logging.info("epoch ended for %s %s", self.dataset.name, last_stats)
        self.cur_epoch += 1


def loadbatch(labels, dataset, samples, params):
    start = time.time()
    # samples = self.samples[index * self.batch_size : (index + 1) * self.batch_size]
    X, y, y_orig = _data(labels, dataset, samples, params)

    logging.debug("%s  Time to get data %s", dataset.name, time.time() - start)
    return X, y, y_orig


def _data(labels, dataset, samples, params, to_categorical=True):
    "Generates data containing batch_size samples"
    # Initialization
    X = np.empty(
        (
            len(samples),
            *params.output_dim,
        )
    )
    y = np.empty((len(samples)), dtype=int)
    data_i = 0
    y_original = []
    for sample in samples:
        label = dataset.mapped_label(sample.label)
        if label not in labels:
            continue
        if params.use_movement:
            try:
                frame_data = dataset.fetch_random_sample(sample)
                overlay = dataset.db.get_overlay(
                    sample.track.clip_id, sample.track.track_id
                )

            except Exception as inst:
                logging.error("Error fetching sample %s %s", sample, inst)
                raise inst
                continue

            if len(frame_data) < 5:
                logging.error(
                    "Important frames filtered for %s %s / %s",
                    sample,
                    len(frame_data),
                    len(sample.track.important_frames),
                )
                continue

            # repeat some frames if need be
            if len(frame_data) < params.square_width ** 2:
                missing = params.square_width ** 2 - len(frame_data)
                indices = np.arange(len(frame_data))
                np.random.shuffle(indices)
                for frame_i in indices[:missing]:
                    frame_data.append(frame_data[frame_i].copy())
            ref = []
            frame_data = sorted(
                frame_data, key=lambda frame_data: frame_data.frame_number
            )

            for frame in frame_data:
                ref.append(sample.track.frame_temp_median[frame.frame_number])
            data = preprocess_movement(
                None,
                frame_data,
                params.square_width,
                None,
                params.channel,
                preprocess_fn=params.model_preprocess,
                augment=params.augment,
                use_dots=params.use_dots,
                reference_level=ref,
                sample=sample,
                overlay=overlay,
            )
        else:
            try:
                data = dataset.fetch_sample(
                    sample, augment=params.augment, channels=params.channel
                )

            except Exception as inst:
                logging.error("Error fetching samples %s %s", sample, inst)
                continue

            data = preprocess_frame(
                data,
                params.output_dim,
                preprocess_fn=params.model_preprocess,
                sample=sample,
            )
        if data is None:
            logging.debug(
                "error pre processing frame (i.e.max and min are the same)sample %s",
                sample,
            )
            continue
        y_original.append(sample.label)
        X[data_i] = data
        y[data_i] = labels.index(label)
        data_i += 1
    # remove data that was null
    X = X[:data_i]
    y = y[:data_i]
    if to_categorical:
        y = keras.utils.to_categorical(y, num_classes=len(labels))
    return np.array(X), y, y_original


# continue to read examples until queue is full
def preloader(q, load_queue, labels, dataset, params):
    """ add a segment into buffer """
    logging.info(
        " -started async fetcher for %s augment=%s",
        dataset.name,
        params.augment,
    )
    while True:
        if not q.full():
            samples = pickle.loads(load_queue.get())
            # datagen.loaded_epochs = samples[0]
            data = []
            for sample_id in samples[1]:
                if params.use_movement:
                    data.append(dataset.segments_by_id[sample_id])
                else:
                    data.append(dataset.frames_by_id[sample_id])

            q.put(loadbatch(labels, dataset, data, params))

        else:
            time.sleep(0.1)

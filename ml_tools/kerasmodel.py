import itertools
import io
import math
import tensorflow as tf
import pickle
import logging
from tensorboard.plugins.hparams import api as hp
from ml_tools import tools

from collections import namedtuple
from ml_tools.datagenerator import DataGenerator
from ml_tools.preprocess import (
    preprocess_frame,
    preprocess_movement,
)
import tensorflow.keras as keras
from tensorflow.keras.layers import *
import numpy as np
import os
import time
import matplotlib.pyplot as plt
import json
from ml_tools.imageprocessing import filtered_is_valid
from classify.trackprediction import TrackPrediction
from sklearn.metrics import confusion_matrix

from ml_tools.hyperparams import HyperParams
import tensorflow_addons as tfa

#
HP_DENSE_SIZES = hp.HParam("dense_sizes", hp.Discrete(["1024 512", ""]))
HP_TYPE = hp.HParam("type", hp.Discrete([14]))

HP_BATCH_SIZE = hp.HParam("batch_size", hp.Discrete([32]))
HP_OPTIMIZER = hp.HParam("optimizer", hp.Discrete(["adam"]))
HP_LEARNING_RATE = hp.HParam("learning_rate", hp.Discrete([0.01, 0.001, 0.0001]))
HP_EPSILON = hp.HParam("epislon", hp.Discrete([1e-7]))  # 1.0 and 0.1 for inception
HP_RETRAIN = hp.HParam("retrain_layer", hp.Discrete([758, 742, -1]))
HP_DROPOUT = hp.HParam("dropout", hp.Discrete([0.1]))

METRIC_ACCURACY = "accuracy"
METRIC_LOSS = "loss"


class KerasModel:
    """ Defines a deep learning model """

    MODEL_NAME = "keras model"
    MODEL_DESCRIPTION = "Using pre trained keras application models"
    VERSION = "0.3.0"

    def __init__(self, train_config=None, labels=None):
        self.model = None
        self.datasets = None
        # dictionary containing current hyper parameters
        self.params = HyperParams()
        if train_config:
            self.log_base = os.path.join(train_config.train_dir, "logs")
            self.log_dir = self.log_base
            os.makedirs(self.log_base, exist_ok=True)
            self.checkpoint_folder = os.path.join(train_config.train_dir, "checkpoints")
            self.params.update(train_config.hyper_params)
        self.labels = labels
        self.preprocess_fn = None
        self.validate = None
        self.train = None
        self.mapped_labels = None
        self.label_probabilities = None

    def base_model(self, input_shape, weights="imagenet"):
        pretrained_model = self.params.model
        if pretrained_model == "resnet":
            return (
                tf.keras.applications.ResNet50(
                    weights=weights,
                    include_top=False,
                    input_shape=input_shape,
                ),
                tf.keras.applications.resnet.preprocess_input,
            )
        elif pretrained_model == "resnetv2":
            return (
                tf.keras.applications.ResNet50V2(
                    weights=weights, include_top=False, input_shape=input_shape
                ),
                tf.keras.applications.resnet_v2.preprocess_input,
            )
        elif pretrained_model == "resnet152":
            return (
                tf.keras.applications.ResNet152(
                    weights=weights, include_top=False, input_shape=input_shape
                ),
                tf.keras.applications.resnet.preprocess_input,
            )
        elif pretrained_model == "vgg16":
            return (
                tf.keras.applications.VGG16(
                    weights=weights,
                    include_top=False,
                    input_shape=input_shape,
                ),
                tf.keras.applications.vgg16.preprocess_input,
            )
        elif pretrained_model == "vgg19":
            return (
                tf.keras.applications.VGG19(
                    weights=weights,
                    include_top=False,
                    input_shape=input_shape,
                ),
                tf.keras.applications.vgg19.preprocess_input,
            )
        elif pretrained_model == "mobilenet":
            return (
                tf.keras.applications.MobileNetV2(
                    weights=weights,
                    include_top=False,
                    input_shape=input_shape,
                ),
                tf.keras.applications.mobilenet_v2.preprocess_input,
            )
        elif pretrained_model == "densenet121":
            return (
                tf.keras.applications.DenseNet121(
                    weights=weights,
                    include_top=False,
                    input_shape=input_shape,
                ),
                tf.keras.applications.densenet.preprocess_input,
            )
        elif pretrained_model == "inceptionresnetv2":
            return (
                tf.keras.applications.InceptionResNetV2(
                    weights=weights,
                    include_top=False,
                    input_shape=input_shape,
                ),
                tf.keras.applications.inception_resnet_v2.preprocess_input,
            )
        elif pretrained_model == "inceptionv3":
            return (
                tf.keras.applications.InceptionV3(
                    weights=weights,
                    include_top=False,
                    input_shape=input_shape,
                ),
                tf.keras.applications.inception_v3.preprocess_input,
            )
        raise Exception("Could not find model" + pretrained_model)

    def get_preprocess_fn(self):
        pretrained_model = self.params.model
        if pretrained_model == "resnet":
            return tf.keras.applications.resnet.preprocess_input

        elif pretrained_model == "resnetv2":
            return tf.keras.applications.resnet_v2.preprocess_input

        elif pretrained_model == "resnet152":
            return tf.keras.applications.resnet.preprocess_input

        elif pretrained_model == "vgg16":
            return tf.keras.applications.vgg16.preprocess_input

        elif pretrained_model == "vgg19":
            return tf.keras.applications.vgg19.preprocess_input

        elif pretrained_model == "mobilenet":
            return tf.keras.applications.mobilenet_v2.preprocess_input

        elif pretrained_model == "densenet121":
            return tf.keras.applications.densenet.preprocess_input

        elif pretrained_model == "inceptionresnetv2":
            return tf.keras.applications.inception_resnet_v2.preprocess_input
        elif pretrained_model == "inceptionv3":
            return tf.keras.applications.inception_v3.preprocess_input
        return None

    def build_model(self, dense_sizes=None, retrain_from=None, dropout=None):

        width = self.params.frame_size
        if self.params.use_movement:
            width = self.params.square_width * self.params.frame_size
        inputs = tf.keras.Input(shape=(width, width, 3), name="input")
        weights = None if self.params.base_training else "imagenet"
        base_model, preprocess = self.base_model((width, width, 3), weights=weights)
        self.preprocess_fn = preprocess
        x = base_model(inputs, training=self.params.base_training)  # IMPORTANT

        if self.params.lstm:
            x = tf.keras.layers.GlobalAveragePooling2D()(x)
            for i in dense_sizes:
                x = tf.keras.layers.Dense(i, activation="relu")(x)
            # gp not sure how many should be pre lstm, and how many post
            cnn = tf.keras.models.Model(inputs, outputs=x)

            self.model = self.add_lstm(cnn)
        else:
            x = tf.keras.layers.GlobalAveragePooling2D()(x)
            if self.params.mvm:
                mvm_inputs = Input((45, 9))
                inputs = [inputs, mvm_inputs]

                mvm_features = Flatten()(mvm_inputs)
                x = Concatenate()([x, mvm_features])

                print(mvm_features.shape)
                # mvm_inputs = tf.reshape(mvm_inputs, [None, 45 * 9])
                # mvm_inputs = Flatten()(mvm_inputs)
                # mvm_features = tf.keras.layers.Dense(i, activation="relu")(x)
                # x = GlobalAveragePooling1D()(x)

            # x = Flatten(x)
            for i in dense_sizes:
                x = tf.keras.layers.Dense(i, activation="relu")(x)
                if dropout:
                    x = tf.keras.layers.Dropout(dropout)(x)

            preds = tf.keras.layers.Dense(
                len(self.labels), activation="softmax", name="prediction"
            )(x)
            self.model = tf.keras.models.Model(inputs, outputs=preds)

        if retrain_from is None:
            retrain_from = self.params.retrain_layer
        if retrain_from:
            for i, layer in enumerate(base_model.layers):
                if isinstance(layer, tf.keras.layers.BatchNormalization):
                    # apparently this shouldn't matter as we set base_training = False
                    layer.trainable = False
                    logging.info("dont train %s %s", i, layer.name)
                else:
                    layer.trainable = i >= retrain_from
        else:
            base_model.trainable = self.params.base_training

        self.model.summary()

        self.model.compile(
            optimizer=self.optimizer(),
            loss=self.loss(),
            metrics=[
                "accuracy",
                tf.keras.metrics.AUC(),
                tf.keras.metrics.Recall(),
                tf.keras.metrics.Precision(),
            ],
        )

    def loss(self):
        softmax = tf.keras.losses.CategoricalCrossentropy(
            label_smoothing=self.params.label_smoothing,
        )
        return softmax

    def optimizer(self):
        if self.params.learning_rate_decay != 1.0:
            learning_rate = tf.compat.v1.train.exponential_decay(
                self.params.learning_rate,
                self.global_step,
                1000,
                self.params["learning_rate_decay"],
                staircase=True,
            )
            tf.compat.v1.summary.scalar("params/learning_rate", learning_rate)
        else:
            learning_rate = self.params.learning_rate  # setup optimizer
        if learning_rate:
            optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        else:
            optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        return optimizer

    def load_weights(self, model_path, meta=True, training=False):
        logging.info("loading weights %s", model_path)
        dir = os.path.dirname(model_path)

        if meta:
            self.load_meta(dir)

        if not self.model:
            self.build_model(
                dense_sizes=self.params.dense_sizes,
                retrain_from=self.params.retrain_layer,
                dropout=self.params.dropout,
            )
        if not training:
            self.model.trainable = False
        self.model.summary()
        self.model.load_weights(dir + "/variables/variables")

    def load_model(self, model_path, training=False):
        logging.info("Loading %s", model_path)
        dir = os.path.dirname(model_path)
        self.model = tf.keras.models.load_model(dir)
        self.load_meta(dir)
        if not training:
            self.model.trainable = False
        self.model.summary()

    def load_meta(self, dir):
        meta = json.load(open(os.path.join(dir, "metadata.txt"), "r"))
        self.params = HyperParams()
        self.params.update(meta["hyperparams"])
        self.labels = meta["labels"]
        self.mapped_labels = meta.get("mapped_labels")
        self.label_probabilities = meta.get("label_probabilities")
        self.preprocess_fn = self.get_preprocess_fn()
        self.type = meta.get("type")

    def save(self, run_name=MODEL_NAME, history=None, test_results=None):
        # create a save point
        self.model.save(
            os.path.join(self.checkpoint_folder, run_name), save_format="tf"
        )
        self.save_metadata(run_name, history, test_results)

    def save_metadata(self, run_name=MODEL_NAME, history=None, test_results=None):
        #  save metadata
        model_stats = {}
        model_stats["name"] = self.MODEL_NAME
        model_stats["description"] = self.MODEL_DESCRIPTION
        model_stats["labels"] = self.labels
        model_stats["hyperparams"] = self.params
        model_stats["training_date"] = str(time.time())
        model_stats["version"] = self.VERSION
        model_stats["mapped_labels"] = self.mapped_labels
        model_stats["label_probabilities"] = self.label_probabilities

        if history:
            model_stats["history"] = history.history
        if test_results:
            model_stats["test_loss"] = test_results[0]
            model_stats["test_acc"] = test_results[1]
        run_dir = os.path.join(self.checkpoint_folder, run_name)
        if not os.path.exists(run_dir):
            os.mkdir(run_dir)
        json.dump(
            model_stats,
            open(
                os.path.join(run_dir, "metadata.txt"),
                "w",
            ),
            indent=4,
        )
        best_loss = os.path.join(run_dir, "val_loss")
        if os.path.exists(best_loss):
            json.dump(
                model_stats,
                open(
                    os.path.join(best_loss, "metadata.txt"),
                    "w",
                ),
                indent=4,
            )
        best_acc = os.path.join(run_dir, "val_acc")
        if os.path.exists(best_acc):
            json.dump(
                model_stats,
                open(
                    os.path.join(best_acc, "metadata.txt"),
                    "w",
                ),
                indent=4,
            )

    def close(self):
        if self.validate:
            self.validate.stop_load()
        if self.train:
            self.train.stop_load()

    def train_model(self, epochs, run_name):
        self.log_dir = os.path.join(self.log_base, run_name)
        os.makedirs(self.log_base, exist_ok=True)

        if not self.model:
            self.build_model(
                dense_sizes=self.params.dense_sizes,
                retrain_from=self.params.retrain_layer,
                dropout=self.params.dropout,
            )
        self.train = DataGenerator(
            self.datasets.train,
            self.labels,
            self.params.output_dim,
            augment=True,
            buffer_size=self.params.buffer_size,
            epochs=epochs,
            batch_size=self.params.batch_size,
            channel=self.params.channel,
            shuffle=self.params.shuffle,
            model_preprocess=self.preprocess_fn,
            load_threads=self.params.train_load_threads,
            use_movement=self.params.use_movement,
            cap_at="bird",
            square_width=self.params.square_width,
            mvm=self.params.mvm,
        )
        self.validate = DataGenerator(
            self.datasets.validation,
            self.labels,
            self.params.output_dim,
            batch_size=self.params.batch_size,
            buffer_size=self.params.buffer_size,
            channel=self.params.channel,
            shuffle=self.params.shuffle,
            model_preprocess=self.preprocess_fn,
            epochs=epochs,
            load_threads=1,
            use_movement=self.params.use_movement,
            cap_at="bird",
            square_width=self.params.square_width,
            mvm=self.params.mvm,
        )
        checkpoints = self.checkpoints(run_name)

        self.save_metadata(run_name)

        weight_for_0 = 1
        weight_for_1 = 1 / 4

        class_weight = {0: weight_for_0, 1: weight_for_1}

        history = self.model.fit(
            self.train,
            validation_data=self.validate,
            epochs=epochs,
            shuffle=False,
            # class_weight=class_weight,
            callbacks=[
                tf.keras.callbacks.TensorBoard(
                    self.log_dir, write_graph=True, write_images=True
                ),
                *checkpoints,
            ],  # log metrics
        )
        self.validate.stop_load()
        self.train.stop_load()
        test_accuracy = None
        if self.datasets.test and self.datasets.test.has_data():
            test = DataGenerator(
                self.datasets.test,
                self.datasets.train.labels,
                self.params.output_dim,
                batch_size=self.params.batch_size,
                channel=self.params.channel,
                use_movement=self.params.use_movement,
                shuffle=True,
                model_preprocess=self.preprocess_fn,
                epochs=1,
                load_threads=self.params.train_load_threads,
                cap_at="bird",
                square_width=self.params.square_width,
                mvm=self.params.mvm,
            )
            test_accuracy = self.model.evaluate(test)
            test.stop_load()
            logging.info("Test accuracy is %s", test_accuracy)
        self.save(run_name, history=history, test_results=test_accuracy)

    def checkpoints(self, run_name):
        val_loss = os.path.join(self.checkpoint_folder, run_name, "val_loss")

        checkpoint_loss = tf.keras.callbacks.ModelCheckpoint(
            val_loss,
            monitor="val_loss",
            verbose=1,
            save_best_only=True,
            save_weights_only=False,
            mode="auto",
        )
        val_acc = os.path.join(self.checkpoint_folder, run_name, "val_acc")

        checkpoint_acc = tf.keras.callbacks.ModelCheckpoint(
            val_acc,
            monitor="val_accuracy",
            verbose=1,
            save_best_only=True,
            save_weights_only=False,
            mode="max",
        )

        val_precision = os.path.join(self.checkpoint_folder, run_name, "val_recall")

        checkpoint_recall = tf.keras.callbacks.ModelCheckpoint(
            val_precision,
            monitor="val_recall",
            verbose=1,
            save_best_only=True,
            save_weights_only=False,
            mode="max",
        )
        earlyStopping = tf.keras.callbacks.EarlyStopping(patience=10)

        return [
            earlyStopping,
            checkpoint_acc,
        ]

    def regroup(self, shuffle=True):
        if not self.mapped_labels:
            logging.warn("Cant regroup without specifying mapped_labels")
            return
        for fld in self.datasets._fields:
            dataset = getattr(self.datasets, fld)
            dataset.regroup(self.mapped_labels, shuffle=shuffle)
            dataset.labels.sort()
        self.set_labels()

    def set_labels(self):
        # preserve label order if needed, this should be used when retraining
        # on a model already trained with our data
        self.labels = self.datasets.train.labels.copy()

    def import_dataset(self, dataset_filename, ignore_labels=None, lbl_p=None):
        """
        Import dataset.
        :param dataset_filename: path and filename of the dataset
        :param ignore_labels: (optional) these labels will be removed from the dataset.
        :return:
        """
        self.label_probabilities = lbl_p
        self.datasets = namedtuple("Datasets", "train, validation, test")
        datasets = pickle.load(open(dataset_filename, "rb"))
        self.datasets.train, self.datasets.validation, self.datasets.test = datasets
        for dataset in datasets:
            dataset.labels.sort()
            dataset.set_read_only(True)
            dataset.lbl_p = lbl_p
            dataset.use_segments = self.params.use_segments
            # dataset.random_segments_only()
            dataset.recalculate_segments()
            # dataset.rebuild_cdf()
            if ignore_labels:
                for label in ignore_labels:
                    dataset.remove_label(label)
        self.labels = self.datasets.train.labels

        if self.mapped_labels:
            self.regroup()
        logging.info(
            "Training samples: {0:.1f}k".format(self.datasets.train.sample_count / 1000)
        )
        logging.info(
            "Validation samples: {0:.1f}k".format(
                self.datasets.validation.sample_count / 1000
            )
        )
        logging.info(
            "Test samples: {0:.1f}k".format(self.datasets.test.sample_count / 1000)
        )
        logging.info("Labels: {}".format(self.labels))

    # GRID SEARCH
    def train_test_model(self, hparams, log_dir, epochs=5):
        # if not self.model:
        dense_size = hparams[HP_DENSE_SIZES].split()
        retrain_layer = hparams[HP_RETRAIN]
        dropout = hparams[HP_DROPOUT]
        if dropout == 0.0:
            dropout = None
        if retrain_layer == -1:
            retrain_layer = None
        if hparams[HP_DENSE_SIZES] == "":
            dense_size = []
        else:
            for i, size in enumerate(dense_size):
                dense_size[i] = int(size)
        self.train.batch_size = hparams.get(HP_BATCH_SIZE, 32)
        self.validate.batch_size = hparams.get(HP_BATCH_SIZE, 32)
        self.train.loaded_epochs = 0
        self.validate.loaded_epochs = 0
        self.build_model(
            dense_sizes=dense_size,
            retrain_from=retrain_layer,
            dropout=dropout,
        )

        opt = None
        learning_rate = hparams[HP_LEARNING_RATE]
        epsilon = hparams[HP_EPSILON]

        if hparams[HP_OPTIMIZER] == "adam":
            opt = tf.keras.optimizers.Adam(learning_rate=learning_rate, epsilon=epsilon)
        else:
            opt = tf.keras.optimizers.SGD(learning_rate=learning_rate, epsilon=epsilon)
        self.model.compile(
            optimizer=opt,
            loss=self.loss(),
            metrics=["accuracy"],
        )
        history = self.model.fit(
            self.train, epochs=epochs, shuffle=False, validation_data=self.validate
        )

        # _, accuracy = self.model.evaluate(self.validate)
        self.train.stop_load()
        # self.validate.stop_load()
        return history

    def test_hparams(self):

        epochs = 6
        type = 12
        batch_size = 32

        dir = self.log_dir + "/hparam_tuning"
        with tf.summary.create_file_writer(dir).as_default():
            hp.hparams_config(
                hparams=[
                    HP_BATCH_SIZE,
                    HP_DENSE_SIZES,
                    HP_LEARNING_RATE,
                    HP_OPTIMIZER,
                    HP_EPSILON,
                    HP_TYPE,
                    HP_RETRAIN,
                    HP_DROPOUT,
                ],
                metrics=[
                    hp.Metric(METRIC_ACCURACY, display_name="Accuracy"),
                    hp.Metric(METRIC_LOSS, display_name="Loss"),
                ],
            )
        session_num = 0
        hparams = {}
        for batch_size in HP_BATCH_SIZE.domain.values:
            for dense_size in HP_DENSE_SIZES.domain.values:
                for retrain in HP_RETRAIN.domain.values:

                    for learning_rate in HP_LEARNING_RATE.domain.values:
                        for type in HP_TYPE.domain.values:
                            for optimizer in HP_OPTIMIZER.domain.values:
                                for epsilon in HP_EPSILON.domain.values:
                                    for dropout in HP_DROPOUT.domain.values:
                                        hparams = {
                                            HP_DENSE_SIZES: dense_size,
                                            HP_BATCH_SIZE: batch_size,
                                            HP_LEARNING_RATE: learning_rate,
                                            HP_OPTIMIZER: optimizer,
                                            HP_EPSILON: epsilon,
                                            HP_TYPE: type,
                                            HP_RETRAIN: retrain,
                                            HP_DROPOUT: dropout,
                                        }
                                        self.train = DataGenerator(
                                            self.datasets.train,
                                            self.datasets.train.labels,
                                            self.params.output_dim,
                                            batch_size=batch_size,
                                            buffer_size=self.params.buffer_size,
                                            channel=self.params.channel,
                                            model_preprocess=self.preprocess_fn,
                                            load_threads=self.params.train_load_threads,
                                            use_movement=self.params.use_movement,
                                            shuffle=True,
                                            cap_at="bird",
                                            square_width=self.params.square_width,
                                        )
                                        self.validate = DataGenerator(
                                            self.datasets.validation,
                                            self.datasets.train.labels,
                                            self.params.output_dim,
                                            batch_size=batch_size,
                                            buffer_size=self.params.buffer_size,
                                            channel=self.params.channel,
                                            model_preprocess=self.preprocess_fn,
                                            epochs=epochs,
                                            use_movement=self.params.use_movement,
                                            shuffle=True,
                                            cap_at="bird",
                                            square_width=self.params.square_width,
                                        )
                                        run_name = "run-%d" % session_num
                                        print("--- Starting trial: %s" % run_name)
                                        print({h.name: hparams[h] for h in hparams})
                                        self.run(dir + "/" + run_name, hparams)
                                        session_num += 1
                                        self.train.stop_load()
                                        self.validate.stop_load()

    def run(self, log_dir, hparams):

        with tf.summary.create_file_writer(log_dir).as_default():
            hp.hparams(hparams)  # record the values used in this trial
            history = self.train_test_model(hparams, log_dir)
            val_accuracy = history.history["val_accuracy"]
            val_loss = history.history["val_loss"]

            for step, accuracy in enumerate(val_accuracy):
                loss = val_loss[step]
                tf.summary.scalar(METRIC_ACCURACY, accuracy, step=step)
                tf.summary.scalar(METRIC_LOSS, loss, step=step)

    @property
    def hyperparams_string(self):
        """ Returns list of hyperparameters as a string. """
        print(self.params)
        return "\n".join(
            ["{}={}".format(param, value) for param, value in self.params.items()]
        )

    def add_lstm(self, cnn):
        input_layer = tf.keras.Input(shape=(None, *self.params.output_dim))
        encoded_frames = tf.keras.layers.TimeDistributed(cnn)(input_layer)
        lstm_outputs = tf.keras.layers.LSTM(
            self.params["lstm_units"],
            dropout=self.params["keep_prob"],
            return_state=False,
        )(encoded_frames)

        hidden_layer = tf.keras.layers.Dense(1024, activation="relu")(lstm_outputs)
        hidden_layer = tf.keras.layers.Dense(512, activation="relu")(hidden_layer)

        preds = tf.keras.layers.Dense(
            len(self.labels), activation="softmax", name="pred"
        )(hidden_layer)
        model = tf.keras.models.Model(input_layer, preds)
        return model

    def classify_track(self, clip, track, keep_all=True):
        track_data = []
        thermal_median = []
        for i, region in enumerate(track.bounds_history):
            frame = clip.frame_buffer.get_frame(region.frame_number)
            cropped_frame = frame.crop_by_region(region)
            track_data.append(cropped_frame)
            thermal_median.append(np.median(frame.thermal))
        return self.classify_track_data(
            track.get_id(), track_data, thermal_median, regions=track.bounds_history
        )

    def classify_track_data(
        self, track_id, data, thermal_median, keep_all=True, regions=None
    ):
        track_prediction = TrackPrediction(track_id, 0, keep_all)
        if self.params.use_movement:
            predictions = self.classify_frames(data, thermal_median, regions=regions)
            for i, prediction in enumerate(predictions):
                track_prediction.classified_frame(i, prediction, None)
        else:
            for i, frame in enumerate(data):
                prediction = self.classify_frame(frame, thermal_median[i])
                track_prediction.classified_frame(i, prediction, None)

        return track_prediction

    def classify_frames(self, data, thermal_median, preprocess=True, regions=None):
        predictions = []

        filtered_data = []
        valid_indices = []
        valid_regions = []
        for i, frame in enumerate(data):
            if filtered_is_valid(frame, ""):
                filtered_data.append(frame)
                valid_indices.append(i)
                valid_regions.append(regions[i])
        frame_sample = valid_indices
        frame_sample.extend(valid_indices)
        np.random.shuffle(frame_sample)
        frames_per_classify = self.params.square_width ** 2
        frames = len(filtered_data)

        n_squares = 3 * math.ceil(float(frames) / frames_per_classify)
        median = np.zeros((frames_per_classify))

        for i in range(n_squares):
            square_data = filtered_data
            seg_frames = frame_sample[:frames_per_classify]
            if len(seg_frames) == 0:
                break
            segment = []
            median = np.zeros((len(seg_frames)))
            # update remaining
            frame_sample = frame_sample[frames_per_classify:]
            seg_frames.sort()
            for i, frame_i in enumerate(seg_frames):
                f = data[frame_i]
                segment.append(f.copy())
                median[i] = thermal_median[frame_i]

            frames = preprocess_movement(
                square_data,
                segment,
                self.params.square_width,
                valid_regions,
                self.params.channel,
                self.preprocess_fn,
                reference_level=median,
            )
            if frames is None:
                continue
            output = self.model.predict(frames[np.newaxis, :])
            predictions.append(output[0])
        return predictions

    def classify_frame(self, frame, thermal_median, preprocess=True):
        if preprocess:
            frame = preprocess_frame(
                frame,
                False,
                thermal_median,
                0,
                self.params.output_dim,
                preprocess_fn=self.preprocess_fn,
            )
            if frame is None:
                return None
        output = self.model.predict(frame[np.newaxis, :])
        return output[0]

    def confusion(self, dataset, filename="confusion.png"):
        dataset.set_read_only(True)
        dataset.use_segments = self.params.use_segments
        test = DataGenerator(
            dataset,
            self.labels,
            self.params.output_dim,
            batch_size=self.params.batch_size,
            channel=self.params.channel,
            use_movement=self.params.use_movement,
            shuffle=True,
            model_preprocess=self.preprocess_fn,
            epochs=1,
            load_threads=self.params.train_load_threads,
            keep_epoch=True,
            cap_samples=True,
            cap_at="bird",
            square_width=self.params.square_width,
        )
        test_pred_raw = self.model.predict(test)
        test.stop_load()
        test_pred = np.argmax(test_pred_raw, axis=1)

        batch_y = test.get_epoch_predictions(0)
        one_hot_y = []
        for batch in batch_y:
            one_hot_y.extend(np.int32(batch))
        one_hot_y = np.array(one_hot_y)
        self.f1(one_hot_y, test_pred_raw)
        # test.epoch_data = None
        cm = confusion_matrix(
            np.argmax(one_hot_y, axis=1), test_pred, labels=np.arange(len(self.labels))
        )
        # Log the confusion matrix as an image summary.
        figure = plot_confusion_matrix(cm, class_names=self.labels)
        plt.savefig(filename, format="png")

    def f1(self, one_hot_y, pred_raw):
        metric = tfa.metrics.F1Score(num_classes=len(self.labels))
        metric.update_state(one_hot_y, pred_raw)
        result = metric.result().numpy()
        print("F1 score")
        by_label = {}
        for i, label in enumerate(self.labels):
            by_label[label] = round(100 * result[i])
        sorted = self.labels.copy()
        sorted.sort()
        for label in sorted:
            print("{} = {}".format(label, by_label[label]))

    def evaluate(self, dataset):
        dataset.set_read_only(True)
        dataset.use_segments = self.params.use_segments

        test = DataGenerator(
            dataset,
            self.labels,
            self.params.output_dim,
            batch_size=self.params.batch_size,
            channel=self.params.channel,
            use_movement=self.params.use_movement,
            shuffle=True,
            model_preprocess=self.preprocess_fn,
            epochs=1,
            load_threads=self.params.train_load_threads,
            cap_samples=True,
            cap_at="bird",
            square_width=self.params.square_width,
        )
        test_accuracy = self.model.evaluate(test)
        test.stop_load()
        logging.info("Test accuracy is %s", test_accuracy)

    def track_confusion(self, dataset, filename="confusion.png"):
        dataset.set_read_only(True)
        dataset.use_segments = self.params.use_segments
        cap_at = len(dataset.tracks_by_label.get("bird", []))
        # cap_at = 1
        predictions = []
        actual = []
        raw_predictions = []
        one_hot = []
        for label in dataset.label_mapping.keys():
            label_tracks = dataset.tracks_by_label.get(label, [])
            label_tracks = [track for track in label_tracks if len(track.segments) > 0]
            sample_tracks = np.random.choice(
                label_tracks, min(len(label_tracks), cap_at), replace=False
            )
            mapped_label = dataset.mapped_label(label)
            for track in sample_tracks:
                track_data = dataset.db.get_track(track.clip_id, track.track_id)

                regions = []
                for region in track.track_bounds:
                    regions.append(tools.Rectangle.from_ltrb(*region))
                track_prediction = self.classify_track_data(
                    track.track_id,
                    track_data,
                    track.frame_temp_median,
                    regions=regions,
                )
                if track_prediction is None or len(track_prediction.predictions) == 0:
                    continue
                avg = np.mean(track_prediction.predictions, axis=0)

                # print(avg.shape, "mean preds are", np.round(100 * avg))
                # print(
                #     mapped_label,
                #     " predictied as ",
                #     self.labels[np.argmax(avg)],
                #     "with",
                #     np.amax(avg) * 100,
                # )
                actual.append(self.labels.index(mapped_label))
                predictions.append(np.argmax(avg))
                raw_predictions.append(avg)

                one_hot.append(
                    keras.utils.to_categorical(actual[-1], num_classes=len(self.labels))
                )

        self.f1(one_hot, raw_predictions)
        # test.epoch_data = None
        cm = confusion_matrix(actual, predictions, labels=np.arange(len(self.labels)))
        # Log the confusion matrix as an image summary.
        print("using ", self.labels, len(self.labels))
        figure = plot_confusion_matrix(cm, class_names=self.labels)
        plt.savefig(filename, format="png")


# from tensorflow examples
def plot_confusion_matrix(cm, class_names):
    """
    Returns a matplotlib figure containing the plotted confusion matrix.

    Args:
      cm (array, shape = [n, n]): a confusion matrix of integer classes
      class_names (array, shape = [n]): String names of the integer classes
    """

    # Normalize the confusion matrix.
    print(cm)
    # cm = np.around(cm.astype("float") / cm.sum(axis=1)[:, np.newaxis], decimals=2)
    # cm = np.nan_to_num(cm)
    figure = plt.figure(figsize=(8, 8))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("Confusion matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)
    #
    cm = np.around(cm.astype("float") / cm.sum(axis=1)[:, np.newaxis], decimals=2)
    cm = np.nan_to_num(cm)

    # Use white text if squares are dark; otherwise black.
    threshold = cm.max() / 2.0
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        color = "white" if cm[i, j] > threshold else "black"
        plt.text(j, i, cm[i, j], horizontalalignment="center", color=color)

    plt.tight_layout()
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    return figure


def log_confusion_matrix(epoch, logs, model, validate, writer):
    # Use the model to predict the values from the validation dataset.
    batch_y = validate.get_epoch_predictions(epoch)
    validate.use_previous_epoch = epoch
    # Calculate the confusion matrix.
    if validate.keep_epoch:
        # x = np.array(x)
        y = []
        for batch in batch_y:
            y.extend(np.argmax(batch, axis=1))
    else:
        y = batch_y

    print("predicting")
    test_pred_raw = model.predict(validate)
    print("predicted")
    validate.epoch_data[epoch] = []

    # reset validation generator will be 1 epoch ahead
    validate.use_previous_epoch = None
    validate.cur_epoch -= 1
    del validate.epoch_data[-1]
    test_pred = np.argmax(test_pred_raw, axis=1)

    cm = confusion_matrix(y, test_pred)

    # Log the confusion matrix as an image summary.
    figure = plot_confusion_matrix(cm, class_names=validate.labels)
    cm_image = plot_to_image(figure)

    # Log the confusion matrix as an image summary.
    with writer.as_default():
        tf.summary.image("Confusion Matrix", cm_image, step=epoch)
    print("done confusion")


def plot_to_image(figure):
    """Converts the matplotlib plot specified by 'figure' to a PNG image and
    returns it. The supplied figure is closed and inaccessible after this call."""
    # Save the plot to a PNG in memory.
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    # Closing the figure prevents it from being displayed directly inside
    # the notebook.
    plt.close(figure)
    buf.seek(0)
    # Convert PNG buffer to TF image
    image = tf.image.decode_png(buf.getvalue(), channels=4)
    # Add the batch dimension
    image = tf.expand_dims(image, 0)
    return image

import cv2
import math
import pickle
from PIL import Image, ImageDraw, ImageFont, ImageColor

import matplotlib.pyplot as plt
from ml_tools.trackdatabase import TrackDatabase

from ml_tools.dataset import dataset_db_path
import tensorflow as tf

import sys
import argparse
from ml_tools.datagenerator import DataGenerator

import json
import logging
import os.path
import time
from typing import Dict

from datetime import datetime
import numpy as np

from classify.trackprediction import Predictions
from load.clip import Clip
from load.cliptrackextractor import ClipTrackExtractor
from ml_tools import tools
from ml_tools.cptvfileprocessor import CPTVFileProcessor
import ml_tools.globals as globs
from ml_tools.newmodel import NewModel
from ml_tools.dataset import Preprocessor, TrackChannels, dataset_db_path
from ml_tools.previewer import Previewer
from track.track import Track
from config.config import Config


class Test:
    def __init__(self, config, model_file):

        # prediction record for each track
        self.FRAME_SKIP = 1
        self.model_file = model_file
        self.config = config
        self.tracker_config = config.tracking
        self.previewer = Previewer.create_if_required(config, config.classify.preview)
        self.enable_per_track_information = False
        self.track_extractor = ClipTrackExtractor(
            self.config.tracking,
            self.config.use_opt_flow
            or config.classify.preview == Previewer.PREVIEW_TRACKING,
            self.config.classify.cache_to_disk,
        )
        self.classifier = None
        # self.load_classifier(model_file)
        # self.predictions = Predictions(self.classifier.labels)

    def load_classifier(self, model_file):
        """
        Returns a classifier object, which is created on demand.
        This means if the ClipClassifier is copied to a new process a new Classifier instance will be created.
        """
        t0 = datetime.now()
        logging.info("classifier loading")

        self.classifier = NewModel(train_config=self.config.train)

        self.classifier.load_model(model_file)
        logging.info("classifier loaded ({})".format(datetime.now() - t0))

    def identify_track(self, clip: Clip, track: Track):
        """
        Runs through track identifying segments, and then returns it's prediction of what kind of animal this is.
        One prediction will be made for every frame.
        :param track: the track to identify.
        :return: TrackPrediction object
        """

        # uniform prior stats start with uniform distribution.  This is the safest bet, but means that
        # it takes a while to make predictions.  When off the first prediction is used instead causing
        # faster, but potentially more unstable predictions.
        UNIFORM_PRIOR = False
        num_labels = len(self.classifier.labels)

        prediction_smooth = 0.1

        smooth_prediction = None
        smooth_novelty = 0

        prediction = 0.0
        novelty = 0.0
        try:
            fp_index = self.classifier.labels.index("false-positive")
        except ValueError:
            fp_index = None

        # go through making classifications at each frame
        # note: we should probably be doing this every 9 frames or so.
        track_prediction = self.predictions.get_or_create_prediction(track)
        for i, region in enumerate(track.bounds_history):
            frame = clip.frame_buffer.get_frame(region.frame_number)
            frame = track.crop_by_region(frame, region)
            # thermal = region.subimage(frame.thermal)
            # note: would be much better for the tracker to store the thermal references as it goes.
            # frame = clip.frame_buffer.get_frame(frame_number)
            # thermal_reference = np.median(frame.thermal)
            # track_data = track.crop_by_region_at_trackframe(frame, i)
            if i % self.FRAME_SKIP == 0:
                # we use a tighter cropping here so we disable the default 2 pixel inset
                # frames = Preprocessor.apply(
                #     [track_data], [thermal_reference], default_inset=0
                # )

                if frame is None:
                    logging.info(
                        "Frame {} of track could not be classified.".format(
                            region.frame_number
                        )
                    )
                    return

                prediction = self.classifier.classify_frame(frame)
                track_prediction.classified_frame(
                    region.frame_number, prediction, smooth_novelty
                )

                continue
                # make false-positive prediction less strong so if track has dead footage it won't dominate a strong
                # score
                if fp_index is not None:
                    prediction[fp_index] *= 0.8

                mass = region.mass

                # we use the square-root here as the mass is in units squared.
                # this effectively means we are giving weight based on the diameter
                # of the object rather than the mass.
                mass_weight = np.clip(mass / 20, 0.02, 1.0) ** 0.5

                # cropped frames don't do so well so restrict their score
                cropped_weight = 0.7 if region.was_cropped else 1.0

                prediction *= mass_weight * cropped_weight

            if smooth_prediction is None:
                if UNIFORM_PRIOR:
                    smooth_prediction = np.ones([num_labels]) * (1 / num_labels)
                else:
                    smooth_prediction = prediction
                # smooth_novelty = 0.5
            else:
                smooth_prediction = (
                    1 - prediction_smooth
                ) * smooth_prediction + prediction_smooth * prediction
                # smooth_novelty = (
                #     1 - prediction_smooth
                # ) * smooth_novelty + prediction_smooth * novelty

            # print(self.classifier.labels[track_prediction.label_at_time(-1)])

        return track_prediction

    def get_meta_data(self, filename):
        """ Reads meta-data for a given cptv file. """
        source_meta_filename = os.path.splitext(filename)[0] + ".txt"
        if os.path.exists(source_meta_filename):

            meta_data = tools.load_clip_metadata(source_meta_filename)

            tags = set()
            for record in meta_data["Tags"]:
                # skip automatic tags
                if record.get("automatic", False):
                    continue
                else:
                    tags.add(record["animal"])

            tags = list(tags)

            if len(tags) == 0:
                tag = "no tag"
            elif len(tags) == 1:
                tag = tags[0] if tags[0] else "none"
            else:
                tag = "multi"
            meta_data["primary_tag"] = tag
            return meta_data
        else:
            return None

    def get_classify_filename(self, input_filename):
        return os.path.splitext(
            os.path.join(
                self.config.classify.classify_folder, os.path.basename(input_filename)
            )
        )[0]

    def process_file(self, filename, **kwargs):
        """
        Process a file extracting tracks and identifying them.
        :param filename: filename to process
        :param enable_preview: if true an MPEG preview file is created.
        """

        if not os.path.exists(filename):
            raise Exception("File {} not found.".format(filename))

        logging.info("Processing file '{}'".format(filename))

        start = time.time()
        clip = Clip(self.tracker_config, filename)
        # base_filename = os.path.splitext(os.path.basename(filename))[0]
        #
        # metadata_filename = os.path.join(
        #     os.path.dirname(filename), base_filename + ".txt"
        # )
        # metadata = tools.load_clip_metadata(metadata_filename)
        #
        # clip.load_metadata(
        #     metadata,
        #     self.config.load.include_filtered_channel,
        #     self.config.load.tag_precedence,
        # )
        self.track_extractor.parse_clip(clip)

        classify_name = self.get_classify_filename(filename)
        destination_folder = os.path.dirname(classify_name)

        if not os.path.exists(destination_folder):
            logging.info("Creating folder {}".format(destination_folder))
            os.makedirs(destination_folder)

        mpeg_filename = classify_name + ".mp4"

        meta_filename = classify_name + ".txt"

        logging.info(os.path.basename(filename) + ":")

        for i, track in enumerate(clip.tracks):
            prediction = self.identify_track(clip, track)
            description = prediction.description(self.classifier.labels)
            logging.info(
                " - [{}/{}] prediction: {}".format(i + 1, len(clip.tracks), description)
            )
        self.predictions.set_important_frames()
        self.save_important_frames(clip, clip.get_id(), self.predictions)
        if self.previewer:
            logging.info("Exporting preview to '{}'".format(mpeg_filename))
            self.previewer.export_clip_preview(mpeg_filename, clip, self.predictions)
        logging.info("saving meta data")
        self.save_metadata(filename, meta_filename, clip)
        self.predictions.clear_predictions()

        if self.tracker_config.verbose:
            ms_per_frame = (
                (time.time() - start) * 1000 / max(1, len(clip.frame_buffer.frames))
            )
            logging.info("Took {:.1f}ms per frame".format(ms_per_frame))

    def save_important_frames(self, clip, filename, predictions):
        for track_id, track_prediction in predictions.prediction_per_track.items():
            track = [track for track in clip.tracks if track.get_id() == track_id]

            if len(track) == 0:
                raise "Couldnt find track {}".format(track_id)
            track = track[0]
            for i in range(1):
                if i == 0:
                    values = track_prediction.best_predictions
                    prefix = "best"
                else:
                    values = track_prediction.clearest_frames
                    prefix = "clear"

                rows = round(len(values) / 5.0) + 1
                fig = plt.figure(figsize=(52, 52))
                for i, frame_i in enumerate(values):
                    axes = fig.add_subplot(rows, 5, i + 1)
                    axes.set_title(
                        "{}-{}".format(
                            frame_i + track.start_frame,
                            track_prediction.get_classified_footer(
                                self.classifier.labels, frame_i
                            ),
                        )
                    )

                    frame = clip.frame_buffer.get_frame(frame_i + track.start_frame)
                    frame = track.crop_by_region(frame, track.bounds_history[frame_i])[
                        0
                    ]
                    frame = np.float32(frame)
                    temp_min = np.amin(frame)
                    temp_max = np.amax(frame)
                    frame = (frame - temp_min) / (temp_max - temp_min)
                    colorized = np.uint8(255.0 * self.previewer.colourmap(frame))
                    plt.imshow(colorized[:, :, :3])
                plt.savefig("{}-{}-{}.png".format(prefix, filename, track_id))
                plt.close(fig)

    def save_metadata(self, filename, meta_filename, clip):
        # if self.cache_to_disk:
        #     clip.frame_buffer.remove_cache()

        # read in original metadata
        meta_data = self.get_meta_data(filename)

        # record results in text file.
        save_file = {}
        save_file["source"] = filename
        start, end = clip.start_and_end_time_absolute()
        save_file["start_time"] = start.isoformat()
        save_file["end_time"] = end.isoformat()
        save_file["algorithm"] = {}
        save_file["algorithm"]["model"] = self.model_file
        save_file["algorithm"]["tracker_version"] = clip.VERSION
        save_file["algorithm"]["tracker_config"] = self.tracker_config.as_dict()
        if meta_data:
            save_file["camera"] = meta_data["Device"]["devicename"]
            save_file["cptv_meta"] = meta_data
            save_file["original_tag"] = meta_data["primary_tag"]
        save_file["tracks"] = []
        for track in clip.tracks:
            track_info = {}
            prediction = self.predictions.prediction_for(track.get_id())
            start_s, end_s = clip.start_and_end_in_secs(track)
            save_file["tracks"].append(track_info)
            track_info["start_s"] = round(start_s, 2)
            track_info["end_s"] = round(end_s, 2)
            track_info["num_frames"] = prediction.num_frames
            track_info["frame_start"] = track.start_frame
            track_info["frame_end"] = track.end_frame
            track_info["label"] = self.classifier.labels[prediction.best_label_index]
            track_info["confidence"] = round(prediction.score(), 2)
            track_info["clarity"] = round(prediction.clarity, 3)
            track_info["average_novelty"] = round(prediction.average_novelty, 2)
            track_info["max_novelty"] = round(prediction.max_novelty, 2)
            track_info["all_class_confidences"] = {}
            for i, value in enumerate(prediction.class_best_score):
                label = self.classifier.labels[i]
                track_info["all_class_confidences"][label] = round(float(value), 3)

            positions = []
            for region in track.bounds_history:
                track_time = round(region.frame_number / clip.frames_per_second, 2)
                positions.append([track_time, region])
            track_info["positions"] = positions

        if self.config.classify.meta_to_stdout:
            print(json.dumps(save_file, cls=tools.CustomJSONEncoder))
        else:
            with open(meta_filename, "w") as f:
                json.dump(save_file, f, indent=4, cls=tools.CustomJSONEncoder)

    def multi_clip(self, db, track_header, filename):
        # frame_i = track_header.important_frames[1]
        frames = db.get_track(track_header.clip_id, track_header.track_id, 0)

        # background = np.float32(db.get_clip_background(track_header.clip_id))
        # print("background multi for", track_header.clip_id)
        #
        # newbackground = background.copy()
        # background[:, :] = 255
        start = 0
        background = np.zeros((120, 160))

        for i, frame in enumerate(frames):
            region = track_header.track_bounds[start + i]
            rect = tools.Rectangle.from_ltrb(*region)
            frame = frame[TrackChannels.filtered]
            subimage = rect.subimage(background)

            subimage[:, :] += np.float32(frame)
        max = np.amax(background)
        background /= max
        background *= 255
        img = Image.fromarray(np.uint8(background))
        img.save(filename + ".png", "PNG")
        return img
        # tools.frame_to_jpg(background, "birdoverlay.jpg")

    def square_clip(self, db, track_header, filename, width=6):
        # frame_i = track_header.important_frames[1]
        frames = db.get_track(track_header.clip_id, track_header.track_id, 0)

        # background = np.float32(db.get_clip_background(track_header.clip_id))
        # print("background multi for", track_header.clip_id)
        #
        # newbackground = background.copy()
        # background[:, :] = 255
        start = 0
        background = np.zeros((288, 288))
        frame_size = 48
        i = 0
        for x in range(width):
            for y in range(width):
                i += 1
                if i >= len(frames):
                    frame = frames[-1]
                else:
                    frame = frames[i]
                frame = frame[TrackChannels.filtered]
                frame = cv2.resize(
                    frame,
                    dsize=(frame_size, frame_size),
                    interpolation=cv2.INTER_LINEAR,
                )
                background[
                    x * frame_size : (x + 1) * frame_size,
                    y * frame_size : (y + 1) * frame_size,
                ] = np.float32(frame)
        max = np.amax(background)
        background /= max
        background *= 255
        img = Image.fromarray(np.uint8(background))
        img.save(filename + ".png", "PNG")
        return img

    # tools.frame_to_jpg(background, "birdoverlay.jpg")
    def save_img(self, data, filename="newmodeltest.png"):
        if len(data.shape) == 2:
            data = data[..., np.newaxis]
            data = np.repeat(data, 3, axis=2)
            # data[:, :, 1] = 0
            # data[:, :, 2] = 0
        print(data.shape)
        img = pillow.Image.fromarray(np.uint8(data))  # ignore alpha

        img.save(filename, "PNG")

    def movement(self, db, track_header, filename):
        frames = db.get_track(track_header.clip_id, track_header.track_id, 0)
        i = 0
        start = 0
        test = np.zeros((120, 160))
        test[:, :] = 255
        test = test[..., np.newaxis]

        test = np.repeat(test, 4, axis=2)

        prev = None
        value = 60
        img = Image.fromarray(np.uint8(test))  # ignore alpha

        d = ImageDraw.Draw(img)
        for i, frame in enumerate(frames):
            region = track_header.track_bounds[start + i]
            rect = tools.Rectangle.from_ltrb(*region)
            x = int(rect.mid_x)
            y = int(rect.mid_y)
            if prev is not None:
                if prev[0] == x and prev[1] == y:
                    value *= 1.1
                else:
                    value = 60
                distance = math.sqrt(pow(prev[0] - x, 2) + pow(prev[1] - y, 2))

                distance *= 21.25
                distance = min(distance, 255)
                d.line(prev + (x, y), fill=(0, int(distance), 0, 255), width=1)
                # d.point([prev], fill=colour)

            prev = (x, y)
            colour = (int(value), 0, 0, 255)
        for i, frame in enumerate(frames):
            region = track_header.track_bounds[start + i]
            rect = tools.Rectangle.from_ltrb(*region)
            x = int(rect.mid_x)
            y = int(rect.mid_y)
            if prev is not None:
                if prev[0] == x and prev[1] == y:
                    value *= 1.1
                else:
                    value = 60
            prev = (x, y)
            colour = (int(value), 0, 0, 255)
            d.point([prev], fill=colour)

        img.save(filename + ".png", "PNG")
        return img
        # self.save_img(background, "{}.png".format(filename))

    def save_db(self, clip_id, track_id):
        db = TrackDatabase(os.path.join(self.config.tracks_folder, "dataset.hdf5"))
        clip_meta = db.get_clip_meta(clip_id)
        print(
            clip_meta["mean_background_value"],
            clip_meta["temp_thresh"],
            clip_meta["mean_background_value"] - clip_meta["temp_thresh"],
        )
        track_meta = db.get_track_meta(clip_id, track_id)
        predictions = db.get_track_predictions(clip_id, track_id)
        track_header = TrackHeader.from_meta(
            clip_id, clip_meta, track_meta, predictions
        )
        labels = db.get_labels()
        track_header.set_important_frames(labels, 0)
        self.multi_clip(db, track_header, clip_meta)
        return
        start_offset = track_meta["start_frame"]
        self.save_trackheader_important(
            db,
            track_header,
            "db" + str(clip_id),
            start_offset,
            clip_meta["temp_thresh"],
        )

    def save_trackheader_important(
        self, db, track_header, filename, start_offset, temp_thresh
    ):
        prefix = "best"
        track_header.important_frames.sort()

        print(track_header.track_number, "predictions", track_header.important_frames)
        rows = round(len(track_header.important_frames) / 5.0) + 1
        fibird_motiong = plt.figure(figsize=(52, 52))
        for i, frame_i in enumerate(track_header.important_frames):

            pred = track_header.predictions[frame_i]
            label = self.classifier.labels[np.argmax(np.array(pred))]
            axes = fig.add_subplot(rows, 5, i + 1)
            axes.set_title(
                "{}-{}-{}-m{}".format(
                    frame_i,
                    label,
                    np.max(track_header.predictions[frame_i]),
                    track_header.frame_mass[frame_i]
                    # track_prediction.get_classified_footer(
                    #     self.classifier.labels, frame_i + start_offset - 1
                    # ),
                )
            )
            frame = db.get_track(
                track_header.clip_id,
                track_header.track_number,
                frame_i,
                frame_i + 1,
            )[0]
            animal = frame[0] * frame[4]
            not_animal = frame[0] - animal
            non_animal_max = np.percentile(not_animal, 90)
            temp_min = np.percentile(animal[animal > 0], 50)
            print(
                "non 80 percent",
                non_animal_max,
                "animal 10 percent",
                temp_min,
                "ignore",
                temp_min - non_animal_max < 0,
            )
            #
            # print("animal stats", np.amax(animal), np.amin(animal[animal > 0]))
            # print("not_animal stats", np.amax(not_animal), np.amin(not_animal))

            frame = np.float32(animal)
            temp_min = np.amin(frame[frame > 0])
            temp_max = np.amax(frame)
            # frame -= temp_min

            print(frame_i, temp_max, temp_min, np.mean(frame), np.std(frame))
            frame = (frame - temp_min) / (temp_max - temp_min)
            colorized = np.uint8(255.0 * self.previewer.colourmap(frame))
            plt.imshow(colorized[:, :, :3])
        plt.savefig(
            "mask{}-{}-{}.png".format(prefix, filename, track_header.track_number)
        )
        plt.close(fig)

    def save_motion_images(self, dataset, label, n=10):
        tracks = dataset.tracks_by_label[label][:n]
        images = []
        for track in tracks:
            # img = self.movement(
            #     dataset.db, track, "samples/{}-{}".format(label, track.unique_id)
            # )
            img, multi = dataset.movement(track)
            # multi = self.multi_clip(
            #     dataset.db, track, "samples/{}multi-{}".format(label, track.unique_id)
            # )
            square = self.square_clip(
                dataset.db, track, "samples/{}square-{}".format(label, track.unique_id)
            )
            max = np.amax(multi)
            multi /= max
            multi *= 255

            multi = Image.fromarray(np.uint8(multi))
            img = Image.fromarray(np.uint8(img))

            if img is not None:
                images.append((track, img.resize((248, 248), Image.NEAREST)))
                images.append((track, multi.resize((248, 248), Image.NEAREST)))
                images.append((track, square))
                print(images[-1])

        n = len(images)
        rows = int(math.sqrt(n))
        columns = math.ceil(n / rows)
        total_width = columns * 248
        total_height = rows * 248

        new_im = Image.new("RGB", (total_width, total_height))
        d = ImageDraw.Draw(new_im)
        fnt = ImageFont.truetype(tools.resource_path("Ubuntu-R.ttf"), 8)

        y_offset = 0
        index = 0
        print(len(images))
        for x in range(rows):
            x_offset = 0
            for y in range(columns):
                if index > len(images):
                    break
                im = images[index]
                print(index, im[1])

                new_im.paste(im[1], (x_offset, y_offset))
                d.text((x_offset, y_offset), im[0].unique_id, fill=(0, 0, 0), font=fnt)

                x_offset += im[1].size[0]
                index += 1

            y_offset += 248

        new_im.save("samples/{}.png".format(label))


def load_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--model-file",
        help="Path to model file to use, will override config model",
    )
    # parser.add_argument(
    #     "source",
    #     help='a CPTV file to process, or a folder name, or "all" for all files within subdirectories of source folder.',
    # )

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
        stream=sys.stderr, level=logging.INFO, format=fmt, datefmt="%Y-%m-%d %H:%M:%S"
    )


def load_dataset(config):
    datasets_filename = dataset_db_path(config)
    datasets = pickle.load(open(datasets_filename, "rb"))
    return datasets


args = load_args()
init_logging()
config = Config.load_from_file(args.config_file)

model_file = config.classify.model
if args.model_file:
    model_file = args.model_file
test = Test(config, model_file)
datasets = load_dataset(config)
test.save_motion_images(datasets[0], "rodent")

# # bird
# test.save_db("603530", "257125")
# test.process_file(args.source)

# test.save_db("606492", "257695")
exit(0)
# /clips/10130/10130
# exit(0)
# raise "EX"
# test.classifier.evaluate()

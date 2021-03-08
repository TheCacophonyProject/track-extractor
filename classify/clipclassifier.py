import json
import logging
import os.path
import time

from datetime import datetime
import numpy as np

from classify.trackprediction import Predictions
from load.clip import Clip
from load.cliptrackextractor import ClipTrackExtractor
from ml_tools import tools
from ml_tools.cptvfileprocessor import CPTVFileProcessor
import ml_tools.globals as globs
from ml_tools.model import Model
from ml_tools.kerasmodel import KerasModel

from ml_tools.preprocess import preprocess_segment
from ml_tools.previewer import Previewer
from track.track import Track


class ClipClassifier(CPTVFileProcessor):
    """ Classifies tracks within CPTV files. """

    # skips every nth frame.  Speeds things up a little, but reduces prediction quality.
    FRAME_SKIP = 1

    def __init__(self, config, tracking_config, model_file, kerasmodel=False):
        """ Create an instance of a clip classifier"""

        super(ClipClassifier, self).__init__(config, tracking_config)
        self.model_file = model_file
        self.kerasmodel = kerasmodel
        # prediction record for each track
        self.predictions = Predictions(self.classifier.labels)

        self.previewer = Previewer.create_if_required(config, config.classify.preview)

        self.start_date = None
        self.end_date = None
        self.cache_to_disk = self.config.classify.cache_to_disk
        # enables exports detailed information for each track.  If preview mode is enabled also enables track previews.
        self.enable_per_track_information = False
        self.track_extractor = ClipTrackExtractor(
            self.config.tracking,
            self.config.use_opt_flow
            or config.classify.preview == Previewer.PREVIEW_TRACKING,
            self.config.classify.cache_to_disk,
            high_quality_optical_flow=self.config.tracking.high_quality_optical_flow,
        )

    def preprocess(self, frame, thermal_reference):
        """
        Applies preprocessing to frame required by the model.
        :param frame: numpy array of shape [C, H, W]
        :return: preprocessed numpy array
        """

        # note, would be much better if the model did this, as only the model knows how preprocessing occurred during
        # training
        frame = np.float32(frame)
        frame[2 : 3 + 1] *= 1 / 256
        frame[0] -= thermal_reference

        return frame

    def identify_track(self, clip: Clip, track: Track):
        """
        Runs through track identifying segments, and then returns it's prediction of what kind of animal this is.
        One prediction will be made for every frame.
        :param track: the track to identify.
        :return: TrackPrediction object
        """
        # go through making classifications at each frame
        # note: we should probably be doing this every 9 frames or so.
        state = None
        if self.kerasmodel:
            track_prediction = self.classifier.classify_track(clip, track)
            self.predictions.prediction_per_track[track.get_id()] = track_prediction
        else:
            track_prediction = self.predictions.get_or_create_prediction(track)

            for i, region in enumerate(track.bounds_history):
                frame = clip.frame_buffer.get_frame(region.frame_number)

                track_data = track.crop_by_region(frame, region)

                # note: would be much better for the tracker to store the thermal references as it goes.
                # frame = clip.frame_buffer.get_frame(frame_number)
                thermal_reference = np.median(frame.thermal)
                # track_data = track.crop_by_region_at_trackframe(frame, i)
                if i % self.FRAME_SKIP == 0:

                    # we use a tighter cropping here so we disable the default 2 pixel inset
                    frames = preprocess_segment(
                        [track_data], [thermal_reference], default_inset=0
                    )

                    if frames is None:
                        logging.info(
                            "Frame {} of track could not be classified.".format(
                                region.frame_number
                            )
                        )
                        continue
                    frame = frames[0]
                    (
                        prediction,
                        novelty,
                        state,
                    ) = self.classifier.classify_frame_with_novelty(frame, state)
                    # make false-positive prediction less strong so if track has dead footage it won't dominate a strong
                    # score

                    # a little weight decay helps the model not lock into an initial impression.
                    # 0.98 represents a half life of around 3 seconds.
                    state *= 0.98

                    # precondition on weight,  segments with small mass are weighted less as we can assume the error is
                    # higher here.
                    mass = region.mass

                    # we use the square-root here as the mass is in units squared.
                    # this effectively means we are giving weight based on the diameter
                    # of the object rather than the mass.
                    mass_weight = np.clip(mass / 20, 0.02, 1.0) ** 0.5

                    # cropped frames don't do so well so restrict their score
                    cropped_weight = 0.7 if region.was_cropped else 1.0
                    track_prediction.classified_frame(
                        region.frame_number,
                        prediction,
                        mass_scale=mass_weight * cropped_weight,
                        novelty=novelty,
                    )
        return track_prediction

    @property
    def classifier(self):
        """
        Returns a classifier object, which is created on demand.
        This means if the ClipClassifier is copied to a new process a new Classifier instance will be created.
        """
        if globs._classifier is None:
            t0 = datetime.now()
            logging.info("classifier loading")
            if self.kerasmodel:
                model = KerasModel(self.config.train)
                model.load_model(self.model_file, training=False)
                globs._classifier = model
            else:
                globs._classifier = Model(
                    train_config=self.config.train,
                    session=tools.get_session(disable_gpu=not self.config.use_gpu),
                )
                globs._classifier.load(self.model_file)

            logging.info("classifier loaded ({})".format(datetime.now() - t0))
        return globs._classifier

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

    def process_file(self, filename):
        """
        Process a file extracting tracks and identifying them.
        :param filename: filename to process
        :param enable_preview: if true an MPEG preview file is created.
        """

        clip, predictions = self.classify_file(filename)

        classify_name = self.get_classify_filename(filename)
        destination_folder = os.path.dirname(classify_name)
        if not os.path.exists(destination_folder):
            logging.info("Creating folder {}".format(destination_folder))
            os.makedirs(destination_folder)
        mpeg_filename = classify_name + ".mp4"
        meta_filename = classify_name + ".txt"
        if self.previewer:
            logging.info("Exporting preview to '{}'".format(mpeg_filename))
            self.previewer.export_clip_preview(mpeg_filename, clip, predictions)
        logging.info("saving meta data")
        self.save_metadata(filename, meta_filename, clip, predictions)

    def classify_file(self, filename):
        if not os.path.exists(filename):
            raise Exception("File {} not found.".format(filename))
        logging.info("Processing file '{}'".format(filename))

        # prediction record for each track
        predictions = Predictions(self.classifier.labels)

        start = time.time()
        clip = Clip(self.tracker_config, filename)
        self.track_extractor.parse_clip(clip)

        for i, track in enumerate(clip.tracks):
            prediction = self.identify_track(clip, track)
            predictions.prediction_per_track[track.get_id()] = prediction
            description = prediction.description(self.classifier.labels)
            logging.info(
                " - [{}/{}] prediction: {}".format(i + 1, len(clip.tracks), description)
            )
        if self.tracker_config.verbose:
            ms_per_frame = (
                (time.time() - start) * 1000 / max(1, len(clip.frame_buffer.frames))
            )
            logging.info("Took {:.1f}ms per frame".format(ms_per_frame))

        return clip, predictions

    def save_metadata(self, filename, meta_filename, clip, predictions):
        if self.cache_to_disk:
            clip.frame_buffer.remove_cache()

        # read in original metadata
        meta_data = self.get_meta_data(filename)

        # record results in text file.
        save_file = {}
        save_file["source"] = filename
        if clip.camera_model:
            save_file["camera_model"] = clip.camera_model
        save_file["background_thresh"] = clip.background_thresh
        start, end = clip.start_and_end_time_absolute()
        save_file["start_time"] = start.isoformat()
        save_file["end_time"] = end.isoformat()
        save_file["algorithm"] = {}
        save_file["algorithm"]["model"] = self.model_file
        save_file["algorithm"]["tracker_version"] = ClipTrackExtractor.VERSION
        save_file["algorithm"]["tracker_config"] = self.tracker_config.as_dict()
        if meta_data:
            save_file["camera"] = meta_data["Device"]["devicename"]
            save_file["cptv_meta"] = meta_data
            save_file["original_tag"] = meta_data["primary_tag"]
        save_file["tracks"] = []
        for track in clip.tracks:
            track_info = {}
            prediction = predictions.prediction_for(track.get_id())
            start_s, end_s = clip.start_and_end_in_secs(track)
            save_file["tracks"].append(track_info)
            track_info["start_s"] = round(start_s, 2)
            track_info["end_s"] = round(end_s, 2)
            track_info["num_frames"] = prediction.num_frames
            track_info["frame_start"] = track.start_frame
            track_info["frame_end"] = track.end_frame
            track_info["label"] = prediction.predicted_tag(self.classifier.labels)
            track_info["confidence"] = round(prediction.max_score, 2)
            track_info["clarity"] = round(prediction.clarity, 3)
            track_info["average_novelty"] = float(round(prediction.average_novelty, 2))
            track_info["max_novelty"] = float(round(prediction.max_novelty, 2))
            track_info["all_class_confidences"] = {}

            # numpy data wont serialize
            prediction_data = []
            for pred in prediction.predictions:
                pred_list = [int(round(p * 100)) for p in pred]
                prediction_data.append(pred_list)
            track_info["predictions"] = prediction_data
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

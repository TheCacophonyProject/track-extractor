import gc
import json
import logging
import os.path
import time

from datetime import datetime
import numpy as np

from classify.trackprediction import Predictions, TrackPrediction
from load.clip import Clip
from load.cliptrackextractor import ClipTrackExtractor
from ml_tools import tools
from ml_tools.cptvfileprocessor import CPTVFileProcessor
import ml_tools.globals as globs
from ml_tools.model import Model
from ml_tools.kerasmodel import KerasModel, is_keras_model

from ml_tools.preprocess import preprocess_segment
from ml_tools.previewer import Previewer
from track.track import Track


class ClipClassifier(CPTVFileProcessor):
    """Classifies tracks within CPTV files."""

    # skips every nth frame.  Speeds things up a little, but reduces prediction quality.
    FRAME_SKIP = 1

    def __init__(self, config, tracking_config, model=None, cache_to_disk=None):
        """Create an instance of a clip classifier"""

        super(ClipClassifier, self).__init__(config, tracking_config)
        self.model = model
        # prediction record for each track

        self.previewer = Previewer.create_if_required(config, config.classify.preview)

        self.start_date = None
        self.end_date = None
        if cache_to_disk is None:
            self.cache_to_disk = self.config.classify.cache_to_disk
        else:
            self.cache_to_disk = cache_to_disk
        # enables exports detailed information for each track.  If preview mode is enabled also enables track previews.
        self.enable_per_track_information = False
        self.track_extractor = ClipTrackExtractor(
            self.config.tracking,
            self.config.use_opt_flow
            or config.classify.preview == Previewer.PREVIEW_TRACKING,
            self.cache_to_disk,
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

    def identify_track(self, classifier, clip: Clip, track: Track):
        """
        Runs through track identifying segments, and then returns it's prediction of what kind of animal this is.
        One prediction will be made for every frame.
        :param track: the track to identify.
        :return: TrackPrediction object
        """
        # go through making classifications at each frame
        # note: we should probably be doing this every 9 frames or so.
        state = None
        if isinstance(classifier, KerasModel):
            track_prediction = classifier.classify_track(clip, track)
        else:
            track_prediction = TrackPrediction(
                track.get_id(), track.start_frame, classifier.labels
            )

            for i, region in enumerate(track.bounds_history):
                frame = clip.frame_buffer.get_frame(region.frame_number)

                cropped = frame.crop_by_region(region)

                # note: would be much better for the tracker to store the thermal references as it goes.
                # frame = clip.frame_buffer.get_frame(frame_number)
                thermal_reference = np.median(frame.thermal)
                if i % self.FRAME_SKIP == 0:

                    # we use a tighter cropping here so we disable the default 2 pixel inset
                    frames, _ = preprocess_segment(
                        [cropped], [thermal_reference], default_inset=0
                    )

                    if frames is None or len(frames) == 0:
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
                    ) = classifier.classify_frame_with_novelty(frame.as_array(), state)
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

    def get_classifier(self, model):
        """
        Returns a classifier object, which is created on demand.
        This means if the ClipClassifier is copied to a new process a new Classifier instance will be created.
        """
        t0 = datetime.now()
        logging.info("classifier loading")
        classifier = None
        if is_keras_model(model.model_file):
            classifier = KerasModel(self.config.train)
            classifier.load_model(model.model_file, model.model_weights)
        else:
            classifier = Model(
                train_config=self.config.train,
                session=tools.get_session(disable_gpu=not self.config.use_gpu),
            )
            classifier.load(model.model_file)

        logging.info("classifier loaded ({})".format(datetime.now() - t0))
        return classifier

    def get_meta_data(self, filename):
        """Reads meta-data for a given cptv file."""
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

        clip, model_predictions = self.classify_file(filename)

        classify_name = self.get_classify_filename(filename)
        destination_folder = os.path.dirname(classify_name)
        if not os.path.exists(destination_folder):
            logging.info("Creating folder {}".format(destination_folder))
            os.makedirs(destination_folder)
        mpeg_filename = classify_name + ".mp4"
        meta_filename = classify_name + ".txt"

        if self.previewer:
            logging.info("Exporting preview to '{}'".format(mpeg_filename))

            self.previewer.export_clip_preview(
                mpeg_filename, clip, list(model_predictions.values())[0]
            )
        logging.info("saving meta data")
        models = [self.model] if self.model else self.config.classify.models
        self.save_metadata(
            filename,
            meta_filename,
            clip,
            model_predictions,
            models,
            self.track_extractor.tracking_time,
        )

    def classify_file(self, filename):
        if not os.path.exists(filename):
            raise Exception("File {} not found.".format(filename))
        logging.info("Processing file '{}'".format(filename))

        # prediction record for each track

        start = time.time()
        clip = Clip(self.tracker_config, filename)
        self.track_extractor.parse_clip(clip)
        predictions_per_model = {}
        if self.model:
            prediction = self.classify_clip(clip, self.model)
            predictions_per_model[self.model.name] = prediction
        else:
            for model in self.config.classify.models:
                prediction = self.classify_clip(clip, model)
                predictions_per_model[model.name] = prediction
        return clip, predictions_per_model

    def classify_clip(self, clip, model):
        start = time.time()

        classifier = self.get_classifier(model)

        predictions = Predictions(classifier.labels, model)
        for i, track in enumerate(clip.tracks):
            prediction = self.identify_track(
                classifier,
                clip,
                track,
            )
            predictions.prediction_per_track[track.get_id()] = prediction
            description = prediction.description()
            logging.info(
                " - [{}/{}] prediction: {}".format(i + 1, len(clip.tracks), description)
            )
        if self.tracker_config.verbose:
            ms_per_frame = (
                (time.time() - start) * 1000 / max(1, len(clip.frame_buffer.frames))
            )
            logging.info("Took {:.1f}ms per frame".format(ms_per_frame))
        predictions.classify_time = time.time() - start
        tools.clear_session()
        del classifier
        gc.collect()

        return predictions

    def save_metadata(
        self,
        filename,
        meta_filename,
        clip,
        predictions_per_model,
        models,
        tracking_time,
    ):
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
        save_file["tracking_time"] = round(tracking_time, 1)
        save_file["algorithm"] = {}
        save_file["algorithm"]["tracker_version"] = ClipTrackExtractor.VERSION
        save_file["algorithm"]["tracker_config"] = self.tracker_config.as_dict()
        if meta_data:
            save_file["camera"] = meta_data["Device"]["devicename"]
            save_file["cptv_meta"] = meta_data
            save_file["original_tag"] = meta_data["primary_tag"]

        save_file["tracks"] = []
        for track in clip.tracks:
            track_info = {}
            start_s, end_s = clip.start_and_end_in_secs(track)
            save_file["tracks"].append(track_info)
            track_info["id"] = track.get_id()
            track_info["start_s"] = round(start_s, 2)
            track_info["end_s"] = round(end_s, 2)
            track_info["num_frames"] = len(track)
            track_info["frame_start"] = track.start_frame
            track_info["frame_end"] = track.end_frame

            positions = []
            for region in track.bounds_history:
                track_time = round(region.frame_number / clip.frames_per_second, 2)
                positions.append([track_time, region])
            track_info["positions"] = positions
            prediction_info = []
            for model, predictions in predictions_per_model.items():
                model_info = {
                    "id": predictions.model.id,
                    "model_file": predictions.model.model_file,
                    "model_name": predictions.model.name,
                }
                prediction = predictions.prediction_for(track.get_id())
                model_info["label"] = prediction.predicted_tag()
                model_info["confidence"] = round(prediction.max_score, 2)
                model_info["clarity"] = round(prediction.clarity, 3)
                model_info["average_novelty"] = float(
                    round(prediction.average_novelty, 2)
                )
                model_info["max_novelty"] = float(round(prediction.max_novelty, 2))
                model_info["all_class_confidences"] = {}
                prediction_data = []
                for pred in prediction.smoothed_predictions:
                    pred_list = [int(round(p * 100)) for p in pred]
                    prediction_data.append(pred_list)
                model_info["predictions"] = prediction_data
                for i, value in enumerate(prediction.class_best_score):
                    label = prediction.labels[i]
                    model_info["all_class_confidences"][label] = round(float(value), 3)
                prediction_info.append(model_info)
            track_info["predictions"] = prediction_info
        model_dictionaries = []
        for model in models:
            model_dic = model.as_dict()
            model_time = [
                predictions.classify_time
                for predictions in predictions_per_model.values()
                if predictions.model == model
            ]
            if len(model_time) > 0:
                model_dic["classify_time"] = round(model_time[0], 1)
            model_dictionaries.append(model_dic)

        save_file["models"] = model_dictionaries
        if self.config.classify.meta_to_stdout:
            print(json.dumps(save_file, cls=tools.CustomJSONEncoder))
        else:
            with open(meta_filename, "w") as f:
                json.dump(save_file, f, indent=4, cls=tools.CustomJSONEncoder)

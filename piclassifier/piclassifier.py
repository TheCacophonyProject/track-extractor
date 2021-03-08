from datetime import datetime
import json
import logging
import os
import time

import psutil
import numpy as np

from classify.trackprediction import Predictions
from load.clip import Clip
from load.cliptrackextractor import ClipTrackExtractor
from ml_tools.preprocess import preprocess_segment
from ml_tools.previewer import Previewer
from ml_tools import tools
from .cptvrecorder import CPTVRecorder
from .motiondetector import MotionDetector
from .processor import Processor


class PiClassifier(Processor):
    """ Classifies frames from leptond """

    PROCESS_FRAME = 3
    NUM_CONCURRENT_TRACKS = 1
    DEBUG_EVERY = 100
    MAX_CONSEC = 3
    # after every MAX_CONSEC frames skip this many frames
    # this gives the cpu a break
    SKIP_FRAMES = 7

    def __init__(self, config, thermal_config, classifier, headers):
        self.headers = headers
        self.frame_num = 0
        self.clip = None
        self.tracking = False
        self.enable_per_track_information = False
        self.rolling_track_classify = {}
        self.skip_classifying = 0
        self.classified_consec = 0
        self.config = config
        self.classifier = classifier
        self.num_labels = len(classifier.labels)

        self.predictions = Predictions(classifier.labels)
        self.preview_frames = thermal_config.recorder.preview_secs * headers.fps
        edge = self.config.tracking.edge_pixels
        self.crop_rectangle = tools.Rectangle(
            edge, edge, headers.res_x - 2 * edge, headers.res_y - 2 * edge
        )

        try:
            self.fp_index = self.classifier.labels.index("false-positive")
        except ValueError:
            self.fp_index = None
        self.track_extractor = ClipTrackExtractor(
            self.config.tracking,
            self.config.use_opt_flow,
            self.config.classify.cache_to_disk,
            keep_frames=False,
            calc_stats=False,
        )
        self.motion = thermal_config.motion
        self.min_frames = thermal_config.recorder.min_secs * headers.fps
        self.max_frames = thermal_config.recorder.max_secs * headers.fps
        self.motion_detector = MotionDetector(
            thermal_config,
            self.config.tracking.dynamic_thresh,
            CPTVRecorder(thermal_config, headers),
            headers,
        )
        self.startup_classifier()

        self._output_dir = thermal_config.recorder.output_dir
        self.meta_dir = os.path.join(thermal_config.recorder.output_dir, "metadata")
        if not os.path.exists(self.meta_dir):
            os.makedirs(self.meta_dir)

    def new_clip(self):
        self.clip = Clip(self.config.tracking, "stream")
        self.clip.video_start_time = datetime.now()
        self.clip.num_preview_frames = self.preview_frames

        self.clip.set_res(self.res_x, self.res_y)
        self.clip.set_frame_buffer(
            self.config.classify_tracking.high_quality_optical_flow,
            self.config.classify.cache_to_disk,
            self.config.use_opt_flow,
            True,
        )

        # process preview_frames
        frames = self.motion_detector.thermal_window.get_frames()
        for frame in frames:
            self.track_extractor.process_frame(self.clip, frame.copy())

    def startup_classifier(self):
        # classifies an empty frame to force loading of the model into memory

        p_frame = np.zeros((5, 48, 48), np.float32)
        self.classifier.classify_frame_with_novelty(p_frame, None)

    def get_active_tracks(self):
        """
        Gets current clips active_tracks and returns the top NUM_CONCURRENT_TRACKS order by priority
        """
        active_tracks = self.clip.active_tracks
        if len(active_tracks) <= PiClassifier.NUM_CONCURRENT_TRACKS:
            return active_tracks
        active_predictions = []
        for track in active_tracks:
            prediction = self.predictions.get_or_create_prediction(
                track, keep_all=False
            )
            active_predictions.append(prediction)

        top_priority = sorted(
            active_predictions,
            key=lambda i: i.get_priority(self.clip.frame_on),
            reverse=True,
        )

        top_priority = [
            track.track_id
            for track in top_priority[: PiClassifier.NUM_CONCURRENT_TRACKS]
        ]
        classify_tracks = [
            track for track in active_tracks if track.get_id() in top_priority
        ]
        return classify_tracks

    def identify_last_frame(self):
        """
        Runs through track identifying segments, and then returns it's prediction of what kind of animal this is.
        One prediction will be made for every active_track of the last frame.
        :return: TrackPrediction object
        """

        prediction_smooth = 0.1

        smooth_prediction = None
        smooth_novelty = None

        prediction = 0.0
        novelty = 0.0
        active_tracks = self.get_active_tracks()
        frame = self.clip.frame_buffer.get_last_frame()
        if frame is None:
            return
        thermal_reference = np.median(frame.thermal)

        for i, track in enumerate(active_tracks):
            track_prediction = self.predictions.get_or_create_prediction(
                track, keep_all=False
            )
            region = track.bounds_history[-1]
            if region.frame_number != frame.frame_number:
                logging.warning(
                    "frame doesn't match last frame {} and {}".format(
                        region.frame_number, frame.frame_number
                    )
                )
            else:
                track_data = track.crop_by_region(frame, region)
                # we use a tighter cropping here so we disable the default 2 pixel inset
                frames = preprocess_segment(
                    [track_data], [thermal_reference], default_inset=0
                )
                if frames is None:
                    logging.warning(
                        "Frame {} of track could not be classified.".format(
                            region.frame_number
                        )
                    )
                    continue
                p_frame = frames[0]
                (
                    prediction,
                    novelty,
                    state,
                ) = self.classifier.classify_frame_with_novelty(
                    p_frame, track_prediction.state
                )
                track_prediction.state = state
                if self.fp_index is not None:
                    prediction[self.fp_index] *= 0.8
                state *= 0.98
                mass = region.mass
                mass_weight = np.clip(mass / 20, 0.02, 1.0) ** 0.5
                cropped_weight = 0.7 if region.was_cropped else 1.0

                prediction *= mass_weight * cropped_weight

                if len(track_prediction.predictions) == 0:
                    if track_prediction.uniform_prior:
                        smooth_prediction = np.ones([self.num_labels]) * (
                            1 / self.num_labels
                        )
                    else:
                        smooth_prediction = prediction
                    smooth_novelty = 0.5
                else:
                    smooth_prediction = track_prediction.predictions[-1]
                    smooth_novelty = track_prediction.novelties[-1]
                    smooth_prediction = (
                        1 - prediction_smooth
                    ) * smooth_prediction + prediction_smooth * prediction
                    smooth_novelty = (
                        1 - prediction_smooth
                    ) * smooth_novelty + prediction_smooth * novelty
                track_prediction.classified_frame(
                    self.clip.frame_on, smooth_prediction, smooth_novelty
                )

    def get_recent_frame(self):
        return self.motion_detector.get_recent_frame()

    def disconnected(self):
        self.end_clip()
        self.motion_detector.disconnected()

    def skip_frame(self):
        self.skip_classifying -= 1

        if self.clip:
            self.clip.frame_on += 1

    def process_frame(self, lepton_frame):
        start = time.time()
        self.motion_detector.process_frame(lepton_frame)
        if self.motion_detector.recorder.recording:
            if self.clip is None:
                self.new_clip()
            self.track_extractor.process_frame(
                self.clip, lepton_frame.pix, self.motion_detector.ffc_affected
            )
            if self.motion_detector.ffc_affected or self.clip.on_preview():
                self.skip_classifying = PiClassifier.SKIP_FRAMES
                self.classified_consec = 0
            elif (
                self.motion_detector.ffc_affected is False
                and self.clip.active_tracks
                and self.skip_classifying <= 0
                and not self.clip.on_preview()
            ):
                self.identify_last_frame()
                self.classified_consec += 1
                if self.classified_consec == PiClassifier.MAX_CONSEC:
                    self.skip_classifying = PiClassifier.SKIP_FRAMES
                    self.classified_consec = 0

        elif self.clip is not None:
            self.end_clip()

        self.skip_classifying -= 1
        self.frame_num += 1
        end = time.time()
        timetaken = end - start
        if (
            self.motion_detector.can_record()
            and self.frame_num % PiClassifier.DEBUG_EVERY == 0
        ):
            logging.info(
                "fps {}/sec time to process {}ms cpu % {} memory % {}".format(
                    round(1 / timetaken, 2),
                    round(timetaken * 1000, 2),
                    psutil.cpu_percent(),
                    psutil.virtual_memory()[2],
                )
            )

    def create_mp4(self):
        previewer = Previewer(self.config, "classified")
        previewer.export_clip_preview(
            self.clip.get_id() + ".mp4", self.clip, self.predictions
        )

    def end_clip(self):
        if self.clip:
            for _, prediction in self.predictions.prediction_per_track.items():
                if prediction.max_score:
                    logging.info(
                        "Clip {} {}".format(
                            self.clip.get_id(),
                            prediction.description(self.predictions.labels),
                        )
                    )
            self.save_metadata()
            self.predictions.clear_predictions()
            self.clip = None
            self.tracking = False

    def save_metadata(self):
        filename = datetime.now().strftime("%Y%m%d.%H%M%S.%f.meta")

        # record results in text file.
        save_file = {}
        start, end = self.clip.start_and_end_time_absolute()
        save_file["start_time"] = start.isoformat()
        save_file["end_time"] = end.isoformat()
        save_file["temp_thresh"] = self.clip.temp_thresh
        save_file["algorithm"] = {}
        save_file["algorithm"]["model"] = self.config.classify.model
        save_file["algorithm"]["tracker_version"] = ClipTrackExtractor.VERSION
        save_file["tracks"] = []
        for track in self.clip.tracks:
            track_info = {}
            prediction = self.predictions.prediction_for(track.get_id())
            start_s, end_s = self.clip.start_and_end_in_secs(track)
            save_file["tracks"].append(track_info)
            track_info["start_s"] = round(start_s, 2)
            track_info["end_s"] = round(end_s, 2)
            track_info["num_frames"] = track.frames
            track_info["frame_start"] = track.start_frame
            track_info["frame_end"] = track.end_frame
            if prediction and prediction.best_label_index is not None:
                track_info["label"] = self.classifier.labels[
                    prediction.best_label_index
                ]
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
                track_time = round(region.frame_number / self.clip.frames_per_second, 2)
                positions.append([track_time, region])
            track_info["positions"] = positions

        with open(os.path.join(self.meta_dir, filename), "w") as f:
            json.dump(save_file, f, indent=4, cls=tools.CustomJSONEncoder)

    @property
    def res_x(self):
        return self.headers.res_x

    @property
    def res_y(self):
        return self.headers.res_y

    @property
    def output_dir(self):
        return self._output_dir

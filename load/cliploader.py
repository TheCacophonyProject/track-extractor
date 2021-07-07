"""
classifier-pipeline - this is a server side component that manipulates cptv
files and to create a classification model of animals present
Copyright (C) 2018, The Cacophony Project

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program. If not, see <http://www.gnu.org/licenses/>.
"""


import os
import logging
import multiprocessing
import time
from multiprocessing import Process, Queue
import traceback
import json
from track.region import Region
from ml_tools import tools

from ml_tools.kerasmodel import KerasModel
from ml_tools.previewer import Previewer
from ml_tools.trackdatabase import TrackDatabase
from .clip import Clip
from .cliptrackextractor import ClipTrackExtractor
from track.track import Track, TrackChannels
from classify.trackprediction import TrackPrediction
from ml_tools.imageprocessing import overlay_image, clear_frame

import numpy as np


def process_job(loader, queue, model_file=None):
    i = 0
    classifier = None
    if model_file is not None:
        classifier = KerasModel()
        classifier.load_model(model_file)
    while True:
        i += 1
        clip_id = queue.get()
        try:
            if clip_id == "DONE":
                break
            else:
                loader.process_file(str(clip_id), classifier=classifier)
            if i % 50 == 0:
                logging.info("%s jobs left", queue.qsize())
        except Exception as e:
            logging.error("Process_job error %s %s", clip_id, e)
            traceback.print_exc()


def prediction_job(queue, db, model_file):
    classifier = KerasModel()
    classifier.load_weights(model_file)
    logging.info("Loaded model")
    while True:
        clip_id = queue.get()
        if clip_id == "DONE":
            break
        else:
            try:
                add_predictions(db, clip_id, classifier)
            except Exception as e:
                logging.error("Process_job error %s %s", clip_id, e)
                traceback.print_exc()


def add_predictions(db, clip_id, classifier):
    tracks = db.get_clip_tracks(clip_id)
    clip_meta = db.get_clip_meta(clip_id)
    crop_region = clip_meta.get("edge_pixels", 0)
    res_x = clip_meta.get("res_x", 160)
    res_y = clip_meta.get("res_y", 120)
    crop_region = Rectangle(edge, edge, res_x - 2 * edge, res_y - 2 * edge)
    overlay = None
    for track in tracks:
        track_data = db.get_track(clip_id, track["id"])
        if classifier.use_movement:
            overlay = db.get_overlay(clip_id, track["id"])
        regions = []
        for i, rect in enumerate(track["bounds_history"]):
            region = Region.region_from_array(rect)
            region.mass = track["mass_history"][i]
            regions.append(region)

        track_prediction = classifier.classify_cropped_data(
            track["id"],
            track["start_frame"],
            track_data,
            clip_meta["frame_temp_median"][track["start_frame"] : track["frames"]],
            regions,
            overlay,
            crop_region,
        )
        db.add_prediction(clip_id, track["id"], track_prediction)


class ClipLoader:
    def __init__(self, config, reprocess=False, calculate_predictions=False):

        self.config = config
        os.makedirs(self.config.tracks_folder, mode=0o775, exist_ok=True)
        self.database = TrackDatabase(
            os.path.join(self.config.tracks_folder, "dataset.hdf5")
        )
        self.calculate_predictions = calculate_predictions
        self.reprocess = reprocess
        self.compression = (
            tools.gzip_compression if self.config.load.enable_compression else None
        )
        self.track_config = config.tracking
        # number of threads to use when processing jobs.
        self.workers_threads = config.worker_threads
        self.previewer = Previewer.create_if_required(config, config.load.preview)
        self.track_extractor = ClipTrackExtractor(
            self.config.tracking,
            self.config.use_opt_flow
            or config.load.preview == Previewer.PREVIEW_TRACKING,
            self.config.load.cache_to_disk,
            high_quality_optical_flow=self.config.load.high_quality_optical_flow,
        )

    def add_predictions(self):
        # run the model on all clips that dont have predictions
        job_queue = Queue()
        processes = []
        for i in range(max(1, self.workers_threads)):
            p = Process(
                target=prediction_job,
                args=(job_queue, self.database, self.config.classify.model),
            )
            processes.append(p)
            p.start()
        clips = self.database.get_all_clip_ids()
        for clip in clips:
            if not self.database.has_prediction(clip):
                job_queue.put(clip)
        logging.info("Processing %d", job_queue.qsize())

        for i in range(len(processes)):
            job_queue.put("DONE")
        for process in processes:
            try:
                process.join()
            except KeyboardInterrupt:
                logging.info("KeyboardInterrupt, terminating.")
                for process in processes:
                    process.terminate()
                exit()

    def process_all(self, root):
        job_queue = Queue()
        processes = []
        for i in range(max(1, self.workers_threads)):
            p = Process(
                target=process_job,
                args=(
                    self,
                    job_queue,
                    self.config.classify.model if self.calculate_predictions else None,
                ),
            )
            processes.append(p)
            p.start()
        if root is None:
            root = self.config.source_folder

        for folder_path, _, files in os.walk(root):
            for name in files:
                if os.path.splitext(name)[1] == ".cptv":
                    full_path = os.path.join(folder_path, name)
                    job_queue.put(full_path)

        logging.info("Processing %d", job_queue.qsize())
        for i in range(len(processes)):
            job_queue.put("DONE")
        for process in processes:
            try:
                process.join()
            except KeyboardInterrupt:
                logging.info("KeyboardInterrupt, terminating.")
                for process in processes:
                    process.terminate()
                exit()

    def _get_dest_folder(self, filename):
        return os.path.join(self.config.tracks_folder, get_distributed_folder(filename))

    def _export_tracks(self, full_path, clip, classifier=None):
        """
        Writes tracks to a track database.
        :param database: database to write track to.
        """
        # overwrite any old clips.
        # Note: we do this even if there are no tracks so there there will be a blank clip entry as a record
        # that we have processed it.
        self.database.create_clip(clip)
        for track in clip.tracks:
            start_time, end_time = clip.start_and_end_time_absolute(
                track.start_s, track.end_s
            )
            original_thermal = []
            cropped_data = []
            for region in track.bounds_history:
                frame = clip.frame_buffer.get_frame(region.frame_number)
                original_thermal.append(frame.thermal)
                cropped = frame.crop_by_region(region)
                # zero out the filtered channel
                if not self.config.load.include_filtered_channel:
                    cropped.filtered = np.zeros(cropped.thermal.shape)
                cropped_data.append(cropped)

            overlay = overlay_image(
                cropped_data,
                track.bounds_history,
                dim=(120, 160),
                require_movement=True,
            )

            important_frames = get_important_frames(
                clip.ffc_frames,
                [bounds.mass for bounds in track.bounds_history],
                self.config.build.train_min_mass,
                cropped_data,
            )
            self.database.add_track(
                clip.get_id(),
                track,
                cropped_data,
                overlay,
                important_frames,
                opts=self.compression,
                original_thermal=original_thermal,
                start_time=start_time,
                end_time=end_time,
            )
            if classifier:
                track_prediction = classifier.classify_track(clip, track)
                track.add_prediction_info(track_prediction)

    def _filter_clip_tracks(self, clip_metadata):
        """
        Removes track metadata for tracks which are invalid. Tracks are invalid
        if they aren't confident or they are in the excluded_tags list.
        Returns valid tracks
        """

        tracks_meta = clip_metadata.get("Tracks", [])
        valid_tracks = [
            track for track in tracks_meta if self._track_meta_is_valid(track)
        ]
        clip_metadata["Tracks"] = valid_tracks
        return valid_tracks

    def _track_meta_is_valid(self, track_meta):
        """
        Tracks are valid if their confidence meets the threshold and they are
        not in the excluded_tags list, defined in the config.
        """
        min_confidence = self.track_config.min_tag_confidence
        track_data = track_meta.get("data")
        if not track_data:
            return False

        track_tags = []
        if "TrackTags" in track_meta:
            track_tags = track_meta["TrackTags"]
        excluded_tags = [
            tag
            for tag in track_tags
            if not tag.get("automatic", False) and tag in self.config.excluded_tags
        ]

        if len(excluded_tags) > 0:
            return False

        track_tag = Track.get_best_human_tag(
            track_tags, self.config.load.tag_precedence, min_confidence
        )
        if track_tag is None:
            return False
        tag = track_tag.get("what")
        confidence = track_tag.get("confidence", 0)
        return tag and tag not in excluded_tags and confidence >= min_confidence

    def process_file(self, filename, classifier=None):
        start = time.time()
        base_filename = os.path.splitext(os.path.basename(filename))[0]

        logging.info(f"processing %s", filename)

        destination_folder = self._get_dest_folder(base_filename)
        # delete any previous files
        tools.purge(destination_folder, base_filename + "*.mp4")

        # read metadata
        metadata_filename = os.path.join(
            os.path.dirname(filename), base_filename + ".txt"
        )

        if not os.path.isfile(metadata_filename):
            logging.error("No meta data found for %s", metadata_filename)
            return

        metadata = tools.load_clip_metadata(metadata_filename)
        if not self.reprocess and self.database.has_clip(str(metadata["id"])):
            if not self.database.has_prediction(str(metadata["id"])) and classifier:
                logging.info("Adding predictions to %s", filename)
                add_predictions(self.database, str(metadata["id"]), classifier)
            else:
                logging.warning("Already loaded %s", filename)
            return

        valid_tracks = self._filter_clip_tracks(metadata)
        if not valid_tracks:
            logging.error("No valid track data found for %s", filename)
            return

        clip = Clip(self.track_config, filename)
        clip.load_metadata(
            metadata,
            self.config.load.include_filtered_channel,
            self.config.load.tag_precedence,
        )

        if not self.track_extractor.parse_clip(clip):
            logging.error("No valid clip found for %s", filename)
            return

        # , self.config.load.cache_to_disk, self.config.use_opt_flow

        if self.track_config.enable_track_output:
            self._export_tracks(filename, clip, classifier)

        # write a preview
        if self.previewer:
            os.makedirs(destination_folder, mode=0o775, exist_ok=True)

            preview_filename = base_filename + "-preview" + ".mp4"
            preview_filename = os.path.join(destination_folder, preview_filename)
            self.previewer.create_individual_track_previews(preview_filename, clip)
            self.previewer.export_clip_preview(preview_filename, clip)

        if self.track_config.verbose:
            num_frames = len(clip.frame_buffer.frames)
            ms_per_frame = (time.time() - start) * 1000 / max(1, num_frames)
            self._log_message(
                "Tracks {}.  Frames: {}, Took {:.1f}ms per frame".format(
                    len(clip.tracks), num_frames, ms_per_frame
                )
            )

    def _log_message(self, message):
        """Record message in stdout.  Will be printed if verbose is enabled."""
        # note, python has really good logging... I should probably make use of this.
        if self.track_config.verbose:
            logging.info(message)


def get_distributed_folder(name, num_folders=256, seed=31):
    """Creates a hash of the name then returns the modulo num_folders"""
    str_bytes = str.encode(name)
    hash_code = 0
    for byte in str_bytes:
        hash_code = hash_code * seed + byte
    return "{:02x}".format(hash_code % num_folders)


def get_important_frames(ffc_frames, mass_history, min_mass=None, frame_data=None):
    clear_frames = []
    lower_mass = np.percentile(mass_history, q=25)
    upper_mass = np.percentile(mass_history, q=75)
    for i, mass in enumerate(mass_history):
        if i in ffc_frames:
            continue
        if (
            min_mass is None
            or mass >= min_mass
            and mass >= lower_mass
            and mass <= upper_mass
        ):
            if frame_data is not None:
                if not clear_frame(frame_data[i]):
                    continue
            clear_frames.append(i)
    return clear_frames

"""

Author Matthew Aitchison

Date December 2017

Handles reading and writing tracks (or segments) to a large database.  Uses HDF5 as a backing store.

"""
import h5py
import os
import logging
import filelock
import datetime
import json
from dateutil.parser import parse as parse_date

from multiprocessing import Lock
import numpy as np
from track.framebuffer import Frame
from track.region import Region

special_datasets = ["background_frame", "predictions", "overlay"]


class HDF5Manager:
    """Class to handle locking of HDF5 files."""

    LOCK_FILE = "/var/lock/classifier-hdf5.lock"

    def __init__(self, db, mode="r"):
        self.mode = mode
        self.f = None
        self.db = db
        self.lock = filelock.FileLock(HDF5Manager.LOCK_FILE, timeout=60 * 3)
        filelock.logger().setLevel(logging.ERROR)

    def __enter__(self):
        # note: we might not have to lock when in read only mode?
        # this could improve performance
        self.lock.acquire()
        self.f = h5py.File(self.db, self.mode)
        return self.f

    def __exit__(self, exc_type, exc_val, exc_tb):
        try:
            self.f.close()
        finally:
            self.lock.release()


class TrackDatabase:
    def __init__(self, database_filename):
        """
        Initialises given database.  If database does not exist an empty one is created.
        :param database_filename: filename of database
        """

        self.database = database_filename

        if not os.path.exists(database_filename):
            logging.info("Creating new database %s", database_filename)
            f = h5py.File(database_filename, "w")
            f.create_group("clips")
            f.close()

    def has_clip(self, clip_id):
        """
        Returns if database contains track information for given clip
        :param clip_id: name of clip
        :return: If the database contains given clip
        """
        with HDF5Manager(self.database) as f:
            clips = f["clips"]
            has_record = clip_id in clips and "finished" in clips[clip_id].attrs
        return has_record

    def has_prediction(self, clip_id):
        with HDF5Manager(self.database, "a") as f:
            clips = f["clips"]
            # has_record = clip_id in clips and "finished" in clips[clip_id].attrs
            clip = clips[clip_id]
            # if has_record:
            return clip.attrs.get("has_prediction", False)
        return False

    def get_labels(self):
        with HDF5Manager(self.database) as f:
            return f.attrs.get("labels", None)

    def set_labels(self, labels):
        with HDF5Manager(self.database, "a") as f:
            f.attrs["labels"] = labels

    def create_clip(self, clip, overwrite=True):
        """
        Creates a clip entry in database.
        :param clip_id: id of the clip
        :param tracker: if provided stats from tracker are used for the clip stats
        :param overwrite: Overwrites existing clip (if it exists).
        """
        print("creating clip {}".format(clip.get_id()))
        clip_id = str(clip.get_id())
        with HDF5Manager(self.database, "a") as f:
            clips = f["clips"]
            if overwrite and clip_id in clips:
                del clips[clip_id]
            group = clips.create_group(clip_id)

            if clip is not None:
                if clip.background is not None:
                    height, width = clip.background.shape
                    background_frame = group.create_dataset(
                        "background_frame",
                        (height, width),
                        chunks=(height, width),
                        dtype=clip.background.dtype,
                    )
                    background_frame[:, :] = clip.background
                group_attrs = group.attrs

                # group_attrs.update(clip.stats)
                group_attrs["filename"] = clip.source_file
                group_attrs["start_time"] = clip.video_start_time.isoformat()
                group_attrs["background_thresh"] = clip.background_thresh

                if clip.res_x and clip.res_y:
                    group_attrs["res_x"] = clip.res_x
                    group_attrs["res_y"] = clip.res_y
                if clip.crop_rectangle:
                    group_attrs["edge_pixels"] = clip.crop_rectangle.left

                group_attrs["mean_background_value"] = clip.stats.mean_background_value
                group_attrs["threshold"] = clip.stats.threshold
                group_attrs["max_temp"] = clip.stats.max_temp
                group_attrs["min_temp"] = clip.stats.min_temp
                group_attrs["mean_temp"] = clip.stats.mean_temp
                group_attrs["filtered_deviation"] = clip.stats.filtered_deviation
                group_attrs["filtered_sum"] = clip.stats.filtered_sum
                group_attrs["temp_thresh"] = clip.stats.temp_thresh

                if not clip.background_is_preview:
                    group_attrs["average_delta"] = clip.stats.average_delta
                    group_attrs["is_static"] = clip.stats.is_static_background
                group_attrs["frame_temp_min"] = clip.stats.frame_stats_min
                group_attrs["frame_temp_max"] = clip.stats.frame_stats_max
                group_attrs["frame_temp_median"] = clip.stats.frame_stats_median
                group_attrs["frame_temp_mean"] = clip.stats.frame_stats_mean

                if clip.device:
                    group_attrs["device"] = clip.device
                group_attrs["frames_per_second"] = clip.frames_per_second
                if clip.location and clip.location.get("coordinates") is not None:
                    group_attrs["location"] = clip.location["coordinates"]
                if clip.tags:
                    clip_tags = []
                    for track in clip.tags:
                        if track["what"]:
                            clip_tags.append(track["what"])
                        elif track["detail"]:
                            clip_tags.append(track["detail"])
                    group_attrs["tags"] = clip_tags
                group_attrs["ffc_frames"] = clip.ffc_frames

            f.flush()
            group.attrs["finished"] = True

    def latest_date(self):
        start_time = None

        with HDF5Manager(self.database) as f:
            clips = f["clips"]
            results = {}
            for clip_id in clips:
                clip_start = clips[clip_id].attrs["start_time"]
                if clip_start:
                    if start_time is None or clip_start > start_time:
                        start_time = clip_start

        return start_time

    def get_all_clip_ids(self):
        """
        Returns a list of clip_id, track_number pairs.
        """
        with HDF5Manager(self.database) as f:
            clips = f["clips"]
            results = {}
            for clip_id in clips:
                results[clip_id] = [track_id for track_id in clips[clip_id]]
        return results

    def get_all_track_ids(self, before_date=None, after_date=None):
        """
        Returns a list of clip_id, track_number pairs.
        """
        with HDF5Manager(self.database) as f:
            clips = f["clips"]
            result = []
            for clip_id in clips:
                clip = clips[clip_id]
                if not clip.attrs.get("finished"):
                    continue
                date = parse_date(clip.attrs["start_time"])
                if before_date and date >= before_date:
                    continue
                if after_date and date < after_date:
                    continue
                for track in clip:
                    if track not in special_datasets:
                        result.append((clip_id, track))
        return result

    def get_track_meta(self, clip_id, track_number):
        """
        Gets metadata for given track
        :param clip_id:
        :param track_number:
        :return:
        """
        with HDF5Manager(self.database) as f:
            dataset = f["clips"][clip_id][str(track_number)]
            result = hdf5_attributes_dictionary(dataset)
            result["id"] = track_number
        return result

    def get_track_predictions(self, clip_id, track_number):
        """
        Gets metadata for given track
        :param clip_id:
        :param track_number:
        :return:
        """
        with HDF5Manager(self.database) as f:
            track = f["clips"][clip_id][str(track_number)]
            if "predictions" in track:
                return track["predictions"][:]
        return None

    def get_clip_background(self, clip_id):
        with HDF5Manager(self.database) as f:
            clip = f["clips"][str(clip_id)]
            if "background_frame" in clip:
                return clip["background_frame"][:]
        return None

    def get_clip_meta(self, clip_id):
        """
        Gets metadata for given clip
        :param clip_id:
        :return:
        """

        with HDF5Manager(self.database) as f:
            dataset = f["clips"][str(clip_id)]
            result = hdf5_attributes_dictionary(dataset)
            result["tracks"] = len(dataset)
        return result

    def get_clip_tracks(self, clip_id):
        """
        Gets metadata for given clip
        :param clip_id:
        :return:
        """
        tracks = []
        with HDF5Manager(self.database) as f:
            dataset = f["clips"][str(clip_id)]
            for track_id in dataset:
                if track_id in special_datasets:
                    continue
                tracks.append(hdf5_attributes_dictionary(dataset[track_id]))
        return tracks

    def get_tag(self, clip_id, track_number):
        with HDF5Manager(self.database) as f:
            clips = f["clips"]
            track_node = clips[str(clip_id)][str(track_number)]
            return track_node.attrs["tag"]

    def get_frame(self, clip_id, track_id, frame, original=False):
        frames = self.get_track(
            clip_id, track_id, frame_numbers=[frame], original=original
        )
        if len(frames) == 1:
            return frames[0]
        return None

    def get_track(
        self,
        clip_id,
        track_number,
        start_frame=None,
        end_frame=None,
        original=False,
        frame_numbers=None,
    ):
        """
        Fetches a track data from database with optional slicing.
        :param clip_id: id of the clip
        :param track_number: id of the track
        :param start_frame: first frame of slice to return (inclusive).
        :param end_frame: last frame of slice to return (exclusive).
        :return: a list of numpy arrays of shape [channels, height, width] and of type np.int16
        """
        with HDF5Manager(self.database) as f:
            clips = f["clips"]
            track_node = clips[str(clip_id)][str(track_number)]
            bounds = track_node.attrs["bounds_history"]
            if start_frame is None:
                start_frame = 0
            if end_frame is None:
                end_frame = track_node.attrs["frames"]

            result = []
            if original:
                track_node = track_node["original"]
            else:
                if "cropped" in track_node:
                    track_node = track_node["cropped"]

            if frame_numbers is None:
                frame_iter = range(start_frame, end_frame)
            else:
                frame_iter = iter(frame_numbers)

            for frame_number in frame_iter:
                frame = track_node[str(frame_number)][:, :, :]

                region = Region.region_from_array(bounds[frame_number])
                result.append(
                    Frame.from_array(
                        frame, frame_number, flow_clipped=True, region=region
                    )
                )
        return result

    def remove_clip(self, clip_id):
        """
        Deletes clip from database.
        Note, as per hdf5 the space will not be recovered.  If many files are deleted repacking the dataset might
        be a good idea.
        :param clip_id: id of clip to remove
        :returns: true if clip was deleted, false if it could not be found.
        """
        with HDF5Manager(self.database, "a") as f:
            clips = f["clips"]
            if clip_id in clips:
                del clips[clip_id]
                return True
            else:
                return False

    def add_prediction(self, clip_id, track_id, track_prediction):
        with HDF5Manager(self.database, "a") as f:
            clip = f["clips"][(str(clip_id))]
            track_node = clip[str(track_id)]
            predicted_tag = track_prediction.predicted_tag()
            if track_prediction.num_frames_classified > 0:
                self.all_class_confidences = track_prediction.class_confidences()
                predictions = np.int16(
                    np.around(100 * np.array(track_prediction.predictions))
                )
                predicted_confidence = int(round(100 * track_prediction.max_score))

                self.add_prediction_data(
                    track_node,
                    predictions,
                    predicted_tag,
                    predicted_confidence,
                    labels=track_prediction.labels,
                )
            clip.attrs["has_prediction"] = True

    def add_prediction_data(
        self, track, predictions, predicted_tag, score, labels=None
    ):
        """
        Add prediction data as a dataset to the track
        data should be  an array of int16 array
        """
        track_attrs = track.attrs
        if predicted_tag is not None:
            track_attrs["correct_prediction"] = track_attrs["tag"] == predicted_tag
            track_attrs["predicted"] = predicted_tag
        track_attrs["predicted_confidence"] = score

        pred_data = track.create_dataset(
            "predictions",
            predictions.shape,
            chunks=predictions.shape,
            dtype=predictions.dtype,
        )
        pred_data[:, :] = predictions
        if labels is not None:
            track_attrs["prediction_classes"] = labels
        track_attrs["has_prediction"] = True

    def add_track(
        self,
        clip_id,
        track,
        cropped_data,
        overlay,
        important_frames,
        opts=None,
        original_thermal=None,
        start_time=None,
        end_time=None,
    ):
        """
        Adds track to database.
        :param clip_id: id of the clip to add track to write
        :param cropped_data: data for track, list of numpy arrays of shape [channels, height, width]
        :param track: the original track record, used to get stats for track
        :param opts: additional parameters used when creating dataset, if not provided defaults to no compression.
        """

        track_id = str(track.get_id())
        if opts is None:
            opts = {}
        frames = len(cropped_data)
        with HDF5Manager(self.database, "a") as f:
            clips = f["clips"]
            clip_node = clips[clip_id]
            has_prediction = False
            track_node = clip_node.create_group(track_id)
            cropped_frame = track_node.create_group("cropped")
            thermal_frame = track_node.create_group("original")

            # write each frame out individually, as they will probably be different sizes.
            original = None
            for frame_i, cropped in enumerate(cropped_data):
                if original_thermal is not None:
                    original = original_thermal[frame_i]
                channels = cropped.channels
                height, width = cropped.shape
                # using a chunk size of 1 for channels has the advantage that we can quickly load just one channel
                chunks = (1, height, width)
                dims = (channels, height, width)
                frame_node = cropped_frame.create_dataset(
                    str(frame_i), dims, chunks=chunks, **opts, dtype=np.int16
                )
                frame_node[:, :, :] = cropped.as_array()
                if original is not None:
                    thermal_node = thermal_frame.create_dataset(
                        str(frame_i),
                        original.shape,
                        chunks=original.shape,
                        **opts,
                        dtype=np.int16,
                    )
                    thermal_node[:, :] = original
            # write out attributes
            if track:
                track_stats = track.get_stats()
                node_attrs = track_node.attrs
                node_attrs["id"] = track_id
                if track.track_tags:
                    node_attrs["track_tags"] = json.dumps(track.track_tags)

                node_attrs["tag"] = track.tag
                node_attrs["frames"] = frames
                node_attrs["start_frame"] = track.start_frame
                node_attrs["end_frame"] = track.end_frame
                if track.predictions is not None:
                    self.add_prediction_data(
                        track_node,
                        track.predictions,
                        track.predicted_tag,
                        track.predicted_confidence,
                        labels=track.prediction_classes,
                    )
                    has_prediction = True
                if track.confidence:
                    node_attrs["confidence"] = track.confidence
                if start_time:
                    node_attrs["start_time"] = start_time.isoformat()
                if end_time:
                    node_attrs["end_time"] = end_time.isoformat()

                for name, value in track_stats._asdict().items():
                    node_attrs[name] = value
                # frame history
                node_attrs["mass_history"] = np.int32(
                    [bounds.mass for bounds in track.bounds_history]
                )
                node_attrs["bounds_history"] = np.int16(
                    [
                        [bounds.left, bounds.top, bounds.right, bounds.bottom]
                        for bounds in track.bounds_history
                    ]
                )

                node_attrs["important_frames"] = np.uint16(important_frames)
                if "overlay" not in track_node:
                    overlay_node = track_node.create_dataset(
                        "overlay", overlay.shape, chunks=overlay.shape, dtype=np.float32
                    )
                else:
                    overlay_node = track_node["overlay"]
                overlay_node[:, :] = overlay

            f.flush()

            # mark the record as have been writen to.
            # this means if we are interupted part way through the track will be overwritten
            clip_node.attrs["finished"] = True
            clip_node.attrs["has_prediction"] = has_prediction

    def get_overlay(self, clip_id, track_id):
        with HDF5Manager(self.database, "r") as f:
            clip = f["clips"][str(clip_id)]
            track = clip[str(track_id)]
            return track["overlay"][:]


def hdf5_attributes_dictionary(dataset):
    result = {}
    for key, value in dataset.attrs.items():
        result[key] = value
    return result

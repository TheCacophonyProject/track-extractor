"""

Author Matthew Aitchison

Date December 2017

Handles reading and writing tracks (or segments) to a large database.  Uses HDF5 as a backing store.

"""
import h5py
import os
import logging
import filelock
import numpy as np
from classify.trackprediction import TrackPrediction
from dateutil.parser import parse as parse_date
from .frame import Frame

special_datasets = ["background_frame", "predictions", "overlay"]


class HDF5Manager:
    """ Class to handle locking of HDF5 files. """

    LOCK_FILE = "/var/lock/classifier-hdf5.lock"
    READ_ONLY = False

    def __init__(self, db, mode="r"):
        self.mode = mode
        self.f = None
        self.db = db
        self.lock = filelock.FileLock(HDF5Manager.LOCK_FILE, timeout=60 * 3)
        filelock.logger().setLevel(logging.ERROR)

    def __enter__(self):
        # note: we might not have to lock when in read only mode?
        # this could improve performance
        if HDF5Manager.READ_ONLY and self.mode != "r":
            raise ValueError("Only read can be done in readonly mode")
        if not HDF5Manager.READ_ONLY:
            self.lock.acquire()
        self.f = h5py.File(self.db, self.mode)
        return self.f

    def __exit__(self, exc_type, exc_val, exc_tb):
        try:
            self.f.close()
        finally:
            if not HDF5Manager.READ_ONLY:
                self.lock.release()


class TrackDatabase:
    def __init__(self, database_filename, read_only=False):
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
        HDF5Manager.READ_ONLY = read_only

    def set_read_only(self, read_only):
        HDF5Manager.READ_ONLY = read_only

    def get_labels(self):
        with HDF5Manager(self.database) as f:
            return f.attrs.get("labels", None)

    def set_labels(self, labels):
        with HDF5Manager(self.database, "a") as f:
            f.attrs["labels"] = labels

    def has_clip(self, clip_id):
        """
        Returns if database contains track information for given clip
        :param clip_id: name of clip
        :return: If the database contains given clip
        """
        with HDF5Manager(self.database) as f:
            clips = f["clips"]
            has_record = clip_id in clips and "finished" in clips[clip_id].attrs
            if has_record:
                return True
        return False

    def has_prediction(self, clip_id):
        with HDF5Manager(self.database, "a") as f:
            clips = f["clips"]
            # has_record = clip_id in clips and "finished" in clips[clip_id].attrs
            clip = clips[clip_id]
            # if has_record:
            return clip.attrs.get("has_prediction", False)
        return False

    def add_overlay(self, clip_id, track_id, overlay):
        with HDF5Manager(self.database, "a") as f:
            clip = f["clips"][str(clip_id)]
            track = clip[str(track_id)]
            chunks = overlay.shape

            dims = overlay.shape
            if "overlay" not in track:
                overlay_node = track.create_dataset(
                    "overlay", dims, chunks=chunks, dtype=np.float32
                )
            else:
                overlay_node = track["overlay"]
            overlay_node[:, :] = overlay

    def get_overlay(self, clip_id, track_id):
        with HDF5Manager(self.database, "r") as f:
            clip = f["clips"][str(clip_id)]
            track = clip[str(track_id)]
            return track["overlay"][:]

    def add_predictions(self, clip_id, model):
        logging.info("Add_prediction waiting")
        with HDF5Manager(self.database, "r") as f:
            clip = f["clips"][str(clip_id)]
            logging.info("adding predictions for %s", clip_id)
            tracks = {}
            for track_id in clip:
                if track_id in special_datasets:
                    continue

                track_node = clip[track_id]
                track_tag = track_node.attrs.get("tag", "")
                if track_tag not in model.labels:
                    if track_tag != "":
                        logging.info(
                            "Tag not in model labels %s", track_node.get("tag")
                        )
                track_data = []
                for frame_number in track_node:
                    if frame_number in special_datasets:
                        continue
                    # we use [:,:,:] to force loading of all data.
                    track_data.append(track_node[str(frame_number)][:, :, :])
                tracks[track_id] = track_data
        clip_predictions = []

        for track_id, track_data in tracks.items():
            logging.info("Predicting %s %d", track_id, len(track_data))

            track_prediction = TrackPrediction(track_id, 0, True)
            for frame in track_data:
                prediction = model.classify_frame(np.copy(frame))
                track_prediction.classified_frame(0, prediction)
            clip_predictions.append(track_prediction)
        logging.info("Saving %s", clip_id)

        with HDF5Manager(self.database, "a") as f:
            clip = f["clips"][str(clip_id)]
            for track_prediction in clip_predictions:
                track_node = clip[str(track_prediction.track_id)]
                preds = np.int16(
                    np.around(100 * np.array(track_prediction.predictions))
                )

                self.add_prediction_data(
                    clip_id,
                    track_node,
                    preds,
                    model.labels[track_prediction.best_label_index],
                    track_prediction.max_score,
                    model.labels,
                )

            clip.attrs["has_prediction"] = True

    def add_prediction_data(
        self, clip_id, track, predictions, predicted_tag, score, labels=None
    ):
        """
        Add prediction data as a dataset to the track
        data should be  an array of int16 array
        """
        track_attrs = track.attrs
        if predicted_tag is not None:
            track_attrs["correct_prediction"] = track_attrs["tag"] == predicted_tag
            track_attrs["predicted"] = predicted_tag
        track_attrs["predicted_confidence"] = int(round(100 * score))

        pred_data = track.create_dataset(
            "predictions",
            predictions.shape,
            chunks=predictions.shape,
            dtype=predictions.dtype,
        )
        pred_data[:, :] = predictions
        if labels is not None:
            track_attrs["prediction_classes"] = labels

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
            # print(f["clips"][str(clip_id)])
            clip = f["clips"][str(clip_id)]
            if "background_frame" in clip:
                return clip["background_frame"][:]
        print("no background")
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
        channels=None,
    ):
        """
        Fetches a track data from database with optional slicing.
        :param clip_id: id of the clip
        :param track_number: id of the track
        :param start_frame: first frame of slice to return (inclusive).
        :param end_frame: last frame of slice to return (exclusive).
        :return: a list of numpy arrays of shape [channels, height, width] and of type np.int16
        """
        # try:
        with HDF5Manager(self.database) as f:
            clips = f["clips"]
            track_node = clips[str(clip_id)][str(track_number)]

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
                if channels is None:
                    frame = track_node[str(frame_number)][:, :, :]
                    result.append(
                        Frame.from_array(frame, frame_number, flow_clipped=True)
                    )
                else:
                    frame = track_node[str(frame_number)][channels, :, :]
                    result.append(Frame.from_channel(frame, channels, frame_number))

        # except:
        # return None
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

    def add_track(
        self,
        clip_id,
        track,
        track_data,
        opts=None,
        start_time=None,
        end_time=None,
        prediction=None,
        prediction_classes=None,
    ):
        """
        Adds track to database.
        :param clip_id: id of the clip to add track to write
        :param track_data: data for track, list of numpy arrays of shape [channels, height, width]
        :param track: the original track record, used to get stats for track
        :param opts: additional parameters used when creating dataset, if not provided defaults to no compression.
        """

        track_id = str(track.get_id())

        frames = len(track_data)
        with HDF5Manager(self.database, "a") as f:
            clips = f["clips"]
            clip_node = clips[clip_id]
            has_prediction = False
            track_node = clip_node.create_group(track_id)
            cropped_frame = track_node.create_group("cropped")
            thermal_frame = track_node.create_group("original")

            # write each frame out individually, as they will probably be different sizes.

            for frame_i, frame_data in enumerate(track_data):
                cropped = frame_data[1]
                original = frame_data[0]
                channels, height, width = cropped.shape

                # using a chunk size of 1 for channels has the advantage that we can quickly load just one channel
                chunks = (1, height, width)

                dims = (channels, height, width)

                if opts is not None:
                    frame_node = cropped_frame.create_dataset(
                        str(frame_i), dims, chunks=chunks, **opts, dtype=np.int16
                    )
                    thermal_node = thermal_frame.create_dataset(
                        str(frame_i),
                        original.shape,
                        chunks=original.shape,
                        **opts,
                        dtype=np.int16,
                    )
                else:
                    frame_node = cropped_frame.create_dataset(
                        str(frame_i), dims, chunks=chunks, dtype=np.int16
                    )
                    thermal_node = thermal_frame.create_dataset(
                        str(frame_i),
                        original.shape,
                        chunks=original.shape,
                        dtype=np.int16,
                    )
                thermal_node[:, :] = original
                frame_node[:, :, :] = cropped
            # write out attributes
            if track:
                track_stats = track.get_stats()
                node_attrs = track_node.attrs
                node_attrs["id"] = track_id
                if track.track_tags:
                    node_attrs["track_tags"] = [
                        track["what"] for track in track.track_tags
                    ]
                node_attrs["tag"] = track.tag
                node_attrs["frames"] = frames
                node_attrs["start_frame"] = track.start_frame
                node_attrs["end_frame"] = track.end_frame
                if track.predictions is not None:
                    self.add_prediction_data(
                        clip_id,
                        track_node,
                        track.predictions,
                        track.predicted_class,
                        track.predicted_confidence,
                        labels=track.prediction_classes,
                    )
                    has_prediction = True

                elif prediction and prediction_classes:
                    preds = np.int16(np.around(100 * np.array(prediction.predictions)))
                    self.add_prediction_data(
                        clip_id,
                        track_node,
                        preds,
                        prediction_classes[prediction.best_label_index],
                        prediction.max_score,
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

            f.flush()

            # mark the record as have been writen to.
            # this means if we are interupted part way through the track will be overwritten
            clip_node.attrs["finished"] = True
            clip_node.attrs["has_prediction"] = has_prediction


def hdf5_attributes_dictionary(dataset):
    result = {}
    for key, value in dataset.attrs.items():
        result[key] = value
    return result

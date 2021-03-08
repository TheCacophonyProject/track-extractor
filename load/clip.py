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


import datetime
import logging
import numpy as np
import os
import pytz
import cv2

from ml_tools.imageprocessing import normalize, detect_objects
from ml_tools.tools import Rectangle
from track.framebuffer import FrameBuffer
from track.track import Track
from track.region import Region
from piclassifier.motiondetector import is_affected_by_ffc


class Clip:
    PREVIEW = "preview"
    FRAMES_PER_SECOND = 9
    local_tz = pytz.timezone("Pacific/Auckland")
    VERSION = 9
    CLIP_ID = 1
    # used when calculating background, mininimum percentage the difference object
    # and background object must overlap i.e. they are a valid object
    MIN_ORIGIN_OVERLAP = 0.80

    def __init__(self, trackconfig, sourcefile, background=None, calc_stats=True):
        self._id = Clip.CLIP_ID
        Clip.CLIP_ID += 1
        Track._track_id = 1
        self.tags = None
        self.disable_background_subtraction = False
        self.frame_on = 0
        self.ffc_affected = False
        self.crop_rectangle = None
        self.num_preview_frames = 0
        self.region_history = []
        self.active_tracks = set()
        self.tracks = []
        self.filtered_tracks = []
        self.from_metadata = False
        self.video_start_time = None
        self.location = None
        self.frame_buffer = None
        self.device = None
        self.background = None
        self.background_calculated = False
        self.res_x = None
        self.res_y = None
        self.background_frames = 0
        self.background_is_preview = trackconfig.background_calc == Clip.PREVIEW
        self.config = trackconfig
        self.frames_per_second = Clip.FRAMES_PER_SECOND

        self.calc_stats = calc_stats
        self.source_file = sourcefile
        self.stats = ClipStats()
        self.camera_model = None
        self.threshold_config = None
        self.track_min_delta = None
        self.track_max_delta = None
        self.background_thresh = None
        # sets defaults
        self.set_model(None)
        if background is not None:
            self.background = background
            self._background_calculated()

    def set_model(self, camera_model):
        self.camera_model = camera_model
        threshold = self.config.motion.threshold_for_model(camera_model)
        if threshold:
            self.threshold_config = threshold
            self.set_motion_thresholds(threshold)

    def set_motion_thresholds(self, threshold):
        self.background_thresh = threshold.background_thresh
        self.temp_thresh = threshold.temp_thresh
        self.stats.threshold = self.background_thresh
        self.track_min_delta = threshold.track_min_delta
        self.track_max_delta = threshold.track_max_delta

    def _background_calculated(self):
        self.stats.mean_background_value = np.average(self.background)
        self.set_temp_thresh()
        self.background_calculated = True

    def on_preview(self):
        return not self.background_calculated

    def update_background(self, frame):
        """ updates the clip background """
        if self.background is None:
            self.background = frame
        else:
            self.background = np.minimum(self.background, frame)
        self.background_frames += 1

    def calculate_initial_diff(self, frame, initial_frames, initial_diff):
        """Compare this frame with the initial frames (to detect movement) and update the initial_diff
        frame (such that it is the maximum difference). This is essentially creating a frame which has
        all the warms moving parts from the initial frame
        """
        if initial_frames is None:
            return np.zeros((frame.shape))
        else:
            diff = initial_frames - frame
            if initial_diff is not None:
                initial_diff = np.maximum(initial_diff, diff)
            else:
                initial_diff = diff
        return initial_diff

    def calculate_background(self, frame_reader):
        """
        Calculate background by reading whole clip and grouping into sets of
        9 frames. Take the average of these 9 frames and use the minimum
        over the sets as the initial background
        Also check for animals in the background by checking for connected components in
        the intital_diff frame - this is the maximum change between first average frame and all other average frames in the clip
        """
        frames = []
        if frame_reader.background_frames > 0:
            for frame in frame_reader:
                if frame.background_frame:
                    frames.append(frame.pix)
                else:
                    break
            frame_average = np.average(frames, axis=0)
            self.update_background(frame_average)
            self._background_calculated()
            return

        initial_frames = None
        initial_diff = None
        for frame in frame_reader:
            ffc_affected = is_affected_by_ffc(frame)
            if ffc_affected:
                continue
            frames.append(frame.pix)
            if len(frames) == 9:
                frame_average = np.average(frames, axis=0)
                self.update_background(frame_average)
                initial_diff = self.calculate_initial_diff(
                    frame_average, initial_frames, initial_diff
                )
                if initial_frames is None:
                    initial_frames = frame_average

                frames = []

        if len(frames) > 0:
            frame_average = np.average(frames, axis=0)
            self.update_background(frame_average)
            initial_diff = self.calculate_initial_diff(
                frame_average, initial_frames, initial_diff
            )
            if initial_frames is None:
                initial_frames = frame_average
        frames = []
        np.clip(initial_diff, 0, None, out=initial_diff)
        initial_frames = self.remove_background_animals(initial_frames, initial_diff)

        self.update_background(initial_frames)
        self._background_calculated()

    def remove_background_animals(self, initial_frame, initial_diff):
        """
        Try and remove animals that are already in the initial frames, by
        checking for connected components in the intital_diff frame
        (this is the maximum change between first frame and all other frames in the clip)
        """
        # remove some noise
        initial_diff[initial_diff < self.background_thresh] = 0
        initial_diff[initial_diff > 255] = 255
        initial_diff = np.uint8(initial_diff)
        initial_diff = cv2.fastNlMeansDenoising(initial_diff, None)

        _, lower_mask, lower_objects = detect_objects(initial_diff, otsus=True)

        max_region = Region(0, 0, self.res_x, self.res_y)
        for component in lower_objects[1:]:
            region = Region(component[0], component[1], component[2], component[3])
            region.enlarge(2, max=max_region)
            if region.width >= self.res_x or region.height >= self.res_y:
                logging.info(
                    "Background animal bigger than max, probably false positive %s %s",
                    region,
                    component[4],
                )
                continue
            background_region = region.subimage(initial_frame)
            norm_back = background_region.copy()
            norm_back, _ = normalize(norm_back, new_max=255)
            sub_components, sub_connected, sub_stats = detect_objects(
                norm_back, otsus=True
            )
            if len(sub_stats) <= 1:
                continue
            overlap_image = region.subimage(lower_mask) * 255
            overlap_pixels = np.sum(sub_connected[overlap_image > 0])
            overlap_pixels = overlap_pixels / float(component[4])

            # filter out components which are too big, or dont match original causes
            # for filtering
            if (
                overlap_pixels < Clip.MIN_ORIGIN_OVERLAP
                or sub_stats[1][4] == 0
                or sub_stats[1][4] == region.area
            ):
                logging.info(
                    "Invalid components mass: %s, components: %s region area %s overlap %s",
                    sub_stats[1][4],
                    sub_components,
                    region.area,
                    overlap_pixels,
                )
                continue

            sub_connected[sub_connected > 0] = 1
            # remove this component from the background by painting with
            # colours of neighbouring pixels
            background_region[:] = cv2.inpaint(
                np.float32(background_region),
                np.uint8(sub_connected),
                3,
                cv2.INPAINT_TELEA,
            )
        return initial_frame

    def _add_active_track(self, track):
        self.active_tracks.add(track)
        self.tracks.append(track)

    def get_id(self):
        return str(self._id)

    def set_temp_thresh(self):
        if self.config.motion.dynamic_thresh:
            min_temp = self.threshold_config.min_temp_thresh
            max_temp = self.threshold_config.max_temp_thresh
            if max_temp:
                self.temp_thresh = min(max_temp, self.stats.mean_background_value)
            else:
                self.temp_thresh = self.stats.mean_background_value
            if min_temp:
                self.temp_thresh = max(min_temp, self.temp_thresh)
            self.stats.temp_thresh = self.temp_thresh
        else:
            self.temp_thresh = self.config.motion.temp_thresh

    def set_video_stats(self, video_start_time):
        """
        Extracts useful statics from video clip.
        """
        self.video_start_time = video_start_time
        self.stats.date_time = video_start_time.astimezone(Clip.local_tz)
        self.stats.is_night = (
            video_start_time.astimezone(Clip.local_tz).time().hour >= 2
        )

    def load_metadata(self, metadata, include_filtered_channel, tag_precedence):
        self._id = metadata["id"]
        device_meta = metadata.get("Device")
        self.tags = metadata.get("Tags")

        if device_meta:
            self.device = device_meta.get("devicename")
        else:
            self.device = os.path.splitext(os.path.basename(self.source_file))[0].split(
                "-"
            )[-1]

        self.location = metadata.get("location")
        tracks = self.load_tracks_meta(
            metadata, include_filtered_channel, tag_precedence
        )
        self.from_metadata = True
        self.tracks = set(tracks)

    def load_tracks_meta(self, metadata, include_filtered_channel, tag_precedence):
        tracks_meta = metadata["Tracks"]
        tracks = []
        # get track data
        for track_meta in tracks_meta:
            track = Track(self.get_id())
            if track.load_track_meta(
                track_meta,
                self.frames_per_second,
                include_filtered_channel,
                tag_precedence,
                self.config.min_tag_confidence,
            ):
                tracks.append(track)
        return tracks

    def start_and_end_in_secs(self, track):
        if track.end_s is None:
            track.end_s = (track.end_frame + 1) / self.frames_per_second

        return (track.start_s, track.end_s)

    def set_frame_buffer(self, high_quality_flow, cache_to_disk, use_flow, keep_frames):
        self.frame_buffer = FrameBuffer(
            self.source_file, high_quality_flow, cache_to_disk, use_flow, keep_frames
        )

    def set_res(self, res_x, res_y):
        self.res_x = res_x
        self.res_y = res_y
        self._set_crop_rectangle()
        for track in self.tracks:
            track.crop_rectangle = self.crop_rectangle

    def _set_crop_rectangle(self):

        edge = self.config.edge_pixels
        self.crop_rectangle = Rectangle(
            edge, edge, self.res_x - 2 * edge, self.res_y - 2 * edge
        )

    def start_and_end_time_absolute(self, start_s=0, end_s=None):
        if not end_s:
            end_s = len(self.frame_buffer.frames) / self.frames_per_second
        return (
            self.video_start_time + datetime.timedelta(seconds=start_s),
            self.video_start_time + datetime.timedelta(seconds=end_s),
        )

    def print_if_verbose(self, info_string):
        if self.config.verbose:
            logging.info(info_string)

    def add_frame(self, thermal, filtered, mask, ffc_affected=False):
        self.frame_buffer.add_frame(
            thermal, filtered, mask, self.frame_on, ffc_affected
        )
        if self.calc_stats:
            self.stats.add_frame(thermal, filtered)


class ClipStats:
    """ Stores background analysis statistics. """

    def __init__(self):
        self.mean_background_value = 0
        self.max_temp = None
        self.min_temp = None
        self.mean_temp = None
        self.frame_stats_min = []
        self.frame_stats_max = []
        self.frame_stats_median = []
        self.frame_stats_mean = []
        self.filtered_deviation = None
        self.filtered_sum = 0
        self.temp_thresh = 0
        self.threshold = None
        self.average_delta = None
        self.is_static_background = None

    def add_frame(self, thermal, filtered):
        f_median = np.median(thermal)
        f_max = np.max(thermal)
        f_min = np.min(thermal)
        f_mean = np.nanmean(thermal)
        self.max_temp = null_safe_compare(self.max_temp, f_max, max)
        self.min_temp = null_safe_compare(self.min_temp, f_min, min)

        self.frame_stats_min.append(f_min)
        self.frame_stats_max.append(f_max)
        self.frame_stats_median.append(f_median)
        self.frame_stats_mean.append(f_mean)
        self.filtered_sum += np.sum(np.abs(filtered))

    def completed(self, num_frames, height, width):
        if num_frames == 0:
            return
        total = num_frames * height * width
        self.filtered_deviation = self.filtered_sum / float(total)
        self.mean_temp = (height * width * sum(self.frame_stats_mean)) / float(total)


def null_safe_compare(a, b, cmp_fn):
    if a is None:
        return b
    elif b is not None:
        return cmp_fn(a, b)
    else:
        return None

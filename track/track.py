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
import math
import numpy as np
from collections import namedtuple

from ml_tools.tools import Rectangle, get_clipped_flow
from track.region import Region
from kalman.kalman import Kalman
from ml_tools.tools import eucl_distance


class TrackChannels:
    """ Indexes to channels in track. """

    thermal = 0
    filtered = 1
    flow_h = 2
    flow_v = 3
    mask = 4
    flow = 5


class Track:
    """ Bounds of a tracked object over time. """

    # keeps track of which id number we are up to.
    _track_id = 1
    # number of frames required before using kalman estimation
    MIN_KALMAN_FRAMES = 18
    # Percentage increase that is considered jitter, e.g. if a region gets
    # 30% bigger or smaller
    JITTER_THRESHOLD = 0.3
    # must change atleast 5 pixels to be considered for jitter
    MIN_JITTER_CHANGE = 5

    def __init__(self, clip_id, id=None, fps=9):
        """
        Creates a new Track.
        :param id: id number for track, if not specified is provided by an auto-incrementer
        """

        if not id:
            self._id = Track._track_id
            Track._track_id += 1
        else:
            self._id = id

        self.clip_id = clip_id
        self.start_frame = None
        self.end_frame = None
        self.start_s = None
        self.end_s = None
        self.fps = fps
        self.current_frame_num = None
        self.frame_list = []
        # our bounds over time
        self.bounds_history = []
        # number frames since we lost target.
        self.frames_since_target_seen = 0
        self.blank_frames = 0

        self.vel_x = []
        self.vel_y = []
        # the tag for this track
        self.tag = "unknown"
        self.prev_frame_num = None
        self.include_filtered_channel = True
        self.confidence = None
        self.max_novelty = None
        self.avg_novelty = None

        self.from_metadata = False
        self.track_tags = None
        self.kalman_tracker = Kalman()
        self.predictions = None
        self.predicted_class = None
        self.predicted_confidence = None

        self.all_class_confidences = None
        self.prediction_classes = None

        self.predicted_mid = None
        self.crop_rectangle = None

    @classmethod
    def from_region(cls, clip, region):
        track = cls(clip.get_id(), fps=clip.frames_per_second)
        track.start_frame = region.frame_number
        track.start_s = region.frame_number / float(clip.frames_per_second)
        track.crop_rectangle = clip.crop_rectangle
        track.add_region(region)
        return track

    def get_id(self):
        return self._id

    def load_track_meta(
        self,
        track_meta,
        frames_per_second,
        include_filtered_channel,
        tag_precedence,
        min_confidence,
    ):
        self.from_metadata = True
        self._id = track_meta["id"]
        self.include_filtered_channel = include_filtered_channel
        data = track_meta["data"]
        self.start_s = data["start_s"]
        self.end_s = data["end_s"]
        self.fps = frames_per_second

        self.predicted_class = data.get("tag")
        self.all_class_confidences = data.get("all_class_confidences", None)
        self.predictions = data.get("predictions")
        if self.predictions:
            self.predictions = np.int16(self.predictions)
            self.predicted_confidence = np.amax(self.predictions)

        self.track_tags = track_meta.get("TrackTags")
        self.prediction_classes = data.get("classes")

        tag = Track.get_best_human_tag(track_meta, tag_precedence, min_confidence)
        if tag:
            self.tag = tag["what"]
            self.confidence = tag["confidence"]
        else:
            return False

        positions = data.get("positions")
        if not positions:
            return False
        self.bounds_history = []
        self.frame_list = []
        for position in positions:
            frame_number = round(position[0] * frames_per_second)
            if self.start_frame is None:
                self.start_frame = frame_number
            self.end_frame = frame_number
            region = Region.region_from_array(position[1], frame_number)
            self.bounds_history.append(region)
            self.frame_list.append(frame_number)
        self.current_frame_num = 0
        return True

    def add_region(self, region):
        if self.prev_frame_num and region.frame_number:
            frame_diff = region.frame_number - self.prev_frame_num - 1
            for _ in range(frame_diff):
                self.add_blank_frame()

        self.bounds_history.append(region)
        self.end_frame = region.frame_number
        self.prev_frame_num = region.frame_number
        self.update_velocity()
        self.frames_since_target_seen = 0
        self.kalman_tracker.correct(region)
        prediction = self.kalman_tracker.predict()

        self.predicted_mid = (prediction[0][0], prediction[1][0])

    def update_velocity(self):
        if len(self.bounds_history) >= 2:
            self.vel_x.append(
                self.bounds_history[-1].mid_x - self.bounds_history[-2].mid_x
            )
            self.vel_y.append(
                self.bounds_history[-1].mid_y - self.bounds_history[-2].mid_y
            )
        else:
            self.vel_x.append(0)
            self.vel_y.append(0)

    def add_frame_for_existing_region(self, frame, mass_delta_threshold, prev_filtered):
        region = self.bounds_history[self.current_frame_num]
        if prev_filtered is not None:
            prev_filtered = region.subimage(prev_filtered)
        filtered = region.subimage(frame.filtered)
        region.calculate_mass(filtered, mass_delta_threshold)
        region.calculate_variance(filtered, prev_filtered)

        if self.prev_frame_num and frame.frame_number:
            frame_diff = frame.frame_number - self.prev_frame_num - 1
            for _ in range(frame_diff):
                self.add_blank_frame()
        self.update_velocity()

        self.prev_frame_num = frame.frame_number
        self.current_frame_num += 1
        self.frames_since_target_seen = 0
        self.kalman_tracker.correct(region)

    def average_mass(self):
        """ Average mass of last 3 frames that weren't blank """
        return np.mean(
            [bound.mass for bound in self.bounds_history if bound.blank == False][-3:]
        )

    def add_blank_frame(self, buffer_frame=None):
        """ Maintains same bounds as previously, does not reset framce_since_target_seen counter """
        if self.frames > Track.MIN_KALMAN_FRAMES:
            region = Region(
                int(self.predicted_mid[0] - self.last_bound.width / 2.0),
                int(self.predicted_mid[1] - self.last_bound.height / 2.0),
                self.last_bound.width,
                self.last_bound.height,
            )
            if self.crop_rectangle:
                region.crop(self.crop_rectangle)
        else:
            region = self.last_bound.copy()
        region.blank = True
        region.mass = 0
        region.pixel_variance = 0
        region.frame_number = self.last_bound.frame_number + 1
        self.bounds_history.append(region)
        self.prev_frame_num = region.frame_number
        self.update_velocity()
        self.blank_frames += 1
        self.frames_since_target_seen += 1
        prediction = self.kalman_tracker.predict()
        self.predicted_mid = (prediction[0][0], prediction[1][0])

    def get_stats(self):
        """
        Returns statistics for this track, including how much it moves, and a score indicating how likely it is
        that this is a good track.
        :return: a TrackMovementStatistics record
        """

        if len(self) <= 1:
            return TrackMovementStatistics()
        # get movement vectors only from non blank regions
        non_blank = [bound for bound in self.bounds_history if not bound.blank]
        mass_history = [int(bound.mass) for bound in non_blank]
        variance_history = [
            bound.pixel_variance for bound in non_blank if bound.pixel_variance
        ]
        movement = 0
        max_offset = 0

        frames_moved = 0
        avg_vel = 0
        first_point = self.bounds_history[0].mid
        for i, (vx, vy) in enumerate(zip(self.vel_x, self.vel_y)):
            region = self.bounds_history[i]
            if not region.blank:
                avg_vel += abs(vx) + abs(vy)
            if i == 0:
                continue

            if region.blank or self.bounds_history[i - 1].blank:
                continue
            if region.has_moved(self.bounds_history[i - 1]) or region.is_along_border:
                distance = (vx ** 2 + vy ** 2) ** 0.5
                movement += distance
                offset = eucl_distance(first_point, region.mid)
                max_offset = max(max_offset, offset)
                frames_moved += 1
        avg_vel = avg_vel / len(mass_history)
        # the standard deviation is calculated by averaging the per frame variances.
        # this ends up being slightly different as I'm using /n rather than /(n-1) but that
        # shouldn't make a big difference as n = width*height*frames which is large.
        max_offset = math.sqrt(max_offset)
        delta_std = float(np.mean(variance_history)) ** 0.5
        jitter_bigger = 0
        jitter_smaller = 0
        for i, bound in enumerate(self.bounds_history[1:]):
            prev_bound = self.bounds_history[i]
            if prev_bound.is_along_border or bound.is_along_border:
                continue
            height_diff = bound.height - prev_bound.height
            width_diff = prev_bound.width - bound.width
            thresh_h = max(
                Track.MIN_JITTER_CHANGE, prev_bound.height * Track.JITTER_THRESHOLD
            )
            thresh_v = max(
                Track.MIN_JITTER_CHANGE, prev_bound.width * Track.JITTER_THRESHOLD
            )
            if abs(height_diff) > thresh_h:
                if height_diff > 0:
                    jitter_bigger += 1
                else:
                    jitter_smaller += 1
            elif abs(width_diff) > thresh_v:
                if width_diff > 0:
                    jitter_bigger += 1
                else:
                    jitter_smaller += 1

        movement_points = (movement ** 0.5) + max_offset
        delta_points = delta_std * 25.0
        jitter_percent = int(
            round(100 * (jitter_bigger + jitter_smaller) / float(self.frames))
        )
        blank_percent = int(round(100.0 * self.blank_frames / self.frames))
        score = (
            min(movement_points, 100)
            + min(delta_points, 100)
            + (100 - jitter_percent)
            + (100 - blank_percent)
        )
        stats = TrackMovementStatistics(
            movement=float(movement),
            max_offset=float(max_offset),
            average_mass=float(np.mean(mass_history)),
            median_mass=float(np.median(mass_history)),
            delta_std=float(delta_std),
            score=float(score),
            region_jitter=jitter_percent,
            jitter_bigger=jitter_bigger,
            jitter_smaller=jitter_smaller,
            blank_percent=blank_percent,
            frames_moved=frames_moved,
            mass_std=float(np.std(mass_history)),
            average_velocity=float(avg_vel),
        )

        return stats

    def smooth(self, frame_bounds: Rectangle):
        """
        Smooths out any quick changes in track dimensions
        :param frame_bounds The boundaries of the video frame.
        """
        if len(self.bounds_history) == 0:
            return

        new_bounds_history = []
        prev_frame = self.bounds_history[0]
        current_frame = self.bounds_history[0]
        next_frame = self.bounds_history[1]

        for i in range(len(self.bounds_history)):

            prev_frame = self.bounds_history[max(0, i - 1)]
            current_frame = self.bounds_history[i]
            next_frame = self.bounds_history[min(len(self.bounds_history) - 1, i + 1)]

            frame_x = current_frame.mid_x
            frame_y = current_frame.mid_y
            frame_width = (
                prev_frame.width + current_frame.width + next_frame.width
            ) / 3
            frame_height = (
                prev_frame.height + current_frame.height + next_frame.height
            ) / 3
            frame = Region(
                int(frame_x - frame_width / 2),
                int(frame_y - frame_height / 2),
                int(frame_width),
                int(frame_height),
            )
            frame.crop(frame_bounds)

            new_bounds_history.append(frame)

        self.bounds_history = new_bounds_history

    def trim(self):
        """
        Removes empty frames from start and end of track
        """
        mass_history = [int(bound.mass) for bound in self.bounds_history]
        start = 0
        while start < len(self) and mass_history[start] <= 2:
            start += 1
        end = len(self) - 1
        while end > 0 and mass_history[end] <= 2:
            if self.frames_since_target_seen > 0:
                self.frames_since_target_seen -= 1
                self.blank_frames -= 1
            end -= 1
        if end < start:
            self.start_frame = 0
            self.bounds_history = []
            self.vel_x = []
            self.vel_y = []
            self.blank_frames = 0
        else:
            self.start_frame += start
            self.bounds_history = self.bounds_history[start : end + 1]
            self.vel_x = self.vel_x[start : end + 1]
            self.vel_y = self.vel_y[start : end + 1]
        self.start_s = self.start_frame / float(self.fps)

    def get_overlap_ratio(self, other_track, threshold=0.05):
        """
        Checks what ratio of the time these two tracks overlap.
        :param other_track: the other track to compare with
        :param threshold: how much frames must be overlapping to be counted
        :return: the ratio of frames that overlap
        """

        if len(self) == 0 or len(other_track) == 0:
            return 0.0

        start = max(self.start_frame, other_track.start_frame)
        end = min(self.end_frame, other_track.end_frame)

        frames_overlapped = 0

        for pos in range(start, end + 1):
            our_index = pos - self.start_frame
            other_index = pos - other_track.start_frame
            if (
                our_index >= 0
                and other_index >= 0
                and our_index < len(self)
                and other_index < len(other_track)
            ):
                our_bounds = self.bounds_history[our_index]
                other_bounds = other_track.bounds_history[other_index]
                overlap = our_bounds.overlap_area(other_bounds) / our_bounds.area
                if overlap >= threshold:
                    frames_overlapped += 1

        return frames_overlapped / len(self)

    def crop_by_region_at_trackframe(self, frame, track_frame_number, clip_flow=True):
        bounds = self.bounds_history[track_frame_number]
        return self.crop_by_region(frame, bounds)

    def crop_by_region(self, frame, region, clip_flow=True, filter_mask_by_region=True):
        thermal = region.subimage(frame.thermal)
        filtered = region.subimage(frame.filtered)
        if frame.flow is not None:
            flow_h = region.subimage(frame.flow_h)
            flow_v = region.subimage(frame.flow_v)
            if clip_flow and not frame.flow_clipped:
                flow_h = get_clipped_flow(flow_h)
                flow_v = get_clipped_flow(flow_v)
        else:
            flow_h = None
            flow_v = None

        mask = region.subimage(frame.mask).copy()
        # make sure only our pixels are included in the mask.
        if filter_mask_by_region:
            mask[mask != region.id] = 0
        mask[mask > 0] = 1
        # stack together into a numpy array.
        # by using int16 we lose a little precision on the filtered frames, but not much (only 1 bit)
        if flow_h is not None and flow_v is not None:
            return np.int16(np.stack((thermal, filtered, flow_h, flow_v, mask), axis=0))
        else:
            empty = np.zeros(filtered.shape)
            return np.int16(np.stack((thermal, filtered, empty, empty, mask), axis=0))
        return frame

    def set_end_s(self, fps):
        self.end_s = (self.end_frame + 1) / fps

    def predicted_velocity(self):
        prev = self.last_bound
        if prev is None or self.nonblank_frames <= Track.MIN_KALMAN_FRAMES:
            return (0, 0)
        pred_vel_x = self.predicted_mid[0] - prev.mid_x
        pred_vel_y = self.predicted_mid[1] - prev.mid_y

        return (pred_vel_x, pred_vel_y)

    @property
    def nonblank_frames(self):
        return self.end_frame + 1 - self.start_frame - self.blank_frames

    @property
    def frames(self):
        return self.end_frame + 1 - self.start_frame

    @property
    def last_mass(self):
        return self.bounds_history[-1].mass

    @property
    def velocity(self):
        return self.vel_x[-1], self.vel_y[-1]

    @property
    def last_bound(self) -> Region:
        return self.bounds_history[-1]

    def __repr__(self):
        return "Track: {} frames# {}".format(self.get_id(), len(self))

    def __len__(self):
        return len(self.bounds_history)

    @classmethod
    def get_best_human_tag(cls, track_meta, tag_precedence, min_confidence=-1):
        """ returns highest precidence non AI tag from the metadata """

        track_tags = track_meta.get("TrackTags", [])
        track_tags = [
            tag
            for tag in track_tags
            if not tag.get("automatic", False)
            and tag.get("confidence") > min_confidence
        ]

        if not track_tags:
            return None

        tag = None
        default_prec = tag_precedence.get("default", 100)
        best = None
        for track_tag in track_tags:
            ranking = cls.tag_ranking(track_tag, tag_precedence, default_prec)

            # if 2 track_tags have same confidence ignore both
            if ranking == best and track_tag["what"] != tag["what"]:
                tag = None
            elif best is None or ranking < best:
                best = ranking
                tag = track_tag
        return tag

    @staticmethod
    def tag_ranking(track_tag, precedence, default_prec):
        """ returns a ranking of tags based of what they are and confidence """

        what = track_tag.get("what")
        confidence = 1 - track_tag.get("confidence", 0)
        prec = precedence.get(what, default_prec)
        return prec + confidence


TrackMovementStatistics = namedtuple(
    "TrackMovementStatistics",
    "movement max_offset score average_mass median_mass delta_std region_jitter jitter_smaller jitter_bigger blank_percent frames_moved mass_std, average_velocity",
)
TrackMovementStatistics.__new__.__defaults__ = (0,) * len(
    TrackMovementStatistics._fields
)

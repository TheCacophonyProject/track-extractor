from cptv import CPTVReader
import cv2
import pytz
import numpy as np
import logging
from ml_tools.tools import Rectangle
from track.framebuffer import FrameBuffer
from track.track import Track
import datetime


class Clip:
    PREVIEW = "preview"
    FRAMES_PER_SECOND = 9

    def __init__(self, trackconfig):
        self.tracks = []
        self.background_is_preview = trackconfig.background_calc == Clip.PREVIEW
        self.config = trackconfig
        self.frames_per_second = Clip.FRAMES_PER_SECOND
        # start time of video
        self.video_start_time = None
        # name of source file
        self.source_file = None

        # per frame temperature statistics for thermal channel
        self.frame_stats_min = []
        self.frame_stats_max = []
        self.frame_stats_median = []
        self.frame_stats_mean = []

        # this buffers store the entire video in memory and are required for fast track exporting
        self.frame_buffer = FrameBuffer()
        self.background_stats = {}
        self.disable_background_subtraction = {}
        self.crop_rectangle = None

    def load_cptv(self, filename):
        """
        Loads a cptv file, and prepares for track extraction.
        """
        self.source_file = filename

        with open(filename, "rb") as f:
            reader = CPTVReader(f)
            local_tz = pytz.timezone("Pacific/Auckland")
            self.video_start_time = reader.timestamp.astimezone(local_tz)
            self.preview_secs = reader.preview_secs
            # self.stats.update(self.get_video_stats())
            # we need to load the entire video so we can analyse the background.
            frames = [np.float32(frame.pix) for frame in reader]
            self.frame_buffer.thermal = np.float32(frames)
            edge = self.config.edge_pixels
            self.crop_rectangle = Rectangle(
                edge,
                edge,
                reader.x_resolution - 2 * edge,
                reader.y_resolution - 2 * edge,
            )

    def parse_clip(self):
        # for now just always calculate as we are using the stats...
        frames = self.frame_buffer.thermal

        # background np.float64[][] filtered calculated here and stats
        background = self._background_from_whole_clip(frames)

        if self.background_is_preview:
            if self.preview_secs > 0:
                # background np.int32[][]
                background = self._background_from_preview(frames)
            else:
                logging.info(
                    "No preview secs defined for CPTV file - using statistical background measurement"
                )

        # create optical flow
        self.opt_flow = cv2.createOptFlow_DualTVL1()
        self.opt_flow.setUseInitialFlow(True)
        if not self.config.high_quality_optical_flow:
            # see https://stackoverflow.com/questions/19309567/speeding-up-optical-flow-createoptflow-dualtvl1
            self.opt_flow.setTau(1 / 4)
            self.opt_flow.setScalesNumber(3)
            self.opt_flow.setWarpingsNumber(3)
            self.opt_flow.setScaleStep(0.5)

        # process each frame
        self.frame_on = 0
        for frame in frames:
            self._process_frame(frame, background)
            self.frame_on += 1

        self.generate_optical_flow()

    def _background_from_preview(self, frame_list):
        number_frames = (
            self.preview_secs * self.frames_per_second - self.config.ignore_frames
        )
        if not number_frames < len(frame_list):
            logging.error("Video consists entirely of preview")
            number_frames = len(frame_list)
        frames = frame_list[0:number_frames]
        background = np.average(frames, axis=0)
        background = np.int32(np.rint(background))
        self.mean_background_value = np.average(background)
        self.threshold = self.config.delta_thresh
        return background

    def _background_from_whole_clip(self, frames):
        """
        Runs through all provided frames and estimates the background, consuming all the source frames.
        :param frames_list: a list of numpy array frames
        :return: background, background_stats
        """

        # note: unfortunately this must be done before any other processing, which breaks the streaming architecture
        # for this reason we must return all the frames so they can be reused

        frames = np.float32(frames)
        # [][] array
        background = np.percentile(frames, q=10, axis=0)
        filtered = np.float32(
            [self._get_filtered_frame(frame, background) for frame in frames]
        )

        delta = np.asarray(frames[1:], dtype=np.float32) - np.asarray(
            frames[:-1], dtype=np.float32
        )
        average_delta = float(np.mean(np.abs(delta)))

        # take half the max filtered value as a threshold
        threshold = float(
            np.percentile(
                np.reshape(filtered, [-1]), q=self.config.threshold_percentile
            )
        )

        # cap the threshold to something reasonable
        if threshold < self.config.min_threshold:
            threshold = self.config.min_threshold
        if threshold > self.config.max_threshold:
            threshold = self.config.max_threshold

        self.background_stats = BackgroundAnalysis()
        self.background_stats.threshold = float(threshold)
        self.background_stats.average_delta = float(average_delta)
        self.background_stats.min_temp = float(np.min(frames))
        self.background_stats.max_temp = float(np.max(frames))
        self.background_stats.mean_temp = float(np.mean(frames))
        self.background_stats.background_deviation = float(np.mean(np.abs(filtered)))
        self.background_stats.is_static_background = (
            self.background_stats.background_deviation
            < self.config.static_background_threshold
        )

        if (
            not self.background_stats.is_static_background
            or self.disable_background_subtraction
        ):
            background = None

        return background

    def _get_filtered_frame(self, thermal, background=None):
        """
        Calculates filtered frame from thermal
        :param thermal: the thermal frame
        :param background: (optional) used for background subtraction
        :return: the filtered frame
        """

        if background is None:
            filtered = thermal - np.median(thermal) - 40
            filtered[filtered < 0] = 0
        elif self.background_is_preview:
            avg_change = int(round(np.average(thermal) - self.mean_background_value))
            filtered = thermal.copy()
            filtered[filtered < self.config.temp_thresh] = 0
            filtered = filtered - background - avg_change
        else:
            background = np.float32(background)
            filtered = thermal - background
            filtered[filtered < 0] = 0
            filtered = filtered - np.median(filtered)
            filtered[filtered < 0] = 0
        return filtered

    def _process_frame(self, thermal, background=None):
        """
        Tracks objects through frame
        :param thermal: A numpy array of shape (height, width) and type uint16
        :param background: (optional) Background image, a numpy array of shape (height, width) and type uint16
            If specified background subtraction algorithm will be used.
        """

        thermal = np.float32(thermal)
        filtered = self._get_filtered_frame(thermal, background)
        frame_height, frame_width = filtered.shape

        mask = np.zeros(filtered.shape)
        edge = self.config.edge_pixels

        # remove the edges of the frame as we know these pixels can be spurious value
        edgeless_filtered = self.crop_rectangle.subimage(filtered)

        thresh = np.uint8(
            blur_and_return_as_mask(edgeless_filtered, threshold=self.threshold)
        )

        dilated = thresh

        # Dilation groups interested pixels that are near to each other into one component(animal/track)
        if self.config.dilation_pixels > 0:
            size = self.config.dilation_pixels * 2 + 1
            kernel = np.ones((size, size), np.uint8)
            dilated = cv2.dilate(dilated, kernel, iterations=1)

        _, small_mask, _, _ = cv2.connectedComponentsWithStats(dilated)

        mask[edge: frame_height - edge, edge: frame_width - edge] = small_mask

        # save frame stats
        self.frame_stats_min.append(np.min(thermal))
        self.frame_stats_max.append(np.max(thermal))
        self.frame_stats_median.append(np.median(thermal))
        self.frame_stats_mean.append(np.mean(thermal))

        # save history
        self.frame_buffer.filtered.append(np.float32(filtered))
        self.frame_buffer.mask.append(np.float32(mask))

    def generate_optical_flow(self):
        if not self.frame_buffer.has_flow:
            self.frame_buffer.generate_optical_flow(
                self.opt_flow, self.config.flow_threshold
            )

    def start_and_end_time_absolute(self, start_s, end_s):
        return (
            self.video_start_time + datetime.timedelta(seconds=start_s),
            self.video_start_time + datetime.timedelta(seconds=end_s),
        )

    def get_stats(self):
        """
        Returns statistics for this track, including how much it moves, and a score indicating how likely it is
        that this is a good track.
        :return: a TrackMovementStatistics record
        """

        if len(self) <= 1:
            return track.TrackMovementStatistics()

        # get movement vectors
        mass_history = [int(bound.mass) for bound in self.bounds_history]
        variance_history = [bound.pixel_variance for bound in self.bounds_history]
        mid_x = [bound.mid_x for bound in self.bounds_history]
        mid_y = [bound.mid_y for bound in self.bounds_history]
        delta_x = [mid_x[0] - x for x in mid_x]
        delta_y = [mid_y[0] - y for y in mid_y]
        vel_x = [cur - prev for cur, prev in zip(mid_x[1:], mid_x[:-1])]
        vel_y = [cur - prev for cur, prev in zip(mid_y[1:], mid_y[:-1])]

        movement = sum((vx ** 2 + vy ** 2) ** 0.5 for vx, vy in zip(vel_x, vel_y))
        max_offset = max((dx ** 2 + dy ** 2) ** 0.5 for dx, dy in zip(delta_x, delta_y))

        # the standard deviation is calculated by averaging the per frame variances.
        # this ends up being slightly different as I'm using /n rather than /(n-1) but that
        # shouldn't make a big difference as n = width*height*frames which is large.
        delta_std = float(np.mean(variance_history)) ** 0.5

        movement_points = (movement ** 0.5) + max_offset
        delta_points = delta_std * 25.0
        score = min(movement_points, 100) + min(delta_points, 100)

        stats = track.TrackMovementStatistics(
            movement=float(movement),
            max_offset=float(max_offset),
            average_mass=float(np.mean(mass_history)),
            median_mass=float(np.median(mass_history)),
            delta_std=float(delta_std),
            score=float(score),
        )

        return stats

    def load_tracks(self, metadata):
        tracks_meta = metadata["tracks"]
        self.tracks = []
        # get track data
        for track_meta in tracks_meta:
            track = Track()
            track.load_track_from_meta(track_meta)

def blur_and_return_as_mask(frame, threshold):
    """
    Creates a binary mask out of an image by applying a threshold.
    Any pixels more than the threshold are set 1, all others are set to 0.
    A blur is also applied as a filtering step
    """
    thresh = cv2.GaussianBlur(np.float32(frame), (5, 5), 0) - threshold
    thresh[thresh < 0] = 0
    thresh[thresh > 0] = 1
    return thresh


class BackgroundAnalysis:
    """ Stores background analysis statistics. """

    def __init__(self):
        self.threshold = None
        self.average_delta = None
        self.max_temp = None
        self.min_temp = None
        self.mean_temp = None
        self.background_deviation = None
        self.is_static = True
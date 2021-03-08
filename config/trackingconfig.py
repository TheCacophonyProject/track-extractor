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

import attr

from config import config
from .defaultconfig import DefaultConfig
from .motionconfig import MotionConfig
from load.cliptrackextractor import ClipTrackExtractor


@attr.s
class TrackingConfig(DefaultConfig):

    background_calc = attr.ib()
    motion = attr.ib()
    preview_ignore_frames = attr.ib()
    threshold_percentile = attr.ib()
    static_background_threshold = attr.ib()
    max_mean_temperature_threshold = attr.ib()
    max_temperature_range_threshold = attr.ib()
    edge_pixels = attr.ib()
    dilation_pixels = attr.ib()
    frame_padding = attr.ib()
    track_smoothing = attr.ib()
    remove_track_after_frames = attr.ib()
    high_quality_optical_flow = attr.ib()
    min_threshold = attr.ib()
    max_threshold = attr.ib()
    flow_threshold = attr.ib()
    max_tracks = attr.ib()
    track_overlap_ratio = attr.ib()
    min_duration_secs = attr.ib()
    track_min_offset = attr.ib()
    track_min_mass = attr.ib()
    aoi_min_mass = attr.ib()
    aoi_pixel_variance = attr.ib()
    cropped_regions_strategy = attr.ib()
    verbose = attr.ib()
    enable_track_output = attr.ib()
    min_tag_confidence = attr.ib()
    moving_vel_thresh = attr.ib()
    min_moving_frames = attr.ib()
    max_blank_percent = attr.ib()
    max_mass_std_percent = attr.ib()

    max_jitter = attr.ib()
    # used to provide defaults
    stats = attr.ib()
    filters = attr.ib()
    areas_of_interest = attr.ib()

    @classmethod
    def load(cls, tracking):
        return cls(
            background_calc=config.parse_options_param(
                "background_calc",
                tracking["background_calc"],
                [ClipTrackExtractor.PREVIEW, "stats"],
            ),
            motion=MotionConfig.load(tracking.get("motion")),
            preview_ignore_frames=tracking["preview_ignore_frames"],
            threshold_percentile=tracking["stats"]["threshold_percentile"],
            min_threshold=tracking["stats"]["min_threshold"],
            max_threshold=tracking["stats"]["max_threshold"],
            static_background_threshold=tracking["static_background_threshold"],
            max_mean_temperature_threshold=tracking["max_mean_temperature_threshold"],
            max_temperature_range_threshold=tracking["max_temperature_range_threshold"],
            edge_pixels=tracking["edge_pixels"],
            dilation_pixels=tracking["dilation_pixels"],
            frame_padding=tracking["frame_padding"],
            track_smoothing=tracking["track_smoothing"],
            remove_track_after_frames=tracking["remove_track_after_frames"],
            high_quality_optical_flow=tracking["high_quality_optical_flow"],
            flow_threshold=tracking["flow_threshold"],
            max_tracks=tracking["max_tracks"],
            moving_vel_thresh=tracking["filters"]["moving_vel_thresh"],
            track_overlap_ratio=tracking["filters"]["track_overlap_ratio"],
            min_duration_secs=tracking["filters"]["min_duration_secs"],
            track_min_offset=tracking["filters"]["track_min_offset"],
            track_min_mass=tracking["filters"]["track_min_mass"],
            cropped_regions_strategy=tracking["areas_of_interest"][
                "cropped_regions_strategy"
            ],
            aoi_min_mass=tracking["areas_of_interest"]["min_mass"],
            aoi_pixel_variance=tracking["areas_of_interest"]["pixel_variance"],
            verbose=tracking["verbose"],
            enable_track_output=tracking["enable_track_output"],
            min_tag_confidence=tracking["min_tag_confidence"],
            min_moving_frames=tracking["min_moving_frames"],
            max_blank_percent=tracking["max_blank_percent"],
            max_mass_std_percent=tracking["max_mass_std_percent"],
            max_jitter=tracking["max_jitter"],
            stats=None,
            filters=None,
            areas_of_interest=None,
        )

    @classmethod
    def get_defaults(cls):
        return cls(
            background_calc=ClipTrackExtractor.PREVIEW,
            motion=MotionConfig.get_defaults(),
            stats={
                "threshold_percentile": 99.9,
                "min_threshold": 30,
                "max_threshold": 50,
            },
            preview_ignore_frames=2,
            max_mean_temperature_threshold=10000,
            max_temperature_range_threshold=10000,
            static_background_threshold=4.0,
            edge_pixels=1,
            frame_padding=4,
            dilation_pixels=2,
            remove_track_after_frames=18,
            track_smoothing=False,
            high_quality_optical_flow=False,
            flow_threshold=40,
            max_tracks=10,
            filters={
                "track_overlap_ratio": 0.5,
                "min_duration_secs": 3.0,
                "track_min_offset": 4.0,
                "track_min_mass": 2.0,
            },
            areas_of_interest={
                "min_mass": 4.0,
                "pixel_variance": 2.0,
                "cropped_regions_strategy": "cautious",
            },
            verbose=False,
            # defaults provided in dictionaries, placesholders to stop init complaining
            aoi_min_mass=4.0,
            aoi_pixel_variance=2.0,
            cropped_regions_strategy="cautious",
            track_min_offset=4.0,
            track_min_mass=2.0,
            threshold_percentile=99.9,
            min_threshold=30,
            max_threshold=50,
            track_overlap_ratio=0.5,
            min_duration_secs=3,
            min_tag_confidence=0.8,
            enable_track_output=True,
            moving_vel_thresh=4,
            min_moving_frames=2,
            max_blank_percent=30,
            max_mass_std_percent=ClipTrackExtractor.MASS_CHANGE_PERCENT,
            max_jitter=20,
        )

    def validate(self):
        return True

    def as_dict(self):
        return attr.asdict(self)

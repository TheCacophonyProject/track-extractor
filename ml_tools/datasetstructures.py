import json
import dateutil
import numpy as np
import logging

from ml_tools.imageprocessing import filtered_is_valid
from ml_tools import tools

FRAMES_PER_SECOND = 9

CPTV_FILE_WIDTH = 160
CPTV_FILE_HEIGHT = 120


class TrackHeader:
    """ Header for track. """

    def __init__(
        self,
        clip_id,
        track_id,
        label,
        start_time,
        num_frames,
        duration,
        camera,
        location,
        score,
        track_bounds,
        frame_temp_median,
        frames_per_second,
        predictions,
        correct_prediction,
        frame_mass,
        start_frame,
        res_x=CPTV_FILE_WIDTH,
        res_y=CPTV_FILE_HEIGHT,
        ffc_frames=None,
    ):
        self.res_x = res_x
        self.res_y = res_y
        self.predictions = predictions
        self.correct_prediction = correct_prediction
        self.filtered_stats = {"segment_mass": 0}
        # reference to clip this segment came from
        self.clip_id = clip_id
        # reference to track this segment came from
        self.track_id = track_id
        # list of segments that belong to this track
        self.segments = []
        # label for this track
        self.label = label
        # date and time of the start of the track
        self.start_time = start_time
        self.start_frame = start_frame
        # duration in seconds
        self.duration = duration
        # camera this track came from
        self.camera = camera

        self.location = location
        # score of track
        self.score = score
        # thermal reference point for each frame.
        self.frame_temp_median = frame_temp_median
        # tracking frame movements for each frame, array of tuples (x-vel, y-vel)
        self.frame_velocity = None
        # original tracking bounds
        self.track_bounds = track_bounds
        # what fraction of pixels are from out of bounds
        self.frame_crop = []
        self.num_frames = num_frames
        self.frames_per_second = frames_per_second
        self.calculate_velocity()
        self.calculate_frame_crop()
        self.important_frames = []
        self.important_predicted = 0
        self.frame_mass = frame_mass
        self.lower_mass = np.percentile(frame_mass, q=25)
        self.upper_mass = np.percentile(frame_mass, q=75)
        self.median_mass = np.median(frame_mass)
        self.mean_mass = np.mean(frame_mass)
        self.ffc_frames = ffc_frames

    def toJSON(self):
        meta_dict = {}
        meta_dict["clip_id"] = int(self.clip_id)
        meta_dict["track_id"] = int(self.track_id)
        meta_dict["camera"] = self.camera
        meta_dict["num_frames"] = int(self.num_frames)
        positions = []
        for region in self.track_bounds:
            positions.append(region.tolist())

        meta_dict["frames_per_second"] = int(self.frames_per_second)
        meta_dict["track_bounds"] = positions
        meta_dict["start_frame"] = int(self.start_frame)
        if self.location is not None:
            meta_dict["location_hash"] = "{}{}".format(
                hash(self.location[0]), hash(self.location[1])
            )
        meta_dict["label"] = self.label

        return json.dumps(meta_dict, indent=3)

    def add_sample(self, sample, use_segments):
        if use_segments:
            self.segments.append(sample)
        else:
            self.important_frames.append(sample)

    def get_sample_frames(self):
        return self.important_frames

    def remove_sample_frame(self, f):
        self.important_frams.remove(f)

    def get_sample_frame(self, i=0, remove=False):
        if len(self.important_frames) == 0:
            return None
        f = self.important_frames[i]

        if remove:
            del self.important_frames[i]
        return f

    @property
    def frame_weight(self):
        return 1 / self.num_sample_frames

    @property
    def num_sample_frames(self):
        return len(self.important_frames)

    # trying to get only clear frames
    def set_important_frames(self, labels, min_mass=None, frame_data=None):
        # this needs more testing
        frames = []
        for i, mass in enumerate(self.frame_mass):
            if self.ffc_frames is not None and i in self.ffc_frames:
                continue
            if (
                min_mass is None
                or mass >= min_mass
                and mass >= self.lower_mass
                and mass <= self.upper_mass
            ):  # trying it out
                if frame_data is not None:
                    if not filtered_is_valid(frame_data[i], self.label):
                        logging.debug(
                            "set_important_frames %s frame %s has no zeros in filtered frame",
                            self.unique_id,
                            i,
                        )
                        continue
                frames.append(i)
        np.random.shuffle(frames)
        for frame in frames:
            f = FrameSample(
                self.clip_id,
                self.track_id,
                frame,
                self.label,
                self.frame_temp_median[frame],
                self.frame_velocity[frame],
            )
            self.important_frames.append(f)

    def calculate_frame_crop(self):
        # frames are always square, but bounding rect may not be, so to see how much we clipped I need to create a square
        # bounded rect and check it against frame size.
        self.frame_crop = []
        for rect in self.track_bounds:
            rect = tools.Rectangle.from_ltrb(*rect)
            rx, ry = rect.mid_x, rect.mid_y
            size = max(rect.width, rect.height)
            adjusted_rect = tools.Rectangle(rx - size / 2, ry - size / 2, size, size)
            self.frame_crop.append(
                get_cropped_fraction(adjusted_rect, self.res_x, self.res_y)
            )

    def calculate_velocity(self):
        frame_center = [
            ((left + right) / 2, (top + bottom) / 2)
            for left, top, right, bottom in self.track_bounds
        ]
        self.frame_velocity = []
        prev = None
        for x, y in frame_center:
            if prev is None:
                self.frame_velocity.append((0.0, 0.0))
            else:
                self.frame_velocity.append((x - prev[0], y - prev[1]))
            prev = (x, y)

    def calculate_segments(
        self,
        mass_history,
        segment_frame_spacing,
        segment_width,
        segment_min_mass=None,
        use_important=True,
        require_movement=False,
        scale=1,
    ):

        self.segments = []
        if use_important and self.num_sample_frames < segment_width:
            # dont want to repeat too many frames
            return
        elif len(mass_history) < segment_width:
            return
        if use_important:
            mid_x = [
                tools.Rectangle.from_ltrb(*bound).mid_x for bound in self.track_bounds
            ]
            mid_y = [
                tools.Rectangle.from_ltrb(*bound).mid_y for bound in self.track_bounds
            ]
            vel_x = [cur - prev for cur, prev in zip(mid_x[1:], mid_x[:-1])]
            vel_y = [cur - prev for cur, prev in zip(mid_y[1:], mid_y[:-1])]

            movement = sum((vx ** 2 + vy ** 2) ** 0.5 for vx, vy in zip(vel_x, vel_y))
            widths = [
                tools.Rectangle.from_ltrb(*bound).width for bound in self.track_bounds
            ]
            if movement < np.median(widths) * 2.0:
                logging.debug("Not enough movment %s %s", self, self.label)
                return

        segment_count = (len(mass_history) - segment_width) // segment_frame_spacing
        segment_count += 1

        if use_important:
            remaining = segment_width - self.num_sample_frames
            segment_count = max(0, (self.num_sample_frames - segment_width) // 9)
            segment_count += 1
            segment_count = int(scale * segment_count)
            # take any segment_width frames, this could be done each epoch
            for i in range(segment_count):
                frames = list(
                    np.random.choice(
                        self.important_frames,
                        min(segment_width, len(self.important_frames)),
                        replace=False,
                    )
                )
                # sample another batch
                if remaining > 0:
                    frames.extend(
                        np.random.choice(
                            self.important_frames,
                            remaining,
                            replace=False,
                        )
                    )
                frames = [frame.frame_num for frame in frames]
                frames.sort()
                mass_slice = mass_history[frames]
                segment_avg_mass = np.mean(mass_slice)
                if segment_avg_mass < 50:
                    segment_weight_factor = 0.75
                elif segment_avg_mass < 100:
                    segment_weight_factor = 1
                else:
                    segment_weight_factor = 1.2
                segment = SegmentHeader(
                    track=self,
                    start_frame=0,
                    frames=segment_width,
                    weight=segment_weight_factor,
                    avg_mass=segment_avg_mass,
                    frame_indices=frames,
                )
                self.segments.append(segment)
            # print("segment count", seg_count, len(self.segments))

            return
        # scan through track looking for good segments to add to our datset
        for i in range(segment_count):
            segment_start = i * segment_frame_spacing

            mass_slice = mass_history[segment_start : segment_start + segment_width]
            segment_avg_mass = np.mean(mass_slice)
            segment_frames = len(mass_slice)

            if segment_frames != segment_width:
                continue
            if segment_min_mass and segment_avg_mass < segment_min_mass:
                self.filtered_stats["segment_mass"] += 1
                continue
            # try to sample the better segments more often
            if segment_avg_mass < 50:
                segment_weight_factor = 0.75
            elif segment_avg_mass < 100:
                segment_weight_factor = 1
            else:
                segment_weight_factor = 1.2

            segment = SegmentHeader(
                track=self,
                start_frame=segment_start,
                frames=segment_width,
                weight=segment_weight_factor,
                avg_mass=segment_avg_mass,
            )

            self.segments.append(segment)

    @property
    def camera_id(self):
        """ Unique name of this track. """
        return "{}-{}".format(self.camera, self.location)

    @property
    def bin_id(self):
        """ Unique name of this track. """
        return "{}-{}".format(self.clip_id, self.track_id)

    @property
    def weight(self):
        """ Returns total weight for all segments in this track"""
        return sum(segment.weight for segment in self.segments)

    @property
    def unique_id(self):
        return "{}-{}".format(self.clip_id, self.track_id)

    @staticmethod
    def from_meta(clip_id, clip_meta, track_meta, predictions=None):
        """ Creates a track header from given metadata. """
        correct_prediction = track_meta.get("correct_prediction", None)
        start_time = dateutil.parser.parse(track_meta["start_time"])
        end_time = dateutil.parser.parse(track_meta["end_time"])
        duration = (end_time - start_time).total_seconds()
        location = clip_meta.get("location")
        num_frames = track_meta["frames"]
        camera = clip_meta["device"]
        frames_per_second = clip_meta.get("frames_per_second", FRAMES_PER_SECOND)
        # get the reference levels from clip_meta and load them into the track.
        track_start_frame = track_meta["start_frame"]
        frame_temp_median = np.float32(
            clip_meta["frame_temp_median"][
                track_start_frame : num_frames + track_start_frame
            ]
        )

        bounds_history = track_meta["bounds_history"]
        ffc_frames = clip_meta.get("ffc_frames", [])
        header = TrackHeader(
            clip_id=int(clip_id),
            track_id=int(track_meta["id"]),
            label=track_meta["tag"],
            start_time=start_time,
            num_frames=num_frames,
            duration=duration,
            camera=camera,
            location=location,
            score=float(track_meta["score"]),
            track_bounds=np.asarray(bounds_history),
            frame_temp_median=frame_temp_median,
            frames_per_second=frames_per_second,
            predictions=predictions,
            correct_prediction=correct_prediction,
            frame_mass=track_meta["mass_history"],
            start_frame=track_start_frame,
            res_x=clip_meta.get("res_x", CPTV_FILE_WIDTH),
            res_y=clip_meta.get("res_y", CPTV_FILE_HEIGHT),
            ffc_frames=ffc_frames,
        )
        return header

    def __repr__(self):
        return self.unique_id


class Camera:
    def __init__(self, camera):
        self.label_to_bins = {}
        self.label_to_tracks = {}

        self.bins = {}
        self.camera = camera
        self.bin_segment_sum = {}
        self.segment_sum = 0
        self.segments = 0
        self.label_frames = {}
        # used to sample bins
        self.bin_i = -1

    def label_tracks(self, label):
        return len(self.label_to_bins[label])

    def sample_frame(self, label):
        bins = self.label_to_bins[label]
        if len(bins) == 0:
            return None, None
        self.bin_i += 1
        self.bin_i = self.bin_i % len(bins)

        bin_id = bins[self.bin_i]
        track = self.bins[bin_id][0]
        f = track.get_sample_frame()
        if len(track.important_frames) == 0 or f is None:
            del bins[self.bin_i]
            del self.bins[bin_id]

        return track, f

    def label_track_count(
        self,
        label,
        max_segments_per_track=None,
    ):
        if label not in self.label_to_tracks:
            return 0
        tracks = self.label_to_tracks[label].values()
        return len(tracks)

    def label_segment_count(
        self,
        label,
        max_segments_per_track=None,
    ):
        if label not in self.label_to_tracks:
            return 0
        tracks = self.label_to_tracks[label].values()
        frames = 0
        for track in tracks:
            if max_segments_per_track:
                frames += max(len(track.segments), max_segments_per_track)
            else:
                frames += len(track.segments)

        return frames

    def label_frame_count(
        self,
        label,
        max_frames_per_track=None,
    ):
        if label not in self.label_to_tracks:
            return 0
        tracks = self.label_to_tracks[label].values()
        frames = 0
        for track in tracks:
            if max_frames_per_track:
                frames += max(len(track.important_frames), max_frames_per_track)
            else:
                frames += len(track.important_frames)

        return frames

    def remove_track(self, track):
        self.segments -= 1
        self.segment_sum -= len(track.segments)
        del self.label_to_tracks["wallaby"][track.unique_id]

    def add_track(self, track_header):
        tracks = self.label_to_tracks.setdefault(track_header.label, {})
        tracks[track_header.unique_id] = track_header
        if track_header.bin_id not in self.bins:
            self.bins[track_header.bin_id] = []
            self.bin_segment_sum[track_header.bin_id] = 0

        if track_header.label not in self.label_to_bins:
            self.label_to_bins[track_header.label] = []
            self.label_frames[track_header.label] = 0

        if track_header.bin_id not in self.label_to_bins[track_header.label]:
            self.label_to_bins[track_header.label].append(track_header.bin_id)

        self.bins[track_header.bin_id].append(track_header)
        self.label_frames[track_header.label] += len(track_header.important_frames)

        segment_length = len(track_header.segments)
        self.bin_segment_sum[track_header.bin_id] += segment_length
        self.segment_sum += segment_length
        self.segments += 1


class FrameSample:
    _frame_id = 1

    def __init__(self, clip_id, track_id, frame_num, label, temp_median, velocity):
        self.id = FrameSample._frame_id
        FrameSample._frame_id += 1
        self.clip_id = clip_id
        self.track_id = track_id
        self.frame_num = frame_num
        self.label = label
        self.temp_median = temp_median
        self.velocity = velocity

    @property
    def unique_track_id(self):
        return "{}-{}".format(self.clip_id, self.track_id)


class SegmentHeader:
    """ Header for segment. """

    _segment_id = 1

    def __init__(
        self,
        track: TrackHeader,
        start_frame,
        frames,
        weight,
        avg_mass,
        frame_indices=None,
    ):
        self.id = SegmentHeader._segment_id
        SegmentHeader._segment_id += 1
        # reference to track this segment came from
        self.track = track
        # first frame of this segment referenced by start of track
        self.start_frame = start_frame
        # length of segment in frames
        self.frames = frames
        # relative weight of the segment (higher is sampled more often)
        self.weight = weight
        # average mass of the segment
        self.avg_mass = avg_mass
        self.frame_indices = frame_indices

    @property
    def unique_track_id(self):
        # reference to clip this segment came from
        return self.track.unique_id

    @property
    def clip_id(self):
        # reference to clip this segment came from
        return self.track.clip_id

    @property
    def label(self):
        # label for this segment
        return self.track.label

    @property
    def name(self):
        """ Unique name of this segment. """
        return self.clip_id + "-" + str(self.track_id) + "-" + str(self.start_frame)

    @property
    def frame_velocity(self):
        # tracking frame velocity for each frame.
        return self.track.frame_velocity[
            self.start_frame : self.start_frame + self.frames
        ]

    @property
    def track_bounds(self):
        # original location of this tracks bounds.
        return self.track.track_bounds[
            self.start_frame : self.start_frame + self.frames
        ]

    @property
    def frame_crop(self):
        # how much each frame has been cropped.
        return self.track.frame_crop[self.start_frame : self.start_frame + self.frames]

    @property
    def frame_temp_median(self):
        # thermal reference temperature for each frame (i.e. which temp is 0)
        return self.track.frame_temp_median[
            self.start_frame : self.start_frame + self.frames
        ]

    @property
    def end_frame(self):
        """ end frame of segment"""
        return self.start_frame + self.frames

    @property
    def track_bin(self):
        """ Unique name of this segments track. """
        return self.track.bin_id

    def __str__(self):
        return "{0} label {1} offset:{2} weight:{3:.1f}".format(
            self.unique_track_id, self.label, self.start_frame, self.weight
        )


def get_cropped_fraction(region: tools.Rectangle, width, height):
    """ Returns the fraction regions mass outside the rect ((0,0), (width, height)"""
    bounds = tools.Rectangle(0, 0, width - 1, height - 1)
    return 1 - (bounds.overlap_area(region) / region.area)

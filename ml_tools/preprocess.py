import cv2
import enum
import numpy as np
import random
from ml_tools import tools
from track.track import TrackChannels
from ml_tools import imageprocessing


# size to scale each frame to when loaded.

MIN_SIZE = 4


class FrameTypes(enum.Enum):
    """Types of frames"""

    thermal_tiled = 0
    filtered_tiled = 1
    flow_tiled = 2
    overlay = 3
    flow_rgb = 4

    @staticmethod
    def is_valid(name):
        return name in FrameTypes.__members__.keys()


def preprocess_segment(
    frames,
    reference_level=None,
    frame_velocity=None,
    augment=False,
    default_inset=2,
    keep_aspect=False,
    frame_size=48,
    crop_rectangle=None,
    keep_edge=False,
):
    """
    Preprocesses the raw track data, scaling it to correct size, and adjusting to standard levels
    :param frames: a list of Frames
    :param reference_level: thermal reference level for each frame in data
    :param frame_velocity: velocity (x,y) for each frame.
    :param augment: if true applies a slightly random crop / scale
    :param default_inset: the default number of pixels to inset when no augmentation is applied.
    """

    if reference_level is not None:
        # -------------------------------------------
        # next adjust temperature and flow levels
        # get reference level for thermal channel
        assert len(frames) == len(
            reference_level
        ), "Reference level shape and data shape not match."

    # -------------------------------------------
    # first we scale to the standard size
    data = []
    flip = False
    if augment:
        contrast_adjust = None
        level_adjust = None
        if random.random() <= 0.75:
            # we will adjust contrast and levels, but only within these bounds.
            # that is a bright input may have brightness reduced, but not increased.
            LEVEL_OFFSET = 4

            # apply level and contrast shift
            level_adjust = float(random.normalvariate(0, LEVEL_OFFSET))
            contrast_adjust = float(tools.random_log(0.9, (1 / 0.9)))
        if random.random() <= 0.50:
            flip = True
    for i, frame in enumerate(frames):
        frame.float_arrays()
        frame_height, frame_width = frame.thermal.shape
        # adjusting the corners makes the algorithm robust to tracking differences.
        # gp changed to 0,1 maybe should be a percent of the frame size
        max_height_offset = int(np.clip(frame_height * 0.1, 1, 2))
        max_width_offset = int(np.clip(frame_width * 0.1, 1, 2))

        top_offset = random.randint(0, max_height_offset) if augment else default_inset
        bottom_offset = (
            random.randint(0, max_height_offset) if augment else default_inset
        )
        left_offset = random.randint(0, max_width_offset) if augment else default_inset
        right_offset = random.randint(0, max_width_offset) if augment else default_inset
        if frame_height < MIN_SIZE or frame_width < MIN_SIZE:
            continue

        frame_bounds = tools.Rectangle(0, 0, frame_width, frame_height)
        # rotate then crop
        if augment and random.random() <= 0.75:

            degrees = random.randint(0, 40) - 20
            frame.rotate(degrees)

        # set up a cropping frame
        crop_region = tools.Rectangle.from_ltrb(
            left_offset,
            top_offset,
            frame_width - right_offset,
            frame_height - bottom_offset,
        )

        # if the frame is too small we make it a little larger
        while crop_region.width < MIN_SIZE:
            crop_region.left -= 1
            crop_region.right += 1
            crop_region.crop(frame_bounds)
        while crop_region.height < MIN_SIZE:
            crop_region.top -= 1
            crop_region.bottom += 1
            crop_region.crop(frame_bounds)
        frame.crop_by_region(crop_region, out=frame)
        frame.resize(
            (frame_size, frame_size),
            keep_aspect=keep_aspect,
            keep_edge=keep_edge,
            crop_rectangle=crop_rectangle,
        )
        if reference_level is not None:
            frame.thermal -= reference_level[i]
            np.clip(frame.thermal, a_min=0, a_max=None, out=frame.thermal)
        frame.unclip_flow()
        frame.normalize()
        if augment:
            if level_adjust is not None:
                frame.thermal += level_adjust
            if contrast_adjust is not None:
                frame.thermal *= contrast_adjust
                frame.filtered *= contrast_adjust
            if flip:
                frame.flip()
        data.append(frame)

    return data, flip


def preprocess_frame(
    data, output_dim, use_thermal=True, augment=False, preprocess_fn=None
):
    if use_thermal:
        channel = TrackChannels.thermal
    else:
        channel = TrackChannels.filtered
    data = data.get_channel(channel)
    data, stats = imageprocessing.normalize(data)
    if not stats[0]:
        return None

    data = data[np.newaxis, :]
    data = np.transpose(data, (1, 2, 0))
    data = np.repeat(data, output_dim[2], axis=2)
    data = imageprocessing.resize_cv(data, output_dim, channel)

    # preprocess expects values in range 0-255
    if preprocess_fn:
        data = data * 255
        data = preprocess_fn(data)
    return data


def preprocess_movement(
    data,
    segment,
    frames_per_row,
    frame_size,
    regions,
    red_type,
    green_type,
    blue_type,
    preprocess_fn=None,
    augment=False,
    keep_aspect=False,
    reference_level=None,
    overlay=None,
    crop_rectangle=None,
    keep_edge=False,
):
    segment, flipped = preprocess_segment(
        segment,
        reference_level=reference_level,
        augment=augment,
        default_inset=0,
        keep_aspect=keep_aspect,
        frame_size=frame_size,
        crop_rectangle=crop_rectangle,
        keep_edge=keep_edge,
    )
    frame_types = {}
    channel_types = set([green_type, blue_type, red_type])
    for type in channel_types:
        if type == FrameTypes.overlay:
            if overlay is None:
                overlay = imageprocessing.overlay_image(
                    data,
                    regions,
                    dim=(frames_per_row * frame_size, frames_per_row * frame_size),
                    require_movement=True,
                )
                channel_data, stats = imageprocessing.normalize(overlay, min=0)
                if not stats[0]:
                    return None
            else:
                channel_data = np.zeros((square.shape[0], square.shape[1]))
                channel_data[: overlay.shape[0], : overlay.shape[1]] = overlay

            if flipped:
                channel_data = np.flip(channel_data, axis=1)
        elif type == FrameTypes.flow_tiled:
            channel_segment = [
                frame.get_channel(TrackChannels.flow) for frame in segment
            ]
            channel_data, success = imageprocessing.square_clip_flow(
                channel_segment, frames_per_row, (frame_size, frame_size)
            )
            if not success:
                return None
        else:
            if type == FrameTypes.thermal_tiled:
                channel = TrackChannels.thermal
            else:
                channel = TrackChannels.filtered

            channel_segment = [frame.get_channel(channel) for frame in segment]
            channel_data, success = imageprocessing.square_clip(
                channel_segment, frames_per_row, (frame_size, frame_size)
            )

            if not success:
                return None

        frame_types[type] = channel_data

    data = np.stack(
        (frame_types[red_type], frame_types[green_type], frame_types[blue_type]), axis=2
    )

    if preprocess_fn:
        data = data * 255
        data = preprocess_fn(data)
    return data

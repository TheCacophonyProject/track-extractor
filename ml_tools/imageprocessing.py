import cv2
import numpy as np
import math

from pathlib import Path
from PIL import Image, ImageDraw
from scipy import ndimage

from ml_tools.tools import eucl_distance
from track.track import TrackChannels


def rotate(image, degrees, mode="nearest", order=1):
    return ndimage.rotate(image, degrees, reshape=False, mode=mode, order=order)


def resize_cv(image, dim, interpolation=None, extra_h=0, extra_v=0):

    return cv2.resize(
        np.float32(image),
        dsize=(dim[0] + extra_h, dim[1] + extra_v),
        interpolation=interpolation if interpolation else cv2.INTER_LINEAR,
    )


def resize_with_aspect(
    frame,
    dim,
    region=None,
    crop_region=None,
    keep_edge=False,
    min_pad=False,
    interpolation=cv2.INTER_LINEAR,
):
    """Resize a numpy frame array while maintaining aspect ratio.
    Pad pixels where needed with minimum frame value or supplied pad value
    If keep edge is true and the frame is the edge of our crop_region make
    sure to keep the resize frame on the edge, otherwise place the original
    frame in the center of our resized frame"""
    if min_pad:
        pad = np.min(frame)
    else:
        pad = 0
    scale_percent = (dim / np.array(frame.shape)).min()
    width = int(frame.shape[1] * scale_percent)
    height = int(frame.shape[0] * scale_percent)
    resize_dim = (width, height)
    resized = np.full(dim, pad, dtype=frame.dtype)
    offset_x = 0
    offset_y = 0
    frame_resized = resize_cv(frame, resize_dim, interpolation=interpolation)
    frame_height, frame_width = frame_resized.shape
    offset_x = (dim[1] - frame_width) // 2
    offset_y = (dim[0] - frame_height) // 2
    if keep_edge and crop_region and region:
        if region.left == crop_region.left:
            offset_x = 0

        elif region.right == crop_region.right:
            offset_x = dim[1] - frame_width

        if region.top == crop_region.top:
            offset_y = 0

        elif region.bottom == crop_region.bottom:
            offset_y = dim[0] - frame_height

    resized[
        offset_y : offset_y + frame_height,
        offset_x : offset_x + frame_width,
    ] = frame_resized

    return resized


def overlay_image(
    frames,
    regions,
    dim,
    require_movement=False,
):
    """Return an image describing the movement by creating a collage of all frames"""
    channel = TrackChannels.filtered

    i = 0
    overlay = np.zeros(dim)

    prev = None
    prev_overlay = None

    # draw movment lines and draw frame overlay
    center_distance = 0
    min_distance = 2
    for i, frame in enumerate(frames):
        region = regions[i]

        x = int(region.mid_x)
        y = int(region.mid_y)

        prev = (x, y)
        # writing overlay image
        if require_movement and prev_overlay:
            center_distance = eucl_distance(
                prev_overlay,
                (
                    x,
                    y,
                ),
            )

        if (
            prev_overlay is None or center_distance > min_distance
        ) or not require_movement:
            frame = frame.get_channel(channel)
            subimage = region.subimage(overlay)
            subimage[:, :] += np.float32(frame)
            center_distance = 0
            min_distance = pow(region.width / 2.0, 2)
            prev_overlay = (x, y)

    return overlay


def square_clip(data, frames_per_row, tile_dim):
    # lay each frame out side by side in rows
    new_frame = np.zeros((frames_per_row * tile_dim[0], frames_per_row * tile_dim[1]))
    i = 0
    success = False
    for x in range(frames_per_row):
        for y in range(frames_per_row):
            if i >= len(data):
                frame = data[-1]
            else:
                frame = data[i]
            frame, stats = normalize(frame)
            if not stats[0]:
                continue
            success = True
            new_frame[
                x * tile_dim[0] : (x + 1) * tile_dim[0],
                y * tile_dim[1] : (y + 1) * tile_dim[1],
            ] = np.float32(frame)
            i += 1

    return new_frame, success


def square_clip_flow(data_flow, frames_per_row, tile_dim, use_rgb=False):
    if use_rgb:
        new_frame = np.zeros(
            (frames_per_row * tile_dim[0], frames_per_row * tile_dim[1], 3)
        )
    else:
        new_frame = np.zeros(
            (frames_per_row * tile_dim[0], frames_per_row * tile_dim[1])
        )

    i = 0
    success = False
    hsv = np.zeros((tile_dim[0], tile_dim[1], 3), dtype=np.float32)
    hsv[..., 1] = 255
    for x in range(frames_per_row):
        for y in range(frames_per_row):
            if i >= len(data_flow):
                flow = data_flow[-1]
            else:
                flow = data_flow[i]
            flow_h = flow[:, :, 0]
            flow_v = flow[:, :, 1]

            mag, ang = cv2.cartToPolar(flow_h, flow_v)
            hsv[..., 0] = ang * 180 / np.pi / 2
            hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
            rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            if use_rgb:
                flow_magnitude = rgb
            else:
                flow_magnitude = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
            frame, norm_success = normalize(flow_magnitude)

            if not norm_success:
                continue
            success = True
            new_frame[
                x * tile_dim[0] : (x + 1) * tile_dim[0],
                y * tile_dim[1] : (y + 1) * tile_dim[1],
            ] = np.float32(frame)
            i += 1
    return new_frame, success


def normalize(data, min=None, max=None, new_max=1):
    """
    Normalize an array so that the values range from 0 -> new_max
    Returns normalized array, stats tuple (Success, min used, max used)
    """
    if data.shape[0] == 0 or data.shape[1] == 0:
        return np.zeros((data.shape)), (False, None, None)
    if max is None:
        max = np.amax(data)
    if min is None:
        min = np.amin(data)
    if max == min:
        if max == 0:
            return np.zeros((data.shape)), (False, max, min)
        return data / max, (True, max, min)
    data -= min
    data = data / (max - min) * new_max
    return data, (True, max, min)


def save_image_channels(data, filename):
    Path(filename).parent.mkdir(parents=True, exist_ok=True)
    r = Image.fromarray(np.uint8(data[:, :, 0] * 255))
    g = Image.fromarray(np.uint8(data[:, :, 1] * 255))
    b = Image.fromarray(np.uint8(data[:, :, 2] * 255))
    concat = np.concatenate((r, g, b), axis=1)
    img = Image.fromarray(np.uint8(concat))
    img.save(filename + ".png")


def detect_objects(image, otsus=True, threshold=0, kernel=(5, 5)):
    image = np.uint8(image)
    image = cv2.GaussianBlur(image, kernel, 0)
    flags = cv2.THRESH_BINARY
    if otsus:
        flags += cv2.THRESH_OTSU
    _, image = cv2.threshold(image, threshold, 255, flags)
    image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    components, small_mask, stats, _ = cv2.connectedComponentsWithStats(image)
    return components, small_mask, stats


def clear_frame(frame):
    # try and remove bad frames by checking for noise
    filtered = frame.filtered
    thermal = frame.thermal
    if len(filtered) == 0 or len(thermal) == 0:
        return False
    thermal_deviation = np.amax(thermal) != np.amin(thermal)
    filtered_deviation = np.amax(filtered) != np.amin(filtered)
    if not thermal_deviation or not filtered_deviation:
        return False

    return True

import cv2
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
import math
from ml_tools.tools import eucl_distance
from track.track import TrackChannels
from scipy import ndimage
from matplotlib import pyplot as plt


def resize_and_pad(
    frame,
    resize_dim,
    new_dim,
    pad=None,
    interpolation=cv2.INTER_LINEAR,
    extra_h=0,
    extra_v=0,
):
    if pad is None:
        pad = np.min(frame)
    resized = np.full(new_dim, pad, dtype=frame.dtype)
    offset = np.int16((np.array(new_dim) - np.array(resize_dim)) / 2.0)
    frame_resized = resize_cv(frame, resize_dim)
    resized[
        offset[1] : offset[1] + frame_resized.shape[0],
        offset[0] : offset[0] + frame_resized.shape[1],
    ] = frame_resized
    return resized


def resize_cv(image, dim, interpolation=cv2.INTER_LINEAR, extra_h=0, extra_v=0):
    return cv2.resize(
        image,
        dsize=(dim[0] + extra_h, dim[1] + extra_v),
        interpolation=interpolation,
    )


def rotate(image, degrees, mode="nearest", order=1):
    return ndimage.rotate(image, degrees, reshape=False, mode=mode, order=order)


def movement_images(
    frames,
    regions,
    dim,
    require_movement=False,
):
    """Return 2 images describing the movement, one has dots representing
    the centre of mass, the other is a collage of all frames
    """
    channel = TrackChannels.filtered

    i = 0
    dots = np.zeros(dim)
    overlay = np.zeros(dim)

    prev = None
    prev_overlay = None
    line_colour = 60
    dot_colour = 120

    img = Image.fromarray(np.uint8(dots))

    d = ImageDraw.Draw(img)
    # draw movment lines and draw frame overlay
    center_distance = 0
    min_distance = 2
    for i, frame in enumerate(frames):
        region = regions[i]

        x = int(region.mid_x)
        y = int(region.mid_y)

        # writing dot image
        if prev is not None:
            d.line(prev + (x, y), fill=line_colour, width=1)
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

    # then draw dots to dot image so they go over the top
    for i, frame in enumerate(frames):
        region = regions[i]
        x = int(region.mid_x)
        y = int(region.mid_y)
        d.point([(x, y)], fill=dot_colour)

    return np.array(img), overlay


def square_clip(data, frames_per_row, tile_dim, type=None):
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


def square_clip_flow(data_flow, frames_per_row, tile_dim, type=None):
    # lay each frame out side by side in rows
    new_frame = np.zeros((frames_per_row * tile_dim[0], frames_per_row * tile_dim[1]))
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
            flow_magnitude = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)

            # flow_magnitude = (
            #     np.linalg.norm(np.float32([flow_h, flow_v]), ord=2, axis=0) / 4.0
            # )
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


def filtered_is_valid(frame, label):
    filtered = frame.filtered
    thermal = frame.thermal
    if len(filtered) == 0 or len(thermal) == 0:
        return False
    thermal_deviation = np.amax(thermal) != np.amin(thermal)
    filtered_deviation = np.amax(filtered) != np.amin(filtered)
    if not thermal_deviation or not filtered_deviation:
        return False
    if label == "false-positive":
        return True

    area = filtered.shape[0] * filtered.shape[1]
    percentile = int(100 - 100 * 16.0 / area)
    threshold = np.percentile(filtered, percentile)
    threshold = max(0, threshold - 40)

    rows = math.floor(0.1 * filtered.shape[0])
    columns = math.floor(0.1 * filtered.shape[1])
    rows = np.clip(rows, 1, 2)
    columns = np.clip(columns, 1, 2)

    top_left = 1 if np.amax(filtered[0:rows][:, 0:columns]) > threshold else 0
    top_right = 1 if np.amax(filtered[0:rows][:, -columns - 1 : -1]) > threshold else 0
    bottom_left = (
        1 if np.amax(filtered[-rows - 1 : -1][:, 0:columns]) > threshold else 0
    )
    bottom_right = (
        1 if np.amax(filtered[-rows - 1 : -1][:, -columns - 1 : -1]) > threshold else 0
    )
    # try and filter out bogus frames where data is on 3 or more corners
    if (top_right + bottom_left + top_left + bottom_right) >= 3:
        return False

    num_less = len(filtered[filtered <= threshold])

    if num_less <= area * 0.05 or np.amax(filtered) == np.amin(filtered):
        return False
    return True

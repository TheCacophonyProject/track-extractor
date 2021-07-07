import attr
import cv2
import numpy as np
from track.track import TrackChannels
from ml_tools.tools import get_clipped_flow
from scipy import ndimage
from ml_tools.imageprocessing import resize_cv, rotate, normalize, resize_with_aspect


@attr.s(slots=True)
class Frame:

    thermal = attr.ib()
    filtered = attr.ib()
    mask = attr.ib()
    frame_number = attr.ib()
    flow = attr.ib(default=None)
    flow_clipped = attr.ib(default=False)
    scaled_thermal = attr.ib(default=None)
    ffc_affected = attr.ib(default=False)
    region = attr.ib(default=None)

    def get_channel(self, channel):
        if channel == TrackChannels.thermal:
            return self.thermal
        elif channel == TrackChannels.filtered:
            return self.filtered
        elif channel == TrackChannels.flow:
            return self.flow
        elif channel == TrackChannels.mask:
            return self.mask
        return None

    @classmethod
    def from_array(
        cls,
        frame_arr,
        frame_number,
        flow_clipped=False,
        ffc_affected=False,
        region=None,
    ):
        flow = None
        if len(frame_arr) == 5:
            flow_h = frame_arr[TrackChannels.flow_h][:, :, np.newaxis]
            flow_v = frame_arr[TrackChannels.flow_v][:, :, np.newaxis]
            flow = np.concatenate((flow_h, flow_v), axis=2)
        return cls(
            frame_arr[TrackChannels.thermal],
            frame_arr[TrackChannels.filtered],
            frame_arr[TrackChannels.mask],
            frame_number,
            flow=flow,
            flow_clipped=flow_clipped,
            ffc_affected=ffc_affected,
            region=region,
        )

    def normalize(self):
        if self.thermal is not None:
            self.thermal, _ = normalize(self.thermal, new_max=255)
        if self.filtered is not None:
            self.filtered, _ = normalize(self.filtered, new_max=255)

    def as_array(self, split_flow=True):
        if self.flow is None:
            return np.asarray([self.thermal, self.filtered, self.mask])
        if split_flow:
            return np.asarray(
                [
                    self.thermal,
                    self.filtered,
                    self.mask,
                    self.flow[:, :, 0],
                    self.flow[:, :, 1],
                ]
            )

        return np.asarray([self.thermal, self.filtered, self.flow, self.mask])

    def generate_optical_flow(self, opt_flow, prev_frame, flow_threshold=40):
        """
        Generate optical flow from thermal frames
        :param opt_flow: An optical flow algorithm
        """
        height, width = self.thermal.shape
        flow = np.zeros([height, width, 2], dtype=np.float32)
        scaled_thermal = self.thermal.copy()
        scaled_thermal[self.mask == 0] = 0
        scaled_thermal, _ = normalize(scaled_thermal, new_max=255)
        scaled_thermal = np.float32(scaled_thermal)

        # threshold = np.median(self.thermal) + flow_threshold
        # scaled_thermal = np.uint8(np.clip(self.thermal - threshold, 0, 255))
        if prev_frame is not None:
            # for some reason openCV spins up lots of threads for this which really slows things down, so we
            # cap the threads to 2
            cv2.setNumThreads(2)
            flow = opt_flow.calc(prev_frame.scaled_thermal, scaled_thermal, flow)
        self.scaled_thermal = scaled_thermal
        self.flow = flow
        if prev_frame:
            prev_frame.scaled_thermal = None

    def unclip_flow(self):
        if self.flow_clipped:
            self.flow *= 1.0 / 256.0
            self.flow_clipped = False

    def clip_flow(self):
        if self.flow is not None:
            self.flow = get_clipped_flow(self.flow)
            self.flow_clipped = True

    def get_flow_split(self, clip_flow=False):
        if self.flow is not None:
            if self.clip_flow and not self.flow_clipped:
                flow_c = get_clipped_flow(self.flow)
                return flow_c[:, :, 0], flow_c[:, :, 1]

            else:
                return self.flow_h, self.flow_v
        return None, None

    def crop_by_region(self, region, out=None):
        # make a new frame cropped by region
        thermal = region.subimage(self.thermal)
        filtered = region.subimage(self.filtered)
        mask = region.subimage(self.mask)
        flow = None
        if self.flow is not None:
            flow = region.subimage(self.flow)
        if out:
            out.thermal = thermal
            out.filtered = filtered
            out.mask = mask
            out.flow = flow
            frame = out
        else:
            frame = Frame(
                thermal,
                filtered,
                mask,
                self.frame_number,
                flow_clipped=self.flow_clipped,
                ffc_affected=self.ffc_affected,
                region=region,
            )
            frame.flow = flow
        return frame

    def resize(self, dim, crop_rectangle=None, keep_aspect=False, keep_edge=False):
        if keep_aspect:
            self.thermal = resize_with_aspect(
                self.thermal,
                dim,
                self.region,
                crop_rectangle,
                keep_edge,
                min_pad=True,
            )
            self.mask = resize_with_aspect(
                self.mask,
                dim,
                self.region,
                crop_rectangle,
                keep_edge,
                interpolation=cv2.INTER_NEAREST,
            )
            self.filtered = resize_with_aspect(
                self.filtered, dim, self.region, crop_rectangle, keep_edge
            )
            if self.flow is not None:
                flow_h = resize_with_aspect(
                    self.flow_h, dim, self.region, crop_rectangle, keep_edge
                )
                flow_v = resize_with_aspect(
                    self.flow_v,
                    dim,
                    self.region,
                    crop_rectangle,
                    keep_edge,
                )
                self.flow = np.stack((flow_h, flow_v), axis=2)

        else:
            self.thermal = resize_cv(self.thermal, dim)
            self.mask = resize_cv(self.mask, dim, interpolation=cv2.INTER_NEAREST)
            if self.flow is not None:
                flow_h = resize_cv(self.flow_h, dim)
                flow_v = resize_cv(self.flow_v, dim)
                self.flow = np.stack((flow_h, flow_v), axis=2)

            self.filtered = resize_cv(self.filtered, dim)

    def rotate(self, degrees):
        self.thermal = rotate(self.thermal, degrees)
        self.mask = rotate(self.mask, degrees)
        if self.flow is not None:
            self.flow = rotate(self.flow, degrees)
        self.filtered = rotate(self.filtered, degrees)

    def float_arrays(self):
        self.thermal = np.float32(self.thermal)
        self.mask = np.float32(self.mask)
        if self.flow is not None:
            self.flow = np.float32(self.flow)
        self.filtered = np.float32(self.filtered)

    def copy(self):
        return Frame(
            self.thermal,
            self.filtered,
            self.mask,
            self.frame_number,
            flow=self.flow,
            flow_clipped=self.flow_clipped,
            ffc_affected=self.ffc_affected,
            region=self.region,
        )

    def flip(self):
        self.thermal = np.flip(self.thermal, axis=1)
        self.mask = np.flip(self.mask, axis=1)
        if self.flow is not None:
            self.flow = np.flip(self.flow, axis=1)
        self.filtered = np.flip(self.filtered, axis=1)

    @property
    def flow_h(self):
        if self.flow is None:
            return None
        return self.flow[:, :, 0]

    @property
    def flow_v(self):
        if self.flow is None:
            return None
        return self.flow[:, :, 1]

    @property
    def channels(self):
        return 5 if self.flow is not None else 3

    @property
    def shape(self):
        return self.thermal.shape

import attr
import cv2
import numpy as np
from track.track import TrackChannels
from ml_tools.tools import get_clipped_flow
from scipy import ndimage
from ml_tools.imageprocessing import resize_cv, rotate, normalize, resize_and_pad


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
    def from_channel(
        cls, frame, channel, frame_number, flow_clipped=False, ffc_affected=False
    ):
        flow = None
        if TrackChannels.thermal == channel:
            thermal = frame
        else:
            thermal = None
        if TrackChannels.filtered == channel:
            filtered = frame
        else:
            filtered = None

        if TrackChannels.mask == channel:
            mask = frame
        else:
            mask = None
        return cls(
            thermal,
            filtered,
            mask,
            frame_number,
            flow=flow,
            flow_clipped=flow_clipped,
            ffc_affected=ffc_affected,
        )

    @classmethod
    def from_array(
        cls, frame_arr, frame_number, flow_clipped=False, ffc_affected=False
    ):
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
        )

    def as_array(self, split_flow=True):
        if split_flow:
            return np.asarray(
                [
                    self.thermal,
                    self.filtered,
                    self.flow[:, :, 0] if self.flow is not None else None,
                    self.flow[:, :, 1] if self.flow is not None else None,
                    self.mask,
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

    def brightness_adjust(self, adjust):
        if self.thermal is not None:
            self.thermal += adjust

    def contrast_adjust(self, adjust):
        if self.thermal is not None:
            self.thermal *= adjust
        if self.filtered is not None:
            self.filtered *= adjust

    def crop_by_region(self, region, out=None):
        # make a new frame cropped by region
        thermal = None
        filtered = None
        mask = None
        flow = None
        if self.thermal is not None:
            thermal = region.subimage(self.thermal)
        if self.filtered is not None:
            filtered = region.subimage(self.filtered)
        if self.mask is not None:
            mask = region.subimage(self.mask)
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
            )
            frame.flow = flow
        return frame

    def resize_with_aspect(self, dim):
        scale_percent = (dim / np.array(self.thermal.shape)).min()
        width = int(self.thermal.shape[1] * scale_percent)
        height = int(self.thermal.shape[0] * scale_percent)
        resize_dim = (width, height)
        if self.thermal is not None:
            self.thermal = resize_and_pad(self.thermal, resize_dim, dim)
        if self.mask is not None:
            self.mask = resize_and_pad(
                self.mask, resize_dim, dim, pad=0, interpolation=cv2.INTER_NEAREST
            )
        if self.filtered is not None:
            self.filtered = resize_and_pad(self.filtered, resize_dim, dim, pad=0)
        if self.flow is not None:
            flow_h = resize_and_pad(self.flow[:, :, 0], resize_dim, dim, pad=0)
            flow_v = resize_and_pad(self.flow[:, :, 1], resize_dim, dim, pad=0)
            self.flow = np.stack((flow_h, flow_v), axis=2)

    def resize(self, dim):
        self.thermal = resize_cv(self.thermal, dim)
        self.mask = resize_cv(self.mask, dim, interpolation=cv2.INTER_NEAREST)
        if self.flow is not None:
            self.flow = resize_cv(self.flow, dim)
        self.filtered = resize_cv(self.filtered, dim)

    def rotate(self, degrees):
        if self.thermal is not None:
            self.thermal = rotate(self.thermal, degrees)
        if self.mask is not None:
            self.mask = rotate(self.mask, degrees)
        if self.flow is not None:
            self.flow = rotate(self.flow, degrees)
        if self.filtered is not None:
            self.filtered = rotate(self.filtered, degrees)

    def float_arrays(self):
        if self.thermal is not None:
            self.thermal = np.float32(self.thermal)
        if self.mask is not None:
            self.mask = np.float32(self.mask)
        if self.flow is not None:
            self.flow = np.float32(self.flow)
        if self.filtered is not None:
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
        )

    def flip(self):
        if self.thermal is not None:
            self.thermal = np.flip(self.thermal, axis=1)
        if self.mask is not None:
            self.mask = np.flip(self.mask, axis=1)
        if self.flow is not None:
            self.flow = np.flip(self.flow, axis=1)
        if self.filtered is not None:
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
    def shape(self):
        return self.thermal.shape

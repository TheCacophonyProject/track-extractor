import h5py
import os
import numpy as np
from multiprocessing import Lock


from ml_tools.tools import get_clipped_flow


class FrameCache:
    def __init__(self, cptv_name, keep_open=True, delete_if_exists=True):
        basename = os.path.splitext(cptv_name)[0]
        self.filename = basename + ".cache"
        self.db = None
        self.keep_open = keep_open
        self.num_farmes = 0
        if delete_if_exists:
            self.delete()

        f = h5py.File(self.filename, "w")
        f.create_group("frames")
        f.close()

    def add_frame(self, frame):
        self.open()
        frames = self.db["frames"]
        frame_group = frames.create_group(str(frame.frame_number))
        frame_group.attrs["ffc_affected"] = frame.ffc_affected

        height, width = frame.thermal.shape

        chunks = (1, height, width)

        dims = (5, height, width)
        frame_node = frame_group.create_dataset("frame", dims, chunks=chunks)
        scaled_flow = get_clipped_flow(frame.flow)
        frame_val = (
            np.int16(frame.thermal),
            np.int16(frame.filtered),
            np.float16(scaled_flow[:, :, 0]),
            np.float16(scaled_flow[:, :, 1]),
            np.int16(frame.mask),
        )
        frame_node[:, :, :] = frame_val
        if not self.keep_open:
            self.close()

    def get_frame(self, frame_number):
        self.open()
        ffc_affected = False
        if str(frame_number) in self.db["frames"]:
            frame_group = self.db["frames"][str(frame_number)]
            frame = frame_group["frame"]
            ffc_affected = frame_group.attrs["ffc_affected"]
        else:
            frame = None
        if not self.keep_open:
            self.close()
        return frame, ffc_affected

    def close(self):
        if self.db:
            self.db.close()
            self.db = None

    def open(self, mode="a"):
        if not self.db:
            self.db = h5py.File(self.filename, mode)

    def delete(self):
        if self.db:
            self.close()
        if os.path.exists(self.filename):
            os.remove(self.filename)

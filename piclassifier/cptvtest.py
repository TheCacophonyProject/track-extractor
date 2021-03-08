#!/usr/bin/python3
import argparse

import socket
import time
import os
from cptv import CPTVReader
import sys
import numpy as np
import h5py

SOCKET_NAME = "/var/run/lepton-frames"
test_cptv = "test.cptv"
test_h5py = "/home/zaza/Cacophony/classifier-pipeline/pithermal65.h5py"


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--cptv", help="a CPTV file to send", default="test.cptv")
    parser.add_argument("--h5", help="a h5py to send")
    parser.add_argument("-clip_id", help="Clip id of h5py file to send")
    args = parser.parse_args()
    return args


def send_cptv(filename, socket):
    """
    Loads a cptv file, and prepares for track extraction.
    """
    with open(filename, "rb") as f:
        reader = CPTVReader(f)
        for i, frame in enumerate(reader):
            f = np.uint16(frame.pix).byteswap()
            socket.sendall(f)
            print("sending frame {}".format(i))


def send_h5py(filename, clip_id, socket):

    db = h5py.File(filename, mode="r")
    clips = db["clips"]
    if clip_id:
        clip = clips[str(clip_id)]
        send_clip(clip, socket)
    else:
        for clip_id in clips:
            send_clip(clips[clip_id], socket)
    db.close()


def send_clip(clip, socket):
    frames = clip["frames"]
    frame_ids = []
    for frame_id in frames:
        frame_ids.append(int(frame_id))

    frame_ids.sort()
    for i in range(1):
        for frame_id in frame_ids:
            f = np.uint16(frames[str(frame_id)]).byteswap()
            socket.sendall(f)


def main():
    args = parse_args()
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)

    try:
        sock.connect(SOCKET_NAME)
        if args.h5:
            print("sending h5 {} clip {}".format(args.h5, args.clip_id))
            send_h5py(args.h5, args.clip_id, sock)
        else:
            print("sending cptv {}".format(args.cptv))
            send_cptv(args.cptv, sock)
    except socket.error as msg:
        print(msg)
        sys.exit(1)
    finally:
        # Clean up the connection
        sock.close()


if __name__ == "__main__":
    main()

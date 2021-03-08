"""
Processes a CPTV file identifying and tracking regions of interest, and saving them in the 'trk' format.
"""

import argparse
import cv2
import os

from ml_tools.logs import init_logging
from config.config import Config
from .cliploader import ClipLoader


def parse_params():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-target",
        default=None,
        help='Target to process, "all" processes all folders, "test" runs test cases, "clean" to remove banned clips from db, or a "cptv" file to run a single source.',
    )

    parser.add_argument(
        "-p",
        "--create-previews",
        action="count",
        help="Create MP4 previews for tracks (can be slow)",
    )
    parser.add_argument(
        "-t",
        "--test-file",
        default="tests.txt",
        help="File containing test cases to run",
    )
    parser.add_argument(
        "-v", "--verbose", action="count", help="Display additional information."
    )
    parser.add_argument("--predictions", action="count", help="Add prediction info")

    parser.add_argument(
        "-r",
        "--reprocess",
        action="count",
        help="Re process clips that already exist in the database",
    )
    parser.add_argument(
        "-i",
        "--show-build-information",
        action="count",
        help="Show openCV build information and exit.",
    )
    parser.add_argument("-c", "--config-file", help="Path to config file to use")
    args = parser.parse_args()

    if args.show_build_information:
        print(cv2.getBuildInformation())
        return None, None

    init_logging()

    config = Config.load_from_file(args.config_file)
    if args.create_previews:
        config.loader.preview = "tracking"
    if args.verbose:
        config.loader.verbose = True

    return config, args


def load_clips(config, args):
    loader = ClipLoader(config, args.reprocess)
    target = args.target
    if target is None:
        target = config.source_folder
    if args.predictions:
        loader.add_predictions()
    else:
        if os.path.splitext(target)[1] == ".cptv":
            loader.process_file(target)
        else:
            loader.process_all(target)


def print_opencl_info():
    """ Print information about opencv support for opencl. """
    if cv2.ocl.haveOpenCL():
        if cv2.ocl.useOpenCL():
            print("OpenCL found and enabled, threads={}".format(cv2.getNumThreads()))
        else:
            print("OpenCL found but disabled")


def main():
    config, args = parse_params()
    if config and args:
        load_clips(config, args)


if __name__ == "__main__":

    # opencv sometimes uses too many threads which can reduce performance.  We are running a worker pool which makes
    # better use of multiple cores, so best to leave thread count per process reasonably low.
    # there is quite a big difference between 1 thread and 2, but after that gains are very minimal, and a lot of
    # cpu time gets wasted, starving the other workers.
    cv2.setNumThreads(2)

    print_opencl_info()

    main()

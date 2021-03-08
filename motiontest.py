import argparse

from cptv import CPTVReader

from config.config import Config
from ml_tools.logs import init_logging
from piclassifier.motiondetector import MotionDetector
from config.thermalconfig import ThermalConfig
from piclassifier.headerinfo import HeaderInfo


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("cptv", help="a CPTV file to detect motion")
    parser.add_argument("-c", "--config-file", help="Path to config file to use")
    parser.add_argument(
        "--thermal-config-file", help="Path to pi-config file (config.toml) to use"
    )

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    init_logging()

    config = Config.load_from_file(args.config_file)
    thermal_config = ThermalConfig.load_from_file(args.thermal_config_file)
    print("detecting on  " + args.cptv)
    with open(args.cptv, "rb") as f:
        reader = CPTVReader(f)

        headers = HeaderInfo(
            res_x=reader.x_resolution,
            res_y=reader.y_resolution,
            fps=9,
            brand="",
            model="",
            frame_size=reader.x_resolution * reader.y_resolution * 2,
            pixel_bits=16,
        )

        motion_detector = MotionDetector(
            thermal_config, config.tracking.motion.dynamic_thresh, None, headers
        )
        for i, frame in enumerate(reader):
            motion_detector.process_frame(frame)


if __name__ == "__main__":
    main()

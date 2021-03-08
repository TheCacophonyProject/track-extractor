import os
import yaml
import argparse

from pathlib import Path
from testconfig import TestRecording, TestConfig
from api import API


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("out_folder", help="Root folder to place downloaded files in.")
    parser.add_argument("user", help="API server username")
    parser.add_argument("password", help="API server password")
    parser.add_argument(
        "-s",
        "--server",
        default="https://api.cacophony.org.nz",
        help="CPTV file server URL",
    )
    parser.add_argument(
        "-t",
        "--test_file",
        default="tracking-tests.yml",
        help="File to save generated tests to",
    )
    parser.add_argument(
        "ids",
        nargs="+",
        help="List of recording ids to download",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    print(args.server)
    api = API(args.user, args.password, args.server)
    out_dir = Path(args.out_folder)
    tests = []
    test_data = TestConfig(tests, args.server, args.out_folder)
    for rec_id in args.ids:
        rec_meta = api.query_rec(rec_id)
        tracks = api.get_tracks(rec_id)
        filename = rec_id.with_suffix(".cptv")
        fullpath = out_dir / filename
        tests.append(
            TestRecording.load_from_meta(rec_meta["recording"], tracks, filename)
        )
        if api.save_file(
            fullpath,
            api._download_signed(rec_meta["downloadRawJWT"]),
        ):
            print("Saved {} - {}.cptv".format(rec_id, fullpath))
        else:
            print("error saving {}".format(rec_id))
    if os.path.exists(args.test_file):
        os.remove(args.test_file)
    with open(args.test_file, "w") as f:
        yaml.dump(test_data, f)


if __name__ == "__main__":
    main()

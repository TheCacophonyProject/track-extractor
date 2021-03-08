import absl.logging
import argparse
import os
import json
import logging
import math
import sys
from pathlib import Path
from ml_tools import tools
from config.config import Config
from classify.clipclassifier import ClipClassifier
from .testconfig import TestConfig
from .api import API

MATCH_ERROR = 1


def match_track(gen_track, expected_tracks):
    score = None
    match = None
    MAX_ERROR = 8
    for track in expected_tracks:
        start_diff = abs(track.start - gen_track.start_s)

        gen_start = gen_track.bounds_history[0]
        distance = tools.eucl_distance(
            (track.start_pos.mid_x, track.start_pos.mid_y),
            (gen_start.mid_x, gen_start.mid_y),
        )
        distance += tools.eucl_distance(
            (track.start_pos.x, track.start_pos.y), (gen_start.x, gen_start.y)
        )
        distance += tools.eucl_distance(
            (track.start_pos.right, track.start_pos.bottom),
            (gen_start.right, gen_start.bottom),
        )
        distance /= 3.0
        distance = math.sqrt(distance)

        # makes it more comparable to start error
        distance /= 4.0
        new_score = distance + start_diff
        if new_score > MAX_ERROR:
            continue
        if score is None or new_score < score:
            match = track
            score = new_score
    return match


class RecordingMatch:
    def __init__(self, filename, id_):
        self.matches = []
        self.unmatched_tracks = []
        self.unmatched_tests = []
        self.filename = filename
        self.number_tracks = 0
        self.id = id_

    def match(self, test, tracks, predictions):
        self.number_tracks += len(test.tracks)

        gen_tracks = sorted(tracks, key=lambda x: x.get_id())
        gen_tracks = sorted(gen_tracks, key=lambda x: x.start_s)

        matched_tests = []
        for i, track in enumerate(gen_tracks):
            prediction = predictions.prediction_for(track.get_id())
            test_track = match_track(track, test.tracks)
            if test_track is not None:
                matched_tests.append(test_track)
                match = Match(
                    test_track, track, prediction.predicted_tag(predictions.labels)
                )
                self.matches.append(match)
            else:
                self.unmatched_tracks.append(
                    (prediction.predicted_tag(predictions.labels), track)
                )
                print(
                    "Unmatched track tag {} start {} end {}".format(
                        prediction.predicted_tag(predictions.labels),
                        track.start_s,
                        track.end_s,
                    )
                )

                self.unmatched_tests = []

        for test_track in test.tracks:
            found = False
            for matched in matched_tests:
                if (
                    matched.start == test_track.start
                    and matched.start_pos == test_track.start_pos
                ):
                    found = True
                    break

            if found is False:
                self.unmatched_tests.append(test_track)

    def print_summary(self):
        matched = [match for match in self.matches if match.tag_match()]
        unmatched = [match for match in self.matches if not match.tag_match()]
        same = [match for match in self.matches if match.status == 0]
        better = [match for match in self.matches if match.status == 1]
        worse = [match for match in self.matches if match.status == -1]
        print("*******Classifying******")
        print(
            "matches {}\tmismatches {}\tunmatched {}".format(
                len(matched), len(unmatched), len(self.unmatched_tracks)
            )
        )
        print("*******Tracking******")
        print("same {} better {}\t worse {}".format(len(same), len(better), len(worse)))
        if len(self.unmatched_tests) > 0:
            print("unmatched tests {}\t ".format(len(self.unmatched_tests)))
        summary = {
            "classify": {"correct": len(matched), "incorrect": len(unmatched)},
            "tracking": {
                "better": len(better),
                "same": len(same),
                "worse": len(worse),
                "unmatched_tracks": len(self.unmatched_tracks),
                "unmatched_tests": len(self.unmatched_tests),
            },
        }
        return summary

    def write_results(self, f):
        f.write("{}{}{}\n".format("-" * 10, "Recording", "-" * 90))
        f.write("Recordings[{}] {}\n\n".format(self.id, self.filename))
        for match in self.matches:
            match.write_results(f)

        if len(self.unmatched_tracks) > 0:
            f.write("Unmatched Tracks\n")
        for (what, track) in self.unmatched_tracks:
            f.write(
                "{} - [{}s]Start-End {} - {}\n".format(
                    what,
                    round(track.end_s - track.start_s, 1),
                    round(track.start_s, 1),
                    round(track.end_s, 1),
                )
            )
        f.write("\n")

        if len(self.unmatched_tests) > 0:
            f.write("Unmatched Tests\n")
        for expected in self.unmatched_tests:
            f.write(
                "{} - Opt[{}s] Start-End {} - {}, Expected[{}s] {} - {}\n".format(
                    expected.tag,
                    round(expected.opt_end - expected.opt_start, 1),
                    expected.opt_start,
                    expected.opt_end,
                    round(expected.end - expected.start, 1),
                    expected.start,
                    expected.end,
                )
            )
        f.write("\n")


class Match:
    def __init__(self, test_track, track, tag):
        expected_length = test_track.opt_end - test_track.opt_start
        self.length_diff = round(expected_length - (track.end_s - track.start_s), 2)
        self.start_diff_s = round(test_track.start - track.start_s, 2)
        self.end_diff_s = round(test_track.end - track.end_s, 2)
        self.opt_start_diff_s = round(test_track.opt_start - track.start_s, 2)
        self.opt_end_diff_s = round(test_track.opt_end - track.end_s, 2)
        self.error = round(abs(self.opt_start_diff_s) + abs(self.opt_end_diff_s), 1)

        if self.error <= test_track.calc_error():
            self.status = 1
        elif self.error < MATCH_ERROR:
            self.status = 0
        else:
            self.status = -1
        self.expected_tag = test_track.tag
        self.got_animal = tag
        self.test_track = test_track
        self.track = track

    def tracking_status(self):
        if self.status == 1:
            return "Better Tracking"
        elif self.status == 0:
            return "Same Tracking"
        return "Worse Tracking"

    def classify_status(self):
        if self.expected_tag == self.got_animal:
            return "Classified Correctly"
        return "Classified Incorrect"

    def write_results(self, f):
        f.write("{}{}{}\n".format("=" * 10, "Track", "=" * 90))

        f.write("{}\t{}\n".format(self.tracking_status(), self.classify_status()))
        f.write("Exepcted:\n")
        f.write(
            "{} - Opt[{}s] Start-End {} - {}, Expected[{}s] {} - {}\n".format(
                self.expected_tag,
                self.test_track.opt_length(),
                self.test_track.opt_start,
                self.test_track.opt_end,
                self.test_track.length(),
                self.test_track.start,
                self.test_track.end,
            )
        )
        f.write("Got:\n")
        f.write(
            "{} - [{}s]Start-End {} - {}\n".format(
                self.got_animal,
                round(self.track.end_s - self.track.start_s, 1),
                round(self.track.start_s, 1),
                round(self.track.end_s, 1),
            )
        )
        f.write("\n")

    def tag_match(self):
        return self.expected_tag == self.got_animal


class TestClassify:
    def __init__(self, args):
        self.test_config = TestConfig.load_from_file(args.tests)

        self.classifier_config = Config.load_from_file(args.classify_config)
        model_file = self.classifier_config.classify.model
        if args.model_file:
            model_file = args.model_file

        path, ext = os.path.splitext(model_file)
        keras_model = False
        if ext == ".pb":
            keras_model = True
        self.clip_classifier = ClipClassifier(
            self.classifier_config,
            self.classifier_config.classify_tracking,
            model_file,
            keras_model,
        )
        # try download missing tests
        if args.user and args.password:
            api = API(args.user, args.password, args.server)
            out_dir = Path(self.test_config.clip_dir)
            if not out_dir.exists():
                out_dir.mkdir()
            for test in self.test_config.recording_tests:
                filepath = out_dir / test.filename
                if not filepath.exists():
                    rec_meta = api.query_rec(test.rec_id)
                    if api.save_file(
                        out_dir / test.filename,
                        api._download_signed(rec_meta["downloadRawJWT"]),
                    ):
                        logging.info("Saved %s", filepath)
        self.results = []

    def run_tests(self, args):
        out_dir = Path(self.test_config.clip_dir)

        for test in self.test_config.recording_tests:
            filepath = out_dir / test.filename
            if not filepath.exists():
                logging.info(
                    " %s not found, add cmd args user and password to download this file from the cacophony server",
                    filepath,
                )
                continue
            logging.info("testing %s ", test.filename)
            clip, predictions = self.clip_classifier.classify_file(filepath)
            rec_match = self.compare_output(clip, predictions, test)
            if self.clip_classifier.previewer:
                mpeg_filename = filepath.with_suffix(".mp4")
                logging.info("Exporting preview to '%s'", mpeg_filename)
                self.clip_classifier.previewer.export_clip_preview(
                    mpeg_filename, clip, predictions
                )
                self.clip_classifier.save_metadata(
                    "", os.path.splitext(test.filename)[0] + ".txt", clip, predictions
                )
            self.results.append(rec_match)

    def write_results(self):
        with open("smoketest-results.txt", "w") as f:
            for res in self.results:
                res.write_results(f)
            f.write("Config\n")
            # yaml.dump(self.classifier_config, f, indent=2)

    def print_summary(self):
        print("===== SUMMARY =====")
        total_summary = None
        total_tracks = 0
        for result in self.results:
            total_tracks += result.number_tracks
            summary = result.print_summary()
            if total_summary is None:
                total_summary = summary
            else:
                for key, value in summary["classify"].items():
                    total_summary["classify"][key] += value

                for key, value in summary["tracking"].items():
                    total_summary["tracking"][key] += value
        if total_summary is None:
            return
        classified_correct = total_summary.get("classify", {}).get("correct", 0)
        tracked_well = total_summary.get("tracking", {}).get("better", 0)
        tracked_well += total_summary.get("tracking", {}).get("same", 0)
        unmatched_tracks = total_summary.get("tracking", {}).get("unmatched_tracks", 0)
        unmatched_tests = total_summary.get("tracking", {}).get("unmatched_tests", 0)

        classified_per = round(100.0 * classified_correct / total_tracks)
        tracked_per = round(100.0 * tracked_well / total_tracks)
        print("===== OVERAL =====")
        print(
            "Classify Results {}% {}/{}".format(
                classified_per, classified_correct, total_tracks
            )
        )
        print(
            "Tracking Results Better/Same {}% {}/{} With {} unmatched tracks (false-positives) and {} missed tracks".format(
                tracked_per,
                tracked_well,
                total_tracks,
                unmatched_tracks,
                unmatched_tests,
            )
        )

    def compare_output(self, clip, predictions, expected):
        rec_match = RecordingMatch(clip.source_file, expected.rec_id)
        rec_match.match(expected, clip.tracks, predictions)
        return rec_match


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--classify_config",
        default="./classifier.yaml",
        help="config file to classify with",
    )

    parser.add_argument(
        "-t",
        "--tests",
        default="smoketest/tracking-tests.yml",
        help="YML file containing tests",
    )

    parser.add_argument(
        "-m",
        "--model-file",
        help="Path to model file to use, will override config model",
    )

    parser.add_argument("--user", help="API server username")
    parser.add_argument("--password", help="API server password")
    parser.add_argument(
        "-s",
        "--server",
        default="https://api.cacophony.org.nz",
        help="CPTV file server URL",
    )
    args = parser.parse_args()
    return args


def init_logging():
    logging.root.removeHandler(absl.logging._absl_handler)
    absl.logging._warn_preinit_stderr = False
    logging.basicConfig(stream=sys.stderr, level=logging.INFO)


def main():
    init_logging()
    args = parse_args()
    test = TestClassify(args)
    test.run_tests(args)
    test.write_results()
    test.print_summary()


if __name__ == "__main__":
    main()


def convert_to_dict(obj):
    return obj.__dict__

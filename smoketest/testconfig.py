import attr
import yaml
from ml_tools.tools import Rectangle


@attr.s
class TestConfig(yaml.YAMLObject):
    yaml_tag = "!TestConfig"
    clip_dir = attr.ib()
    recording_tests = attr.ib()
    server = attr.ib()

    @classmethod
    def load_from_file(cls, filename):
        with open(filename) as stream:
            tests = yaml.load(stream, Loader=yaml.Loader)
        for test in tests.recording_tests:
            for track in test.tracks:
                track.start_pos = Rectangle.from_ltrb(*track.start_pos[1])
                track.end_pos = Rectangle.from_ltrb(*track.end_pos[1])
        return tests


@attr.s
class TestRecording(yaml.YAMLObject):
    yaml_tag = "!TestRecording"

    rec_id = attr.ib()
    filename = attr.ib()
    device_id = attr.ib()
    device = attr.ib()
    group_id = attr.ib()
    group = attr.ib()
    tracks = attr.ib()

    @classmethod
    def load_from_meta(cls, rec_meta, track_meta, filepath):
        rec_id = rec_meta["id"]
        filename = str(filepath.with_suffix(".cptv").absolute())
        device_id = rec_meta["Device"]["id"]
        device = rec_meta["Device"]["devicename"]
        group_id = rec_meta["GroupId"]
        group = rec_meta["Group"]["groupname"]
        tracks = []

        for track in track_meta["tracks"]:
            tag = get_best_tag(track)
            if tag is None:
                tag = {}
            test_track = TestTrack.load_from_meta(track, tag)
            tracks.append(test_track)
        return cls(rec_id, filename, device_id, device, group_id, group, tracks)


@attr.s
class TestTrack(yaml.YAMLObject):
    yaml_tag = "!TestTrack"
    id = attr.ib()
    track_id = attr.ib()
    tag = attr.ib()
    start = attr.ib()
    end = attr.ib()
    opt_start = attr.ib()
    opt_end = attr.ib()
    start_pos = attr.ib()
    end_pos = attr.ib()
    confidence = attr.ib()
    expected = attr.ib()

    @classmethod
    def load_from_meta(cls, track, tag):
        start = track["data"]["start_s"]
        end = track["data"]["end_s"]
        return cls(
            track_id=track["id"],
            tag=tag.get("what"),
            start=start,
            end=end,
            opt_start=start,
            opt_end=end,
            start_pos=track["data"]["positions"][0],
            end_pos=track["data"]["positions"][-1],
            confidence=tag.get("confidence", None),
            expected=True,
        )

    def calc_error(self):
        score = self.opt_start - self.start
        score += self.opt_end - self.end
        score = round(score, 1)
        return score

    def opt_length(self):
        return round(self.opt_end - self.opt_start, 1)

    def length(self):
        return round(self.end - self.start, 1)


def get_best_tag(track, min_confidence=0.6):
    """ returns highest precidence tag from the metadata """

    track_tags = track.get("TrackTags", [])
    track_tags = [tag for tag in track_tags if tag.get("confidence") > min_confidence]

    if not track_tags:
        return None

    # sort by original model so it has first pick if confidence is the same
    track_tags = sorted(
        track_tags,
        key=lambda x: 0 if model_name(x) == "Original" else 1,
    )

    track_tags = sorted(track_tags, key=lambda x: -1 * x["confidence"])
    manual_tags = [tag for tag in track_tags if not tag.get("automatic", False)]
    if len(manual_tags) > 0:
        return manual_tags[0]

    return track_tags[0]


def model_name(x):
    data = x.get("data")
    if not data:
        return "Original"
    return data.get("name", "Original")

"""
Processes a CPTV file identifying and tracking regions of interest, and saving them in the 'trk' format.
"""

# we need to use a non GUI backend.  AGG works but is quite slow so I used SVG instead.
import matplotlib
matplotlib.use("SVG")

import matplotlib.pyplot as plt
import matplotlib.animation as manimation
import pickle
import os
import Tracker
import ast
import glob
import argparse


def purge(dir, pattern):
    for f in glob.glob(os.path.join(dir, pattern)):
        os.remove(os.path.join(dir, f))

def find_file(root, filename):
    """
    Finds a file in root folder, or any subfolders.
    :param root: root folder to search file
    :param filename: exact time of file to look for
    :return: returns full path to file or None if not found.
    """
    for root, dir, files in os.walk(root):
        if filename in files:
            return os.path.join(root, filename)
    return None


class TrackEntry:
    """ Database entry for a track """

    def __init__(self):
        pass

class TrackerTestCase():
    def __init__(self):
        self.source = None
        self.tracks = []



class CPTVTrackExtractor:
    """
    Handles extracting tracks from CPTV files.
    Maintains a database recording which files have already been processed, and some statistics parameters used
    during processing.
    """

    # version number.  Recorded into stats file when a clip is processed.
    VERSION = 1

    # all files will be reprocessed
    OM_ALL = 'all'

    # any clips with a lower version than the current will be reprocessed
    OM_OLD_VERSION = 'version'

    # no clips will be overwritten
    OM_NONE = 'none'

    def __init__(self, out_folder):

        self.hints = {}
        self.colormap = plt.cm.jet
        self.verbose = True
        self.out_folder = out_folder
        self.overwrite_mode = CPTVTrackExtractor.OM_OLD_VERSION
        self.enable_previews = False

        self._init_MpegWriter()

    def _init_MpegWriter(self):
        """ setup our MPEG4 writer.  Requires FFMPEG to be installed. """
        plt.rcParams['animation.ffmpeg_path'] = 'C:\\ffmpeg\\bin\\ffmpeg.exe'
        try:
            self.MpegWriter = manimation.writers['ffmpeg']
        except:
            self.log_warning("FFMPEG is not installed.  MPEG output disabled")
            self.MpegWriter = None


    def load_custom_colormap(self, filename):
        """ Loads a custom colormap used for creating MPEG previews of tracks. """

        if not os.path.exists(filename):
            return

        self.colormap = pickle.load(open(filename, 'rb'))


    def load_hints(self, filename):
        """ Read in hints file from given path.  If file is not found an empty hints dictionary set."""

        self.hints = {}

        if not os.path.exists(filename):
            return

        f = open(filename)
        for line_number, line in enumerate(f):
            line = line.strip()
            # comments
            if line == '' or line[0] == '#':
                continue
            try:
                (filename, file_max_tracks) = line.split()[:2]
            except:
                raise Exception("Error on line {0}: {1}".format(line_number, line))
            self.hints[filename] = int(file_max_tracks)

    def process(self, root_folder):
        """ Process all files in root folder.  CPTV files are expected to be found in folders corresponding to their
            class name. """

        folders = [os.path.join(root_folder, f) for f in os.listdir(root_folder) if
                   os.path.isdir(os.path.join(root_folder, f))]

        for folder in folders:
            print("Processing folder {0}".format(folder))
            self.process_folder(folder)
        print("Done.")

    def process_folder(self, folder_path, tag = None):
        """ Extract tracks from all videos in given folder.
            All tracks will be tagged with 'tag', which defaults to the folder name if not specified."""

        if tag is None:
            tag = os.path.basename(folder_path).upper()

        for file_name in os.listdir(folder_path):
            full_path = os.path.join(folder_path, file_name)
            if os.path.isfile(full_path) and os.path.splitext(full_path )[1].lower() == '.cptv':
                self.process_file(full_path, tag, self.enable_previews)

    def log_message(self, message):
        """ Record message in log.  Will be printed if verbose is enabled. """
        # note, python has really good logging... I should probably make use of this.
        if self.verbose: print(message)

    def log_warning(self, message):
        """ Record warning message in log.  Will be printed if verbose is enabled. """
        # note, python has really good logging... I should probably make use of this.
        print("Warning:",message)


    def needs_processing(self, stats_filename):
        """
        Opens a stats file and checks if this clip needs to be processed.
        :param stats_filename: the full path and filename of the stats file for the clip in question.
        :return: returns true if file should be overwritten, false otherwise
        """

        # if no stats file exists we haven't processed file, so reprocess
        if not os.path.exists(stats_filename):
            return True

        # otherwise check what needs to be done.
        if self.overwrite_mode == CPTVTrackExtractor.OM_ALL:
            return True
        elif self.overwrite_mode == CPTVTrackExtractor.OM_NONE:
            return False

        # read in stats file.
        stats = Tracker.load_tracker_stats(stats_filename)
        try:
            stats = Tracker.load_tracker_stats(stats_filename)
        except Exception as e:
            self.log_warning("Invalid stats file "+stats_filename+" error:"+str(e))
            return True

        if self.overwrite_mode == CPTVTrackExtractor.OM_OLD_VERSION:
            return stats['version'] < CPTVTrackExtractor.VERSION

        raise Exception("Invalid overwrite mode {0}".format(self.overwrite_mode))

    def process_file(self, full_path, tag, create_preview_file = False):
        """
        Extract tracks from specific file, and assign given tag.
        :param full: path: path to CPTV file to be processed
        :param tag: the tag to assign all tracks from this CPTV files
        :param create_preview_file: if enabled creates an MPEG preview file showing the tracking working.  This
            process can be quite time consuming.
        :returns the tracker object
        """

        base_filename = os.path.splitext(os.path.split(full_path)[1])[0]
        cptv_filename = base_filename + '.cptv'
        preview_filename = base_filename + '-preview' + '.mp4'
        stats_filename = base_filename + '.txt'

        destination_folder = os.path.join(self.out_folder, tag.lower())

        stats_path_and_filename = os.path.join(destination_folder, stats_filename)

        # read additional information from hints file
        if cptv_filename in self.hints:
            max_tracks = self.hints[cptv_filename ]
            if max_tracks == 0:
                return
        else:
            max_tracks = None

        # make destination folder if required
        try:
            os.stat(destination_folder)
        except:
            self.log_message(" Making path " + destination_folder)
            os.mkdir(destination_folder)

        # check if we have already processed this file
        if self.needs_processing(stats_path_and_filename ):
            self.log_message("Processing {0} [{1}]".format(cptv_filename , tag))
        else:
            return

        # delete any previous files
        purge(destination_folder, base_filename + "*.trk")
        purge(destination_folder, base_filename + "*.mp4")
        purge(destination_folder, base_filename + "*.txt")

        # read metadata
        meta_data_filename = os.path.splitext(full_path)[0] + ".dat"
        if os.path.exists(meta_data_filename):

            meta_data = ast.literal_eval(open(meta_data_filename,'r').read())

            tag_count = len(meta_data['Tags'])

            # we can only handle one tagged animal at a time here.
            if tag_count != 1:
                return

            confidence = meta_data['Tags'][0]['confidence']
        else:
            print(" - Warning: no metadata found for file.")

        tracker = Tracker.Tracker(full_path)
        tracker.max_tracks = max_tracks
        tracker.tag = tag
        # save some additional stats
        tracker.stats['confidence'] = confidence
        tracker.stats['version'] = CPTVTrackExtractor.VERSION

        tracker.extract()

        tracker.export(os.path.join(self.out_folder, tag, cptv_filename), use_compression=True)

        if create_preview_file:
            tracker.display(os.path.join(self.out_folder, tag.lower(), preview_filename), self.colormap)

        tracker.save_stats(stats_path_and_filename)

        return tracker

    def run_test(self, source_folder, test: TrackerTestCase):
        """ Runs a specific test case. """

        # find the file.  We looking in all the tag folder to make life simpler when creating the test file.
        source_file = find_file(source_folder, test.source)

        if source_file is None:
            print("Could not find {0} in root folder {1}".format(test.source, source_folder))
            return

        tracker = self.process_file(source_file, 'test', create_preview_file=True)

        # read in stats files and see how we did
        if len(tracker.tracks) != len(test.tracks):
            print("[Fail] {0} Incorrect number of tracks, expected {1} found {2}".format(test.source, len(test.tracks), len(tracker.tracks)))
        else:
            print("[PASS] {0}".format(test.source))


    def run_tests(self, source_folder, tests_file):
        """ Processes file in test file and compares results to expected output. """

        tests = []
        test = None

        # we need to make sure all tests are redone every time.
        self.overwrite_mode = CPTVTrackExtractor.OM_ALL

        # load in the test data
        for line in open(tests_file, 'r'):
            line = line.strip()
            if line == '':
                continue
            if line[0] == '#':
                continue

            if line.split()[0].lower() == 'track':
                if test == None:
                    raise Exception("Can not have track before source file.")
                _, expeced_length, expected_movement = line.split()
                test.tracks.append((expeced_length, expected_movement))
            else:
                test = TrackerTestCase()
                test.source = line
                tests.append(test)

        print("Found {0} test cases".format(len(tests)))

        for test in tests:
            self.run_test(source_folder, test)


def parse_params():

    parser = argparse.ArgumentParser()

    parser.add_argument('tag', default='all', help='Tag to process, "all" processes all tags, "test" runs test cases')

    parser.add_argument('-o', '--output-folder', default="d:\cac\\tracks", help='Folder to output tracks to')
    parser.add_argument('-s', '--source-folder', default="d:\\cac\out", help='Source folder root with class folders containing CPTV files')
    parser.add_argument('-c', '--color-map', default="custom_colormap.dat", help='Colormap to use when exporting MPEG files')
    parser.add_argument('-p', '--enable-previews', action='store_true', help='Enables preview MPEG files (can be slow)')
    parser.add_argument('-t', '--test-file', default='tests.txt', help='File containing test cases to run')

    args = parser.parse_args()

    # setup extractor

    extractor = CPTVTrackExtractor(args.output_folder)

    # this colormap is specially designed for heat maps
    extractor.load_custom_colormap(args.color_map)

    # load hints.  Hints are a way to give extra information to the tracker when necessary.
    extractor.load_hints("hints.txt")

    extractor.enable_previews = args.enable_previews

    if args.tag.lower() == 'test':
        print("Running test suite")
        extractor.run_tests(args.source_folder, args.test_file)
        return

    print('Processing tag "{0}"'.format(args.tag))

    if extractor.enable_previews:
        print(" -previews enabled.")

    if args.tag.lower() == 'all':
        extractor.process(args.source_folder)
        return
    else:
        extractor.process_folder(os.path.join(args.source_folder, args.tag), args.tag)
        return


def main():
    parse_params()


main()


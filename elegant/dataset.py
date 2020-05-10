"""Tools for producing torch datasets from elegant experiment directories.

Main entry point is the TimepointList class, which organizes a list of
worms/timepoints across multiple experiments and simplifies loading images and
annotations (&c.) therefrom. A TimepointList can be produced from annotations
dictionaries from one or more experiments, and loaded from / saved to text files.

The TimepointList also facilitates producing test/train splits in which a single
worm position is never split across groups.

Example showing usage of a TimepointList as a helper for a PyTorch Dataset
subclass that provides worm frame transforms of mode-normalized brightfield
images.

from torch.utils import data
from zplib.image import colorize
import freeimage

from elegant import process_images
from elegant import worm_spline
from elegant import dataset

class WormFrameDataset(data.Dataset):
    def __init__(self, timepoint_list):
        super().__init__()
        self.timepoint_list = timepoint_list

    def __len__(self):
        return len(self.timepoint_list)

    def __getitem__(self, i):
        return worm_frame_image(i, image_shape=(1000, 200))

    def normalized_bf_image(self, i):
        bf = freeimage.read(self.timepoint_list.image_path(i, 'bf'))
        mode = process_images.get_image_mode(bf, optocoupler=self.timepoint_list.optocoupler(i))
        # map image image intensities in range (100, 2*mode) to range (0, 2)
        bf = colorize.scale(bf, min=100, max=2*mode, output_max=2)
        # now shift range to (-1, 1)
        bf -= 1
        return bf

    def worm_frame_image(self, i, image_shape):
        bf = self.normalized_bf_image(i)
        annotations = self.timepoint_list.timepoint_annotations(i)
        center_tck, width_tck = annotations['pose']
        reflect = False
        if 'keypoints' in annotations and 'vulva' in annotations['keypoints']:
            x, y = annotations['keypoints']['vulva']
            reflect = y < 0
        image_width, image_height = image_shape
        worm_frame = worm_spline.to_worm_frame(bf, center_tck, width_tck,
            sample_distance=image_height//2, standard_length=image_width, reflect_centerline=reflect)
        mask = worm_spline.worm_frame_mask(width_tck, worm_frame.shape)
        worm_frame[mask == 0] = 0
        return worm_frame

worm_data = WormFrameDataset(dataset.TimepointList.from_file('path/to/worm_data.txt'))
"""

import collections
import itertools
import functools
import pathlib
import random

from . import load_data


class TimepointList(tuple):
    """Manage a set of worm positions / timepoints across multiple experiments
    to simplify loading data therefrom for e.g. traning ML models.

    Features include:
    1) Converting one or more (filtered) sets of annotations to TimepointList,
    2) Splitting sets of annotations into multiple TimepointLists (for e.g.
       test/train splits) such that no worm position is split across sets,
    3) Loading from / saving to text files.

    Example of constructing and saving TimepointLists:

    from elegant import dataset
    from elegant import load_data

    experiments = ['/path/to/exp1', '/path/to/exp2']
    def has_pose(position_name, position_annotations, timepoint_annotations):
        keep = []
        for tp in timepoint_annotations.values():
            pose = tp.get('pose')
            # make sure pose is not None, and center/width tcks are both not None
            good = pose is not None and pose[0] is not None and pose[1] is not None
            keep.append(good)
        return keep

    positions = [load_data.read_annotations(exp) for exp in experiments]
    filtered = [load_data.filter_annotations(pos, has_pose) for pos in positions]
    timepoint_list = dataset.TimepointList.from_annotations(*filtered)
    timepoint_list.to_file('path/to/output.txt')
    train, test, validate = dataset.TimepointList.split_annotations(*filtered, fractions=[0.5, 0.3, 0.2])
    """

    @classmethod
    def from_annotations(cls, *positions_dicts):
        """Convert one or more positions dictionaries to a TimepointList.

        Parameters:
            *positions_dicts: one or more positions dictionaries as returned
                by load_data.read_positions (&c.)

        Returns: TimepointList instance
        """
        positions_list = _flatten_to_positions_list(positions_dicts)
        timepoint_paths = _flatten_to_timepoint_paths(positions_list)
        return cls(timepoint_paths)

    @classmethod
    def split_annotations(cls, *positions_dicts, fractions=[0.75, 0.25], random_seed=0):
        """Split one or more positions dictionaries to multiple TimepointLists.

        Positions are split across multiple TimepointLists based on a list of
        fractions, which controls the fraction of the total number of timepoints
        (not total number of worm positions) found in each TimepointList, under
        the constraint that no worm position is split across TimepointLists.

        Worm positions are randomly shuffled (based on a fixed, specified random
        seed, for reproducibility) before being split.

        Parameters:
            *positions_dicts: one or more positions dictionaries as returned
                by load_data.read_positions (&c.)
            fractions: list that must sum to 1 that specifies the approximate
                fraction of the total number of timepoints which will be in the
                corresponding TimepointList.
            random_seed: string or integer providing random seed for reproducible
                shuffling of the positions.

        Returns: list of TimepointList instance of same length as fractions.
        """
        positions_list = _flatten_to_positions_list(positions_dicts)
        assert sum(fractions) == 1
        # shuffle with a new Random generator from the specified seed. (Don't just re-seed the
        # default random generator, because that would mess up other random streams if in use.)
        random.Random(random_seed).shuffle(positions_list)
        all_timepoints = (timepoints for experiment_root, position, timepoints in positions_list)
        total_timepoints = sum(map(len, all_timepoints))
        target_sizes = [fraction * total_timepoints for fraction in fractions]
        subsets = [[] for _ in fractions]
        # assign positions to each subset in round-robin fashion to try to keep
        # things balanced
        round_robin = itertools.cycle(zip(target_sizes, subsets))
        for position in positions_list:
            # find the next subset that still needs more timepoints
            while True:
                target_size, subset = next(round_robin)
                if len(subset) < target_size:
                    break
            subset += _flatten_to_timepoint_paths([position])
        return [cls(subset) for subset in subsets]

    @classmethod
    def from_file(cls, path):
        """Load TimepointList from a text file.

        The file is expected to be lines pseudo-paths of the form:
        /path/to/experiment_root/position_name/timepoint_name
        where:
        /path/to/experiment_root/position_name
        is a valid path to a position directory containing one or more image
        files with the provided timepoint_name prefix.

        Parameter:
            path: path to a text file containing pseudo-paths as specified above.

        Returns: TimepointList instance.
        """
        path = pathlib.Path(path)
        return cls(path.read_text().strip('\n').split('\n'))

    def __new__(cls, timepoint_paths):
        """Organize a set of timepoints across multiple experiments and positions.

        Parameter:
            timepoint_paths: list of pseudo-paths that specify a valid experiment
                directory, position directory and timepoint name, as follows:
                /path/to/experiment_root/position_name/timepoint_name
                where:
                /path/to/experiment_root/position_name
                is a valid path to a position directory containing one or more
                image files with the provided timepoint_name prefix.
        """
        timepoints = []
        for timepoint_path in timepoint_paths:
            timepoint_path = pathlib.Path(timepoint_path)
            timepoint = timepoint_path.name
            position_dir = timepoint_path.parent
            position = position_dir.name
            experiment_root = position_dir.parent
            timepoints.append(_Timepoint(experiment_root, position, timepoint))
        return super().__new__(cls, timepoints)

    def to_file(self, path):
        """Write list of timepoints to a file.

        Parameter:
            path: path to a text file to write.
        """
        path = pathlib.Path(path)
        path.write_text('\n'.join(str(t.experiment_root / t.position / t.timepoint) for t in self))

    def position_dir(self, i):
        """Return the position directory corresponding to the timepoint with index i."""
        t = self[i]
        return t.experiment_root / t.position

    def image_path(self, i, image_type):
        """Return the path to the requested image type for the timepoint with index i.

        Example:
            bf_image_path = timepoint_list.image_path(i, 'bf')
        """
        t = self[i]
        return t.experiment_root / t.position / f'{t.timepoint} {image_type}.png'

    def experiment_metadata(self, i):
        """Return the metadata dict for the experiment corresponding to the timepoint with index i."""
        return _experiment_data(self[i].experiment_root).metadata

    def optocoupler(self, i):
        """Return the optocoupler for the experiment corresponding to the timepoint with index i."""
        return self.experiment_metadata(i)['optocoupler']

    def position_annotations(self, i):
        """Return the position annotations dict corresponding to the timepoint with index i."""
        t = self[i]
        return _experiment_data(t.experiment_root).annotations[t.position][0]

    def timepoint_annotations(self, i):
        """Return the timepoint annotations dict corresponding to the timepoint with index i."""
        t = self[i]
        return _experiment_data(t.experiment_root).annotations[t.position][1][t.timepoint]


_Timepoint = collections.namedtuple('_Timepoint', ('experiment_root', 'position', 'timepoint'))
_ExperimentData = collections.namedtuple('_ExperimentData', ('metadata', 'annotations'))

def _flatten_to_positions_list(positions_dicts):
    """Convert a list of positions dictionaries as returned by load_data.read_annotations()
    into a list of (experiment_root, position_name, timepoint_names) triples,
    where experiment_root is a pathlib.Path instance, position_name is a string,
    and timepoint_names is a list of strings.
    """
    positions_list = []
    for positions in positions_dicts:
        experiment_root = pathlib.Path(positions.experiment_root)
        for position, (position_annotations, timepoint_annotations) in positions.items():
            timepoints = list(timepoint_annotations.keys())
            positions_list.append((experiment_root, position, timepoints))
    return positions_list

def _flatten_to_timepoint_paths(positions_list):
    """Convert a list of (experiment_root, position_name, timepoint_names) triples
    as produced by _flatten_to_positions_list into a list of
    experiment_root/position_name/timepoint_name pseudo-paths.
    """
    timepoint_paths = []
    for experiment_root, position, timepoints in positions_list:
        for timepoint in timepoints:
            timepoint_paths.append(experiment_root / position / timepoint)
    return timepoint_paths

@functools.lru_cache(maxsize=64)
def _experiment_data(experiment_root):
    """Read and cache experiment metadata and annotations"""
    metadata = load_data.read_metadata(experiment_root)
    annotations = load_data.read_annotations(experiment_root)
    return _ExperimentData(metadata, annotations)
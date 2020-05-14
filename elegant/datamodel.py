import itertools
import json
import operator
import pathlib
import pickle
import random

from zplib import datafile

class _DataclassBase:
    def _compare(self, other, method):
        try:
            return method(self._cmpkey(), other._cmpkey())
        except (AttributeError, TypeError):
            # _cmpkey not implemented, or returns different type, so can't compare with "other".
            return NotImplemented

    def __lt__(self, other):
        return self._compare(other, operator.lt)

    def __le__(self, other):
        return self._compare(other, operator.le)

    def __eq__(self, other):
        return self._compare(other, operator.eq)

    def __ge__(self, other):
        return self._compare(other, operator.ge)

    def __gt__(self, other):
        return self._compare(other, operator.gt)

    def __ne__(self, other):
        return self._compare(other, operator.ne)

    def __hash__(self):
        return hash(self._cmpkey())

    def _cmpkey(self):
        return tuple(getattr(self, field) for field in self._FIELDS)

    def __repr__(self):
        return self.__class__.__qualname__ + '(' + ', '.join([f"{getattr(self, f)!r}" for f in self._FIELDS]) + ' )'


class Experiment(_DataclassBase):
    _FIELDS = ('path',)
    def __init__(self, path, annotation_dir='annotations'):
        super().__init__()
        self.path = pathlib.Path(path)
        self.annotation_dir = self.path / annotation_dir
        self.metadata_file = self.path / 'experiment_metadata.json'
        self.name = self.path.name
        self._positions = None
        self._metadata = None

    def __repr__(self):
        return self.__class__.__qualname__ + f'({self.name!r})'

    @property
    def metadata(self):
        """metadata dict for the experiment."""
        if self._metadata is None:
            self._metadata = json.loads(self.metadata_file.read_text())
        return self._metadata

    def write_metadata(self):
        datafile.json_encode_atomic_legible_to_file(self.metadata, self.metadata_file)

    @property
    def positions(self):
        if self._positions is None:
            self._positions = {}
        for name in sorted(self.metadata['positions'].keys()):
            position = Position(self, name)
            if position.metadata_file.exists():
                self._positions[name] = position
        return self._positions

    def __iter__(self):
        return iter(self.positions.values())

    def __len__(self):
        return len(self.positions)

    def filter(self, position_filter=None, timepoint_filter=None):
        if position_filter is not None and isinstance(position_filter, callable):
            position_filter = [position_filter]
        if timepoint_filter is not None and isinstance(timepoint_filter, callable):
            timepoint_filter = [timepoint_filter]
        for position in list(self):
            if position_filter is not None and not all(pf(position) for pf in position_filter):
                del self.positions[position.name]
            else:
                for timepoint in list(position):
                    if timepoint_filter is not None and not all(tf(timepoint) for tf in timepoint_filter):
                        del position.timepoints[timepoint.name]


class Position(_DataclassBase):
    _FIELDS = ('experiment', 'name')
    def __init__(self, experiment, name):
        super().__init__()
        if not isinstance(experiment, Experiment):
            experiment = Experiment(experiment)
        self.experiment = experiment
        self.name = name
        self.path = self.experiment.path / self.name
        self.metadata_file = self.path / 'position_metadata.json'
        self.annotation_file = self.experiment.annotation_dir / (self.name + '.pickle')
        self._timepoints = None
        self._annotations = None

    def _load_metadata(self):
        """metadata dict for the position."""
        metadata = {timepoint.pop('timepoint'): timepoint
            for timepoint in json.loads(self.metadata_file.read_text())}
        if self._timepoints is None:
            self._timepoints = {name: Timepoint(self, name) for name in metadata}
        for timepoint in self:
            timepoint._metadata = metadata[timepoint.name]
        # NB: if we have metadata for a timepoint not in our dict, it must be
        # because it was deleted from the dict by some filtering operation.
        # Don't attempt to detect or warn about this case.
        return metadata

    def write_metadata(self):
        metadata = [dict(timepoint=timepoint.name, **timepoint.metadata) for timepoint in self]
        datafile.json_encode_atomic_legible_to_file(metadata, self.metadata_file)

    @property
    def timepoints(self):
        if self._timepoints is None:
            # turns out that _load_metadata is the most sensible place to init timepoints
            self._load_metadata()
        return self._timepoints

    def __iter__(self):
        return iter(self.timepoints.values())

    def __len__(self):
        return len(self.timepoints)

    def _load_annotations(self):
        if self.annotation_file.exists():
            position_annotations, timepoint_annotations = pickle.loads(self.annotation_file.read_bytes())
        else:
            position_annotations, timepoint_annotations = {}, {}
        # make sure _annotations is a plain ol' dict in sorted order
        self._annotations = {name: annotations for name, annotations in sorted(position_annotations.items())}
        for timepoint in self:
            timepoint._annotations = timepoint_annotations.get(timepoint.name, {})
        # NB: if we have annotations for a timepoint not in our dict, it must be
        # because it was deleted from the dict by some filtering operation.
        # Don't attempt to detect or warn about this case.

    @property
    def annotations(self):
        if self._annotations is None:
            self._load_annotations()
        return self._annotations

    def write_annotations(self):
        timepoint_annotations = {timepoint.name: timepoint.annotations for timepoint in self}
        self.annotation_file.write_bytes(pickle.dumps((self.annotations, timepoint_annotations)))

class Timepoint(_DataclassBase):
    _FIELDS = ('position', 'name')
    _METADATA_FILENAME = 'position_metadata.json'
    def __init__(self, position, name):
        super().__init__()
        self.position = position
        self.name = name
        self._annotations = None
        self._metadata = None

    @property
    def annotations(self):
        if self._annotations is None:
            self.position._load_annotations()
        return self._annotations

    @property
    def metadata(self):
        if self._metadata is None:
            self.position._load_metadata()
        return self._metadata

    @property
    def path(self):
        return self.position.path / self.name

    def image_path(self, image_type, suffix='png'):
        """Return the path to the requested image type for this timepoint."""
        return self.position.path / f'{self.name} {image_type}.{suffix}'


def flatten(containers):
    """Given a list of Experiments, return an iterator over all the Positions.
    Given an Experiment (or a list of Positions, which is how an Experiment behaves),
    return an iterator over all the Timepoints."""
    return itertools.chain.from_iterable(containers)


class TimepointList(tuple):
    """Manage a set of worm positions / timepoints across multiple experiments
    to simplify loading data therefrom for e.g. traning ML models.
    Features include:
    1) Converting one or more (filtered) Experiments to a TimepointList,
    2) Splitting one or more Experiments into multiple TimepointLists (for e.g.
       test/train splits) such that no worm position is split across sets,
    3) Loading from / saving to text files.
    Example of constructing and saving TimepointLists:
    from elegant import datamodel
    experiments = [datamodel.Experiment(path) for path in ('/path/to/exp1', '/path/to/exp2')]
    def has_pose(timepoint):
        pose = timepoint.get('pose')
        # make sure pose is not None, and center/width tcks are both not None
        return pose is not None and pose[0] is not None and pose[1] is not None
    for experiment in experiments:
        experiment.filter(timepoint_filter=has_pose)
    timepoint_list = dataset.TimepointList.from_experiments(*experiments)
    timepoint_list.to_file('path/to/output.txt')
    train, test, validate = dataset.TimepointList.split_annotations(*experiments, fractions=[0.5, 0.3, 0.2])
    """
    @classmethod
    def from_experiments(cls, *experiments):
        return cls(flatten(flatten(experiments)))

    @classmethod
    def split_experiments(cls, *experiments, fractions=[0.75, 0.25], random_seed=0):
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
        positions = list(flatten(experiments))
        assert sum(fractions) == 1
        # shuffle with a new Random generator from the specified seed. (Don't just re-seed the
        # default random generator, because that would mess up other random streams if in use.)
        random.Random(random_seed).shuffle(positions)
        total_timepoints = sum(map(len, positions))
        target_sizes = [fraction * total_timepoints for fraction in fractions]
        subsets = [[] for _ in fractions]
        # assign positions to each subset in round-robin fashion to try to keep
        # things balanced
        round_robin = itertools.cycle(zip(target_sizes, subsets))
        for position in positions:
            # find the next subset that still needs more timepoints
            while True:
                target_size, subset = next(round_robin)
                if len(subset) < target_size:
                    break
            subset += list(position)
        return [cls(flatten(subset)) for subset in subsets]

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
        paths = pathlib.Path(path).read_text().strip('\n').split('\n')
        experiments = {}
        positions = {}
        timepoints = []
        for path in paths:
            path = pathlib.Path(path)
            timepoint_name = path.name
            position_dir = path.parent
            position_name = position_dir.name
            experiment_root = position_dir.parent
            if experiment_root not in experiments:
                experiments[experiment_root] = Experiment(experiment_root)
            experiment = experiments[experiment_root]
            pos_key = (experiment_root, position_name)
            if pos_key not in positions:
                positions[pos_key] = Position(experiment, position_name)
            position = positions[pos_key]
            timepoints.append(Timepoint(position, timepoint_name))
        return cls(timepoints)

    def to_file(self, path):
        """Write list of timepoints to a file.

        Parameter:
            path: path to a text file to write.
        """
        pathlib.Path(path).write_text('\n'.join(str(t.path) for t in self))

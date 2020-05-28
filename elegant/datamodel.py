import itertools
import json
import operator
import pathlib
import pickle
import random

from zplib import datafile

class _DataclassBase:
    """Basic methods for "data classes" that have a defined set of fields with
    which to compare and hash class instances."""
    _FIELDS = () # subclasses should provide a tuple of field names
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
        return self.__class__.__qualname__ + '(' + ', '.join([f"{getattr(self, f)!r}" for f in self._FIELDS]) + ')'


class Experiment(_DataclassBase):
    """Class that represents an experiment.

    Experiment instances provide an iterable API and a dict-like API to
    access positions by either lexicographical order or name, respectively:
    experiment = Experiment('/path/to/experiment/root')
    for position in experiment:
        # do something
    num_positions = len(experiment)
    pos_3 = experiment.positions['003']

    Experiments also support lazy-loading of the metadata file as needed via
    the metadata attribute.
    """
    _FIELDS = ('path',)
    def __init__(self, path, annotation_dir='annotations'):
        """Create an Experiment instance for data at a given path.

        Parameters:
            path: path to experiment root directory (containing
                experiment_metadata.json)
            annotation_dir: subdirectory from which to load position and
                timepoint annotations.
        """
        super().__init__()
        self.path = pathlib.Path(path).resolve()
        self.annotation_dir = self.path / annotation_dir
        self.metadata_file = self.path / 'experiment_metadata.json'
        self.name = self.path.name
        self._positions = None
        self._metadata = None

    def __repr__(self):
        return self.__class__.__qualname__ + f'({self.name!r})'

    @property
    def metadata(self):
        """Metadata dict for the experiment, read from backing file if not
        presently cached."""
        if self._metadata is None:
            self._metadata = json.loads(self.metadata_file.read_text())
        return self._metadata

    def write_metadata(self):
        """Write the current metadata dictionary back to the metadata file."""
        datafile.json_encode_atomic_legible_to_file(self.metadata, self.metadata_file)

    @property
    def positions(self):
        """"Dict of Position objects associated with the Experiment, read from
        backing file if not presently cached."""
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

    def __contains__(self, position):
        if not isinstance(position, Position):
            return False
        return position.experiment.path == self.path and position.name in self.positions

    def filter(self, position_filter=None, timepoint_filter=None):
        """Delete positions/timepoints from an Experiment instance (but importantly
        not from on disk) based on filter functions.

        Example: To retain only non-excluded timepoints and positions, and to
            further retain only positions that have been fully stage-annotated
            up to the time of death:
                experiment.filter(position_filter=(filter_excluded, filter_staged),
                    timepoint_filter=filter_excluded)

        Parameters:
            position_filter: function or list of functions to be called on each
                Position object contained in the Experiment as
                position_filter(position). If the result of (any of) the position
                filter function(s) evaluates to false, the Position is removed
                from the Experiment.
            timepoint_filter: function or list of functions to be called on each
                Position object contained in the Experiment as
                timepoint_filter(position). The filter must return a list of
                containing True or False for each timepoint in the Position.
                If any of the timepoint filter functions return False for a
                particular timepoint, it will be removed.

        """
        if position_filter is None:
            position_filter = []
        elif callable(position_filter):
            position_filter = [position_filter]
        if timepoint_filter is None:
            timepoint_filter = []
        elif callable(timepoint_filter):
            timepoint_filter = [timepoint_filter]

        for position in list(self):
            for pf in position_filter:
                if not pf(position):
                    del self.positions[position.name]
                    break
            else: # The else clause of for loop executes if the for completes without a break.
                # So in this case, else executes if the position was not deleted...
                tf_results = [tf(position) for tf in timepoint_filter]
                # now transpose wth zip(*tf_results), resulting in a list of results
                # for each timepoint rather than a list of results for each filter-function
                # and then use all() to boolean-and the results together.
                keeps = map(all, zip(*tf_results))
                for timepoint, keep in zip(list(position), keeps):
                    if not keep:
                        del position.timepoints[timepoint.name]
                if len(position) == 0:
                    del self.positions[position.name]

    def diff_vs_disk(self):
        on_disk = self.__class__(self.path)
        removed_positions = []
        removed_timepoints = []
        for position in on_disk:
            in_self = self.positions.get(position.name)
            if in_self is None:
                removed_positions.append(position)
            else:
                for timepoint in position:
                    if timepoint.name not in in_self.timepoints:
                        removed_timepoints.append(timepoint)
        return removed_positions, removed_timepoints

    def purge_from_disk(self, positions, timepoints):
        pass

    @property
    def all_timepoints(self):
        return flatten(self)

def filter_excluded_positions(position):
    """Position filter for Experiment.filter() to remove excluded positions."""
    # if no annotation present, assume not excluded
    return not position.annotations.get('exclude', False)

def filter_excluded_timepoints(position):
    """Tiimepoint filter for Experiment.filter() to remove excluded timepoints."""
    # if no annotation present, assume not excluded
    return [not timepoint.annotations.get('exclude', False) for timepoint in position]

def filter_staged(position):
    """Position filter for Experiment.filter() to include worms that have been
    stage-annotated fully, are noted as "dead", and have at least one non-dead timepoint."""
    stages = [timepoint.annotations.get('stage') for timepoint in position]
    # NB: all(stages) below is True iff there is a non-None, non-empty-string
    # annotation for each stage.
    return all(stages) and stages[-1] == 'dead' and stages[0] != 'dead'

def filter_to_be_staged(position):
    """Position for Experiment.filter() to include worms that still need to be
    stage-annotated fully."""
    stages = [timepoint.annotations.get('stage') for timepoint in position]
    # NB: all(stages) below is True iff there is a non-None, non-empty-string
    # annotation for each stage.
    return not all(stages) or stages[-1] != 'dead'

def filter_living_timepoints(position):
    """Timepoint filter to exclude all timepoints annotated as "egg" or "dead", except the last "egg"
    and/or the first "dead". (The non-excluded "egg" and "dead" allow us to define the hatch and
    death times precisely.)"""
    stages = [timepoint.annotations.get('stage') for timepoint in position]
    n = len(stages)
    trim_eggs = max(0, stages.count('egg') - 1)
    trim_dead = max(0, stages.count('dead') - 1)
    retain = n - trim_eggs - trim_dead
    return [False] * trim_eggs + [True] * retain + [False] * trim_dead


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

    def __repr__(self):
        return self.__class__.__qualname__ + f'({self.experiment.name!r}, {self.name!r})'

    def _load_metadata(self):
        """metadata dict for the position."""
        try:
            metadata_list = json.loads(self.metadata_file.read_text())
        except FileNotFoundError:
            metadata_list = []
        metadata = {timepoint.pop('timepoint'): timepoint for timepoint in metadata_list}
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

    def __contains__(self, timepoint):
        if not isinstance(timepoint, Timepoint):
            return False
        position = timepoint.position
        return (position.experiment.path == self.experiment.path and
                position.name == self.name and
                timepoint.name in self.timepoints)

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
        self._timepoint_annotations = {name: annotation for name, annotation in sorted(timepoint_annotations.items())}
        for timepoint in self:
            # timepoint.annotations is just a reference to annotation dicts stores in self._timepoint_annotations
            timepoint._annotations = self._timepoint_annotations.setdefault(timepoint.name, {})
        # NB: if we have annotations for a timepoint not in our dict, it must be
        # because it was deleted from the dict by some filtering operation.
        # Don't attempt to detect or warn about this case.

    @property
    def annotations(self):
        if self._annotations is None:
            self._load_annotations()
        return self._annotations

    def write_annotations(self):
        """Write contents of position and timepoint annotation dicts to disk.

        Note that if a timepoint was removed by a filtering operation, write_annotations will NOT
        remove those annotations."""
        self.annotation_file.write_bytes(pickle.dumps((self.annotations, self._timepoint_annotations)))

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


class Timepoints(tuple):
    """Manage a set of worm positions / timepoints across multiple experiments
    to simplify loading data therefrom for e.g. traning ML models.
    Features include:
    1) Converting one or more (filtered) Experiments to a Timepoints,
    2) Splitting one or more Experiments into multiple Timepointss (for e.g.
       test/train splits) such that no worm position is split across sets,
    3) Loading from / saving to text files.
    Example of constructing and saving Timepointss:
    from elegant import datamodel
    experiments = [datamodel.Experiment(path) for path in ('/path/to/exp1', '/path/to/exp2')]
    def has_pose(timepoint):
        pose = timepoint.get('pose')
        # make sure pose is not None, and center/width tcks are both not None
        return pose is not None and pose[0] is not None and pose[1] is not None
    for experiment in experiments:
        experiment.filter(timepoint_filter=has_pose)
    timepoint_list = dataset.Timepoints.from_experiments(*experiments)
    timepoint_list.to_file('path/to/output.txt')
    train, test, validate = dataset.Timepoints.split_annotations(*experiments, fractions=[0.5, 0.3, 0.2])
    """
    @classmethod
    def from_experiments(cls, *experiments):
        return cls(flatten(flatten(experiments)))

    @classmethod
    def split_experiments(cls, *experiments, fractions=[0.75, 0.25], random_seed=0):
        """Split one or more positions dictionaries to multiple Timepointss.

        Positions are split across multiple Timepointss based on a list of
        fractions, which controls the fraction of the total number of timepoints
        (not total number of worm positions) found in each Timepoints, under
        the constraint that no worm position is split across Timepointss.

        Worm positions are randomly shuffled (based on a fixed, specified random
        seed, for reproducibility) before being split.

        Parameters:
            *positions_dicts: one or more positions dictionaries as returned
                by load_data.read_positions (&c.)
            fractions: list that must sum to 1 that specifies the approximate
                fraction of the total number of timepoints which will be in the
                corresponding Timepoints.
            random_seed: string or integer providing random seed for reproducible
                shuffling of the positions.

        Returns: list of Timepoints instance of same length as fractions.
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
        return list(map(cls, subsets))

    @classmethod
    def from_file(cls, path):
        """Load Timepoints from a text file.

        The file is expected to be lines of pseudo-paths of the form:
        /path/to/experiment_root/position_name/timepoint_name
        where:
        /path/to/experiment_root/position_name
        is a valid path to a position directory containing one or more image
        files with the provided timepoint_name prefix.

        Parameter:
            path: path to a text file containing pseudo-paths as specified above.

        Returns: Timepoints instance.
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

        The file is written as lines of pseudo-paths of the form:
        /path/to/experiment_root/position_name/timepoint_name

        Parameter:
            path: path to a text file to write.
        """
        pathlib.Path(path).write_text('\n'.join(str(t.path) for t in self))

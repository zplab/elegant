import itertools
import json
import operator
import pathlib
import pickle
import random
import shutil

from zplib import datafile

class _DataclassBase:
    """Basic methods for "data classes" that have a defined set of fields with
    which to compare and hash class instances."""
    _FIELDS = () # subclasses should provide a tuple of field names

    def _cmpkey(self):
        return tuple(getattr(self, field) for field in self._FIELDS)

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

    def __iter__(self):
        return iter(self.positions.values())

    def __len__(self):
        return len(self.positions)

    def __contains__(self, position):
        # Below will be False if dict.get returns None, i.e. don't have a position of that name
        # otherwise, will compare positions for equality, which chains up to comparing experiments too...
        return position is not None and self.positions.get(getattr(position, 'name', None)) == position

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

    @property
    def all_timepoints(self):
        """Iterator over all timepoints in all positions in the experiment."""
        return flatten(self)

    def add_position(self, name, coords):
        """Add a position at the given xyz coordinates to the experiment."""
        position = self.positions[name] = Position(self, name)
        self._positions = dict(sorted(self.positions.items())) # make sure positions dict is in sorted order by keys
        self.metadata['positions'][name] = coords
        return position

    def write_to_disk(self):
        """Write all metadata and annotations for the experiment and all positions/timepoints."""
        self.write_metadata()
        for position in self:
            position.write_metadata()
            position.write_annotations()

    def filter(self, position_filter=None, timepoint_filter=None):
        """Delete positions/timepoints from an Experiment instance (but importantly
        not from on disk) based on filter functions.

        Example: To retain only non-excluded timepoints and positions, and to
            further retain only positions that have been fully stage-annotated
            up to the time of death:
                experiment.filter(position_filter=(filter_excluded_positions, filter_staged),
                    timepoint_filter=filter_excluded_timepoints)

        Parameters:
            position_filter: function or list of functions to be called on each
                Position object contained in the Experiment as
                position_filter(position). Each function must return bool (to
                keep or remove the whole position) or a list of bools, one for
                each timepoint in the position.
            timepoint_filter: function or list of functions to be called on each
                Timepoint object contained in the Experiment as
                timepoint_filter(timepoint). The filter must return a bool
                representing whether to keep or remove the timepoint. If every
                timepoint is removed, the position will be as well.

        Returns: filtered_positions, filtered_timepoints
            filtered_positions: list of Position instances removed (including
                those removed because all timepoints for that position were removed)
            filtered_timepoints: list of Timepoint instances removed (including
                those whose removal caused the removal of the entire Position)

        """
        if position_filter is None:
            position_filter = []
        elif callable(position_filter):
            position_filter = [position_filter]
        if timepoint_filter is None:
            timepoint_filter = []
        elif callable(timepoint_filter):
            timepoint_filter = [timepoint_filter]

        filtered_positions = []
        filtered_timepoints = []
        for position in list(self):
            results = []
            for pf in position_filter:
                result = pf(position)
                if isinstance(result, bool):
                    result = [result] * len(position)
                results.append(result)
            for tf in timepoint_filter:
                results.append([tf(timepoint) for timepoint in position])
            # now transpose wth zip(*results), resulting in a list of results
            # for each timepoint rather than a list of results for each filter-function
            # and then use all() to boolean-and the results together.
            keep_timepoints = list(map(all, zip(*results)))
            for timepoint, keep in zip(list(position), keep_timepoints):
                if not keep:
                    filtered_timepoints.append(position.timepoints.pop(timepoint.name))
            if not any(keep_timepoints):
                filtered_positions.append(self.positions.pop(position.name))
        return filtered_positions, filtered_timepoints

    def purge_filtered(self, filtered_positions, filtered_timepoints, dry_run=False, backup_dirname=None):
        """Delete positions and timepoints from disk that had been filtered.

        Parameters:
            filtered_positions: list of positions to remove, as returned by filter()
            filtered_timepoints: list of timepoints to remove, as returned by filter()
                NB: timepoints may already be in one of the filtered_positions; this
                duplication will be handled correctly.
            dry_run: passed on to position/timepoint.purge_from_disk() calls.
            backup_dirname: passed on to position.purge_from_disk() call.
        """
        for position in filtered_positions:
            position.purge_from_disk(dry_run, backup_dirname)
        filtered_positions = set(filtered_positions)
        for timepoint in filtered_timepoints:
            if timepoint.position not in filtered_positions:
                timepoint.purge_from_disk(dry_run)

    def purge_timepoint(self, timepoint_name, dry_run=False):
        """Remove a specific named timepoint from every position on disk and in memory"""
        for position in self:
            if timepoint_name in position.timepoints:
                position[timepoint_name].purge_from_disk(dry_run=dry_run)
                if not dry_run:
                    del position[timepoint_name]

        timepoint_idx = self.metadata['timepoints'].index(timepoint_name)
        if not dry_run:
            for metadata_list in ['durations', 'timestamps', 'timepoints']:
                del self.metadata[metadata_list][timepoint_idx]
            for metadata_dict in ['brightfield metering', 'fluorescent metering', 'humidity', 'temperature']:
                del self.metadata[metadata_dict][timepoint_name]
            self.write_metadata()


def filter_excluded(position_or_timepoint):
    """Position or timepoint filter for Experiment.filter() to remove excluded positions/timepoints."""
    # if no annotation present, assume not excluded
    return not position_or_timepoint.annotations.get('exclude', False)

def filter_staged(position):
    """Position filter for Experiment.filter() to include only worms that have been
    stage-annotated fully, are noted as "dead", and have at least one non-dead timepoint."""
    stages = [timepoint.annotations.get('stage') for timepoint in position]
    # NB: all(stages) below is True iff there is a non-None, non-empty-string
    # annotation for each stage.
    return all(stages) and stages[-1] == 'dead' and stages[0] != 'dead'

def filter_to_be_staged(position):
    """Position filter for Experiment.filter() to include only worms that still need to be
    stage-annotated fully."""
    stages = [timepoint.annotations.get('stage') for timepoint in position]
    # NB: all(stages) below is True iff there is a non-None, non-empty-string
    # annotation for each stage.
    return not all(stages) or stages[-1] != 'dead'

def make_living_filter(keep_eggs, keep_dead):
    """Return a position filter for Experiment.filter() that retains only the
    last keep_eggs timepoints staged as 'egg' and the first keep_dead timepoints
    staged as 'dead'.
    """
    def living_filter(position):
        """Position filter to exclude all timepoints annotated as 'egg' or 'dead', except the last {keep_eggs} 'egg'
        and/or the first {keep_dead} 'dead' annotations. (The non-excluded 'egg' and 'dead' allow us to define the hatch and
        death times precisely.)"""
        stages = [timepoint.annotations.get('stage') for timepoint in position]
        trim_eggs = max(0, stages.count('egg') - keep_eggs)
        trim_dead = max(0, stages.count('dead') - keep_dead)
        retain = len(position) - trim_eggs - trim_dead
        return [False] * trim_eggs + [True] * retain + [False] * trim_dead
    living_filter.__doc__ = living_filter.__doc__.format(**locals())
    return living_filter

filter_living_timepoints = make_living_filter(keep_eggs=1, keep_dead=1)

def filter_has_pose(timepoint):
    """Timepoint filter for Experiment.filter() to include only worms where the
    centerline and widths have been fully defined."""
    pose = timepoint.annotations.get('pose')
    # make sure pose is not None, and center/width tcks are both not None
    return pose is not None and pose[0] is not None and pose[1] is not None


class Position(_DataclassBase):
    """Class that represents a specific Position within an experiment."""
    _FIELDS = ('experiment', 'name')
    def __init__(self, experiment, name):
        """To add a new position to an Experiment instance, use add_position() instead
        of constructing a Position directly. Direct construction is only appropriate
        for Positions that already exist in the backing files on disk."""
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
        self._metadata = None

    def __contains__(self, timepoint):
        # Below will be False if dict.get returns None, i.e. don't have a timepoint of that name
        # otherwise, will compare timepoints for equality, which chains up to comparing positions and experiments too...
        return timepoint is not None and self.timepoints.get(getattr(timepoint, 'name', None)) == timepoint

    def __iter__(self):
        return iter(self.timepoints.values())

    def __len__(self):
        return len(self.timepoints)

    def __repr__(self):
        return self.__class__.__qualname__ + f'({self.experiment.name!r}, {self.name!r})'

    @property
    def timepoints(self):
        """"Dict of Timepoint objects associated with the Position, read from
        backing file if not presently cached."""
        if self._timepoints is None:
            # turns out that _load_metadata is the most sensible place to init timepoints
            self._load_metadata()
        return self._timepoints

    def _load_metadata(self):
        if self.metadata_file.exists():
            metadata_list = json.loads(self.metadata_file.read_text())
        else:
            metadata_list = []
        self._metadata = {timepoint.pop('timepoint'): timepoint for timepoint in metadata_list}
        if self._timepoints is None:
            self._timepoints = {name: Timepoint(self, name) for name in self._metadata}
        for timepoint in self:
            # timepoint.metadata is just a reference to metadata dicts stored in self._metadata
            timepoint._metadata = self._metadata.setdefault(timepoint.name, {})
            # NB: if we have metadata for a timepoint not in our dict, it must be
            # because it was deleted from the dict by some filtering operation.
            # Don't attempt to detect or warn about this case.

    def write_metadata(self):
        """Write the current set of metadata for all timepoints back to the metadata file."""
        if self._metadata is None:
            self._load_metadata()
        # read from self._metadata instead of gathering from all timepoints in self
        # in case some timepoints have been filtered just temporarily -- don't want to
        # blow away their metadata
        metadata_list = [dict(timepoint=name, **rest) for name, rest in sorted(self._metadata.items())]
        datafile.json_encode_atomic_legible_to_file(metadata_list, self.metadata_file)

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
        """Dictionary of position-level annotation data."""
        if self._annotations is None:
            self._load_annotations()
        return self._annotations

    def write_annotations(self):
        """Write contents of position and timepoint annotation dicts to disk.

        Note that if a timepoint was removed by a filtering operation, write_annotations will NOT
        remove those annotations."""
        self.annotation_file.write_bytes(pickle.dumps((self.annotations, self._timepoint_annotations)))

    def add_timepoint(self, name):
        """Add a new timepoint of the given name to the Position."""
        timepoint = self.timepoints[name] = Timepoint(self, name)
        self._timepoints = dict(sorted(self.timepoints.items())) # make sure timepoints dict is in sorted order by keys
        timepoint._metadata = self.metadata[name] = {}
        self._load_annotations()
        timepoint._annotations = self._timepoint_annotations[name] = {}
        return timepoint

    def purge_from_disk(self, dry_run=False, backup_dirname=None):
        """Remove an entire position from disk, including metadata, annotations, and derived data.

        Parameters:
            dry_run: if True, do not actually delete any files or entries in metadata/annotations.
            backup_dirname: if not None, name of a directory to copy the position
                data/metadata/annotations into.
        """
        annotation_dir = self.experiment.annotation_dir
        if backup_dirname is not None and not dry_run:
            # don't actually backup if just doing a dry run
            backup_root = self.path / backup_dirname
            backup_position = backup_root / self.name
            backup_position.mkdir(parents=True, exist_ok=True)
            backup_annotations = backup_root / annotation_dir.name / self.name
            backup_annotations.mkdir(parents=True, exist_ok=True)

            shutil.copy(self.metadata_file, backup_position / self.metadata_file.name)
            shutil.copy(self.annotation_file, backup_annotations / self.annotation_file.name)
            backup_metadata = backup_root / self.experiment.metadata_file.name
            if not backup_metadata.exists():
                shutil.copy(self.experiment.metadata_file, backup_metadata)

        _maybe_delete(self.path, dry_run)
        _maybe_delete(self.annotation_file, dry_run)
        for data_dir in sorted(self.experiment.path.glob(f'derived_data/*/{self.name}')):
            _maybe_delete(data_dir, dry_run)
        if not dry_run:
            del self.experiment.metadata['positions'][self.name]
            self.experiment.write_metadata()

class Timepoint(_DataclassBase):
    """Class to represent a timepoint of a specific position within an experiment"""
    _FIELDS = ('position', 'name')
    def __init__(self, position, name):
        """To add a new timepoint to a Position instance, use add_timepoint() instead
        of constructing a Timepoint directly. Direct construction is only appropriate
        for Timepoints that already exist in the backing files on disk."""
        super().__init__()
        self.position = position
        self.name = name
        self._annotations = None
        self._metadata = None

    @property
    def annotations(self):
        """Annotation dictionary pertaining to the specific timepoint.

        NB: annotations for all timepoints are stored in a single, per-position
        file, so to write changes to this dict to disk, use Position.write_annotations()
        """
        if self._annotations is None:
            self.position._load_annotations()
        return self._annotations

    @property
    def metadata(self):
        """Metadata dictionary pertaining to the specific timepoint.

        NB: metadata for all timepoints are stored in a single, per-position
        file, so to write changes to this dict to disk, use Position.write_metadata()
        """
        if self._metadata is None:
            self.position._load_metadata()
        return self._metadata

    @property
    def path(self):
        """Pseudo-path representing the prefix for all data files for this timepoint"""
        return self.position.path / self.name

    def image_path(self, image_type, suffix='png'):
        """Return the path to the requested image type for this timepoint."""
        return self.position.path / f'{self.name} {image_type}.{suffix}'

    def purge_from_disk(self, dry_run=False):
        """Remove this position from disk, including metadata, annotations, and derived data.

        Parameters:
            dry_run: if True, do not actually delete any files or entries in metadata/annotations.
        """
        images_for_timepoint = sorted(self.position.path.glob(f'{self.name} *'))
        derived_for_timepoint = sorted((self.position.experiment.path / 'derived_data').glob(f'*/{self.position.name}/{self.name} *'))
        for f in images_for_timepoint + derived_for_timepoint:
            _maybe_delete(f, dry_run)

        if not dry_run:
            del self.position._metadata[self.name]
            self.position.write_metadata()
            del self.position._timepoint_annotations[self.name]
            self.position.write_annotations()


def _maybe_delete(path, dry_run):
    path = pathlib.Path(path)
    prefix = '[DRY RUN] ' if dry_run else ''
    if path.is_dir():
        print(f'{prefix}removing directory {path}/')
        if not dry_run:
            shutil.rmtree(path)
    else:
        print(f'{prefix}deleting file {path}')
        if not dry_run:
            path.unlink()

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
    2) Splitting one or more Experiments into multiple Timepoints instances (for
       e.g. test/train splits) such that no worm position is split across sets,
    3) Loading from / saving to text files.
    Example of constructing and saving Timepoints instances:
        from elegant import datamodel
        experiments = [datamodel.Experiment(path) for path in ('/path/to/exp1', '/path/to/exp2')]
        for experiment in experiments:
            experiment.filter(timepoint_filter=datamodel.has_pose)
        timepoint_list = dataset.Timepoints.from_experiments(*experiments)
        timepoint_list.to_file('path/to/output.txt')
        train, test, validate = dataset.Timepoints.split_annotations(*experiments, fractions=[0.5, 0.3, 0.2])
    """
    @classmethod
    def from_experiments(cls, *experiments):
        """Flatten all of the timepoints from one or more Experiment instances into a Timepoints instance."""
        return cls(flatten(flatten(experiments)))

    @classmethod
    def split_experiments(cls, *experiments, fractions=[0.75, 0.25], random_seed=0):
        """Split one or more positions dictionaries to multiple Timepoints instances.

        Positions are split across multiple Timepointss based on a list of
        fractions, which controls the fraction of the total number of timepoints
        (not total number of worm positions) found in each Timepoints, under
        the constraint that no worm position is split across Timepointss.

        Worm positions are randomly shuffled (based on a fixed, specified random
        seed, for reproducibility) before being split.

        Parameters:
            *experiments: one or more Experiment instances.
            fractions: list (which must sum to 1) that specifies the approximate
                fraction of the total number of timepoints which will be in the
                corresponding Timepoints.
            random_seed: string or integer providing random seed for reproducible
                shuffling of the positions.

        Returns: list of Timepoints instances of same length as fractions.
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
            for target_size, subset in round_robin:
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

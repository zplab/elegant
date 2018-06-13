# This code is licensed under the MIT License (see LICENSE file for details)

import collections
import pathlib
import pickle

def scan_experiment_dir(experiment_root, channels='bf', timepoint_filter=None, image_ext='png'):
    """Read files from a 'worm corral' experiment for further processing or analysis.

    Directory structure is assumed to be an experimental directory, with one or
    more position directories within. Inside each position directory are image
    files, where each image filename begins with a timepoint name (usually an
    ISO timestamp), followed by a space, and then a channel name for that
    timepoint. Each image file ends with an extension, ususally .png or .tif.

    Graphically:
    EXP_DIR/
        POS1/
            TIMEPOINT1 CHANNEL1.EXT
            TIMEPOINT1 CHANNEL2.EXT
            TIMEPOINT2 CHANNEL1.EXT
            TIMEPOINT2 CHANNEL2.EXT
            ...
        POS2/
            TIMEPOINT1 CHANNEL1.EXT
            TIMEPOINT1 CHANNEL2.EXT
            TIMEPOINT2 CHANNEL1.EXT
            TIMEPOINT2 CHANNEL2.EXT
            ...
        ...

    If multiple channels are specified, they will be loaded as image layer
    stacks.

    Parameters:
        experiment_root: the path to an experimental directory.
        channels: list/tuple of image names to load for each timepoint, or
            a single string.
        timepoint_filter: if not None, this must be a function defined as:
            timepoint_filter(position_name, timepoint_name) -> bool
            This function takes the position name and the timepoint name, and
            returns True or False, depending on whether to load those images.
            This can be used to only load a range of positions or timepoints.
        image_ext: the image filename extension.

    Returns: an ordered dictionary mapping position names to position dicts,
        where each position dict is another ordered dictionary mapping timepoint
        names to a list of image files, one image file per channel requested.

    Examples of using timepoint_filter:

    To load only images after a certain date:
        positions = read_annotations('path/to/experiment')
        def timepoint_filter(position_name, timepoint_name):
            return timepoint_name > '2017-07-05'
        files = scan_experiment_directory(experiment_directory,
            timepoint_filter=timepoint_filter)

    To load only positions that weren't previously annotated as "exclude":
        experiment_root = 'path/to/experiment'
        positions = read_annotations(experiment_root)
        good_positions = filter_annotations(positions, filter_excluded)
        def timepoint_filter(position_name, timepoint_name):
            return position_name in good_positions
        files = scan_experiment_directory(experiment_root, timepoint_filter=timepoint_filter)

    For simplicity, this last example is also pre-packaged into the
    scan_good_positions() function:
        files = scan_good_positions('path/to/experiment', filter_excluded)
    """
    experiment_root = pathlib.Path(experiment_root)
    positions = collections.OrderedDict()
    if isinstance(channels, str):
        channels = [channels]
    for image_path in sorted(experiment_root.glob('*/* {}.{}'.format(channels[0], image_ext))):
        position_name = image_path.parent.name
        if position_name not in positions:
            positions[position_name] = collections.OrderedDict()
        timepoints = positions[position_name]
        timepoint_name = image_path.stem.split(' ')[0]
        if timepoint_filter is None or timepoint_filter(position_name, timepoint_name):
            channel_images = []
            for channel in channels:
                image_path = experiment_root / position_name / (timepoint_name + ' {}.{}'.format(channel, image_ext))
                if not image_path.exists():
                    raise RuntimeError('File not found: '.format(str(image_path)))
                channel_images.append(image_path)
            timepoints[timepoint_name] = channel_images
    return positions

def scan_positions(experiment_root, position_filter, channels='bf', image_ext='png'):
    """Load positions whose annotations meet a specified criterion.

    Parameters:
        experiment_root: the path to an experimental directory.
        position_filter: filter-function suitable to pass to filter_annotations()
            (see documentation for filter_annotations.)
        channels: list/tuple of image names to load for each timepoint, or
            a single string.
        image_ext: the image filename extension.

    Returns: an ordered dictionary mapping position names to position dicts,
        where each position dict is another ordered dictionary mapping timepoint
        names to a list of image files, one image file per channel requested.
        Internally, the annotations for the experiment are loaded with
        read_annotations(), and only those positions for which position_filter()
        returns true are loaded.

    Example:
    To load only positions that weren't previously annotated as "exclude":
        files = scan_good_positions('path/to/experiment', filter_excluded)
    """
    positions = read_annotations(experiment_root)
    selected_positions = filter_annotations(positions, position_filter)
    def timepoint_filter(position_name, timepoint_name):
        return position_name in selected_positions and timepoint_name in positions[position_name][1]
    return scan_experiment_directory(experiment_directory, channels, timepoint_filter, image_ext)

def add_position_to_flipbook(ris_widget, position):
    """Add images from a single ordered position dictionary (as returned by
    scan_experiment_dir) to the ris_widget flipbook.

    To wait for all the image loading tasks to finish:
        from concurrent import futures
        futs = add_position_to_flipbook(ris_widget, positions['001'])
        futures.wait(futs)"""
    return ris_widget.add_image_files_to_flipbook(position.values(), page_names=position.keys())

def read_annotations(experiment_root):
    """Read annotation data from an experiment directory.

    Parameters:
        experiment_root: the path to an experimental directory.

    Returns: an ordered dictionary mapping position names to annotations,
        where each annotation is a (position_annotations, timepoint_annotations)
        pair. In this, position_annotations is a dict of "global" per-position
        annotation information, while timepoint_annotations is an ordered dict
        mapping timepoint names to annotation dictionaries (which themselves map
        strings to annotation data).

    Example:
        positions = read_annotations('my_experiment')
        position_annotations, timepoint_annotations = positions['009']
        life_stage = timepoint_annotations['2017-04-23t0122']['stage']
    """
    experiment_root = pathlib.Path(experiment_root)
    positions = collections.OrderedDict()
    for annotation_file in sorted(experiment_root.glob('annotations/*.pickle')):
        worm_name = annotation_file.stem
        positions[worm_name] = read_annotation_file(annotation_file)
    return positions

def read_annotation_file(annotation_file):
    """Read a single annotation file.

    Parameter:
        annotation_file: path to an existing annotation file

    Returns: (position_annotations, timepoint_annotations)
        position_annotations: dict of "global" per-position annotations
        timepoint_annotations: ordered dict mapping timepoint names to
            annotation dictionaries (which map strings to annotation data)
    """
    annotation_file = pathlib.Path(annotation_file)
    with annotation_file.open('rb') as af:
        annotations = pickle.load(af)
    if isinstance(annotations, (dict, collections.OrderedDict)):
        # read in simple or old-style annotation dicts, with no position_annotations
        # or with them stashed in a __global__ key
        position_annotations = annotations.pop('__global__', {})
        timepoint_annotations = annotations
    else:
        position_annotations, timepoint_annotations = annotations
    timepoint_annotations = collections.OrderedDict(sorted(timepoint_annotations.items()))
    return position_annotations, timepoint_annotations

def write_annotations(experiment_root, positions):
    """Converse of read_annotations(): write a set of annotation files back,
    from a positions dictionary like that returned by read_annotations().
    """
    annotation_dir = pathlib.Path(experiment_root) / 'annotations'
    for position_name, (position_annotations, timepoint_annotations) in positions.items():
        annotation_file = annotation_dir / f'{position_name}.pickle'
        write_annotation_file(annotation_file, position_annotations, timepoint_annotations)

def write_annotation_file(annotation_file, position_annotations, timepoint_annotations):
    """Write a single annotation file.

    Parameters:
        annotation_file: path to an existing annotation file
        position_annotations: dict of "global" per-position annotations
        timepoint_annotations: ordered dict mapping timepoint names to
            annotation dictionaries (which map strings to annotation data)
    """
    annotation_file = pathlib.Path(annotation_file)
    annotation_file.parent.mkdir(exist_ok=True)
    with annotation_file.open('wb') as af:
        # convert from  OrderedDict or defaultdict or whatever to plain dict for output
        pickle.dump((dict(position_annotations), dict(timepoint_annotations)), af)

def merge_annotations(positions, positions2):
    """Merge two position dictionaries (as returned by e.g. read_annotations).

    All annotations from 'positions2' will be merged in-place into 'positions'.
    """
    for position, (position_annotations2, timepoint_annotations2) in positions2.items():
        position_annotations, timepoint_annotations = positions.setdefault(position, ({}, {}))
        position_annotations.update(position_annotations2)
        for timepoint, annotations2 in timepoint_annotations2.items():
            timepoint_annotations.setdefault(timepoint, {}).update(annotations2)

def filter_annotations(positions, position_filter):
    """Filter annotation dictionary for an experiment based on some criteria.

    Parameters:
        positions: dict mapping position names to annotations (e.g. as returned
            by read_annotations)
        position_filter: A function that determines whether a given position
            should be included in the output. The function's signature must be:
            position_filter(position_name, position_annotations, timepoint_annotations)
            where position_name is the name of the position, position_annotations
            is a dict of "global" annotations for that position, and timepoint_annotations
            is a dict mapping timepoint names to annotation dicts. (These are
            just as returned by read_annotations &c.)
            The function must return a bool or a list of bools. If a bool is
            returned, the whole position will be included; if a list of bools,
            then only the timepoints for which matching list entry is True will
            be included.

    Returns: OrderedDict of the subset of supplied positions/timepoints for
        which position_filter returned True.

    Examples:
    Basic usage: read in a position dict, define a position_filter function, and
    then call filter_annotations(). Sample example position_filter functions are below.

        positions = read_annotations('/path/to/exp/root')
        def position_filter(position_name, position_annotations, timepoint_annotations):
            ...
        new_positions = filter_annotations(positions, position_filter)

    To exclude all positions with the "exclude" keyword set:
        def position_filter(position_name, position_annotations, timepoint_annotations):
            return position_annotations.get('exclude', True) # if not present, assume not excluded

    To exclude based on whether a string appears or not in the "notes" field:
        def position_filter(position_name, position_annotations, timepoint_annotations):
            return 'bagged' not in position_annotations.get('notes', '')

    To get only worms that have not been fully annotated with life-stages:
        def position_filter(position_name, position_annotations, timepoint_annotations):
            stages = [tp.get('stage') for tp in timepoint_annotations.values()]
            # NB: all(stages) below is True iff there is a non-None, non-empty-string
            # annotation for each stage.
            done = all(stages) and stages[-1] == 'dead'
            return not done

    Several useful filters are pre-defined in this module as filter_*. For example,
    filter_excluded() excludes all positions with the "exclude" keyword set, just
    as above. It's simple to use that filter:
        positions = read_annotations('/path/to/exp/root')
        new_positions = filter_annotations(positions, filter_excluded)
    """
    selected_positions = collections.OrderedDict()
    for position_name, annotations in positions.items():
        position_annotations, timepoint_annotations = annotations
        include = position_filter(position_name, position_annotations, timepoint_annotations)
        try:
            include = list(include)
        except TypeError:
            include = [include] * len(timepoint_annotations)
        assert len(include) == len(timepoint_annotations)
        selected_timepoints = collections.OrderedDict(tpa for tpa, i in zip(timepoint_annotations.items(), include) if i)
        if len(selected_timepoints) > 0:
            selected_positions[position_name] = position_annotations, selected_timepoints
    return selected_positions

def filter_excluded(position_name, position_annotations, timepoint_annotations):
    """Filter-function for filter_annotations() to return non-excluded worms."""
    return position_annotations.get('exclude', True) # if not present, assume not excluded

def filter_staged(position_name, position_annotations, timepoint_annotations):
    """Filter-function for filter_annotations() to return worms that have been
    stage-annotated fully and are noted as "dead"."""
    stages = [tp.get('stage') for tp in timepoint_annotations.values()]
    # NB: all(stages) below is True iff there is a non-None, non-empty-string
    # annotation for each stage.
    return all(stages) and stages[-1] == 'dead'

def filter_good_complete(position_name, position_annotations, timepoint_annotations):
    """Filter-function for filter_annotations() to return only non-excluded worms
    which have been completely annotated with life stages."""
    return (filter_excluded(position_name, position_annotations, timepoint_annotations) and
        filter_staged(position_name, position_annotations, timepoint_annotations))

def filter_good_incomplete(position_name, position_annotations, timepoint_annotations):
    """Filter-function for filter_annotations() to return only non-excluded worms
    which haven't been completely annotated with life stages."""
    return (filter_excluded(position_name, position_annotations, timepoint_annotations) and
        not filter_staged(position_name, position_annotations, timepoint_annotations))

def filter_living_timepoints(position_name, position_annotations, timepoint_annotations):
    """Filter-function for filter_annotations() to return only non-excluded worms
    which have been completely annotated with life stages; of these, all timepoints
    annotated as "egg" or "dead", except the last "egg" and first "dead" will be
    excluded. (The non-excluded "egg" and "dead" allow us to define the hatch and
    death times more carefully.)"""
    if not filter_excluded(position_name, position_annotations, timepoint_annotations):
        return False
    stages = [tp.get('stage') for tp in timepoint_annotations.values()]
    if not all(stages) or stages[-1] != 'dead':
        return False
    good_times = []
    n = len(stages)
    for i, stage in enumerate(stages):
        if stage == 'egg':
            keep = i < n-1 and stages[i+1] != 'egg'
        elif stage == 'dead':
            keep = i > 0 and stages[i-1] != 'dead'
        else:
            keep = True
        good_times.append(keep)
    return good_times
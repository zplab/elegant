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
        channels: list/tuple of image names to acquire for each timepoint, or
            a single string.
        name_prefix: By default, the flipbook will show the position and
            timepoint name for each set of channels loaded. If a name_prefix
            is supplied, it will be listed first. This is useful for
            distinguishing among experiment directories if this function is
            called multiple times.
        timepoint filter: if not None, this must be a function that takes two
            parameters: the position name and the timepoint name, and which
            returns True or False, depending on whether to load those images.
            This can be used to only load a range of positions or timepoints.
            For example, the following will load only timepoints before the
            given date:
                def timepoint_filter(position_name, timepoint_name):
                    return timepoint_name < '2017-07-05'
        image_ext: the image filename extension.

    Returns: an ordered dictionary mapping position names to position dicts,
        where each position dict is another ordered dictionary mapping timepoint
        names to a list of image files, one image file per channel requested.

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
        pickle.dump((position_annotations, timepoint_annotations), af)

def add_position_to_flipbook(rw, position):
    """ Add images from a single ordered position dictionary (as returned by
    scan_experiment_dir) to the ris_widget flipbook.

    To wait for all the image loading tasks to finish:
        from concurrent import futures
        futs = add_position_to_flipbook(rw, positions['001'])
        futures.wait(futs)"""
    rw.add_image_files_to_flipbook(position.values(), page_names=position.keys())
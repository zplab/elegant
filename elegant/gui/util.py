# This code is licensed under the MIT License (see LICENSE file for details)

import pathlib

def add_experiment_images(rw, exp_dir, channels=('bf',), name_prefix='', timepoint_filter=None, image_ext='png'):
    """Add files from a 'worm corral' experiment to the flipbook.

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
        rw: a ris_widget.RisWidget() instance
        exp_dir: the path to an experimental directory.
        channels: image names to load for each timepoint.
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
                def timepoint_filter(position, timepoint):
                    return timepoint < '2017-07-05'
        image_ext: the image filename extension.

    Returns: the list of futures for the image-reading tasks provided by
        rw.add_image_files_to_flipbook()

    To wait for all the image loading tasks to finish:
        from concurrent import futures
        futs = add_experiment_images(rw, exp_dir)
        futures.wait(futs)
    """
    exp_dir = pathlib.Path(exp_dir)
    image_stacks = []
    names = []
    if name_prefix != '':
        name_prefix += '/'
    for image_path in sorted(exp_dir.glob('*/* {}.{}'.format(channels[0], image_ext))):
        pos = image_path.parent.name
        timepoint = image_path.stem.split(' ')[0]
        if timepoint_filter is None or timepoint_filter(pos, timepoint):
            names.append('{}{}/{}'.format(name_prefix, pos, timepoint))
            image_stack = []
            for channel in channels:
                image_path = exp_dir / pos / (timepoint + ' {}.{}'.format(channel, image_ext))
                if not image_path.exists():
                    raise RuntimeError('File not found: '.format(str(image_path)))
                image_stack.append(image_path)
            image_stacks.append(image_stack)
    return rw.add_image_files_to_flipbook(image_stacks, names)
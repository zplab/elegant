# This code is licensed under the MIT License (see LICENSE file for details)

import argparse
import os
import tempfile
import pickle
import pathlib

import freeimage

from . import load_data
from . import segment_images
from . import process_data

def compress_pngs(experiment_root, timepoints=None, level=freeimage.IO_FLAGS.PNG_Z_DEFAULT_COMPRESSION):
    """Recompress image files from an experiment directory.

    Parameters:
        experiment_root: top-level experiment directory
        timepoints: list of timepoints to compress (or list of glob expressions
            to match multiple timepoints). If None, compress all.
        level: flag from freeimage.IO_FLAGS for compression level, or integer
            from 1-9 for compression level.
    """
    timepoint_filter = None
    if timepoints is not None:
        timepoint_set = set(timepoints)
        def timepoint_filter(position_name, timepoint_name):
            return timepoint_name in timepoint_set
    positions = load_data.scan_experiment_dir(experiment_root, channels=None,
         timepoint_filter=timepoint_filter)
    to_compress = [image_path for position_name, timepoint_name, image_path in
         load_data.flatten_positions(positions)]
    for i, image_path in enumerate(to_compress):
        print(f'Compressing {image_path.parent}/{image_path.name} ({i}/{len(to_compress)})')
        image = freeimage.read(image_path)
        with tempfile.NamedTemporaryFile(dir=image_path.parent,
                prefix=image_path.stem + 'compressing_', suffix='.png',
                delete=False) as temp:
            freeimage.write(image, temp.name, flags=level)
        os.replace(temp.name, image_path)

def compress_main(argv=None):
    parser = argparse.ArgumentParser(description="re-compress image files from experiment")
    parser.add_argument('experiment_root', help='the experiment to compress')
    parser.add_argument('timepoints', nargs="*", metavar='timepoint', default=argparse.SUPPRESS,
        help='timepoint(s) to compress')
    parser.add_argument('--level', type=int, default=6, choices=range(1, 10),
        metavar='[1-9]', help="compression level 1-9 (more than 6 doesn't do much)")
    args = parser.parse_args(argv)
    compress_pngs(**args.__dict__)

def segment_experiment(experiment_root, model, channels='bf', use_gpu=True, overwrite_existing=False):
    """Segment all 'bf' image files from an experiment directory.

    For more complex needs, use segment_images.segment_positions. This function
    is largely a simple example of its usage.

    Parameters:
        experiment_root: top-level experiment directory
        model: path to a model file, or name of a model packaged with
            the matlab tool. (If there is no '/' in this parameter, it is
            assumed to be a model name rather than a path.)
        channels: list/tuple of image channels to segment, or a single channel
            as a string.
        use_gpu: whether or not to use the GPU to perform the segmentations
        overwrite_existing: if False, the segmenter will not be run on existing
            mask files.
    """
    experiment_root = pathlib.Path(experiment_root)
    positions = load_data.scan_experiment_dir(experiment_root, channels=channels)
    segment_images.segment_positions(positions, model, use_gpu, overwrite_existing)

    mask_root = pathlib.Path(experiment_root) / 'derived_data' / 'mask'
    with (mask_root / 'notes.txt').open('w') as notes_file:
        notes_file.write(f'These masks were segmented with model {model}')

    process_data.annotate(experiment_root, [process_data.annotate_poses])

def segment_main(argv=None):
    parser = argparse.ArgumentParser(description="segment image files from experiment")
    parser.add_argument('experiment_root', help='the experiment to segment')
    parser.add_argument('model', help='model name or path')
    parser.add_argument('--channel', '-c', action='append', default=argparse.SUPPRESS,
        dest='channels', metavar='CHANNEL',
        help='image channel to segment; can be specified multiple times. If not specified, segment "bf" images only')
    parser.add_argument('--overwrite', dest='overwrite_existing', action='store_true',
        help="don't skip existing masks")
    parser.add_argument('--no-gpu', dest='use_gpu', action='store_false',
        help="disable GPU usage")
    args = parser.parse_args(argv)
    segment_experiment(**args.__dict__)

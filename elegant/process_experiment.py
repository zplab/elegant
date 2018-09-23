# This code is licensed under the MIT License (see LICENSE file for details)

import argparse
import pathlib
import re
import threading

import prompt_toolkit
import freeimage

from zplib.image import threaded_io

from . import load_data
from . import segment_images
from . import process_data
from . import worm_widths

def compress_pngs(experiment_root, timepoints=None,
    level=freeimage.IO_FLAGS.PNG_Z_DEFAULT_COMPRESSION, num_threads=4):
    """Recompress image files from an experiment directory.

    Parameters:
        experiment_root: top-level experiment directory
        timepoints: list of timepoints to compress (or list of glob expressions
            to match multiple timepoints). If None, compress all.
        level: flag from freeimage.IO_FLAGS for compression level, or integer
            from 1-9 for compression level.
        num_threads: number of threads to use.
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
    _VerboseCompressor(level, num_threads, to_compress)

class _VerboseCompressor(threaded_io.PNG_Compressor):
    def __init__(self, level, num_threads, image_paths):
        super().__init__(level, num_threads)
        self.n = len(image_paths)
        self.compressed = 0
        self.compressed_lock = threading.Lock()
        self.wait_first_error(self.compress(image_paths))

    def _compress(self, image_path):
        with self.compressed_lock:
            self.compressed += 1
            print(f'Compressing {image_path.parent}/{image_path.name} ({self.compressed}/{self.n})')
        super()._compress(image_path)

def compress_main(argv=None):
    parser = argparse.ArgumentParser(description="re-compress image files from experiment")
    parser.add_argument('experiment_root', help='the experiment to compress')
    parser.add_argument('timepoints', nargs="*", metavar='timepoint', default=argparse.SUPPRESS,
        help='timepoint(s) to compress')
    parser.add_argument('--level', type=int, default=6, choices=range(1, 10),
        metavar='[1-9]', help="compression level 1-9 (more than 6 doesn't do much)")
    parser.add_argument('--threads', type=int, default=4, choices=range(1, 10),
        metavar='[1-10]', dest='num_threads', help="number of threads to use")
    args = parser.parse_args(argv)
    compress_pngs(**args.__dict__)

def segment_experiment(experiment_root, model, channels='bf', use_gpu=True, overwrite_existing=False):
    """Segment all 'bf' image files from an experiment directory and annotate poses.

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
            mask files, nor will existing annotations be modified even if new
            mask files are generated for a timepoint.
    """
    experiment_root = pathlib.Path(experiment_root)
    positions = load_data.scan_experiment_dir(experiment_root, channels=channels)
    mask_root = experiment_root / 'derived_data' / 'mask'
    segment_images.segment_positions(positions, model, mask_root, use_gpu, overwrite_existing)
    annotations = load_data.read_annotations(experiment_root)
    metadata = load_data.read_metadata(experiment_root)
    age_factor = metadata.get('age_factor', 1) # see if there is an "age factor" stashed in the metadata...
    width_estimator = worm_widths.WidthEstimator.from_experiment_metadata(metadata, age_factor)
    segment_images.annotate_poses_from_masks(positions, mask_root, annotations,
        overwrite_existing, width_estimator)
    load_data.write_annotations(experiment_root, annotations)

class _ListAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        models = segment_images.get_model_names()
        if len(models) == 0:
            print('No models installed.')
        else:
            print('Avaliable models:')
            print('\n    '.join(models))
        parser.exit()

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
    parser.add_argument('--list-models', '-l', default=argparse.SUPPRESS, nargs=0,
        action=_ListAction, help="list available models")
    args = parser.parse_args(argv)
    segment_experiment(**args.__dict__)

def update_metadata_file(experiment_root, nominal_temperature, objective, optocoupler, filter_cube,
        fluorescence_flatfield_lamp=None, **kws):
    metadata = load_data.read_metadata(experiment_root)
    metadata.update(dict(nominal_temperature=nominal_temperature,
        objective=objective, optocoupler=optocoupler, filter_cube=filter_cube,
        fluorescence_flatfield_lamp=fluorescence_flatfield_lamp, **kws))
    load_data.write_metadata(metadata, experiment_root)

def auto_update_metadata_file(experiment_root, nominal_temperature, optocoupler=None, acquire_file=None, **kws):
    experiment_root = pathlib.Path(experiment_root)
    if acquire_file is not None:
        if not acquire_file.endswith('.py'):
            acquire_file += '.py'
        acquire_file = experiment_root / acquire_file
    else:
        # take the alphabetically first python file
        acquire_file = list(sorted(experiment_root.glob('*.py')))[0]
    contents = acquire_file.read_text()
    objective = int(re.search(r'^    OBJECTIVE\s*=\s*(\d+)', contents, flags=re.MULTILINE)[1])
    filter_cube = re.search(r'''^    FILTER_CUBE\s*=\s*['"](.+)['"]''', contents, flags=re.MULTILINE)[1]
    fluorescence_flatfield_lamp = re.search(r'''^    FLUORESCENCE_FLATFIELD_LAMP\s*=\s*['"](.+)['"]''', contents, flags=re.MULTILINE)
    # above will not match 'FLUORESCENCE_FLATFIELD_LAMP = None', so instead the match object will just be None.
    # match object will also be None if there was just no FLUORESCENCE_FLATFIELD_LAMP line at all...
    if fluorescence_flatfield_lamp is not None:
        fluorescence_flatfield_lamp = fluorescence_flatfield_lamp[1]
    if optocoupler is None:
        optocoupler = 1 if objective == 5 else 0.7
    print('**********')
    print(f'Read the following from {acquire_file.parent}/{acquire_file.name}:')
    print(f'objective = {objective}')
    print(f'optocoupler = {optocoupler}')
    print(f'filter_cube = {filter_cube}')
    print(f'fluorescence_flatfield_lamp = {fluorescence_flatfield_lamp}')
    response = prompt_toolkit.shortcuts.confirm('press y if correct, or n/control-C to cancel ')
    if response:
        print('updating metadata')
        update_metadata_file(experiment_root, nominal_temperature, objective,
            optocoupler, filter_cube, fluorescence_flatfield_lamp, **kws)
    else:
        print('canceling')

def update_metadata_main(argv=None):
    parser = argparse.ArgumentParser(description="update experiment metadata file")
    parser.add_argument('experiment_root', help='the experiment to segment')
    parser.add_argument('-t', '--temp', dest='nominal_temperature', type=float,
        help="nominal experiment temperature", required=True)
    parser.add_argument('-o', '--optocoupler', type=float,
        help="optocoupler; if not specified will be 1 for a 10x objective and 0.7 for 5x")
    parser.add_argument('-s', '--script', dest='acquire_file',
        help="filename of acquisition script to parse; if not specified will be the alphabetically first python file in the experiment directory")
    args = parser.parse_args(argv)
    auto_update_metadata_file(**args.__dict__)
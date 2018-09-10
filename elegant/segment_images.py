import pathlib
import os
import numpy
import pkg_resources
import subprocess
import tempfile

import freeimage

from . import process_data, load_data

MATLAB_RUNTIME = '/usr/local/MATLAB/MATLAB_Runtime/v94'
SEGMENT_EXECUTABLE = 'processImageBatch'

def segment_images(images_and_outputs, model, use_gpu=True):
    """Segment images with the external matlab tool.

    Parameters:
        images_and_outputs: list of pairs of (input_path, output_path)
        model: path to a model file, or name of a model packaged with
            the matlab tool. (If there is no '/' in this parameter, it is
            assumed to be a model name rather than a path.)
        use_gpu: whether or not to use the GPU to perform the segmentations

    Returns: subprocess.CompletedProcess instance with relevant information
        from the matlab segmenter run. (Useful attributes include returncode,
        stdout, and stderr.)
    """
    if '/' not in model:
        model = pkg_resources.resource_filename('worm_segmenter','models/' + model)
    mcr_root = pathlib.Path(MATLAB_RUNTIME)
    ld_path = os.pathsep.join(str(mcr_root / mat_dir / 'glnxa64') for mat_dir in ('runtime', 'bin', 'sys/os', 'extern/bin'))
    env = dict(LD_LIBRARY_PATH=ld_path)
    segmenter = pkg_resources.resource_filename('worm_segmenter', SEGMENT_EXECUTABLE)

    with tempfile.NamedTemporaryFile(mode='w') as temp:
        for image_file, mask_file in images_and_outputs:
            temp.write(str(image_file)+'\n')
            temp.write(str(mask_file)+'\n')
        temp.flush()
        # process = subprocess.run([segmenter, temp.name, str(int(bool(use_gpu))), model],
        #     capture_output=True, text=True, env=env)
        process = subprocess.run([segmenter, temp.name, str(int(bool(use_gpu))), model], env=env) # quick hack for python 3.6 since capture_output and text not on python3.6
    return process

def segment_positions(positions, model, use_gpu=True, overwrite_existing=False, mask_root=None):
    """Segment image files from an experiment directory.

    Runs the external matlab segmenter on positions from an experiment directory,
    then runs the pose-finding

    Parameters:
        positions: positions dictionary, as returned by load_data.scan_experiment_dir.
        model: path to a model file, or name of a model packaged with
            the matlab tool. (If there is no '/' in this parameter, it is
            assumed to be a model name rather than a path.)
        use_gpu: whether or not to use the GPU to perform the segmentations
        overwrite_existing: if False, the segmenter will not be run on existing
            mask files.
        mask_root: root directory into which to save generated masks; if None, defaults
            to the standard mask root in 'derived_data'

    Returns: subprocess.CompletedProcess instance with relevant information
        from the matlab segmenter run. (Useful attributes include returncode,
        stdout, and stderr.)
    """

    images_and_outputs = []
    for position_name, timepoint_name, image_path in load_data.flatten_positions(positions):
        if mask_root is None:
            mask_root = image_path.parent.parent / 'derived_data' / 'mask'
        mask_path = mask_root / position_name / (image_path.stem + '.png')
        if overwrite_existing or not mask_path.exists():
            mask_path.parent.mkdir(exist_ok=True, parents=True)
            images_and_outputs.append((image_path, mask_path))
    process = segment_images(images_and_outputs, model, use_gpu)
    return process

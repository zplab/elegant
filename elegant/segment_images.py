# This code is licensed under the MIT License (see LICENSE file for details)

import pathlib
import os
import pkg_resources
import subprocess
import tempfile

import numpy
from scipy import ndimage
from sklearn import mixture

import freeimage
import zplib.image.mask as zpl_mask

from . import load_data
from . import worm_spline
from . import process_images

try:
    import worm_segmenter
    HAS_SEGMENTER = True
except ImportError:
    HAS_SEGMENTER = False


MATLAB_RUNTIME = '/usr/local/MATLAB/MATLAB_Runtime/v94'
SEGMENT_EXECUTABLE = 'processImageBatch'

def get_model_names():
    if not HAS_SEGMENTER:
        return []
    model_dir = pathlib.Path(worm_segmenter.__file__) / 'models'
    if not model_dir.exists():
        return []
    return sorted(f.name for f in model_dir.iterdir() if f.is_file() and not f.name.startswith('.'))

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
        if not HAS_SEGMENTER:
            raise ValueError('Model name without path specified but "worm_segmenter" module with models not installed.')
        model = pkg_resources.resource_filename('worm_segmenter', 'models/' + model)
    mcr_root = pathlib.Path(MATLAB_RUNTIME)
    ld_path = os.pathsep.join(str(mcr_root / mat_dir / 'glnxa64') for mat_dir in ('runtime', 'bin', 'sys/os', 'extern/bin'))
    env = dict(LD_LIBRARY_PATH=ld_path)
    segmenter = pkg_resources.resource_filename('worm_segmenter', SEGMENT_EXECUTABLE)

    with tempfile.NamedTemporaryFile(mode='w') as temp:
        for image_file, mask_file in images_and_outputs:
            temp.write(str(image_file)+'\n')
            temp.write(str(mask_file)+'\n')
        temp.flush()
        # TODO: When all machines are on python 3.7, can use "capture_output=True" instead of stdout and stderr
        # and can use the more clear "text" instead of "universal_newlines":
        # process = subprocess.run([segmenter, temp.name, str(int(bool(use_gpu))), model],
        #     env=env, capture_output=True, text=True)
        process = subprocess.run([segmenter, temp.name, str(int(bool(use_gpu))), model],
            env=env, universal_newlines=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return process

def segment_positions(positions, model, mask_root, use_gpu=True, overwrite_existing=False):
    """Segment image files from a position dictionary and write mask files.

    Parameters:
        positions: positions dictionary, as returned by load_data.scan_experiment_dir.
        model: path to a model file, or name of a model packaged with
            the matlab tool. (If there is no '/' in this parameter, it is
            assumed to be a model name rather than a path.)
        mask_root: root directory into which to save generated masks
        use_gpu: whether or not to use the GPU to perform the segmentations
        overwrite_existing: if False, the segmenter will not be run on existing
            mask files.

    Returns: subprocess.CompletedProcess instance with relevant information
        from the matlab segmenter run. (Useful attributes include returncode,
        stdout, and stderr.)
    """
    images_and_outputs = []
    for position_name, timepoint_name, image_path in load_data.flatten_positions(positions):
        mask_path = mask_root / position_name / (image_path.stem + '.png')
        if overwrite_existing or not mask_path.exists():
            mask_path.parent.mkdir(exist_ok=True, parents=True)
            images_and_outputs.append((image_path, mask_path))
    process = segment_images(images_and_outputs, model, use_gpu)
    return process

def annotate_poses_from_masks(positions, mask_root, annotations, overwrite_existing=False, width_estimator=None):
    """Extract worm poses from mask files and add them to an annotation dictionary.

    Poses from brightfield-derived masks will be stored as the annotation "pose".
    All other masks will be named as '{image_type} pose', where image_type is
    the image channel name (e.g. "gfp"). In addition, these poses will also be
    stored with the same name, but with ' [original]' appended. This way, it is
    possible to determine if and how much a user has modified a pose from the
    original annotation.

    Parameters:
        positions: positions dictionary, as returned by load_data.scan_experiment_dir.
        mask_root: root directory into which to save generated masks.
        annotations: annotation dictionary, as returned by load_data.read_annotations.
        overwrite_existing: if False, pose annotations that already exist will
            not be modified. If no '[original]' annotation exists, that will be
            added regardless, however.
        width_estimator: WidthEstimator instance to use to perform PCA smoothing
            of the derived pose. If None, no smoothing will be employed. Width
            smoothing requires knowledge of the age of a given worm. If no
            age is annotated for the worm at a given timepoint, an age will be
            estimated from the annotated timestamps using the assumption that
            the worm hatched at the first timepoint taken.
    """
    for position_name, timepoint_name, image_path in load_data.flatten_positions(positions):
        mask_path = mask_root / position_name / (image_path.stem + '.png')
        position_annotations, timepoint_annotations = annotations.setdefault(position_name, ({}, {}))
        current_annotation = timepoint_annotations.setdefault(timepoint_name, {})
        image_type = mask_path.stem.split(' ', 1)[1]
        if image_type == 'bf':
            annotation = 'pose'
        else:
            annotation = f'{image_type} pose'
        original_annotation = annotation + ' [original]'
        center_tck, width_tck = annotations.get(annotation, (None, None))
        need_annotation = need_original = False
        if overwrite_existing or center_tck is None:
            need_annotation = True
        elif original_annotation not in annotations:
            need_original = True
        if need_original or need_annotation:
            mask = freeimage.read(mask_path) > 0
            pose = _get_pose(mask, timepoint_annotations, timepoint_name, width_estimator)
            if need_annotation:
                current_annotation[annotation] = pose
            current_annotation[original_annotation] = pose

def _get_pose(mask, timepoint_annotations, timepoint_name, width_estimator):
    center_tck, width_tck = worm_spline.pose_from_mask(mask)
    if width_estimator is not None and width_tck is not None:
        current_annotation = timepoint_annotations[timepoint_name]
        age = None
        if 'age' in current_annotation:
            age = current_annotation['age']
        else:
            first_annotation = timepoint_annotations[sorted(timepoint_annotations.keys())[0]]
            if 'timestamp' in current_annotation and 'timestamp' in first_annotation:
                age = (current_annotation['timestamp'] - first_annotation['timestamp']) / 3600 # age in hours
        width_tck = width_estimator.pca_smooth_widths(width_tck, width_estimator.width_profile_for_age(age))
    return center_tck, width_tck

def find_lawn_from_images(images, optocoupler):
    """Find bacterial lawn from a set of images."""
    images = [process_images.pin_image_mode(image, optocoupler=optocoupler) for image in images]
    lawns = [find_lawn_in_image(image, optocoupler) for image in images]
    return numpy.bitwise_or.reduce(lawns, axis=0)

def find_lawn_in_image(image, optocoupler, return_model=False):
    """Find a lawn in an image use Gaussian mixture modeling (GMM)

    This lawn maker models an image (i.e. its pixel intensities) as as mixture
        of two Gaussian densities. Each corresponds to either the background & lawn.

    Parameters:
        image: numpy ndarray of the image to find the lawn from
        optocoupler: optocoupler magnification (as a float) used for the specified image
        return model: if True, return the GMM fit to the images.

    Returns: lawn_mask or (lawn_mask, gmm_model)
        lawn mask: bool ndarray
        gmm_model: if return_model is True, also return the fitted GMM model
    """
    filtered_image = ndimage.filters.median_filter(image, size=(3,3), mode='constant')
    vignette_mask = process_images.vignette_mask(optocoupler, image.shape)

    img_data = filtered_image[vignette_mask]
    gmm = mixture.GaussianMixture(n_components=2)
    gmm.fit(numpy.expand_dims(img_data, 1))

    # Calculate boundary point for label classification as intensity threshold
    gmm_support = numpy.linspace(0, 2**16-1, 2**16)

    labels = gmm.predict(numpy.reshape(gmm_support, (-1,1)))
    thr = numpy.argmax(numpy.abs(numpy.diff(labels)))
    lawn_mask = (filtered_image < thr) & vignette_mask

    # Smooth the lawn mask by eroding, grabbing the largest object, and dilating back
    lawn_mask = ndimage.morphology.binary_erosion(lawn_mask, iterations=10)
    lawn_mask = zpl_mask.get_largest_object(lawn_mask)
    lawn_mask = ndimage.morphology.binary_fill_holes(lawn_mask)
    lawn_mask = ndimage.morphology.binary_dilation(lawn_mask, iterations=10)

    if return_model:
        return lawn_mask, gmm
    else:
        return lawn_mask

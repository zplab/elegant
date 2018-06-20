# This code is licensed under the MIT License (see LICENSE file for details)

import pathlib
import itertools
import multiprocessing
import json
import numpy
import collections

from zplib.curve import spline_geometry
import freeimage

from . import worm_data
from . import load_data
from . import worm_spline
from . import measure_fluor

DERIVED_ROOT = 'derived_data'


def update_annotations(experiment_root):
    """Run prior to manually annotating an experiment directory, in order to
    update the annotations dictionaries with all relevant data that can be
    automatically extracted.
    """
    annotate(experiment_root, [TimestampAnnotator(), PoseFromMaskAnnotator()])

def annotate(experiment_root, annotators):
    """Apply one or more measurement functions to produce annotations for manual review.

    Parameters:
        experiment_root: the path to an experimental directory.
        annotators: list of annotation instances to apply to each timepoint. Each
            will be called as:
                annotator.annotate(experiment_root, position, timepoint, metadata, annotations)
            where:
                experiment_root: as above
                positions: name of position
                timepoint: name of timepoint
                metadata: metadata dict for the timepoint from 'position_metadata.json'
                annotations: annotation dictionary for the timepoint
            The function should add anything relevant to the annotations dictionary;
            and the function may use the contents of that dictionary to decide
            whether to overwrite existing annotations with new ones or not.
            The return value of the measure function is ignored.
    """
    experiment_root = pathlib.Path(experiment_root)
    positions = load_data.read_annotations(experiment_root)
    for metadata_path in sorted(experiment_root.glob('*/position_metadata.json')):
        with metadata_path.open('r') as f:
            position_metadata = json.load(f)
        position = metadata_path.parent.name
        position_annotations, timepoint_annotations = positions.setdefault(position, ({}, {}))
        for metadata in position_metadata:
            timepoint = metadata['timepoint']
            annotations = timepoint_annotations.setdefault(timepoint, {})
            for annotator in annotators:
                annotator.annotate(experiment_root, position, timepoint, metadata, annotations)
    load_data.write_annotations(experiment_root, positions)

class TimestampAnnotator:
    def annotate(self, experiment_root, position, timepoint, metadata, annotations):
        annotations['timestamp'] = metadata['timestamp']

class PoseFromMaskAnnotator:
    def __init__(self, mask_name='bf', dest_annotation='pose'):
        self.mask_name = mask_name
        self.dest_annotation = dest_annotation

    def annotate(self, experiment_root, position, timepoint, metadata, annotations):
        mask_path = experiment_root / DERIVED_ROOT / 'mask' / position / f'{timepoint} {self.mask_name}.png'
        if mask_path.exists():
            center_tck, width_tck = annotations.get('pose', (None, None))
            if center_tck is None:
                mask = freeimage.read(mask_path) > 0
                annotations[self.dest_annotation] = worm_spline.pose_from_mask(mask)



def measure_worms(experiment_root, positions, measures, measurement_name, n_jobs=1):
    """Apply one or more measurement functions to all or some worms in an experiment.

    The measurements are class instances that produce and save files (such as
    image-segmentation tools), or produce timecourse data from image data (or
    other files produces) and/or from experimental annotations. Any timecourse
    data produced is saved into a named .tsv file for each position, with the
    worm simply named as the position name.

    measure_worms can be run more than once on a given experiment_root with
    different subsets of positions and measurements, if some measurements don't
    make sense to apply to all timepoints (for example), or if some measurements
    are run later on (or are run in parallel on the cluster, perhaps). To prevent
    collisions, the output data produced by different runs of measure_worms
    should be identified with distinct values of the measurement_name parameter.

    The function collate_data can then be used to merge all these .tsv files
    together into one authoriatative file for the experimental measurements.

    Parameters:
        experiment_root: the path to an experimental directory.
        positions: dict mapping position names to annotations (e.g. as returned
            by load_data.read_annotations). Only the positions and timepoints
            listed will be measured, so load_data.filter_annotations can be used
            to restrict measurements to a useful subset. In particular, use of
            load_data.filter_living_timepoints filter-function with
            filter_annotations is recommended:
                positions = load_data.read_annotations('path/to/experiment')
                to_measure = load_data.filter_annotations(positions, load_data.filter_living_timepoints)
        measures: list of measurements to apply to each timepoint. A measurement
            must (1) have an attribute 'feature_names' which is a list of the
            names of the measurement(s) it produces, and (2) have a 'measure'
            method that will be called as:
                data = measure(position_root, derived_root, timepoint, annotations)
            where the parameters are as follows:
                position_root: path to directory of data files for the position
                    (i.e. experiment_root / position_name)
                derived_root: path to directory where output of measurements is
                    to be written. Meaasurement functions that produce files
                    should write them into a subdirectory of derived_root named
                    based on the measurement, organized as follows:
                    '{derived_root}/{name}/{position_name}/{timepoint}.ext'
                    where position_name can be obtained as position_root.name.
                timepoint: name of the timepoint to be measured
                annotations: annotation dict for this position at the given
                    timepoint
            and the return value must be a list of measurements of the same
            length as the feature_names attribute, or nan if that measurement
            cannot be made on the given worm and timepoint (e.g. missing data).
            If a measurement produces no data (e.g. it only produces file output,
            such as segmentation masks), then feature_names may be None, and
            the return value of measure must also be None.
        measurement_name: the name of this set of measurements. The .tsv files
            produced by this function (one for each position) will be written
            to '{experiment_root}/derived_data/measurements/{measurement_name}'
        n_jobs: if > 1, divide the provided positions dictionary into this many
            parts and run the measurements in parallel via the multiprocessing
            library.

    Below is an example of performing some measurements on all worms and others
    on only a subset. (In practice, however, it generally makes more sense to
    apply all measures to all data. A properly-written measurement function
    will return None when the required data is missing, producing .tsv files
    with lots of blank spaces. This is just fine and is simpler.)

    positions = load_data.read_annotations('path/to/experiment')
    to_measure = load_data.filter_annotations(positions, load_data.filter_living_timepoints)
    measures = [BasicMeasurements(), PoseMeasurements(microns_per_pixel=5)]
    measure_worms(experiment_root, to_measure, measures, 'core_measures')
    def filter_adult(position_name, position_annotations, timepoint_annotations):
        return [tp['stage'] == 'adult' for tp in timepoint_annotations.values()]
    adult_timepoints = load_data.filter_annotations(positions, filter_adult)
    adult_gfp = FluorMeasurements('gfp')
    measure_worms(experiment_root, adult_timepoints, [adult_gfp], 'gfp_measures')
    """
    if n_jobs > 1:
        _multiprocess_measure(experiment_root, positions, measures, measurement_name, n_jobs)
        return
    experiment_root = pathlib.Path(experiment_root)
    derived_root = experiment_root / DERIVED_ROOT
    feature_names = ['timepoint']
    for measure in measures:
        if measure.feature_names is not None:
            feature_names += measure.feature_names
    data = []
    for position_name, (position_annotations, timepoint_annotations) in positions.items():
        position_root = experiment_root / position_name
        timepoint_data = []
        for timepoint, annotations in timepoint_annotations.items():
            timepoint_features = [timepoint]
            timepoint_data.append(timepoint_features)
            for measure in measures:
                features = measure.measure(position_root, derived_root, timepoint, annotations)
                if features is None:
                    assert measure.feature_names is None
                else:
                    assert len(features) == len(measure.feature_names)
                    timepoint_features.extend(features)
        data.append((position_name, timepoint_data))
    if len(feature_names) > 1: # more than just the timepoint column in the data
        data_root = derived_root / 'measurements' / measurement_name
        data_root.mkdir(parents=True, exist_ok=True)
        worms = worm_data.Worms()
        for position_name, timepoint_data in data:
            worms.append(worm_data.Worm(position_name, feature_names, timepoint_data))
        worms.write_timecourse_data(data_root, multi_worm_file=False)

def _multiprocess_measure(experiment_root, positions, measures, measurement_name, n_jobs):
    job_position_names = numpy.array_split(list(positions.keys()), n_jobs)
    job_positions = [{name:positions[name] for name in names} for names in job_position_names]
    job_args = [(experiment_root, job_pos, measures, measurement_name) for job_pos in job_positions]
    with multiprocessing.Pool(processes=n_jobs) as pool:
        pool.starmap(measure_worms, job_args)

def collate_data(experiment_root):
    """Gather all .tsv files produced by measurement runs into a single file.

    This function will concatenate all individual-worm .tsv files for all of the
    different measure_worms runs (which each output their .tsv files into a
    different subdirectory of '{experiment_root}/derived_data/measurements')
    into a single master-file of timecourse data:
        {experiment_root}/derived_data/measurements/{experiment_root.name} timecourse.tsv
    If possible, lifespans and other spans will be calculated for the worms,
    with the results stored in a master-file of summary data:
        {experiment_root}/derived_data/measurements/{experiment_root.name} summary.tsv

    The worms in these files will be renamed as:
        '{experiment_root.name} {position_name}'
    """
    experiment_root = pathlib.Path(experiment_root)
    experiment_name = experiment_root.name
    derived_root = experiment_root / DERIVED_ROOT
    measurement_root = derived_root / 'measurements'
    measurements = []
    for measurement_dir in measurement_root.iterdir():
        files = list(measurement_dir.glob('*.tsv'))
        if len(files) > 0:
            measurements.append(worm_data.read_worms(*files, name_prefix=experiment_name+' ', calculate_lifespan=False))
    worms = measurements[0]
    for other_measurement in measurements[1:]:
        worms.merge_in(other_measurement)
    for w in worms:
        try:
            w.calculate_ages_and_spans()
        except (NameError, ValueError):
            print(f'could not calculate lifespan for worm {w.name}')
    worms.write_timecourse_data(derived_root / f'{experiment_name} timecourse.tsv', multi_worm_file=True, error_on_missing=False)
    worms.write_summary_data(derived_root / f'{experiment_name} summary.tsv', error_on_missing=False)

class BasicMeasurements:
    """Provide data columns for the timepoint's UNIX timestamp and the annotated stage.

    Each position/timepoint to be measured must have a stage annotated and
    load_data.annotate_timestamps must have been run on the experiment root
    in order to generate the timestamp annotations needed. Otherwise an error
    will be raised.

    This data is required to calculate lifespans and other spans, and should be
    considered the minimum set of useful information about each worm.
    """
    feature_names = ['timestamp', 'stage']
    def measure(self, position_root, derived_root, timepoint, annotations):
        return annotations['timestamp'], annotations['stage']

class PoseMeasurements:
    """Provide data columns based on annotated worm pose information.

    Given the pose data, each worm's length, volume, surface_area, and maximum
    width, plus the area of the 2D projection of the worm into the image plane
    (i.e. the area of the worm region in the image).

    If no pose annotation is present, Nones are returned.

    Note: the correct microns_per_pixel conversion factor passed to the
    constructor of this class.
    """
    feature_names = ['length', 'volume', 'surface_area', 'projected_area', 'max_width']

    def __init__(self, microns_per_pixel):
        self.microns_per_pixel = microns_per_pixel

    def measure(self, position_root, derived_root, timepoint, annotations):
        center_tck, width_tck = annotations.get('pose', (None, None))
        if center_tck is None:
            return [numpy.nan] * len(self.feature_names)
        elif width_tck is None:
            length = spline_geometry.arc_length(center_tck) * self.microns_per_pixel
            return [length] + [numpy.nan] * (len(feature_names) - 1)
        else:
            projected_area = spline_geometry.area(center_tck, width_tck) * self.microns_per_pixel**2
            volume, surface_area = spline_geometry.volume_and_surface_area(center_tck, width_tck)
            volume *= self.microns_per_pixel**3
            surface_area *= self.microns_per_pixel**2
            length, max_width = spline_geometry.length_and_max_width(center_tck, width_tck)
            length *= self.microns_per_pixel
            max_width *= 2 * self.microns_per_pixel # the "width_tck" is really more like a radius,
            # storing the distance from the centerline to the edge. Double it to generate a real width.
            return [length, volume, surface_area, projected_area, max_width]

class FluorMeasurements:
    """Provide data columns based on a fluorescent images.

    This measurement applies the measure_fluor.subregion_measures function to
    a specific fluorescent image (i.e. gfp or autofluorescence) at the
    provided timepoint.

    If the image required can't be found or there is no pose data and no mask
    file, Nones are returned for the measurement data.
    However, pose data is preferred, and will be
    looked up from the annotation dictionary with the provided annotation name
    ('pose' by default).

    Note: this class must be instantiated to be used as a measurement. The
    constructor takes the following parameters:
        image_type: the name of the images to load, e.g. 'gfp' or 'autofluorescence'.
            Images files will be loaded as:
            {experiment_root}/{position_name}/{timepoint} {image_type}.png
        pose_annotation: name of the annotation that the pose for this image,
            'pose' by defauly.
        mask_name: name of the mask file to read if no pose is found; 'bf' by
            default. Mask files are expected to be organized as follows:
            {experiment_root}/derived_data/masks/{position_name}/{timepoint} {mask_name}.png
        write_masks: if True (default is False), write out a colorized
            representation of the expression, high_expression and over_99
            regions as:
            {experiment_root}/derived_data/fluor_region_masks/{position_name}/{timepoint} {image_type}.png
    """

    def __init__(self, image_type, pose_annotation='pose', mask_name='bf', write_masks=False):
        self.image_type = image_type
        self.pose_annotation = pose_annotation
        self.mask_name = mask_name
        self.write_masks = write_masks

    @property
    def feature_names(self):
        return [f'{self.image_type}_{name}' for name in measure_fluor.SUBREGION_FEATURES]

    def measure(self, position_root, derived_root, timepoint, annotations):
        image_file = position_root / f'{timepoint} {self.image_type}.png'
        if not image_file.exists():
            return [numpy.nan] * len(self.feature_names)

        image = freeimage.read(image_file)
        flatfield = freeimage.read(position_root.parent / 'calibrations' / f'{timepoint} fl_flatfield.tiff')
        center_tck, width_tck = annotations.get(self.pose_annotation, (None, None))
        if center_tck is None or width_tck is None:
            mask_file = derived_root / 'masks' / position_root.name / f'{timepoint} {mask_name}.png'
            if mask_file.exists():
                print(f'No pose data found for {position_root.name} at {timepoint}; falling back to mask file.')
            else:
                print(f'No mask file or pose data found for {position_root.name} at {timepoint}.')
                return [numpy.nan] * len(self.feature_names)
            mask = freeimage.read(mask_file)
        else:
            # NB: it's WAY faster to regenerate the mask from the splines than to read it in,
            # even if the file is cached in RAM. Strange but true.
            mask = worm_spline.lab_frame_mask(center_tck, width_tck, image.shape)

        mask = mask > 0
        image = image.astype(numpy.float32) * flatfield
        data, region_masks = measure_fluor.subregion_measures(image, mask)

        if self.write_masks:
            color_mask = measure_fluor.colorize_masks(mask, region_masks)
            out_dir = derived_root / 'fluor_region_masks' / position_root.name
            out_dir.mkdir(parents=True, exist_ok=True)
            freeimage.write(color, out_dir / f'{timepoint} {self.image_type}.png')
        return data

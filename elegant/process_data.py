# This code is licensed under the MIT License (see LICENSE file for details)

import pathlib
import multiprocessing
import json
import numpy
import datetime

from zplib.curve import spline_geometry
import freeimage

from . import worm_data
from . import load_data
from . import worm_spline
from . import measure_fluor
from . import process_images
from . import segment_images

DERIVED_ROOT = 'derived_data'

def update_annotations(experiment_root):
    """Run prior to manually annotating an experiment directory, in order to
    update the annotations dictionaries with all relevant data that can be
    automatically extracted from the experiment acquisition metadata files.
    """
    annotate(experiment_root, [annotate_timestamps, annotate_z], [annotate_stage_pos])

def annotate(experiment_root, annotators=[], position_annotators=[]):
    """Apply one or more measurement functions to produce annotations for manual review.

    Parameters:
        experiment_root: the path to an experimental directory.
        annotators: list of annotation functions to call for each timepoint. Each
            will be called with parameters:
                experiment_root: as above
                position: name of position
                timepoint: name of timepoint
                metadata: metadata dict for the timepoint from 'position_metadata.json'
                annotations: annotation dictionary for the timepoint
            The function should add anything relevant to the annotations dictionary;
            and the function may use the contents of that dictionary to decide
            whether to overwrite existing annotations with new ones or not.
            The return value of the measure function is ignored.
        position_annotators: list of annotation functions to call for each position
            with parameters:
                experiment_root: as above
                position: name of position
                metadata: metadata dict from 'experiment_metadata.json'
                annotations: annotation dictionary for the position

    """
    experiment_root = pathlib.Path(experiment_root)
    positions = load_data.read_annotations(experiment_root)
    with (experiment_root / 'experiment_metadata.json').open('r') as f:
        experiment_metadata = json.load(f)
    for metadata_path in sorted(experiment_root.glob('*/position_metadata.json')):
        with metadata_path.open('r') as f:
            position_metadata = json.load(f)
        position = metadata_path.parent.name
        position_annotations, timepoint_annotations = positions.setdefault(position, ({}, {}))
        for annotator in position_annotators:
            annotator(experiment_root, position, experiment_metadata, position_annotations)
        for metadata in position_metadata:
            timepoint = metadata['timepoint']
            annotations = timepoint_annotations.setdefault(timepoint, {})
            for annotator in annotators:
                annotator(experiment_root, position, timepoint, metadata, annotations)
    load_data.write_annotations(experiment_root, positions)

def annotate_timestamps(experiment_root, position, timepoint, metadata, annotations):
    annotations['timestamp'] = metadata['timestamp']

def annotate_z(experiment_root, position, timepoint, metadata, annotations):
    annotations['stage_z'] = metadata.get('fine_z', numpy.nan)

def annotate_stage_pos(experiment_root, position, metadata, annotations):
    x, y, z = metadata['positions'][position]
    annotations['stage_x'] = x
    annotations['stage_y'] = y
    annotations['starting_stage_z'] = z

def annotate_lawn(experiment_root, position, metadata, annotations, num_images_for_lawn=3):
    '''Position annotator used to find the lawn and associated metadata about it'''

    print(f'Working on position {position}')
    position_root = experiment_root / position
    lawn_mask_root = experiment_root / 'derived_data' / 'lawn_masks'
    lawn_mask_root.mkdir(parents=True, exist_ok=True)

    microns_per_pixel = process_images.microns_per_pixel(metadata['objective'],metadata['optocoupler'])

    position_images = load_data.scan_experiment_dir(experiment_root)[position]
    first_imagepaths = sorted(position_root.glob('* bf.png'))[:num_images_for_lawn]

    first_images = [freeimage.read(str(image_path)) for image_path in first_imagepaths]
    first_images = [process_images.pin_image_mode(image, optocoupler=metadata['optocoupler'])
        for image in first_images]

    individual_lawns = [segment_images.find_lawn(image, metadata['optocoupler']) for image in first_images]
    lawn_mask = numpy.bitwise_or.reduce(individual_lawns, axis=0)

    freeimage.write(lawn_mask.astype('uint8')*255, str(lawn_mask_root / f'{position}.png'))
    annotations['lawn_area'] = lawn_mask.sum() * microns_per_pixel**2

def set_hatch_time(experiment_root, year, month, day, hour):
    """Manually set a hatch-time for all worms in an experiment.

    This is useful in cases where the full life-cycle from hatch onward was not
    imaged or is not being annotated. Precisions of minutes and seconds are not
    used because there can't be that much precision across a whole experiment
    directory.

    Parameters:
        experiment_root: the path to an experimental directory.
        year, month, day, hour: integer values for the time of hatching
    """
    hatch_timestamp = datetime.datetime(year, month, day, hour).timestamp()
    positions = load_data.read_annotations(experiment_root)
    for position_name, (position_annotations, timepoint_annotations) in positions.items():
        position_annotations['hatch_timestamp'] = hatch_timestamp
    load_data.write_annotations(experiment_root, positions)

def annotate_ages_from_timestamps_and_stages(experiment_root, stage_annotation='stage', unhatched_stage='egg'):
    """Annotate the age of each worm from previously-annotated life stages.

    Parameters:
        experiment_root: the path to an experimental directory.
        stage_annotation: name of the annotation containing the life stages
        unhatched_stage: name of the life stage where worms are unhatched. Used
            when hatch_timestamp is None to determine the hatch time on a per-
            worm basis.
    """
    positions = load_data.read_annotations(experiment_root)
    for position_name, (position_annotations, timepoint_annotations) in positions.items():
        _update_ages(timepoint_annotations, position_annotations, stage_annotation, unhatched_stage, force=True)
    load_data.write_annotations(experiment_root, positions)

def _update_ages(timepoint_annotations, position_annotations, stage_annotation='stage', unhatched_stage='egg', force=False):
    hatch_timestamp = position_annotations.get('hatch_timestamp')
    if hatch_timestamp is None:
        for annotations in timepoint_annotations:
            if 'timestamp' not in annotations:
                return
            timestamp = annotations['timestamp']
            page_stage = annotations.get(stage_annotation, unhatched_stage)
            if page_stage != unhatched_stage:
                hatch_timestamp = timestamp
                break
        if hatch_timestamp is None:
            if 'hatch_timestamp' in position_annotations:
                del position_annotations['hatch_timestamp']
            return
        else:
            position_annotations['hatch_timestamp'] = hatch_timestamp
    for annotations in timepoint_annotations:
        if 'age' in annotations and not force:
            continue
        timestamp = annotations.get('timestamp')
        if timestamp is not None:
            annotations['age'] = (timestamp - hatch_timestamp) / 3600 # age in hours

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
                data = measure(position_root, timepoint, annotations, before, after)
            where the parameters are as follows:
                position_root: path to directory of data files for the position
                    (i.e. experiment_root / position_name)
                timepoint: name of the timepoint to be measured
                annotations: annotation dict for this position at the given
                    timepoint
                before, after: annotation dictionaries (or None) for the
                    previous and next timepoints, if any. This enables
                    measurements based on the changes between annotations.
            and the return value must be a list of measurements of the same
            length as the feature_names attribute, or nan if that measurement
            cannot be made on the given worm and timepoint (e.g. missing data).
            If a measurement produces no data (e.g. it only produces file output,
            such as segmentation masks), then feature_names may be None, and
            the return value of measure must also be None.
            If a measurement wishes to write out data file(s), they should be
            stored as:
            '{experiment_root}/derived_data/{name}/{position_name}/{timepoint}.ext'
            where 'name' is a unique name for this type of data, and
            'position_name' can be obtained as position_root.name. If multiple
            related datafiles need to be saved, they cound be saved as:
            '{timepoint} {suffix}.ext' for whatever set of suffixes are needed.
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
    feature_names = ['timepoint']
    for measure in measures:
        if measure.feature_names is not None:
            feature_names += measure.feature_names
    data = []
    for position_name, (position_annotations, timepoint_annotations) in positions.items():
        position_root = experiment_root / position_name
        timepoint_data = []
        timepoints = sorted(timepoint_annotations.keys())
        befores = [None] + [timepoint_annotations[t] for t in timepoints[:-1]]
        afters = [timepoint_annotations[t] for t in timepoints[1:]] + [None]
        for timepoint, before, after in zip(timepoints, befores, afters):
            annotations = timepoint_annotations[timepoint]
            timepoint_features = [timepoint]
            timepoint_data.append(timepoint_features)
            for measure in measures:
                features = measure.measure(position_root, timepoint, annotations, before, after)
                if features is None:
                    assert measure.feature_names is None
                else:
                    assert len(features) == len(measure.feature_names)
                    timepoint_features.extend(features)
        data.append((position_name, timepoint_data))
    if len(feature_names) > 1: # more than just the timepoint column in the data
        data_root = experiment_root / DERIVED_ROOT / 'measurements' / measurement_name
        data_root.mkdir(parents=True, exist_ok=True)
        worms = worm_data.Worms()
        for position_name, timepoint_data in data:
            worms.append(worm_data.Worm(position_name, feature_names, timepoint_data))
        worms.write_timecourse_data(data_root, multi_worm_file=False)

def _multiprocess_measure(experiment_root, positions, measures, measurement_name, n_jobs):
    job_position_names = numpy.array_split(list(positions.keys()), n_jobs)
    job_positions = [{name: positions[name] for name in names} for names in job_position_names]
    job_args = [(experiment_root, job_pos, measures, measurement_name) for job_pos in job_positions]
    with multiprocessing.Pool(processes=n_jobs) as pool:
        pool.starmap(measure_worms, job_args)

def collate_data(experiment_root, position_features=('stage_x', 'stage_y', 'starting_stage_z')):
    """Gather all .tsv files produced by measurement runs into a single file.

    This function will concatenate all individual-worm .tsv files for all of the
    different measure_worms runs (which each output their .tsv files into a
    different subdirectory of '{experiment_root}/derived_data/measurements')
    into a single master-file of timecourse data:
        {experiment_root}/derived_data/measurements/{experiment_root.name} timecourse.tsv
    If possible, lifespans and other spans will be calculated for the worms,
    with the results stored in a master-file of summary data:
        {experiment_root}/derived_data/measurements/{experiment_root.name} summary.tsv
    Any features named in the position_features parameter will be transfered
    from the annotations for that position to the worm summary data as well.

    The worms in these files will be renamed as:
        '{experiment_root.name} {position_name}'
    """
    experiment_root = pathlib.Path(experiment_root)
    positions = load_data.read_annotations(experiment_root)
    experiment_name = experiment_root.name
    derived_root = experiment_root / DERIVED_ROOT
    measurement_root = derived_root / 'measurements'
    measurements = []
    name_prefix = experiment_name + ' '
    for measurement_dir in measurement_root.iterdir():
        files = list(measurement_dir.glob('*.tsv'))
        if len(files) > 0:
            measurements.append(worm_data.read_worms(*files, name_prefix=name_prefix, calculate_lifespan=False))
    worms = measurements[0]
    for other_measurement in measurements[1:]:
        worms.merge_in(other_measurement)
    for w in worms:
        try:
            w.calculate_ages_and_spans()
        except (NameError, ValueError):
            print(f'could not calculate lifespan for worm {w.name}')
        position_annotations, timepoint_annotations = positions.get(w.name[len(name_prefix):], ({}, {}))
        for feature in position_features:
            if feature in position_annotations:
                setattr(w, feature, position_annotations[feature])

    worms.write_timecourse_data(derived_root / f'{experiment_name} timecourse.tsv', multi_worm_file=True, error_on_missing=False)
    worms.write_summary_data(derived_root / f'{experiment_name} summary.tsv', error_on_missing=False)

class BasicMeasurements:
    """Provide basic standard data columns for timestamp, life stage, and z position.

    Each position/timepoint to be measured must have a stage annotated and
    update_annotations() must have been run on the experiment root
    in order to generate the timestamp and z-value annotations needed.
    Otherwise an error will be raised.

    This data is required to calculate lifespans and other spans, and should be
    considered the basic set of useful information about each worm.
    """
    feature_names = ['timestamp', 'stage', 'stage_z']
    def measure(self, position_root, timepoint, annotations, before, after):
        return annotations['timestamp'], annotations['stage'], annotations['stage_z']

class PoseMeasurements:
    """Provide data columns based on annotated worm pose information.

    Given the pose data, each worm's length, volume, surface_area, and maximum
    width, plus the area of the 2D projection of the worm into the image plane
    (i.e. the area of the worm region in the image).

    If no pose annotation is present, Nones are returned.

    Note: the correct microns_per_pixel conversion factor MUST passed to the
    constructor of this class.
    """
    feature_names = ['length', 'volume', 'surface_area', 'projected_area', 'max_width', 'centroid_dist', 'rms_dist']

    def __init__(self, microns_per_pixel, pose_annotation='pose'):
        self.microns_per_pixel = microns_per_pixel
        self.pose_annotation = pose_annotation

    def measure(self, position_root, timepoint, annotations, before, after):
        center_tck, width_tck = annotations.get(self.pose_annotation, (None, None))
        measures = {}
        if center_tck is not None:
            if width_tck is None:
                measures['length'] = spline_geometry.arc_length(center_tck) * self.microns_per_pixel
            else:
                measures['projected_area'] = spline_geometry.area(center_tck, width_tck) * self.microns_per_pixel**2
                volume, surface_area = spline_geometry.volume_and_surface_area(center_tck, width_tck)
                measures['volume'] = volume * self.microns_per_pixel**3
                measures['surface_area'] = surface_area * self.microns_per_pixel**2
                length, max_width = spline_geometry.length_and_max_width(center_tck, width_tck)
                measures['length'] = length * self.microns_per_pixel
                # the "width_tck" is really more like a radius,
                # storing the distance from the centerline to the edge.
                # Double it to generate a real width.
                measures['max_width'] = max_width * 2 * self.microns_per_pixel
            centroid_distances = []
            rmsds = []
            for adjacent in (before, after):
                if adjacent is not None:
                    adj_center_tck, adj_width_tck = adjacent.get(self.pose_annotation, (None, None))
                    if adj_center_tck is not None:
                        centroid_distances.append(spline_geometry.centroid_distance(center_tck, adj_center_tck, num_points=300))
                        rmsds.append(spline_geometry.rmsd(center_tck, adj_center_tck, num_points=300))
            if len(rmsds) > 0:
                measures['centroid_dist'] = numpy.mean(centroid_distances) * self.microns_per_pixel
                measures['rms_dist'] = numpy.mean(rmsds) * self.microns_per_pixel
        return [measures.get(feature, numpy.nan) for feature in self.feature_names]

class _FluorMeasureBase:
    def __init__(self, image_type, write_masks):
        self.image_type = image_type
        self.write_masks = write_masks

    @property
    def feature_names(self):
        return [f'{self.image_type}_{name}' for name in measure_fluor.SUBREGION_FEATURES]

    def measure(self, position_root, timepoint, annotations, before, after):
        derived_root = position_root.parent / DERIVED_ROOT
        image_file = position_root / f'{timepoint} {self.image_type}.png'
        if not image_file.exists():
            return [numpy.nan] * len(self.feature_names)

        image = freeimage.read(image_file)
        flatfield = freeimage.read(position_root.parent / 'calibrations' / f'{timepoint} fl_flatfield.tiff')
        image = image.astype(numpy.float32) * flatfield
        mask = self.get_mask(position_root, derived_root, timepoint, annotations, image.shape)
        if mask is None:
            return [numpy.nan] * len(self.feature_names)
        if mask.sum() == 0:
            print(f'No worm region defined for {position_root.name} at {timepoint}')

        data, region_masks = measure_fluor.subregion_measures(image, mask)

        if self.write_masks:
            color_mask = measure_fluor.colorize_masks(mask, region_masks)
            out_dir = derived_root / 'fluor_region_masks' / position_root.name
            out_dir.mkdir(parents=True, exist_ok=True)
            freeimage.write(color_mask, out_dir / f'{timepoint} {self.image_type}.png')
        return data

    def get_mask(self, position_root, derived_root, timepoint, annotations, shape):
        raise NotImplementedError


class FluorMeasurements(_FluorMeasureBase):
    """Provide data columns based on a fluorescent images and pose data.

    This measurement applies the measure_fluor.subregion_measures function to
    a specific fluorescent image (i.e. gfp or autofluorescence) at the
    provided timepoint.

    If the image required can't be found or there is no pose data, Nones are
    returned for the measurement data.

    Note: this class must be instantiated to be used as a measurement. The
    constructor takes the following parameters:
        image_type: the name of the images to load, e.g. 'gfp' or 'autofluorescence'.
            Images files will be loaded as:
            {experiment_root}/{position_name}/{timepoint} {image_type}.png
        pose_annotation: name of the annotation that the pose for this image,
            'pose' by default.
        mask_name: name of the mask file to read if no pose is found; 'bf' by
            default. Mask files are expected to be organized as follows:
            {experiment_root}/derived_data/mask/{position_name}/{timepoint} {mask_name}.png
        write_masks: if True (default is False), write out a colorized
            representation of the expression, high_expression and over_99
            regions as:
            {experiment_root}/derived_data/fluor_region_masks/{position_name}/{timepoint} {image_type}.png
    """

    def __init__(self, image_type, pose_annotation='pose', write_masks=False):
        self.pose_annotation = pose_annotation
        super().__init__(image_type, write_masks)

    def get_mask(self, position_root, derived_root, timepoint, annotations, shape):
        center_tck, width_tck = annotations.get(self.pose_annotation, (None, None))
        if center_tck is None or width_tck is None:
            print(f'No pose data found for {position_root.name} at {timepoint}.')
            return None
        else:
            # NB: it's WAY faster to regenerate a mask from the splines than to read it in,
            # even if the file is in the disk cache. Strange but true.
            return worm_spline.lab_frame_mask(center_tck, width_tck, shape)


class MaskFluorMeasurements(_FluorMeasureBase):
    """Provide data columns based on a fluorescent images and mask images.

    This measurement applies the measure_fluor.subregion_measures function to
    a specific fluorescent image (i.e. gfp or autofluorescence) at the
    provided timepoint.

    If the image required or corresponding mask can't be found, Nones are
    returned for the measurement data.

    Note: this class must be instantiated to be used as a measurement. The
    constructor takes the following parameters:
        image_type: the name of the images to load, e.g. 'gfp' or 'autofluorescence'.
            Images files will be loaded as:
            {experiment_root}/{position_name}/{timepoint} {image_type}.png
        mask_name: name of the mask file to read if no pose is found; 'bf' by
            default. Mask files are expected to be organized as follows:
            {experiment_root}/derived_data/mask/{position_name}/{timepoint} {mask_name}.png
        write_masks: if True (default is False), write out a colorized
            representation of the expression, high_expression and over_99
            regions as:
            {experiment_root}/derived_data/fluor_region_masks/{position_name}/{timepoint} {image_type}.png
    """

    def __init__(self, image_type, mask_name='bf', write_masks=False):
        self.mask_name = mask_name
        super().__init__(image_type, write_masks)

    def get_mask(self, position_root, derived_root, timepoint, annotations, shape):
        mask_file = derived_root / 'mask' / position_root.name / f'{timepoint} {self.mask_name}.png'
        if not mask_file.exists():
            print(f'No mask file found for {position_root.name} at {timepoint}.')
            return None
        else:
            mask = freeimage.read(mask_file)
            assert mask.shape == shape
            return mask

class LawnMeasurements:
    feature_names = ['summed_lawn_intensity', 'median_lawn_intensity', 'background_intensity']
    def __init__(self):
        self._optocouplers = {}

    def measure(self, position_root, timepoint, annotations, before, after):
        measures = {}
        experiment_root, position_name = position_root.parent, position_root.name
        derived_root = experiment_root / DERIVED_ROOT

        if experiment_root not in self._optocouplers:
            self._optocouplers[experiment_root] = load_data.read_metadata(experiment_root)['optocoupler']
        optocoupler = self._optocouplers[experiment_root]

        timepoint_imagepath = position_root / (timepoint + ' bf.png')
        timepoint_image = freeimage.read(timepoint_imagepath)
        rescaled_image = process_images.pin_image_mode(timepoint_image, optocoupler=optocoupler)

        lawn_mask = freeimage.read(derived_root / 'lawn_masks' / f'{position_name}.png').astype('bool')
        vignette_mask = process_images.vignette_mask(optocoupler, timepoint_image.shape)

        # Remove the animal from the lawn if possible.
        center_tck, width_tck = annotations.get('pose', (None, None))
        if center_tck is not None:
            lawn_mask &= worm_spline.lab_frame_mask(center_tck, width_tck, timepoint_image.shape).astype('bool')

        measures['summed_lawn_intensity'] = numpy.sum(rescaled_image[lawn_mask])
        measures['median_lawn_intensity'] = numpy.median(rescaled_image[lawn_mask])
        measures['background_intensity'] = numpy.median(rescaled_image[~lawn_mask & vignette_mask])

        return [measures[feature_name] for feature_name in self.feature_names]
=======
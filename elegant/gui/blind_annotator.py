# This code is licensed under the MIT License (see LICENSE file for details)

import random

from . import experiment_annotator

class BlindedExperimentAnnotator(experiment_annotator.ExperimentAnnotator):
    def __init__(self, ris_widget, position_names, position_data, annotation_fields, start_position_index=0, readonly=False):
        """ExperimentAnnotator subclass that disguises worm and image names in the
        GUI to allow for blinded annotation.

        Parameters:
            ris_widget: a RisWidget object
            position_names, position_data: as returned by shuffle_positions(),
                a list of the names of each position and a separate list of
                timepoint dictionaries for each position. Because multiple
                experiments should be shuffled together (which might have duplicate
                position names), two lists are required here, instead of an
                ordered dict as in the case where only one experiment is examined.
            annotation_fields: list of annotation fields to pass to
                ris_widget.add_annotator()
            start_position_index: index of starting position to load first.
                To change to an arbitrary position mid-stream, call the
                load_position_index() method.
        """

        experiment_name = f'Blinded ({len(positions_dicts)} experiments)'
        positions = (position_names, position_data)
        super().__init__(ris_widget, experiment_name, positions, annotation_fields, start_position=None, readonly=False)

    def _init_positions(self, positions):
        self.position_names, self.positions = positions

    def load_timepoints(self):
        timepoint_paths = self.timepoints.values()
        page_names = list(map(str, range(len(self.timepoints))))
        image_names = [[image_path.name for image_path in image_paths] for image_paths in timepoint_paths]
        self.pos_label.setText(f'{self.position_i+1}/{len(self.positions)}')
        return self.ris_widget.add_image_files_to_flipbook(timepoint_paths, page_names, image_names)


def shuffle_positions(*position_dicts):
    """Shuffle multiple positions dictionaries for blinded annotation.

    parameters:
        position_dicts: one or more ordered dictionaries of position names
            mapping to timepoint data, as returned by
            load_data.scan_experiment_dir().

    returns: position_names, position_data
        position_names: list of position names
        position_data: list of timepoint dicts, where each timepoint dict is an
            OrderedDict mapping timepoint names to image paths.
    """
    positions = []
    for position_dict in position_dicts:
        for position_name, timepoint_dicts in position_dict.items():
            positions.append((position_name, timepoint_dicts))
    random.shuffle(positions)
    position_names, position_data = zip(*positions)
    return position_names, position_data
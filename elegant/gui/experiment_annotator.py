# This code is licensed under the MIT License (see LICENSE file for details)

import pathlib
import pickle

from PyQt5 import Qt

from ris_widget import shared_resources
from ris_widget.overlay import base

from . import util

class ExperimentAnnotator:
    def __init__(self, ris_widget, experiment_root, worms, annotation_fields):
        """Set up a GUI for annotating an entire experiment.

        Annotations for each experiment position (i.e. each worm) are loaded
        and saved from an 'annotations' directory in each experiment directory.
        This directory will be created as needed. Upon switching between
        positions, the annotations for the previous position will be saved out
        as a pickle file named based on the position name. The pickle file will
        contain a dictionary mapping timepoint names to the annotation
        dictionaries produced by  ris_widget annotators and saved in the
        'annotations' attribute of each flipbook page.

        Keyboard shortcuts:
            [ and ] switch between worms
            control-s (command-s on macs) saves annotations


        Parameters:
            ris_widget: a RisWidget object
            experiment_root: path to an experiment directory
            positions: ordered dictionary of position names (i.e. worm names)
                mapping to ordered dictionaries of timepoint names, each of
                which maps to a list of image paths to load for each timepoint.
                Note: util.scan_experiment_dir() provides exactly these data.
            annotation_fields: list of annotation fields to pass to
                ris_widget.add_annotator()

        """
        self.ris_widget = ris_widget
        ris_widget.add_annotator(annotation_fields)
        self.experiment_root = pathlib.Path(experiment_root)
        self.annotations_dir = self.experiment_root / 'annotations'
        self.position_names = list(positions.keys())
        self.positions = list(positions.values())
        self.position_i = None
        self.flipbook = ris_widget.flipbook
        self.listener = NavListener(ris_widget)

        self.widget = Qt.QGroupBox(self.experiment_root.name)
        layout = Qt.QVBoxLayout()
        layout.setSpacing(0)
        self.widget.setLayout(layout)
        worm_info = Qt.QHBoxLayout()
        worm_info.setSpacing(11)
        self.pos_label = Qt.QLabel()
        worm_info.addWidget(self.pos_label)
        self._add_button(worm_info, 'Save Annotations', self.save_annotations)
        layout.add_layout(worm_info)
        nav_buttons = Qt.QHBoxLayout()
        nav_buttons.setSpacing(11)
        self._add_button(nav_buttons, '\N{LEFTWARDS ARROW TO BAR} Worm', self.prev_worm)
        self._add_button(nav_buttons, '\N{UPWARDS ARROW} Time', self.prev_timepoint)
        self._add_button(nav_buttons, '\N{DOWNWARDS ARROW} Time', self.next_timepoint)
        self._add_button(nav_buttons, 'Worm \N{RIGHTWARDS ARROW TO BAR}', self.next_worm)
        layout.add_layout(nav_buttons)
        ris_widget.annotator.widget.layout().addWidget(self.widget)

        Qt.QShortcut(Qt.Qt.Key_BracketLeft, ris_widget.annotator.widget, self.prev_worm)
        Qt.QShortcut(Qt.Qt.Key_BracketRight, ris_widget.annotator.widget, self.next_worm)
        Qt.QShortcut(Qt.QKeySequence.Save, ris_widget.annotator.widget, self.save_annotations)

        self.load_position(0)

    def _add_button(self, layout, title, callback):
        button = Qt.QPushButton(title)
        button.clicked.connect(callback)
        layout.addWidget(button)

    def load_position(self, i):
        if self.position_i is not None:
            self.save_annotations()
        self.position_i = i
        self.ris_widget.flipbook_pages.clear()
        if self.position_i == i:
            return []
        if i is not None:
            self.timepoints = self.positions[i]
            self.timepoint_indices = {name: i for i, name in enumerate(self.timepoints.keys())}
            self.pos_label.setText('Worm {}'.format(self.position_name))
            self.load_annotations()
            return util.add_position_to_flipbook(self.ris_widget, self.timepoints)
        else:
            self.pos_label.setText('-')
            return []

    @property
    def position_name(self):
        return self.position_names[self.position_i])

    @property
    def annotation_file(self):
        return self.annotations_dir / (self.position_name + '.pickle')

    def load_annotations(self):
        try:
            with self.annotation_file.open('rb') as af:
                all_annotations = pickle.load(af)
        except FileNotFoundError:
            return
        assert set(self.timepoints.keys()).issuperset(annotations.keys())
        for timepoint_name, annotations in all_annotations.items():
            page_i = self.timepoint_indices[timepoint_name]
            self.ris_widget.flipbook_pages[page_i].annotations = annotations
        self.ris_widget.annotator.update_fields()

    def save_annotations(self):
        assert len(self.timepoints) == len(self.ris_widget.flipbook_pages)
        all_annotations = {}
        for timepoint_name, page in zip(self.timepoints.keys(), self.ris_widget.flipbook_pages):
            if hasattr(page, 'annotations'):
                all_annotations[timepoint_name] = page.annotations
        self.annotations_dir.mkdir(exist_ok=True)
        with self.annotation_file.open('wb') as af:
            pickle.dump(all_annotations, af)

    def prev_timepoint(self):
        self.flipbook.focus_prev_page()

    def next_timepoint(self):
        self.flipbook.focus_next_page()

    def prev_worm(self):
        return self.load_position(max(0, self.position_i - 1))

    def next_worm(self):
        return self.load_position(min(len(self.positions), self.position_i + 1))


class NavListener(base.SceneListener):
    def __init__(self, ris_widget):
        self.flipbook = ris_widget.flipbook
        super().__init__(ris_widget)
    QGRAPHICSITEM_TYPE = shared_resources.generate_unique_qgraphicsitem_type()

    def sceneEventFilter(self, watched, event):
        if event.type() == Qt.QEvent.GraphicsSceneMousePress and event.modifiers() & Qt.Qt.ControlModifier:
            self._mouse_y = event.pos().y()
            return True
        if event.type() == Qt.QEvent.GraphicsSceneMouseMove and event.modifiers() & Qt.Qt.ControlModifier:
            delta = event.pos().y() - self._mouse_y
            if abs(delta) > 20:
                if delta > 0:
                    self.flipbook.focus_next_page()
                else:
                    self.flipbook.focus_prev_page()
                return True
        return False

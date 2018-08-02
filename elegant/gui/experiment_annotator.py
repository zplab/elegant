# This code is licensed under the MIT License (see LICENSE file for details)

import atexit
import pathlib
import pickle

from PyQt5 import Qt

from ris_widget import shared_resources
from ris_widget.overlay import base

from .. import load_data

class ExperimentAnnotator:
    def __init__(self, ris_widget, experiment_name, positions, annotation_fields, start_position=None, readonly=False):
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
            experiment_name: name of experiment to display. If a filesystem
                path, only the last component is shown.
            positions: ordered dictionary of position names (i.e. worm names)
                mapping to ordered dictionaries of timepoint names, each of
                which maps to a list of image paths to load for each timepoint.
                Note: load_data.scan_experiment_dir() provides exactly this.
            annotation_fields: list of annotation fields to pass to
                ris_widget.add_annotator()
            start_position: name of starting position to load first (e.g. "009").
                if none specified, load the first position. To change to an
                arbitrary position mid-stream, call the load_position() method.

        """
        self.ris_widget = ris_widget
        self.readonly = readonly
        ris_widget.add_annotator(annotation_fields)
        self.annotation_fields = annotation_fields
        self._init_positions(positions)

        # skip positions with no timepoints listed (don't do this in _init_positions
        # because a subclass might override it and we don't want to duplicate logic)
        position_names, positions = [], []
        for name, timepoints in zip(self.position_names, self.positions):
            if len(timepoints) > 0:
                position_names.append(name)
                positions.append(timepoints)
        self.position_names = position_names
        self.positions = positions

        self.position_names_to_indices = {name: i for i, name in enumerate(self.position_names)}
        self.position_i = None
        self.flipbook = ris_widget.flipbook

        self.main_zoom = ZoomListener(ris_widget)
        if hasattr(ris_widget, 'alt_view'):
            self.alt_zoom = ZoomListener(ris_widget.alt_view)

        widget = Qt.QGroupBox(pathlib.Path(experiment_name).name)
        layout = Qt.QVBoxLayout(widget)
        layout.setSpacing(0)
        worm_info = Qt.QHBoxLayout()
        worm_info.setSpacing(11)
        self.pos_editor = Qt.QLineEdit()
        self.pos_editor.editingFinished.connect(self._on_pos_editing_finished)
        maxlen = max(map(len, self.position_names))
        self.pos_editor.setMaxLength(maxlen)
        self.pos_editor.setAlignment(Qt.Qt.AlignCenter)
        w = self.pos_editor.fontMetrics().boundingRect('0'*maxlen).width()
        self.pos_editor.setFixedWidth(w + 14)
        worm_info.addWidget(self.pos_editor)
        self.pos_label = Qt.QLabel()
        worm_info.addWidget(self.pos_label, stretch=1)
        save = self._add_button(worm_info, 'Save', self.save_annotations)
        if readonly:
            save.setEnabled(False)
        self.exclude = Qt.QCheckBox(text='Exclude')
        worm_info.addWidget(self.exclude)
        layout.addLayout(worm_info)
        nav_buttons = Qt.QHBoxLayout()
        nav_buttons.setSpacing(11)
        self._prev_button = self._add_button(nav_buttons, '\N{LEFTWARDS ARROW TO BAR}', self.prev_position)
        self._add_button(nav_buttons, '\N{UPWARDS ARROW}', self.prev_timepoint)
        self._add_button(nav_buttons, '\N{DOWNWARDS ARROW}', self.next_timepoint)
        self._next_button = self._add_button(nav_buttons, '\N{RIGHTWARDS ARROW TO BAR}', self.next_position)
        layout.addLayout(nav_buttons)
        self.notes = Qt.QPlainTextEdit()
        row_height = self.notes.fontMetrics().lineSpacing()
        self.notes.setFixedHeight(3 * row_height)
        self.notes.setPlaceholderText('notes')
        layout.addWidget(self.notes)
        ris_widget.annotator.layout().insertRow(0, widget)

        Qt.QShortcut(Qt.Qt.Key_BracketLeft, ris_widget.annotator, self.prev_position, context=Qt.Qt.ApplicationShortcut)
        Qt.QShortcut(Qt.Qt.Key_BracketRight, ris_widget.annotator, self.next_position, context=Qt.Qt.ApplicationShortcut)
        Qt.QShortcut(Qt.QKeySequence.Save, ris_widget.annotator, self.save_annotations, context=Qt.Qt.ApplicationShortcut)

        if start_position is None:
            self.load_position_index(0)
        else:
            self.load_position(start_position)
        atexit.register(self.save_annotations)

    def _init_positions(self, positions):
        # this function allows subclasses to re-interpret the positions parameter
        self.position_names = list(positions.keys())
        self.positions = list(positions.values())

    def _add_button(self, layout, title, callback):
        button = Qt.QPushButton(title)
        button.clicked.connect(callback)
        layout.addWidget(button)
        return button

    def _on_pos_editing_finished(self):
        name = self.pos_editor.text()
        if name not in self.position_names_to_indices:
            self.pos_editor.setText(self.position_name)
            Qt.QMessageBox.warning(self.ris_widget.qt_object, 'Unknown Position', f'Position "{name}" is not defined.')
        elif name != self.position_name:
            self.load_position(name)

    def load_position(self, name):
        self.load_position_index(self.position_names_to_indices[name])

    def load_position_index(self, i):
        num_positions = len(self.position_names)
        if i is not None and i < 0:
            i += num_positions
        if not (i is None or 0 <= i < num_positions):
            raise ValueError('Invalid position index')
        if self.position_i == i:
            return []
        if self.position_i is not None:
            self.save_annotations()
        self.position_i = i
        self.ris_widget.flipbook_pages.clear()
        self.position_annotations = {}
        if i is not None:
            self._prev_button.setEnabled(i != 0)
            self._next_button.setEnabled(i != num_positions - 1)
            self.timepoints = self.positions[i]
            self.timepoint_indices = {name: i for i, name in enumerate(self.timepoints.keys())}
            self.pos_editor.setText(self.position_name)
            self.pos_label.setText(f'({i+1}/{len(self.positions)})')
            timepoint_names = self.timepoints.keys()
            futures = self.load_timepoints()
            for timepoint_name, page in zip(timepoint_names, self.ris_widget.flipbook_pages):
                page._timepoint_name = timepoint_name
            self.load_annotations()
            for field in self.annotation_fields:
                field.position_annotations = self.position_annotations
            if '__last_timepoint_annotated__' in self.position_annotations:
                timepoint_name = self.position_annotations['__last_timepoint_annotated__']
                i = self.timepoint_indices.get(timepoint_name, 0)
            elif '__last_page_annotated__' in self.position_annotations:
                # older, deprecated style. TODO: delete this elif stanza after everyone has
                # updated their annotation dicts...
                i = self.position_annotations['__last_page_annotated__']
                if i > len(self.ris_widget.flipbook_pages):
                    i = 0
            else:
                i = 0
            self.ris_widget.flipbook.current_page_idx = i
            self.ris_widget.flipbook.pages_view.setFocus()
            return futures
        else:
            self.pos_editor.setText('')
            self.pos_label.setText('')
            return []

    def load_timepoints(self):
        return load_data.add_position_to_flipbook(self.ris_widget, self.timepoints)

    @property
    def position_name(self):
        return self.position_names[self.position_i]

    @property
    def experiment_root(self):
        first_image = next(iter(self.timepoints.values()))[0]
        return first_image.parent.parent

    @property
    def annotation_file(self):
        return self.experiment_root / 'annotations' / (self.position_name + '.pickle')

    def load_annotations(self):
        try:
            self.position_annotations, self.timepoint_annotations = load_data.read_annotation_file(self.annotation_file)
        except FileNotFoundError:
            self.position_annotations = {}
            self.timepoint_annotations = {}
        self.exclude.setChecked(self.position_annotations.get('exclude', False)) # Set to include by default
        self.notes.setPlainText(self.position_annotations.get('notes', ''))
        unknown_timepoints = []
        for timepoint_name, annotations in self.timepoint_annotations.items():
            if timepoint_name not in self.timepoint_indices:
                unknown_timepoints.append(timepoint_name)
            else:
                page_i = self.timepoint_indices[timepoint_name]
                self.ris_widget.flipbook_pages[page_i].annotations = dict(annotations) # make a copy
        if len(unknown_timepoints) > 0:
            print('Annotations were found for some timepoints that have not been loaded in the annotator.')
            print('These annotations will be preserved.')
        self.ris_widget.annotator.update_fields()

    def save_annotations(self):
        if self.readonly == True:
            return
        for page in self.ris_widget.flipbook_pages:
            self.timepoint_annotations[page._timepoint_name] = getattr(page, 'annotations', {})
        self.position_annotations['notes'] = self.notes.toPlainText()
        self.position_annotations['exclude'] = self.exclude.isChecked()
        current_timepoint_name = list(self.timepoints.keys())[self.ris_widget.flipbook.current_page_idx]
        self.position_annotations['__last_timepoint_annotated__'] = current_timepoint_name
        load_data.write_annotation_file(self.annotation_file, self.position_annotations, self.timepoint_annotations)

    def prev_timepoint(self):
        self.flipbook.focus_prev_page()

    def next_timepoint(self):
        self.flipbook.focus_next_page()

    def prev_position(self):
        return self.load_position_index(max(0, self.position_i - 1))

    def next_position(self):
        last_i = len(self.position_names) - 1
        return self.load_position_index(min(self.position_i + 1, last_i))


class ZoomListener(base.SceneListener):
    QGRAPHICSITEM_TYPE = shared_resources.generate_unique_qgraphicsitem_type()

    def __init__(self, ris_widget):
        self.image_view = ris_widget.image_view
        super().__init__(ris_widget)

    def sceneEventFilter(self, watched, event):
        if event.type() == Qt.QEvent.GraphicsSceneMousePress and event.modifiers() & Qt.Qt.ControlModifier:
            self._mouse_y = self.image_view.mapFromScene(event.pos()).y()
            return True
        if event.type() == Qt.QEvent.GraphicsSceneMouseMove and event.modifiers() & Qt.Qt.ControlModifier:
            y = self.image_view.mapFromScene(event.pos()).y()
            delta = y - self._mouse_y
            if abs(delta) > 15:
                self._mouse_y = y
                if delta > 0:
                    self.image_view.change_zoom(zoom_in=True)
                else:
                    self.image_view.change_zoom(zoom_in=False)
            return True
        return False

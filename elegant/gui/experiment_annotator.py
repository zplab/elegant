# This code is licensed under the MIT License (see LICENSE file for details)

import atexit
import pathlib
import collections
import random

from PyQt5 import Qt

from ris_widget import shared_resources
from ris_widget.overlay import base

from .. import datamodel
from .. import process_data

def _get_display_name(obj):
    if hasattr(obj, 'display_name'):
        return obj.display_name
    elif hasattr(obj, 'name'):
        return obj.name
    else:
        return ''

def shuffle_and_blind_experiments(*experiments, random_seed=0):
    """Shuffle one or more datamodel.Experiments and blind the names.

    datamodel.Position instances from the provided experiment(s) are randomly
    shuffled (based on a fixed, specified random seed, for reproducibility) and
    their names obscured for blind annotation.

    Parameters:
        *experiments: one or more datamodel.Experiment instances, or lists of
            datamodel.Position instances.
        random_seed: string or integer providing random seed for reproducible
            shuffling of the positions.

    Returns: list datamodel.Position instances.
    """
    positions = collections.UserList(datamodel.flatten(experiments))
    # needs to be a UserList so a display_name attribute can be set
    positions.display_name = 'blinded positions'
    # shuffle with a new Random generator from the specified seed. (Don't just re-seed the
    # default random generator, because that would mess up other random streams if in use.)
    random.Random(random_seed).shuffle(positions)
    for i, position in enumerate(positions):
        position.display_name = str(i)
        for j, timepoint in enumerate(position):
            timepoint.display_name = str(j)
    return positions

class ExperimentAnnotator:
    def __init__(self, ris_widget, positions, annotation_fields, start_position=None, readonly=False, channels='bf'):
        """Set up a GUI for annotating an entire experiment.

        Produce annotations for experimental positions (i.e. each worm). Loading
        and saving annotations is handled by the underlying datamodel classes;
        the entries in the annotations dictionaries will be those in the ris_widget
        flibook's annotations dictionaries (the 'annotations' attribute of each
        flipbook page), as produced by ris_widget annotators.

        Keyboard shortcuts:
            [ and ] switch between worms
            control-s (command-s on macs) saves annotations


        Parameters:
            ris_widget: a RisWidget object
            positions: a sequence of datamodel.Position instances. Most commonly
                a datamodel.Experiment instance, but could also be a list/tuple
                of Position instances. Each Position and each Timepoint within
                that position will be displayed using the 'display_name' attribute
                if present, else the 'name' attribute.
            annotation_fields: list of annotation fields to pass to
                ris_widget.add_annotator()
            start_position: name of starting position to load first (e.g. "009").
                if none specified, load the first position. To change to an
                arbitrary position mid-stream, call the load_position() method.
            readonly: do not write changes to annotation files on disk
            channels: image channel or list of channels to load to the flipbook from
                each timepoint
        """
        self.ris_widget = ris_widget
        self.readonly = readonly
        ris_widget.add_annotator(annotation_fields)
        for field in annotation_fields:
            # give fields a reference to the annotator should they need that information
            field.experiment_annotator = self
        if isinstance(channels, str):
            channels = [channels]
        self.channels = channels

        self.positions = [position for position in positions if len(position) > 0]
        self.position_names_to_indices = {_get_display_name(position): i for i, position in enumerate(self.positions)}

        if not self.readonly:
            # update the annotations file(s)
            for experiment in set(position.experiment for position in self.positions):
                process_data.update_annotations(experiment.path)

        self.position_i = None
        self.flipbook = ris_widget.flipbook

        # add a listener to zoom the view when dragging with the control key held down
        # (for graphics tablet cases with no mouse wheel to zoom)
        self.main_zoom = ZoomListener(ris_widget)
        if hasattr(ris_widget, 'alt_view'):
            self.alt_zoom = ZoomListener(ris_widget.alt_view)

        display_name = _get_display_name(positions)
        widget = Qt.QGroupBox(display_name)
        layout = Qt.QVBoxLayout(widget)
        layout.setSpacing(0)
        worm_info = Qt.QHBoxLayout()
        worm_info.setSpacing(11)
        self.pos_editor = Qt.QLineEdit()
        self.pos_editor.editingFinished.connect(self._on_pos_editing_finished)
        maxlen = max(map(len, self.positions))
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
        num_positions = len(self.positions)
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
        if i is not None:
            self._prev_button.setEnabled(i != 0)
            self._next_button.setEnabled(i != num_positions - 1)
            self.position = self.positions[i]
            self.pos_editor.setText(self.position_name)
            self.pos_label.setText(f'({i+1}/{len(self.positions)})')
            futures = self.add_images_to_flipbook()
            self.load_annotations()
            last_i = 0
            if '__last_timepoint_annotated__' in self.position.annotations:
                last_timepoint_name = self.position.annotations['__last_timepoint_annotated__']
                for i, timepoint in enumerate(self.position):
                    # use actual timepoint.name, not display_name, since the former is authoritative
                    if timepoint.name == last_timepoint_name:
                        last_i = i
                        break
            self.ris_widget.flipbook.current_page_idx = last_i
            self.ris_widget.flipbook.pages_view.setFocus()
            return futures
        else:
            self.position = None
            self.pos_editor.setText('')
            self.pos_label.setText('')
            return []

    def add_images_to_flipbook(self):
        page_names, image_paths, image_names = [], [], []
        for timepoint in self.position:
            page_names.append(_get_display_name(timepoint))
            image_paths.append([timepoint.image_path(channel) for channel in self.channels])
            image_names.append(self.channels)
        return self.ris_widget.add_image_files_to_flipbook(image_paths, page_names, image_names)

    @property
    def position_name(self):
        return _get_display_name(self.position)

    def load_annotations(self):
        self.exclude.setChecked(self.position.annotations.get('exclude', False)) # Set to include by default
        self.notes.setPlainText(self.position.annotations.get('notes', ''))
        for page, timepoint in zip(self.ris_widget.flipbook_pages, self.position):
            annotations = timepoint.annotations
            if self.readonly:
                # make a copy so we can't save out changes even if we wanted
                annotations = dict(annotations)
            page.annotations = annotations
            page.timepoint = timepoint
        self.ris_widget.annotator.update_fields()

    def save_annotations(self):
        if self.readonly:
            return
        self.position.annotations['notes'] = self.notes.toPlainText()
        self.position.annotations['exclude'] = self.exclude.isChecked()
        self.position.annotations['__last_timepoint_annotated__'] = self.ris_widget.flipbook.current_page.timepoint.name
        self.position.write_annotations()

    def prev_timepoint(self):
        self.flipbook.focus_prev_page()

    def next_timepoint(self):
        self.flipbook.focus_next_page()

    def prev_position(self):
        return self.load_position_index(max(0, self.position_i - 1))

    def next_position(self):
        last_i = len(self.positions) - 1
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

# This code is licensed under the MIT License (see LICENSE file for details)

import collections
import pickle
import pkg_resources
from PyQt5 import Qt

from zplib.curve import interpolate
from ris_widget.qwidgets import annotator

from .spline_overlay import spline_outline
from .. import edge_detection
from .. import worm_widths

class PoseAnnotation(annotator.AnnotationField):
    ENABLABLE = True

    @classmethod
    def from_experiment_metadata(cls, metadata, ris_widget, name='pose', age_factor=1):
        width_estimator = worm_widths.WidthEstimator.from_experiment_metadata(metadata, age_factor)
        return cls(ris_widget, name, width_estimator, metadata['objective'], metadata['optocoupler'])

    def __init__(self, ris_widget, name='pose', width_estimator=None, objective=5, optocoupler=1):
        """Annotation field to record worm positions.

        Shortcuts:
            Note: these shortcuts apply to the centerline or width spline based
                on which sub-window was last clicked in.
            f / shift-f: increase/decrease overall smoothing factor of the
            centerline or width spline
            s: perform a smoothing operation on the centerline or width spline
            r: reverse the spline direction
            escape: start drawing centerline or width spline if none is extant
            delete: delete selected centerline or width spline
            shift while dragging: "fine control" mode to warp smaller areas.
            double-click: append a new endpoint to the centerline
            control-z / shift-control-z (command-z / shift-command-z on mac):
                undo / redo spline edits.

        Parameters:
            ris_widget: RisWidget instance
            name: name that the annotations will be stored in.
            width_estimator: worm_widths.WidthEstimator instance, or None.
            objective: magnification (as a float) of the objective used
            optocoupler: magnification (as a float) of the optocoupler used
        """
        self.ris_widget = ris_widget
        self.outline = spline_outline.SplineOutline(ris_widget, Qt.QColor(0, 255, 0, 128))
        self.outline.geometry_change_callbacks.append(self.on_geometry_change)
        self.undo_stack = collections.deque(maxlen=100)
        self.redo_stack = collections.deque(maxlen=100)
        self.width_estimator = width_estimator
        self.objective = objective
        self.optocoupler = optocoupler
        super().__init__(name, default=(None, None))

    def init_widget(self):
        self.widget = Qt.QGroupBox(self.name)
        layout = Qt.QVBoxLayout()
        self._hbox_spacing = self.widget.style().layoutSpacing(Qt.QSizePolicy.PushButton, Qt.QSizePolicy.PushButton, Qt.Qt.Horizontal)
        layout.setSpacing(0)
        self.widget.setLayout(layout)

        self.show_centerline = Qt.QCheckBox('Center')
        self.show_centerline.setChecked(True)
        self.show_centerline.toggled.connect(self.show_or_hide_centerline)
        self.show_outline = Qt.QCheckBox('Outline')
        self.show_outline.setChecked(True)
        self.show_outline.toggled.connect(self.show_or_hide_outline)
        self._add_row(layout, Qt.QLabel('Show:'), self.show_centerline, self.show_outline)

        self.undo_button = Qt.QPushButton('Undo')
        self.undo_button.clicked.connect(self.undo)
        Qt.QShortcut(Qt.QKeySequence.Undo, self.widget, self.undo, context=Qt.Qt.ApplicationShortcut)
        self.redo_button = Qt.QPushButton('Redo')
        self.redo_button.clicked.connect(self.redo)
        Qt.QShortcut(Qt.QKeySequence.Redo, self.widget, self.redo, context=Qt.Qt.ApplicationShortcut)
        self._add_row(layout, self.undo_button, self.redo_button)

        self.draw_center_button = Qt.QPushButton('Center')
        self.draw_center_button.setCheckable(True)
        self.draw_center_button.clicked.connect(self.draw_centerline)
        self.draw_width_button = Qt.QPushButton('Widths')
        self.draw_width_button.setCheckable(True)
        self.draw_width_button.clicked.connect(self.draw_widths)
        self._add_row(layout, Qt.QLabel('Draw:'), self.draw_center_button, self.draw_width_button)

        self.smooth_center_button = Qt.QPushButton('Center')
        self.smooth_center_button.clicked.connect(self.outline.center_spline.smooth)
        self.smooth_width_button = Qt.QPushButton('Widths')
        self.smooth_width_button.clicked.connect(self.outline.width_spline.smooth)
        self._add_row(layout, Qt.QLabel('Smooth:'), self.smooth_center_button, self.smooth_width_button)

        self.default_button = Qt.QPushButton('Default')
        self.default_button.clicked.connect(self.set_widths_to_default)
        self.pca_button = Qt.QPushButton('PCA')
        self.pca_button.clicked.connect(self.pca_smooth_widths)
        self._add_row(layout, Qt.QLabel('Widths:'), self.default_button, self.pca_button)

        self.auto_center_button = Qt.QPushButton('All')
        self.auto_center_button.clicked.connect(self.auto_center)
        self.auto_widths_button = Qt.QPushButton('Widths')
        self.auto_widths_button.clicked.connect(self.auto_widths)
        self._add_row(layout, Qt.QLabel('Auto:'), self.auto_center_button, self.auto_widths_button)

        self.reverse_button = Qt.QPushButton('Reverse')
        self.reverse_button.clicked.connect(self.outline.reverse_spline)
        Qt.QShortcut(Qt.Qt.Key_R, self.widget, self.outline.reverse_spline, context=Qt.Qt.ApplicationShortcut)

        self.fine_mode = Qt.QCheckBox('Fine')
        self.fine_mode.setChecked(False)
        self.fine_mode.toggled.connect(self.outline.set_fine_warp)

        lock_warp = Qt.QCheckBox('Lock')
        lock_warp.setChecked(False)
        lock_warp.toggled.connect(self.set_locked)
        self._add_row(layout, lock_warp, self.fine_mode, self.reverse_button)

    def _add_row(self, layout, *widgets):
        hbox = Qt.QHBoxLayout()
        hbox.setSpacing(self._hbox_spacing)
        layout.addLayout(hbox)
        for widget in widgets:
            sp = Qt.QSizePolicy(Qt.QSizePolicy.Ignored, Qt.QSizePolicy.Preferred)
            widget.setSizePolicy(sp)
            hbox.addWidget(widget, stretch=1)

    def on_geometry_change(self, tcks):
        center_tck, width_tck = tcks
        self.show_or_hide_centerline(self.show_centerline.isChecked())
        if not (self.outline.center_spline.warping or self.outline.width_spline.warping):
            self.undo_stack.append(self.get_annotation()) # put current value on the undo stack
            self.redo_stack.clear()
            self._enable_buttons()
            self.update_annotation((center_tck, width_tck))

    def update_widget(self, tcks):
        # called when switching pages
        if tcks is None:
            tcks = None, None
        self.undo_stack.clear()
        self.redo_stack.clear()
        self.outline.geometry = tcks
        self._enable_buttons()

    def undo(self):
        self._undo_redo(self.undo_stack, self.redo_stack)

    def redo(self):
        self._undo_redo(self.redo_stack, self.undo_stack)

    def _undo_redo(self, from_stack, to_stack):
        if len(from_stack) > 0:
            to_stack.append(self.get_annotation())
            new_state = from_stack.pop()
            self.outline.geometry = new_state
            self._enable_buttons()
            self.update_annotation(new_state)

    def _enable_buttons(self):
        center_tck = self.outline.center_spline.geometry
        width_tck = self.outline.width_spline.geometry
        has_center = center_tck is not None
        has_center_and_widths = has_center and width_tck is not None
        unlocked = not self.outline.center_spline.locked
        self.undo_button.setEnabled(len(self.undo_stack) > 0 and unlocked)
        self.redo_button.setEnabled(len(self.redo_stack) > 0 and unlocked)
        self.smooth_center_button.setEnabled(has_center and unlocked)
        self.smooth_width_button.setEnabled(has_center_and_widths and unlocked)
        self.draw_center_button.setEnabled(unlocked)
        self.draw_center_button.setChecked(self.outline.center_spline.drawing)
        self.draw_width_button.setEnabled(has_center and unlocked)
        self.draw_width_button.setChecked(self.outline.width_spline.drawing)
        self.default_button.setEnabled(self.width_estimator is not None and has_center and unlocked)
        self.pca_button.setEnabled(self.width_estimator is not None and has_center_and_widths and unlocked)
        self.reverse_button.setEnabled(has_center and unlocked)
        self.auto_center_button.setEnabled(has_center and unlocked)
        self.auto_widths_button.setEnabled(has_center and unlocked)
        self.fine_mode.setEnabled(unlocked)

    def set_locked(self, locked):
        self.outline.set_locked(locked)
        self._enable_buttons()

    def _change_geometry(self, center_tck=None, width_tck=None):
        """Cause a geometry change programmatically. This function takes care
        of updating the GUI and the annotation, and adding the new geometry to
        the undo stack."""
        if center_tck is None:
            center_tck = self.outline.center_spline.geometry
        if width_tck is None:
            width_tck = self.outline.width_spline.geometry
        self.outline.geometry = center_tck, width_tck
        # now tell the outline to let all listeners (including us) know that
        # the geometry has changed. This will lead to the annotation and undo
        # stack getting properly updated via our on_geometry_change()
        self.outline._geometry_changed()

    def get_default_widths(self):
        if self.width_estimator is None:
            return None
        else:
            return self.width_estimator.width_tck_for_age(self.page.annotations.get('age'))

    def set_widths_to_default(self):
        self._change_geometry(width_tck=self.get_default_widths())

    def _pca_smooth_widths(self, width_tck):
        if self.width_estimator is None:
            return None
        mean_widths = self.width_estimator.width_profile_for_age(self.page.annotations.get('age'))
        return self.width_estimator.pca_smooth_widths(width_tck, mean_widths)

    def pca_smooth_widths(self):
        width_tck = self._pca_smooth_widths(self.outline.width_spline.geometry)
        if width_tck is not None:
            self._change_geometry(width_tck=width_tck)

    def _fit_to_image(self):
        width_tck = self.outline.width_spline.geometry
        if width_tck is None:
            width_tck = self.get_default_widths()
        center_tck, width_tck = edge_detection.detect_edges(
            image=self.ris_widget.image.data, center_tck=self.outline.center_spline.geometry,
            width_tck=width_tck, objective=self.objective, optocoupler=self.optocoupler)
        smooth_width_tck = self._pca_smooth_widths(width_tck)
        if smooth_width_tck is not None:
            width_tck = smooth_width_tck
        return center_tck, width_tck

    def auto_center(self):
        center_tck, width_tck = self._fit_to_image()
        self._change_geometry(center_tck, width_tck)

    def auto_widths(self):
        center_tck, width_tck = self._fit_to_image()
        self._change_geometry(width_tck=width_tck)

    def draw_centerline(self, draw):
        center_tck, width_tck = self.get_annotation()
        if draw:
            if width_tck is None:
                width_tck = self.get_default_widths()
            self.outline.geometry = None, width_tck
            self.outline.center_spline.start_drawing()
        else: # draw operation canceled by clicking button again
            self.outline.geometry = center_tck, width_tck
        self._enable_buttons()

    def draw_widths(self, draw):
        center_tck, width_tck = self.get_annotation()
        if draw:
            self.outline.geometry = center_tck, None
            self.outline.width_spline.start_drawing()
        else: # draw operation canceled by clicking button again
            self.outline.geometry = center_tck, width_tck
        self._enable_buttons()

    def show_or_hide_centerline(self, show):
        # 1: For the lab frame of reference:
        # if show, then show the centerline.
        # if not, then only show if there is *no* centerline set: this way,
        # the line will be shown during manual drawing but hid once that line
        # is converted to a spline tck.
        if show or self.outline.center_spline.geometry is None:
            self.outline.center_spline.setPen(self.outline.center_spline.display_pen)
        else:
            # "hide" by setting transparent pen. This still allows for dragging
            # the hidden centerline -- which using its setVisible method prevents.
            self.outline.center_spline.setPen(Qt.QPen(Qt.Qt.transparent))
        # 2: hide or show midline in worm frame of reference
        self.outline.width_spline.midline.setVisible(show and self.outline.center_spline.geometry is not None)

    def show_or_hide_outline(self, show):
        self.outline.setVisible(show) # in lab frame of reference
        self.outline.width_spline.setVisible(show) # in worm frame


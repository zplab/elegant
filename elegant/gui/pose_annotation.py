# This code is licensed under the MIT License (see LICENSE file for details)

import collections

from PyQt5 import Qt
import numpy
from ris_widget import internal_util
from ris_widget.qwidgets import annotator
from ris_widget.overlay import spline_outline

class PoseAnnotation(annotator.AnnotationField):
    ENABLABLE = True

    def __init__(self, ris_widget, name='pose', mean_widths=None, width_pca_basis=None):
        self.ris_widget = ris_widget
        self.outline = spline_outline.SplineOutline(ris_widget, Qt.QColor(0, 255, 0, 128))
        self.centerline = self.outline.center_spline
        self.widths = self.outline.width_spline
        self.centerline.geometry_change_callbacks.append(self.on_centerline_change)
        self.widths.geometry_change_callbacks.append(self.on_widths_change)
        self._ignore_geometry_change = internal_util.Condition()
        self.undo_stack = collections.deque(maxlen=100)
        self.redo_stack = collections.deque(maxlen=100)
        if mean_widths is None:
            self.widths_calculator = None
        else:
            if not isinstance(mean_widths, dict):
                mean_widths = {0: mean_widths}
            if width_pca_basis is not None:
                if not numpy.allclose((width_pca_basis**2).sum(axis=1), numpy.ones(len(width_pca_basis))):
                    raise ValueError('a unit-length (non-normalized) PCA basis must be provided')
            self.widths_calculator = PCAWidthCalculator(mean_widths, width_pca_basis, self.widths)
        super().__init__(name)

    def init_widget(self):
        self.widget = Qt.QGroupBox(self.name)
        self._current_row = 0
        layout = Qt.QGridLayout()
        self.widget.setLayout(layout)

        self.show_centerline = Qt.QCheckBox('Show Center')
        self.show_centerline.setChecked(True)
        self.show_centerline.toggled.connect(self.show_or_hide_centerline)
        self.show_outline = Qt.QCheckBox('Show Outline')
        self.show_outline.setChecked(True)
        self.show_outline.toggled.connect(self.outline.setVisible)
        self._add_row(layout, self.show_centerline, self.show_outline)

        self.undo_button = Qt.QPushButton('Undo')
        self.undo_button.clicked.connect(self.undo)
        Qt.QShortcut(Qt.QKeySequence.Undo, self.widget, self.undo)
        self.redo_button = Qt.QPushButton('Redo')
        self.redo_button.clicked.connect(self.redo)
        Qt.QShortcut(Qt.QKeySequence.Redo, self.widget, self.redo)
        self._add_row(layout, self.undo_button, self.redo_button)

        self.draw_center_button = Qt.QPushButton('Draw Center')
        self.draw_center_button.setCheckable(True)
        self.draw_center_button.clicked.connect(self.draw_centerline)
        self.draw_width_button = Qt.QPushButton('Draw Widths')
        self.draw_width_button.setCheckable(True)
        self.draw_width_button.clicked.connect(self.draw_widths)
        self._add_row(layout, self.draw_center_button, self.draw_width_button)

        self.smooth_center_button = Qt.QPushButton('Smooth Center')
        self.smooth_center_button.clicked.connect(self.centerline.smooth)
        self.smooth_width_button = Qt.QPushButton('Smooth Widths')
        self.smooth_width_button.clicked.connect(self.widths.smooth)
        self._add_row(layout, self.smooth_center_button, self.smooth_width_button)

        self.default_button = Qt.QPushButton('Default Widths')
        self.default_button.clicked.connect(self.set_widths_to_default)
        self.pca_button = Qt.QPushButton('PCA(Widths)')
        self.pca_button.clicked.connect(self.pca_smooth_widths)
        self._add_row(layout, self.default_button, self.pca_button)

        self.reverse_button = Qt.QPushButton('Reverse')
        self.reverse_button.clicked.connect(self.reverse_spline)
        Qt.QShortcut(Qt.Qt.Key_R, self.widget, self.reverse_spline)
        self.fine_mode = Qt.QCheckBox('Fine Warping')
        self.fine_mode.setChecked(False)
        self.fine_mode.toggled.connect(self.toggle_fine_mode)
        self._add_row(layout, self.reverse_button, self.fine_mode)

    def _add_row(self, layout, *widgets):
        if len(widgets) == 1:
            layout.addWidget(widgets[0], self._current_row, 0, 1, -1, Qt.Qt.AlignCenter)
        else:
            for i, widget in enumerate(widgets):
                layout.addWidget(widget, self._current_row, i)
        self._current_row += 1

    def get_age_from_page(self, flipbook_page):
        """Used for age-specific calculation of mean width profiles. If only
        one profile is provided, not needed. Otherwise, subclass / monkeypatch
        to define a function that identifies the age of the worm based on a
        flipbook page (e.g. looks up something based on that page's .name
        attribute, or something in its .annotations dictionary...)"""
        return None

    def default_annotation_for_page(self, page):
        center_tck = None
        width_tck = self.get_default_widths()
        return center_tck, width_tck

    def get_default_widths(self):
        if self.widths_calculator is None:
            return None
        else:
            age = self.get_age_from_page(self.page)
            return self.widths_calculator.mean_width_tck_for_age(age)

    def on_centerline_change(self, center_tck):
        self.show_or_hide_centerline(self.show_centerline.isChecked())
        self.on_geometry_change(center_tck, self.widths.geometry)

    def on_widths_change(self, width_tck):
        self.on_geometry_change(self.centerline.geometry, width_tck)

    def on_geometry_change(self, center_tck, width_tck):
        if not (self._ignore_geometry_change or self.centerline.warping or self.widths.warping):
            self.undo_stack.append(self.get_annotation()) # put current value on the undo stack
            self.redo_stack.clear()
            self._enable_buttons(center_tck, width_tck)
            self.update_annotation((center_tck, width_tck))

    def update_widget(self, value):
        # called when switching pages
        if value is None:
            value = None, None
        center_tck, width_tck = value
        self.undo_stack.clear()
        self.redo_stack.clear()
        self.ris_widget.alt_view.image_view.zoom = self.ris_widget.image_view.zoom
        self._update_widget(center_tck, width_tck)

    def _update_widget(self, center_tck, width_tck):
        # called by update_widget and also when undoing / redoing
        self._enable_buttons(center_tck, width_tck)
        with self._ignore_geometry_change:
            self.centerline.geometry = center_tck
            self.widths.geometry = width_tck

    def undo(self):
        if len(self.undo_stack) > 0:
            self.redo_stack.append(self.get_annotation())
            new_state = self.undo_stack.pop()
            self._update_widget(*new_state)
            self.update_annotation(new_state)

    def redo(self):
        if len(self.redo_stack) > 0:
            self.undo_stack.append(self.get_annotation())
            new_state = self.redo_stack.pop()
            self._update_widget(*new_state)
            self.update_annotation(new_state)

    def _enable_buttons(self, center_tck, width_tck):
        has_center = center_tck is not None
        has_center_and_widths = has_center and width_tck is not None
        self.undo_button.setEnabled(len(self.undo_stack) > 0)
        self.redo_button.setEnabled(len(self.redo_stack) > 0)
        self.smooth_center_button.setEnabled(has_center)
        self.smooth_width_button.setEnabled(has_center_and_widths)
        self.draw_center_button.setChecked(self.centerline.drawing)
        self.draw_width_button.setEnabled(has_center)
        self.draw_width_button.setChecked(self.widths.drawing)
        self.default_button.setEnabled(self.widths_calculator is not None and has_center)
        self.pca_button.setEnabled(self.widths_calculator is not None and has_center_and_widths)
        self.reverse_button.setEnabled(has_center)

    def set_widths_to_default(self):
        self.widths.geometry = self.get_default_widths()

    def pca_smooth_widths(self):
        age = self.get_age_from_page(self.page)
        self.widths.geometry = self.widths_calculator.pca_smooth_widths(age)

    def draw_centerline(self, draw):
        with self._ignore_geometry_change:
            self.centerline.geometry = None
        if draw:
            self.centerline.start_drawing()

    def draw_widths(self, draw):
        with self._ignore_geometry_change:
            self.widths.geometry = None
        if draw:
            self.widths.start_drawing()

    def reverse_spline(self):
        if self.centerline.geometry is not None:
            if self.widths.geometry is not None:
                # Ignore this change so as not to add the width reversal to the
                # undo stack separately from the centerline reversal. The latter
                # will cause both reversals to be added to the stack.
                with self._ignore_geometry_change:
                    self.widths.reverse_spline()
            self.centerline.reverse_spline()

    def show_or_hide_centerline(self, show):
        # if show, then show the centerline.
        # if not, then only show if there is *no* centerline set: this way,
        # the line will be shown during manual drawing but hid once that line
        # is converted to a spline tck.
        if show or self.centerline.geometry is None:
            self.centerline.setPen(self.centerline.display_pen)
        else:
            # "hide" by setting transparent pen. This still allows for dragging
            # the hidden outline -- which using its hide() method prevents.
            self.centerline.setPen(Qt.QPen(Qt.Qt.transparent))

    def toggle_fine_mode(self, v):
        self.centerline.fine_warp = v
        self.widths.fine_warp = v


class PCAWidthCalculator:
    def __init__(self, ages_to_widths, width_pca_basis, width_spline):
        """ages_to_widths: dict mapping worm ages to mean width profiles for
                that age. May have any number of entries.
        """
        ages, self.widths = zip(*sorted(ages_to_widths.items()))
        self.ages = numpy.array(ages)
        self.width_pca_basis = width_pca_basis
        self.width_spline = width_spline

    def mean_width_tck_for_age(self, age):
        """get the default width spline for a given age"""
        return self.width_spline.calculate_tck(self.mean_width_for_age(age))

    def mean_width_for_age(self, age):
        """get the default width profile for a given age"""
        if len(self.ages) == 1:
            # if there's only one choice, return it
            return self.widths[0]
        # interpolate the ages to find the index we want to look up / calculate
        age_i = numpy.interp(age, self.ages, numpy.arange(len(self.ages)))
        if age_i == int(age_i):
            # if the index is a whole number, return that value:
            return self.widths[age_i]
        # since the index is not an int, get the bracketing indices
        i_low, i_high = int(numpy.floor(age_i)), int(numpy.ceil(age_i))
        w_low = self.widths[i_low]
        w_high = self.widths[i_high]
        # linearly interpolate between the two width profiles to get the
        # best estimate for the given age
        return w_low * (i_high - age_i) + w_high * (age_i - i_low)

    def pca_smooth_widths(self, age):
        if self.width_pca_basis is None:
            return None
        mean_widths = self.mean_width_for_age(age)
        x = numpy.linspace(0, 1, len(mean_widths))
        widths = self.width_spline.evaluate_tck(x)
        pca_projection = numpy.dot(widths - mean_widths, self.width_pca_basis.T)
        pca_reconstruction = mean_widths + numpy.dot(pca_projection, self.width_pca_basis)
        return self.width_spline.calculate_tck(pca_reconstruction, x)

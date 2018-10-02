# This code is licensed under the MIT License (see LICENSE file for details)

from PyQt5 import Qt
import numpy

from zplib.curve import interpolate
from ris_widget import shared_resources

from . import center_spline

class WidthSpline(center_spline.CenterSpline, Qt.QGraphicsPathItem):
    QGRAPHICSITEM_TYPE = shared_resources.generate_unique_qgraphicsitem_type()
    SPLINE_POINTS = 200
    SMOOTH_BASE = 1
    BANDWIDTH = 4

    def __init__(self, ris_widget, pen=None, geometry=None):
        self._tck_x = numpy.linspace(0, 1, self.SPLINE_POINTS)
        self.image_shape = None
        super().__init__(ris_widget, pen, geometry)
        self.layer = None
        self.draw_midline = True
        self.midline = Qt.QGraphicsPathItem(self.parentItem())
        midline_pen = Qt.QPen(self.display_pen)
        midline_pen.setStyle(Qt.Qt.DotLine)
        self.midline.setPen(midline_pen)
        self.parentItem().bounding_rect_changed.connect(self._update_image_shape)
        self._update_image_shape()

    def _update_image_shape(self):
        # bounding rect change means that the image at layers[0] has changed in some way
        self.image_shape = None
        layers = self.ris_widget.layer_stack.layers
        if len(layers) > 0 and layers[0].image is not None:
            self.image_shape = layers[0].image.data.shape
        self._update_path()

    def remove(self):
        super().remove()
        self.parentItem().bounding_rect_changed.disconnect(self._update_image_shape)
        self.ris_widget.image_scene.removeItem(self.midline)

    def _update_path(self):
        self.path = Qt.QPainterPath()
        if self.image_shape is not None:
            width, height = self.image_shape
            centerline_y = height / 2
            self.path.moveTo(0, centerline_y)
            if self._tck is not None:
                image_x = self._tck_x * width
                points = self.evaluate_tck()
                for x, y in zip(image_x, centerline_y - points):
                    self.path.lineTo(x, y)
                for x, y in zip(image_x[::-1], centerline_y + points[::-1]):
                    self.path.lineTo(x, y)
                self.path.closeSubpath()
            if self.draw_midline:
                midline_path = Qt.QPainterPath()
                midline_path.moveTo(0, centerline_y)
                midline_path.lineTo(width, centerline_y)
                self.midline.setPath(midline_path)
        self.setPath(self.path)

    def _update_points(self):
        if self._tck is None:
            self._points = numpy.empty_like(self._tck_x)
            self._points.fill(numpy.nan)
            self._hand_drawn = True
        else:
            self._points = numpy.maximum(self.evaluate_tck(), 0.1)
            self._hand_drawn = False

    def _generate_tck_from_points(self, smoothing=None):
        x = self._tck_x
        widths = self._points
        if self._hand_drawn:
            # un-filled widths may be nan
            good_widths = numpy.isfinite(widths)
            x = x[good_widths]
            widths = widths[good_widths]
        if len(widths) > 4:
            if smoothing is None:
                smoothing = self._smoothing if self._hand_drawn else None
            tck = self.calculate_tck(widths, x, smoothing=smoothing)
        else:
            tck = None
        self._set_tck(tck)
        self._geometry_changed()

    def calculate_tck(self, widths, x=None, smoothing=None):
        if x is None:
            x = numpy.linspace(0, 1, len(widths))
        if smoothing is None:
            smoothing = self.REFIT_SMOOTH
        return interpolate.fit_nonparametric_spline(x, widths, smoothing=smoothing * len(widths))

    def evaluate_tck(self, x=None):
        if x is None:
            x = self._tck_x
        return interpolate.spline_evaluate(self._tck, x)

    def _add_point(self, pos):
        if self.image_shape is None:
            return
        width, height = self.image_shape
        centerline_y = height / 2
        x, y = pos.x(), pos.y()
        if self._last_pos is not None:
            last_x, y_sign = self._last_pos
            if abs(x - last_x) < 6:
                return
        else:
            # invert widths if we started out past the centerline
            y_sign = 1 if y < centerline_y else -1
        if not 0 <= x <= width:
            return
        x_i = round((len(self._points) - 1) * x / width)
        self._points[x_i] = max(y_sign * (centerline_y - y), 0.1)
        self._last_pos = (x, y_sign)
        # now draw points
        good_points = numpy.isfinite(self._points)
        xs = self._tck_x[good_points] * width
        ys = self._points[good_points]
        self.path = Qt.QPainterPath()
        self.path.moveTo(xs[0], centerline_y - ys[0])
        for x, y in zip(xs[1:], centerline_y - ys[1:]):
            self.path.lineTo(x, y)
        self.path.moveTo(xs[0], centerline_y + ys[0])
        for x, y in zip(xs[1:], centerline_y + ys[1:]):
            self.path.lineTo(x, y)
        self.setPath(self.path)

    def _start_warp(self, pos):
        self._warp_start = pos.y()
        self._update_points()
        self._warp_points = self._points

    def _warp_spline(self, pos):
        self._last_pos = pos
        if self.image_shape is None:
            return
        width, height = self.image_shape
        centerline_y = height / 2
        bandwidth_factor = 0.5 if self.fine_warp else 1
        bandwidth = bandwidth_factor / self.BANDWIDTH
        distances = self._tck_x - pos.x() / width
        warp_coefficients = numpy.exp(-(distances/bandwidth)**2)
        displacement = self._warp_start - pos.y()
        if self._warp_start > centerline_y:
            displacement *= -1
        displacements = displacement * warp_coefficients
        self._points = numpy.maximum(self._warp_points + displacements, 0.1)
        self._generate_tck_from_points()

    def _extend_endpoint(self, pos):
        # no endpoint-extending for width splines...
        pass
# This code is licensed under the MIT License (see LICENSE file for details)

from PyQt5 import Qt
import numpy

from zplib.image import colorize
from ris_widget.qwidgets import annotator
from ris_widget.overlay import identified_point_set
from ris_widget import split_view

class KeypointAnnotation(annotator.AnnotationField):
    def __init__(self, ris_widget, keypoint_names, colors=None, name='keypoints', worm_frame=True, auto_advance=False):
        num_points = len(keypoint_names)
        if colors is None:
            colors = colorize.color_map(numpy.linspace(0, 1, num_points), spectrum_max=0.8)
        self.colors = colors
        self.keypoint_names = keypoint_names
        if worm_frame:
            if not hasattr(ris_widget, 'alt_view'):
                split_view.split_view(ris_widget)
            self.ris_widget = ris_widget.alt_view
            self.center_y_origin = True
            # bounding rect change means that the image shape has changed in some way
            self.ris_widget.image_scene.layer_stack_item.bounding_rect_changed.connect(self._new_image_shape)
        else:
            self.ris_widget = ris_widget
            self.center_y_origin = False
        qcolors = [Qt.QColor(*c) for c in colors]
        pen = Qt.QPen(Qt.QColor(255, 255, 255, 84))
        pen.setWidth(3)
        pen.setCosmetic(True)
        self.point_set = identified_point_set.IdentifiedPointSet(self.ris_widget, num_points, qcolors, pen)
        self.point_set.geometry_change_callbacks.append(self.on_geometry_change)
        self._auto_advance = auto_advance
        super().__init__(name, default={name: None for name in keypoint_names})

    def _new_image_shape(self):
        if self.page is None:
            named_points = None
        else:
            named_points = self.page.annotations.get(self.name)
        self.update_widget(named_points)

    def init_widget(self):
        self.widget = Qt.QGroupBox(self.name)
        layout = Qt.QHBoxLayout()
        self.widget.setLayout(layout)
        self.labels = [Qt.QLabel(name) for name in self.keypoint_names]
        for label in self.labels:
            label.setAlignment(Qt.Qt.AlignCenter)
            layout.addWidget(label)

    def on_geometry_change(self, point_list):
        image = self.ris_widget.image
        if image is None:
            return
        image_height = image.data.shape[1]
        new_points = []
        for point in point_list:
            if point is None:
                new_points.append(None)
            else:
                x, y = point
                if self.center_y_origin:
                    y -= image_height/2
                new_points.append((x, y))
        named_points = dict(zip(self.keypoint_names, new_points))
        self.update_text(named_points)
        # call last, because may produce an auto-advance
        self.update_annotation(named_points)

    def update_widget(self, named_points):
        if named_points is None:
            point_list = None
        else:
            point_list = [named_points.get(name) for name in self.keypoint_names]
            image = self.ris_widget.image
            if image is None:
                point_list = None
            else:
                image_height = image.data.shape[1]
                new_points = []
                for point in point_list:
                    if point is None:
                        new_points.append(None)
                    else:
                        x, y = point
                        if self.center_y_origin:
                            y += image_height/2
                        new_points.append((x, y))
                    point_list = new_points
        self.point_set.geometry = point_list
        self.update_text(named_points)

    def update_text(self, named_points):
        if named_points is None:
            named_points = {name: None for name in self.keypoint_names}
        for label, (r, g, b), name in zip(self.labels, self.colors, self.keypoint_names):
            point = named_points.get(name)
            if point is None:
                style = 'color: gray'
            else:
                style = f'color: rgb({r}, {g}, {b}); font-weight: bold'
            label.setStyleSheet(style)

    def auto_advance(self, old_named_points, new_named_points):
        # advance only when going from not-enough-points to enough-points
        if not self._auto_advance or new_named_points is None:
            return False
        if old_named_points is None:
            prev_remaining = len(self.keypoint_names)
        else:
            prev_remaining = len([v for v in old_named_points.values() if v is None])
        new_remaining = len([v for v in new_named_points.values() if v is None])
        return prev_remaining > 0 and new_remaining == 0

# This code is licensed under the MIT License (see LICENSE file for details)


from PyQt5 import Qt
import numpy
from zplib.image import colorize
from ris_widget.qwidgets import annotator
from ris_widget.overlay import identified_point_set

class KeypointAnnotation(annotator.AnnotationField):
    def __init__(self, ris_widget, keypoint_names, colors=None, name='keypoints', auto_advance=False):
        num_points = len(keypoint_names)
        if colors is None:
            colors = colorize.color_map(numpy.linspace(0, 1, num_points), spectrum_max=0.8)
        self.colors = colors
        self.keypoint_names = keypoint_names
        qcolors = [Qt.QColor(*c) for c in colors]
        pen = Qt.QPen(Qt.QColor(255, 255, 255, 84))
        pen.setWidth(3)
        pen.setCosmetic(True)
        self.point_set = identified_point_set.IdentifiedPointSet(ris_widget, num_points, qcolors, pen)
        self.point_set.geometry_change_callbacks.append(self.on_geometry_change)
        self._auto_advance = auto_advance
        super().__init__(name)

    def init_widget(self):
        self.widget = Qt.QGroupBox(self.name)
        layout = Qt.QHBoxLayout()
        self.widget.setLayout(layout)
        self.labels = [Qt.QLabel(name) for name in self.keypoint_names]
        for label in self.labels:
            label.setAlignment(Qt.Qt.AlignCenter)
            layout.addWidget(label)

    def on_geometry_change(self, point_list):
        named_points = dict(zip(self.keypoint_names, point_list))
        self.update_text(named_points)
        # call last, because may produce an auto-advance
        self.update_annotation(named_points)

    def update_widget(self, named_points):
        if named_points is None:
            point_list = None
        else:
            point_list = [named_points[name] for name in self.keypoint_names]
        self.point_set.geometry = point_list
        self.update_text(named_points)

    def update_text(self, named_points):
        if named_points is None:
            named_points = {name: None for name in self.keypoint_names}
        for label, color, name in zip(self.labels, self.colors, self.keypoint_names):
            point = named_points[name]
            if point is None:
                style = 'color: gray'
            else:
                style = 'color: rgb({}, {}, {}); font-weight: bold'.format(*color)
            label.setStyleSheet(style)

    def auto_advance(self, old_named_points, new_named_points):
        if not self._auto_advance or old_named_points is None or new_named_points is None:
            return False
        old_nones = len([v for v in old_named_points.values() if v is None])
        new_nones = len([v for v in new_named_points.values() if v is None])
        return old_nones > 0 and new_nones == 0

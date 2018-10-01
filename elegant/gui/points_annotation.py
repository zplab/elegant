# This code is licensed under the MIT License (see LICENSE file for details)

from PyQt5 import Qt
from ris_widget.qwidgets import annotator
from ris_widget.overlay import point_set

class PointsAnnotation(annotator.AnnotationField):
    def __init__(self, ris_widget, name, color=(0, 255, 0, 255)):
        brush = Qt.QBrush(Qt.QColor(*color))
        self.point_set = point_set.PointSet(ris_widget, brush)
        self.point_set.geometry_change_callbacks.append(self.on_geometry_change)
        super().__init__(name)

    def init_widget(self):
        self.widget = Qt.QGroupBox(self.name)
        layout = Qt.QHBoxLayout()
        self.widget.setLayout(layout)
        self.label = Qt.QLabel()
        layout.addWidget(self.label)
        self.clear_button = Qt.QPushButton('Clear')
        self.clear_button.clicked.connect(self.clear)
        layout.addWidget(self.clear_button)

    def clear(self):
        self.update_widget(None)
        self.update_annotation(None)

    def on_geometry_change(self, points):
        # convert points to tuples so we can hash them into a set (just to clarify that it's unordered)
        points = None if points is None else set(map(tuple, points))
        self.update_annotation(points)
        self.update_gui(points)

    def update_widget(self, points):
        self.point_set.geometry = points
        self.update_gui(points)

    def update_gui(self, points):
        num_points = 0 if points is None else len(points)
        self.clear_button.setEnabled(num_points > 0)
        text = f'{num_points} point'
        if num_points != 1:
            text += 's'
        self.label.setText(text)

# This code is licensed under the MIT License (see LICENSE file for details)

from PyQt5 import Qt
from ris_widget.qwidgets import annotator

class TimepointAnnotations:
    def __init__(self):
        self.fields = [annotator.BoolField('exclude'), annotator.StringField('notes')]
        exclude, notes = [field.widget for field in self.fields]
        exclude.setText('Exclude')
        notes.setPlaceholderText('notes')
        self.widget = Qt.QGroupBox('timepoint annotations')
        layout = Qt.QHBoxLayout()
        self.widget.setLayout(layout)
        layout.addWidget(notes)
        layout.addWidget(exclude)

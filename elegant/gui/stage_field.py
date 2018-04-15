# This code is licensed under the MIT License (see LICENSE file for details)

import itertools
from PyQt5 import Qt
from ris_widget.qwidgets import annotator

class StageField(annotator.AnnotationField):
    FIRST_COLOR = (255, 255, 255)
    LAST_COLOR = (184, 184, 184)
    COLOR_CYCLE = itertools.cycle([(184, 255, 184), (255, 255, 184), (184, 184, 255), (255, 184, 184), (255, 184, 255)])

    def __init__(self, name='stage', stages=['egg', 'larva', 'adult', 'dead'], transitions=['hatch', 'adult', 'dead'], shortcuts=None):
        """Annotate the life-stage of a worm.

        Parameters:
            name: annotation name
            stages: list of life stages to annotate, generally starting with egg
                and ending with dead.
            transitions: names of the transitions between life stages.
            shortcuts: shortcut keys to select the different transitions; if not
                specified, the first letter of each transition will be used.
        """
        assert len(transitions) == len(stages) - 1
        self.stages = stages
        self.transitions = transitions
        if shortcuts is None:
            # take the first letter of each as the shortcut
            shortcuts = [transition[0] for transition in transitions]
        self.shortcuts = shortcuts
        self.colors = {stages[0]: self.FIRST_COLOR, stages[-1]: self.LAST_COLOR}
        self.colors.update(zip(stages[1:-1], self.COLOR_CYCLE))
        super().__init__(name)

    def init_widget(self):
        self.widget = Qt.QGroupBox(self.name)
        layout = Qt.QHBoxLayout()
        self.widget.setLayout(layout)
        self.label = Qt.QLabel()
        layout.addWidget(self.label)
        for transition, key, next_stage in zip(self.transitions, self.shortcuts, self.stages[1:]):
            button = Qt.QPushButton(transition)
            callback = self._make_transition_callback(next_stage)
            button.clicked.connect(callback)
            layout.addWidget(button)
            Qt.QShortcut(key, self.widget, callback, context=Qt.Qt.ApplicationShortcut)

    def _make_transition_callback(self, next_stage):
        def callback():
            self.set_stage(next_stage)
        return callback

    def set_stage(self, stage):
        self.update_annotation(stage)
        stage_i = self.stages.index(stage)
        # i will always be > 0
        prev_stage = self.stages[stage_i - 1]
        fb_i = self.flipbook.pages.index(self.page)
        for page_fb_i, page in enumerate(self.flipbook.pages):
            page_stage = self.get_annotation(page, setdefault=True)
            if page_stage is None:
                if page_fb_i < fb_i:
                    new_page_stage = prev_stage
                else:
                    # page is current page or later
                    new_page_stage = stage
            else:
                # page_stage is not None
                page_stage_i = self.stages.index(page_stage)
                if page_fb_i < fb_i and page_stage_i >= stage_i:
                    new_page_stage = prev_stage
                elif page_fb_i > fb_i and page_stage_i < stage_i:
                    # page is current page or later
                    new_page_stage = stage
                else:
                    new_page_stage = page_stage
            page.annotations[self.name] = new_page_stage
        self.update_widget(stage)

    def update_widget(self, value):
        if value is None:
            self.label.setText('')
        elif value not in self.stages:
            raise ValueError('Value {} not in list of stages.'.format(value))
        else:
            self.label.setText(value)
        self.recolor_pages()

    def recolor_pages(self):
        for page in self.flipbook.pages:
            stage = self.get_annotation(page)
            if stage is None:
                page.color = None
            else:
                page.color = self.colors[stage]

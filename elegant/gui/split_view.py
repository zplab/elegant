from ris_widget import split_view

def split_ris_widget(ris_widget):
    if not hasattr(ris_widget, 'alt_view'):
        split_view.split_view(ris_widget, stretch_factors=(4, 1))
    return ris_widget.alt_view

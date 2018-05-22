import pickle
import pathlib
from collections import OrderedDict

def filter_positions_by_kw(annotations, selection_kws, invert_selection=False):
    """Filter positions for an experiment based on notes in corresponding annotations
    
        Parameters
            annotations - An OrderedDict mapping position names to corresponding
                annotations (returned by load_data.read_annotations)
            annotation_kws - Optional ist of str containing kws used to select positions
                (a given position's annotations must require ALL of the specified kws)
            invert_selection - bool flag indicating whether to select filter
                the selected positions in or out (True/False respectively)
    """
    
    selected_positions = []
    for position_name, position_annotations in annotations.items():
        if any([kw in position_annotation['notes'] for kw in selection_kws]):
            selected_positions.append(position_name)
    if not invert_selection:
        return selected_positions
    else:
        return [position 
            for position in selected_positions 
            if position not in annotations.keys()]

def filter_positions(annotations, selection_criteria

def print_formatted_list(string_list):
    return "[\'" + "\',\'".join(string_list) + "\']"

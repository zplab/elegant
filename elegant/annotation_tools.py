import pickle
import pathlib
from collections import OrderedDict

def filter_positions(annotations, selection_criteria, invert_selection=False):
    """Filter positions for an experiment based on defined selection criteria

    Parameters
        annotations - An OrderedDict mapping position names to corresponding
            annotations (returned by load_data.read_annotations)
        selection_criteria - A function taking in a set of global and timepoint
            annotations (returned as a tuple by load_data.read_annotations)
            and returning a bool indicating whether those annotations satisfy
            the criteria
        invert_selection - bool flag indicating whether to select filter
            the selected positions in or out (True/False respectively)
    """
    selected_positions = []
    for position_name, position_annotations in annotations.items():
        if selection_criteria(position_annotations):
            selected_positions.append(position_name)
    
    if not invert_selection:
        return selected_positions
    else:
        return [position 
            for position in selected_positions 
            if position not in annotations.keys()]

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
    def selection_by_kw(position_annotations, selection_kws):
        global_annotations, timepoint_annotations = position_annotations
        return any([kw in global_annotations['notes'] for kw in selection_kws])
    
    return filter_positions(
        annotations, 
        lambda position_annotations: selection_by_kw(position_annotations, selection_kws), 
        invert_selection=invert_selection)

def check_complete_annotations(annotations, stages):
    """Check that a set of annotations are complete 
        
        Parameters
            annotations - An OrderedDict mapping position names to corresponding
                annotations (returned by load_data.read_annotations)
            stages - A iterable containing the stages that should be annotated
                for this experiment (e.g. could be ('larva','adult','dead')
                for a complete experiment, but only ('larva', 'adult') for 
                an ongoing experiment)
        Returns
            bad_positions - a list of positions with incomplete annotations
    """
    def check_for_stages(position_annotations, stages):
        global_annotations, timepoint_annotations = position_annotations
        return all([stage in timepoint_annotations.values() for stage in timepoint_annotations])
    
    return filter_positions(
        annotations,
        lambda position_annotations: check_for_stages(position_annotations, stages)
        invert_selection=invert_selection)
    
        

def print_formatted_list(string_list):
    return "[\'" + "\',\'".join(string_list) + "\']"

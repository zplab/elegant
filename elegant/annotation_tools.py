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
    
    Returns
        OrderedDict representing the subset of supplied positions satisfying
            selection criteria (i.e. mapping positions to annotations)
    """
    selected_positions = OrderedDict()
    for position_name, position_annotations in annotations.items():
        if selection_criteria(position_annotations):
            selected_positions[position_name] = position_annotations
    
    if invert_selection:
        selected_positions = OrderedDict([(position_name, position_annotations)
            for position_name, position_annotations in annotations.items()
            if position_name not in selected_positions])
    
    return selected_positions

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
    def check_kws(position_annotations, selection_kws):
        global_annotations, timepoint_annotations = position_annotations
        return any([kw in global_annotations['notes'] for kw in selection_kws])
    select_by_kw = lambda position_annotations: check_kws(position_annotations, selection_kws)
    
    return filter_positions(
        annotations, 
        select_by_kw, 
        invert_selection=invert_selection)

def filter_excluded(annotations):
    """Filter positions for an experiment based on exclusion flag in corresponding annotations

    Parameters
        annotations - An OrderedDict mapping position names to corresponding
            annotations (returned by load_data.read_annotations)
    """
    
    def select_excluded(position_annotations):
        global_annotations, timepoint_annotations = position_annotations
        return global_annotations['exclude'] is False
    
    return filter_positions(
        annotations,
        select_excluded)
    

def check_stage_annotations(annotations, stages):
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
        stage_annotations = [timepoint_annotation['stage'] 
            for timepoint_annotation in timepoint_annotations.values()]
        return all([stage in stage_annotations for stage in stages])
    select_by_stage_annotation = lambda position_annotations: check_for_stages(position_annotations, stages)
    
    return filter_positions(
        annotations,
        select_by_stage_annotation,
        invert_selection=True) # Get positions whose stages are not all annotated

def print_formatted_list(string_list):
    return "[\'" + "\',\'".join(string_list) + "\']"

#==================================
# How to load filtered positions into annotator
from elegant import load_data

# Wrapper around scan_experiment_directory
def filter_positions_for_annotation(experiment_dir, positions_to_load, **sed_kws):
    positions = load_data.scan_experiment_dir(experiment_dir, **sed_kws)
    return OrderedDict([(position_name, position_images) 
        for position_name, position_images in positions
        if position_name in positions_to_load])

# Doing things with a "function factory"
def build_timepoint_filter(base_func, **var_assignments)
    def wrapper(position_name, timepoint_name):
        return base_func(position_name, timepoint_name, **var_assignments)
    return wrapper

def timepoint_filter_byposition(position_name, timepoint_name, positions_to_load):
    return position_name in positions_to_load

'''
# To run with scan_experiment_directory
experiment_directory = ...
annotations = load_data.read_annotations(experiment_directory)
timepoint_filter = build_timepoint_filter(timepoint_filter_byposition,
    positions_to_load=filter_excluded(check_stage_annotations(annotations,['larva','adult','dead'])))
load_data.scan_experiment_directory(experiment_directory, timepoint_filter=timepoint_filter)
'''


from collections import OrderedDict

def filter_positions(annotations, selection_criteria):
    """Filter positions for an experiment based on defined selection criteria

    Parameters
        annotations - An OrderedDict mapping position names to annotations 
            (as returned by load_data.read_annotations)
        selection_criteria - A function taking in a set of position annotations 
            (as returned by load_data.read_annotations)and returning a bool 
            indicating whether those annotations satisfy the criteria; e.g.,
            
                def selection_criteria(position_annotations):
                    global_positions, timepoint_annotations = position_annotations
                    return global_positions['exclude'] is False
            
            See associated functions in this module for examples of how to 
                implement appropriate functions (esp. closures).
        invert_selection - bool flag indicating whether to select filter
            the selected positions in or out (True/False respectively)
    
    Returns
        OrderedDict of the subset of supplied positions satisfying selection 
            criteria (i.e. mapping positions to annotations)
    """
    selected_positions = OrderedDict()
    for position_name, position_annotations in annotations.items():
        if selection_criteria(position_annotations):
            selected_positions[position_name] = position_annotations
    
    return selected_positions

def filter_positions_by_kw(annotations, selection_kws, invert_selection=False):
    """Filter positions for an experiment based on the presence of kws
        in the 'notes' field of its annotations
    
        Parameters
            annotations - An OrderedDict mapping position names to corresponding
                annotations (returned by load_data.read_annotations)
            annotation_kws - Optional ist of str containing kws used to select positions
                (a given position's annotations must require ALL of the specified kws)
            invert_selection - bool flag indicating whether to select filter
                the selected positions in or out (True/False respectively)
        Returns
            OrderedDict of the subset of supplied positions satisfying selection 
                criteria (i.e. mapping positions to annotations)
    """
    
    # Create a suitable function to use with filter_positions using a closure
    def select_by_kw(position_annotations):
        global_annotations, timepoint_annotations = position_annotations
        return any([kw in global_annotations['notes'] for kw in selection_kws])
    
    return filter_positions(
        annotations, 
        select_by_kw, 
        invert_selection=invert_selection)

def filter_excluded(annotations):
    """Filter positions for an experiment based on exclusion flag in its
        annotations

        Parameters
            annotations - An OrderedDict mapping position names to annotations 
                (returned by load_data.read_annotations)
        
        Returns
            OrderedDict of the subset of supplied positions satisfying selection 
                criteria (i.e. mapping positions to annotations)
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
    
    # Create a suitable function to use with filter_positions using a closure
    def select_by_stage_annotation(position_annotations):
        global_annotations, timepoint_annotations = position_annotations
        stage_annotations = [timepoint_annotation['stage'] 
            for timepoint_annotation in timepoint_annotations.values()]
        return all([stage in stage_annotations for stage in stages])
    
    return filter_positions(
        annotations,
        select_by_stage_annotation,
        invert_selection=True) # Get positions whose stages are not all annotated

def build_position_filter(positions_to_load):
    """Generates a position filtering function that can be used with
        experiment_annotator.ExperimentAnnotator to load a subset of positions
        from an experiment
        
        Parameters
            positions_to_load - iterable of positions to select for within
                a set of annotations
        
        Returns
            filter_by_position - a function to be used with the timepoint_filter
                keyword parameter; this function is implemented as a closure - 
                a function in which some of its local variables are defined 
                ahead of time using prespecified values (in this case, using 
                positions_to_load; to use this function with the annotator,
                one can call it as in the following example syntax:
            
            experiment_directory = 'path/to/experiment'
            annotations = load_data.read_annotations(experiment_directory)
            positions_to_load = filter_excluded(annotations)
            timepoint_filter = build_position_filter(positions_to_load)
            load_data.scan_experiment_directory(experiment_directory, 
                timepoint_filter=timepoint_filter)
    """
    def filter_by_position(position_name, timepoint_name):
        return position_name in positions_to_load
    return filter_by_position


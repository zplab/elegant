import pickle
import pathlib
from collections import OrderedDict

def compile_annotations(annotation_dir):
    """Load all annotations in a specified directory
    
    Parameters:
        annotation_dir
    Returns
        OrderedDict mapping position names to dict of annotations"""
    
    annotations = OrderedDict()
    for position_fp in pathlib.Path(annotation_dir).iterdir():
        with position_fp.open('rb') as position_file:
            annotations[position_fp.stem] = pickle.load(position_file)
    return annotations

def filter_positions(annotations, good_annotation_kws=None, bad_annotation_kws=None,return_str=True):
    """Filter positions for an experiment based on notes in corresponding annotations
    
        Parameters
            annotations - An OrderedDict mapping position names to corresponding
                annotations (returned by compile_annotations)
            good_annotation_kws - List of str containing kws to screen in good positions
                (a given position's annotations must require ALL of the specified kws)
            bad_annotation_kws - List of str containing kws used to screen out bad positions
            return_str - bool dictating how to return skip positions; if True,
                return a formated str (i.e. for pasting into experiment_metadata's);
                if False, return a list
    """
    
    skip_positions = []
    for position_name, position_annotations in annotations.items():
        if ((good_annotation_kws is None or 
                all([kw in position_annotation['notes'] for kw in good_annotation_kws])) and
            (bad_annotation_kws is None or
                any([kw in position_annotation['notes'] for kw in bad_annotation_kws]))):
            skip_positions.append(position_name)
    if return_str:
        return "[\'" + "\',\'".join(skip_positions) + "\']"
    else
        return skip_positions

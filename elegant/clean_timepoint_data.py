import pathlib
import shutil
import json
import datetime

import numpy

from zplib import datafile
from elegant import load_data

def remove_timepoint_for_position(experiment_root, position, timepoint, dry_run=False):
    """Removes all memory of a given timepoint for a given position (files, metadata, annotations entries)

    Parameters
        experiment_root - str/pathlib.Path pointing to experiment directory
        position - str for position to modify
        timepoint - str with standard format for timepoint acquisition (YYYY-MM-DDtHHMM)
        dry_run - bool flag that toggles taking action (if False, only specifies when offending files are found)

    Example Usage

        experiment_root = /path/to/experiment
        position = '000'
        timepoint = '2018-07-30t1200'
        dry_run = True  # No deleting; just see what happens
        remove_timepoint_for_position(experiment_root, position, timepoint, dry_run=dry_run)
    """

    experiment_root = pathlib.Path(experiment_root)

    offending_files = [img_file for img_file in (experiment_root / position).iterdir() if timepoint in str(img_file.name)]
    if len(offending_files) > 0:
        print(f'Found offending files for position {position}: f{offending_files}')
        if not dry_run:
            [img_file.unlink() for img_file in offending_files]

    md_file = (experiment_root / position /'position_metadata.json')
    with md_file.open() as md_fp:
        pos_md = json.load(md_fp)
    has_bad_timepoint = any([timepoint_info['timepoint'] == timepoint for timepoint_info in pos_md])
    if has_bad_timepoint:
        print('Found offending entry in position_metadata for: '+str(position))
        if not dry_run:
            pos_md = [timepoint_data for timepoint_data in pos_md if timepoint_data['timepoints'] != timepoint]
            datafile.json_encode_atomic_legible_to_file(pos_md, experiment_root / position / 'position_metadata.json')   # Write out new position_metadata

    position_annotation_file = experiment_root / 'annotations' / f'{position}.pickle'
    if position_annotation_file.exists():
        general_annotations,timepoint_annotations = load_data.read_annotation_file(position_annotation_file)

        if timepoint in timepoint_annotations:
            print(f'Found offending entry in annotation for position {position}')
            if not dry_run:
                del timepoint_annotations[timepoint]
                load_data.write_annotation_file(position_annotation_file,
                    general_annotations, timepoint_annotations)

def remove_timepoint_from_experiment(experiment_root, timepoint, dry_run=False):
    """Removes information about a timepoint from all positions in an experiment (as well as experiment_metadata)

    This function is useful for when a data for an entire timepoint is totally unusable
        (e.g. metering fails hard, one or more files get fatally overwritten).

    Parameters
        experiment_root - str/pathlib.Path pointing to experiment directory
        timepoint - str with standard format for timepoint acquisition (YYYY-MM-DDtHHMM)
        dry_run - bool flag that toggles taking action (if False, only specifies when offending files are found);
            this flag is exposed for corresponding functionality through remove_timepoint_for_position

    Example Usage
        experiment_root = /path/to/experiment
        timepoint = '2018-07-30t1200'
        remove_timepoint_for_position(experiment_root, position, timepoint)
    """

    experiment_root = pathlib.Path(experiment_root)
    positions = [metadata_file.parent for metadata_file in sorted(experiment_root.glob('*/position_metadata.json'))]

    for position in positions:
        remove_timepoint_for_position(experiment_root, position, timepoint, dry_run=dry_run)

    # Handle experiment_metadata
    # Not sure if this is just unnecessary tidying up, since expt_metadata isn't explicitly used for anything
    # in the new codebase, but...
    md_file = (experiment_root/'experiment_metadata.json')
    with md_file.open() as md_fp:
        expt_md = json.load(md_fp)

    try:
        timepoint_idx = expt_md['timepoints'].index(timepoint)
    except ValueError:  # Offending timepoint didn't make it into metadata
        print('Timepoint not found in experiment_metadata')
        return

    for list_entry_type in ['durations','timestamps','timepoints']:
        del expt_md[list_entry_type][timepoint_idx]

    for dict_entry_type in ['brightfield metering', 'fluorescent metering', 'humidity', 'temperature']:
        expt_md[dict_entry_type] = {key:val for key,val in expt_md[dict_entry_type].items() if key != timepoint}

    datafile.json_encode_atomic_legible_to_file(expt_md,experiment_root/'experiment_metadata.json')

def remove_excluded_positions(experiment_root,dry_run=False):
    """Deletes excluded positions from an experiment directory

    Parameters
        experiment_root - str/pathlib.Path to experiment
        dry_run - bool flag that toggles taking action (if False, only specifies when offending files are found)
    """

    experiment_root = pathlib.Path(experiment_root)
    annotations = load_data.read_annotations(experiment_root)
    good_annotations = load_data.filter_annotations(annotations, load_data.filter_excluded)
    excluded_positions = sorted(set(annotations.keys()).difference(set(good_annotations.keys())))
    for position in excluded_positions:
        if (experiment_root / position).exists():
            print(f'Found an excluded position to delete {position}')
            if not dry_run:
                shutil.rmtree(str(experiment_root / position))

def remove_dead_timepoints(experiment_root, postmortem_timepoints, dry_run=False):
    """Deletes excess timepoints in an experiment where worms are dead

    Parameters
        experiment_root - str/pathlib.Path to experiment
        postmortem_time - Number of timepoints to keep past the annotated death timepoint;
            useful for keeping extra timepoints in case one ever wants to validate
            death for the previously made annotations; postmortem_timepoints should be an int >= 1
        dry_run - bool flag that toggles taking action (if False, only specifies when offending files are found);
            this flag is exposed for corresponding functionality through remove_timepoint_for_position
    """

    if postmortem_timepoints < 1:
        raise ValueError('postmortem_timepoints should be >= 1')
    elif type(postmortem_timepoints) is not int:
        raise ValueError('postmortem_timepoints must be an integer')

    experiment_root = pathlib.Path(experiment_root)
    annotations = load_data.read_annotations(experiment_root)
    good_annotations = load_data.filter_annotations(annotations, load_data.filter_excluded)

    for position, position_annotations in good_annotations.items():
        general_annotations, timepoint_annotations = position_annotations
        timepoint_stages = [timepoint_info['stage'] for timepoint_info in timepoint_annotations.values()]
        death_timepoint_index = timepoint_stages.index('dead')

        for timepoint_num, timepoint in enumerate(timepoint_annotations):
            timepoints_after_death = timepoint_num - death_timepoint_index # positive values for timepoints after death
            if timepoints_after_death > postmortem_timepoints:
                remove_timepoint_for_position(experiment_root, position, timepoint, dry_run=dry_run)

import pathlib
import shutil
import json
import datetime

import numpy as np

from zplib import datafile
from elegant import load_data

def remove_timepoint(experiment_root, position, timepoint, dry_run=False):
    '''Removes all memory of a given timepoint for a given position (files, metadata, annotations entries)

    Parameters
        experiment_root - str/pathlib.Path pointing to experiment directory
        position - str for position to modify
        timepoint - str with standard format for timepoint acquisition (YYYY-MM-DDtHHMM)
        dry_run - bool flag that toggles taking action (if False, only specifies when offending files are found)
    '''

    if type(experiment_root) is str: experiment_root = pathlib.Path(experiment_root)

    [im_file for img_file in (experiment_root / position).iterdir() if timepoint in str(im_file.name)]
    if len(offending_files) > 0:
        print('Found offending files for: '+ str(position))
        if not dry_run:
            [im_file.unlink() for im_file in offending_files]

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

def purge_timepoint_from_experiment(experiment_root, timepoint, dry_run=False):
    '''Removes information about a timepoint from all positions in an experiment (as well as experiment_metadata)

    This function is useful for when a data for an entire timepoint is totally unusable
        (e.g. metering fails hard, one or more files get fatally overwritten).

    Parameters
        experiment_root - str/pathlib.Path pointing to experiment directory
        timepoint - str with standard format for timepoint acquisition (YYYY-MM-DDtHHMM)
        dry_run - bool flag that toggles taking action (if False, only specifies when offending files are found)
    '''

    positions = [sub_dir.name for sub_dir in sorted(experiment_root.iterdir()) if sub_dir.name.isnumeric()]
    for position in positions:
        remove_timepoint(experiment_root, position, timepoint, dry_run=dry_run)

    # Handle experiment_metadata
    # Not sure if this is just unnecessary tidying up, since expt_metadata isn't explicitly used for anything
    # in the new codebase, but...
    md_file = (experiment_root/'experiment_metadata.json')
    with md_file.open() as md_fp:
        expt_md = json.load(md_fp)

    try:
        timepoint_idx = expt_md['timepoints'].index(timepoint)
        for list_entry_type in ['durations','timestamps','timepoints']
            del expt_md[list_entry_type][timepoint_idx]

        for dict_entry_type in ['brightfield_metering', 'fluorescent_metering', 'humidity', 'temperature']:
            expt_md[dict_entry_type] = {key:val for key,val in expt_md[dict_entry_type] if key != timepoint}

        datafile.json_encode_atomic_legible_to_file(expt_md,experiment_root/'experiment_metadata.json')
    except:  # Offending timepoint didn't make it into metadata
        pass

def remove_dead_timepoints(experiment_root, postmortem_time, delete_excluded=False):
    '''Deletes excess timepoints in an experiment where worms are dead

    Parameters
        experiment_root - str/pathlib.Path to experiment
        postmortem_time - Number of hours of data to keep past the annotated death timepoint;
            useful for keeping extra timepoints in case one ever wants to validate
            death for the previously made annotations
        delete_excluded - bool flag for whether to delete excluded positions; if True
            deletes folders for each position, but keeps the relevant annotation as a
            record of what happened at that position
    '''

    experiment_root = pathlib.Path(experiment_root)
    annotations = load_data.read_annotations(experiment_root)
    good_annotations = load_data.filter_annotations(annotations, load_data.filter_excluded)

    if delete_excluded:
        excluded_positions = sorted(set(annotations.keys()).difference(set(good_annotations.keys())))
        for position in excluded_positions:
            if (experiment_root / position).exists():
                shutil.rmtree(str(experiment_root / position))

    for position, position_annotations in good_annotations.items():
        general_annotations, timepoint_annotations = position_annotations
        timepoint_keys = list(timepoint_annotations.keys())
        timepoint_stages = [timepoint_info['stage'] for timepoint_info in timepoint_annotations.values()]
        death_timepoint = timepoint_keys[timepoint_stages.index('dead')]

        for timepoint in timepoint_keys:
            time_since_death = (_extract_datetime_fromstr(timepoint_label) - _extract_datetime_fromstr(death_timepoint))/3600
            if time_since_death > postmortem_time:
                remove_timepoint(experiment_root, position, timepoint)

def _extract_datetime_fromstr(time_str):
    '''Converts standard experimental timepoint string to time representation (seconds since epoch)'''
    return datetime.datetime.strptime(time_str,'%Y-%m-%dt%H%M').timestamp()

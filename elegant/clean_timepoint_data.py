import pathlib
import shutil
import json
import datetime

import numpy

from zplib import datafile

from . import load_data

def remove_timepoint_for_position(experiment_root, position, timepoint, dry_run=False):
    """Removes all memory of a given timepoint for a given position (files, metadata, annotations entries)

    Parameters:
        experiment_root: str/pathlib.Path pointing to experiment directory
        position: str for position to modify
        timepoint: str with standard format for timepoint acquisition (YYYY-MM-DDtHHMM)
        dry_run: bool flag that toggles taking action (if False, only specifies when offending files are found)

    Example Usage:
        experiment_root = /path/to/experiment
        position = '000'
        timepoint = '2018-07-30t1200'
        dry_run = True  # No deleting; just see what happens
        remove_timepoint_for_position(experiment_root, position, timepoint, dry_run=dry_run)
    """

    experiment_root = pathlib.Path(experiment_root)

    files_to_remove = [img_file for img_file in (experiment_root / position).iterdir() if img_file.name.startswith(timepoint)]
    if len(files_to_remove) > 0:
        print(f'Found files for removal for position {position}: {files_to_remove}')
        if not dry_run:
            [img_file.unlink() for img_file in files_to_remove]

    md_file = (experiment_root / position /'position_metadata.json')
    with md_file.open() as md_fp:
        pos_md = json.load(md_fp)

    acquired_timepoints = [timepoint_info['timepoint'] for timepoint_info in pos_md]
    try:
        offending_timepoint_index = acquired_timepoints.index(timepoint)
        print(f'Found entry for removal in position_metadata for position {position}')
        if not dry_run:
            del pos_md[offending_timepoint_index]
            datafile.json_encode_atomic_legible_to_file(pos_md, md_file)   # Write out new position_metadata
    except ValueError:  # bad timepoint wasn't found
        pass

    position_annotation_file = experiment_root / 'annotations' / f'{position}.pickle'
    if position_annotation_file.exists():
        general_annotations,timepoint_annotations = load_data.read_annotation_file(position_annotation_file)

        if timepoint in timepoint_annotations:
            print(f'Found entry for removal in annotation for position {position}')
            if not dry_run:
                del timepoint_annotations[timepoint]
                load_data.write_annotation_file(position_annotation_file,
                    general_annotations, timepoint_annotations)

def remove_timepoint_from_experiment(experiment_root, timepoint, dry_run=False):
    """Removes information about a timepoint from all positions in an experiment (as well as experiment_metadata)

    This function is useful for when a data for an entire timepoint is totally unusable
        (e.g. metering fails hard, one or more files get fatally overwritten).

    Parameters:
        experiment_root: str/pathlib.Path pointing to experiment directory
        timepoint: str with standard format for timepoint acquisition (YYYY-MM-DDtHHMM)
        dry_run: bool flag that toggles taking action (if False, only specifies when offending files are found);
            this flag is exposed for corresponding functionality through remove_timepoint_for_position

    Example Usage:
        experiment_root = /path/to/experiment
        timepoint = '2018-07-30t1200'
        remove_timepoint_for_position(experiment_root, position, timepoint)
    """

    experiment_root = pathlib.Path(experiment_root)
    positions = [metadata_file.parent for metadata_file in sorted(experiment_root.glob('*/position_metadata.json'))]

    for position in positions:
        remove_timepoint_for_position(experiment_root, position, timepoint, dry_run=dry_run)

    md_file = (experiment_root/'experiment_metadata.json')
    with md_file.open() as md_fp:
        expt_md = json.load(md_fp)

    try:
        timepoint_idx = expt_md['timepoints'].index(timepoint)
    except ValueError:
        print('Timepoint not found in experiment_metadata')
        return

    for list_entry_type in ['durations','timestamps','timepoints']:
        del expt_md[list_entry_type][timepoint_idx]

    for dict_entry_type in ['brightfield metering', 'fluorescent metering', 'humidity', 'temperature']:
        del expt_md[dict_entry_type][timepoint]

    datafile.json_encode_atomic_legible_to_file(expt_md, md_file)

def remove_excluded_positions(experiment_root,dry_run=False):
    """Deletes excluded positions from an experiment directory

    This function deletes position folders from the specified experiment directory,
    but saves position_metadata and annotations into the 'excluded_positions' subfolder.
    The 'excluded_positions' subfolder has parallel structure to the standard experiment_root,
    with one 'annotations' folder for annotations and each position having its own folder containing metadata.

    Parameters:
        experiment_root: str/pathlib.Path to experiment
        dry_run: bool flag that toggles taking action (if False, only specifies when offending files are found)
    """

    experiment_root = pathlib.Path(experiment_root)
    annotations = load_data.read_annotations(experiment_root)
    good_annotations = load_data.filter_annotations(annotations, load_data.filter_excluded)
    excluded_positions = sorted(set(annotations.keys()).difference(set(good_annotations.keys())))

    for position in excluded_positions:
        if (experiment_root / position).exists():
            print(f'Found an excluded position to delete {position}')
            if not dry_run:
                (experiment_root / 'excluded_positions' / position).mkdir(parents=True,exist_ok=True)
                (experiment_root / 'excluded_positions' / 'annotations').mkdir(parents=True,exist_ok=True)

                shutil.copy(str(experiment_root / position / 'position_metadata.json'),
                    str(experiment_root / 'excluded_positions' / position / 'position_metadata.json'))
                shutil.copy(str(experiment_root / 'annotations' / f'{position}.pickle'),
                    str(experiment_root / 'excluded_positions' / 'annotations' / f'{position}.pickle'))
                shutil.copy(str(experiment_root /  'experiment_metadata.json'),
                    str(experiment_root / 'excluded_positions' / 'experiment_metadata_old.json')) # Back this up JIC....

                shutil.rmtree(str(experiment_root / position))
                (experiment_root / 'annotations' / f'{position}.pickle').unlink()

                # Load/save atomically for each position to minimize the chance of failing oneself into a bad state
                with (experiment_root /  'experiment_metadata.json').open('r') as md_file:
                    expt_md = json.load(md_file)
                del expt_md['positions'][position]
                datafile.json_encode_atomic_legible_to_file(expt_md, md_file)


def remove_dead_timepoints(experiment_root, postmortem_timepoints, dry_run=False):
    """Deletes excess timepoints in an experiment where worms are dead

    Parameters:
        experiment_root: str/pathlib.Path to experiment
        postmortem_time: Number of timepoints to keep past the annotated death timepoint;
            useful for keeping extra timepoints in case one ever wants to validate
            death for the previously made annotations; postmortem_timepoints should be an int >= 1
        dry_run: bool flag that toggles taking action (if False, only specifies when offending files are found);
            this flag is exposed for corresponding functionality through remove_timepoint_for_position
    """

    if postmortem_timepoints < 1:
        raise ValueError('postmortem_timepoints should be >= 1')

    experiment_root = pathlib.Path(experiment_root)
    annotations = load_data.read_annotations(experiment_root)
    good_annotations = load_data.filter_annotations(annotations, load_data.filter_excluded)

    for position, position_annotations in good_annotations.items():
        general_annotations, timepoint_annotations = position_annotations
        timepoint_stages = [timepoint_info['stage'] for timepoint_info in timepoint_annotations.values()]
        try:
            death_timepoint_index = timepoint_stages.index('dead')
        except ValueError:
            print(f'No death timepoint found for position {position}; skipping position.')
            continue

        for timepoint_num, timepoint in enumerate(timepoint_annotations):
            timepoints_after_death = timepoint_num - death_timepoint_index # positive values for timepoints after death
            if timepoints_after_death > postmortem_timepoints:
                remove_timepoint_for_position(experiment_root, position, timepoint, dry_run=dry_run)

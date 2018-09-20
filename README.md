# elegant
tools and pipelines for zplab C. elegans data

These tools assume the following directory structure for experiments:

```
experiment_root/
    position_0/
    ...
    position_n/ 
    annotations/
        position_0.pickle
        ...
        position_n.pickle
    derived_data/
        data_class_1/
            position_0/
            ...
            position_n/
        data_class_2/
            position_0.xyz
            ...
            position_n.xyz
```
The position directories can be identified by the presence of a `position_metadata.json` file in them. If an `experiment_metadata.json` file exists, its contents can also be used to find the position names, but the former is easier. The names of the position directories are arbitrary. We've been using number-strings, but don't assume that.

The annotations directory contains a pickle file for each position, or perhaps no file. Don't manipulate this directly; use the tools in elegant.load_data to read, write, filter, and merge annotation dictionaries.

The derived_data directory will contain anything else written out by manual or automated tools. Each distinct type of output data should get its own subdirectory, with output organized by position name within the subdirectory. If a class of output data requires different files for each timepoint output (e.g. making worm masks), then there should be a subdirectory for each position, within which the files will be organized by timepoint just as in the position directories. Alternatively, if a class of output data is just a summary of the whole position, individually-named position data files should be used.

In particular, tools for converting brightfield images to masks should save the output as `experiment_root/derived_data/masks/{position}/{timepoint}.png`. (If you manually edit the masks, you may choose to save the original as a **new class** of data in `experiment_root/derived_data/original_masks/{position}/{timepoint}.png`. The same pattern should hold for any manually-changed data. But don't save it side-by-side the old data with some different prefix or suffix. This could confuse code written to traverse these directories.)

Overall, the workflow is:
    1. set up experiment on scope
    2. as experiment is running, use the annotator to mark positions as "exclude" if they're bad, and to annotate when worms are dead (in either case, the scope will stop acquiring images)
    3. run `elegant.process_data.update_annotations()` and then perform manual annotations of the data.
    4. run `elegant.process_data.measure_worms()` to make measurements of interest (can work locally or on cluster)
    5. run `elegant.process_data.collate_data()` to produce authoritative files of all the measurements.
    6. load these files with `elegant.worm_data.read_worms()` to begin interactive data analysis.

Here is simple example for starting up an annotator with a few standard annotation types:

    from ris_widget import ris_widget
    from elegant.gui import pose_annotation
    from elegant.gui import stage_field
    from elegant.gui import timepoint_annotations
    from elegant.gui import keypoint_annotation
    from elegant import load_data

    exp_root = '/path/to/data'
    
    rw = ris_widget.RisWidget()
    metadata = load_data.read_metadata(exp_root)
    pa = pose_annotation.PoseAnnotation.from_experiment_metadata(rw, metadata)
    st = stage_field.StageField()
    ta = timepoint_annotations.TimepointAnnotations()
    ka = keypoint_annotation.KeypointAnnotation(rw.alt_view, ['pharynx'], center_y_origin=True, auto_advance=True)
    positions = load_data.scan_experiment_dir(exp_root)
    from elegant.gui import experiment_annotator
    ea = experiment_annotator.ExperimentAnnotator(rw, exp_root, positions, [pa, st, ta, ka])

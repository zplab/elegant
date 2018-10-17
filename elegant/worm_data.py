# This code is licensed under the MIT License (see LICENSE file for details)

import numpy
import glob
import pathlib
import collections

from zplib.scalar_stats import moving_mean_std
from zplib.scalar_stats import regress
from zplib.image import colorize
from zplib import datafile

TIME_UNITS = dict(days=24, hours=1, minutes=1/60, seconds=1/3600)

def read_worms(*path_globs, name_prefix='', delimiter='\t', summary_globs=None,
    calculate_lifespan=True, last_timepoint_is_first_dead=True, time_units='hours'):
    """Read worm data files into a Worms object.

    Each file can be either a single-worm file (where the "timepoint" identifier
    is the first column), or a multi-worm file (which has timepoint data for
    multiple worms concatenated, and has "name" as the column to disambiguate
    each worm).

    In the case of a single-worm file, the name of the worm is the name of the
    input file (minus any file extension), with an optional prefix prepended.
    In the case of a multi-worm file, the name of the worm is given in the
    "name" column, again optionally prepended by a prefix.

    The prefix is useful to distinguish animals from different experimental runs
    or genotypes, &c.

    Each data file must have at a minimum a "timepoint" column and an "age" column.
    For single-worm files, the "timepoint" column need not be labeled in the
    header, as it is assumed to be the first column. Multi-worm files must have
    the first column labeled "name" or "worm", and a labeled "timepoint" column
    elsewhere.

    Summary files containing worm-level data (i.e. lifespan &c., as opposed to
    the timepoint-level annotations in the worm files) can also be provided.
    If no lifespan data is present from a summary file, it will be calculated
    if requested. If 'timestamp' and 'stage' columns are present, the various
    ages (age, adult_age, ghost_age) and spans (lifespan, adultspan, &c.) will
    be calculated therefrom. Failing that, if an 'age' column is available,
    the lifespan will be calculated from that, as controlled by the
    'last_timepoint_is_first_dead' paramter.

    Parameters:
        *path_globs: all non-keyword arguments are treated as paths to worm data
            files, or as glob expressions matching multiple such data files.
        name_prefix: if a string, this will be used as the prefix (see above).
            If a function, it will be called with the pathlib.Path of the input
            file in question, so that the file name and/or name of the parent
            directories can be used to compute the prefix.
        delimiter: controls whether the data are assumed to be tab, comma, or
            otherwise delimited.
        summary_globs: if not None, either a single glob expression (or file path)
            or a list of same. These will be interpreted as summary data files
            to load (i.e. files of worm-level data, such as lifespan &c.,
            rather than timepoint-level annotations).
        calculate_lifespan: if True and if no lifespan data is present, then
            calculate lifespans (and other spans potentially), as described.
        last_timepoint_is_first_dead: if True, the last timepoint in the data
            file is assumed to represent the first time the worm was annotated
            as dead. Otherwise, the last timepoint is assumed to represent the
            last time the worm was known to be alive. Only used when lifespans
            are auto-calculated and when there are no 'timestamp' and 'stage'
            columns.
        time_units: the unit of time for the values in the "age" column and any
            column ending in "_age". This unit will also be used for summary
            columns ending in "span". All time units will be converted into
            hours internally. (The rescale_time() method can be used to
            change units later, but this is at your own risk: all time-based
            analysis code can and should assume time units of hours.)
            Must be one of 'days', 'hours', 'minutes', or 'seconds'.

    Returns: Worms object, sorted by lifespan (if possible)

    Examples:
        worms = read_worms('/path/to/spe-9/datafiles/*.csv', name_prefix='spe-9 ', delimiter=',')

        def get_genotype(path):
            # assume that the grandparent path contains the genotype
            return path.parent.parent.name + ' '
        worms = read_worms('/path/to/*/datafiles/*.tsv', name_prefix=get_genotype)
    """
    worms = Worms()
    for path_glob in path_globs:
        paths = glob.glob(str(path_glob), recursive=True)
        if len(paths) == 0:
            raise FileNotFoundError(f'"{path_glob}" matches no files.')
        for path in map(pathlib.Path, paths):
            if callable(name_prefix):
                prefix = name_prefix(path)
            else:
                prefix = name_prefix
            for name, header, data in _read_datafile(path, prefix, delimiter):
                worms.append(Worm(name, header, data, time_units))
    if isinstance(summary_globs, (str, pathlib.Path)):
        summary_globs = [summary_globs]
    elif summary_globs is None:
        summary_globs = []
    for path_glob in summary_globs:
        paths = glob.glob(str(path_glob), recursive=True)
        if len(paths) == 0:
            raise FileNotFoundError(f'"{path_glob}" matches no files.')
        for path in paths:
            worms.read_summary_data(path, delimiter=delimiter, time_units=time_units)

    if calculate_lifespan and not all(hasattr(w, 'lifespan') for w in worms):
        if all(hasattr(w.td, 'timepoint') and hasattr(w.td, 'stage') for w in worms):
            for w in worms:
                try:
                    w.calculate_ages_and_spans()
                except (NameError, ValueError):
                    print(f'could not calculate lifespan for worm {w.name}')
        elif all(hasattr(w.td, 'age') for w in worms):
            for w in worms:
                try:
                    w.calculate_lifespan_simple(last_timepoint_is_first_dead)
                except (NameError, ValueError):
                    print(f'could not calculate lifespan for worm {w.name}')
    if all(hasattr(w, 'lifespan') for w in worms):
        worms.sort('lifespan')
    return worms

def _read_datafile(path, prefix, delimiter):
    """Iterate over a single- or multi-worm datafile, yielding (name, header, data)
    triplets corresponding to each worm in the file."""
    header, data = datafile.read_delimited(path, delimiter=delimiter)
    header = [colname.replace(' ', '_') for colname in header]
    is_multi_worm = header[0] in ('name', 'worm')
    if not is_multi_worm:
        name = prefix + path.stem
        header[0] = 'timepoint'
        yield name, header, data
    else:
        header[1] = 'timepoint'
        worm_rows = []
        current_name = None
        for name, *row in data:
            if current_name is None:
                current_name = name
            if current_name != name:
                yield prefix + current_name, header[1:], worm_rows
                current_name = name
                worm_rows = []
            worm_rows.append(row)
        yield prefix + current_name, header[1:], worm_rows

def meta_worms(grouped_worms, *features, age_feature='age', summary_features=('lifespan',), min_age=-numpy.inf, max_age=numpy.inf, smooth=0.4):
    """Calculate average trends across groups of worms, returning a new Worms object.

    Given a set of timecourse features and a set of worms grouped by some criterion,
    calculate the average trajectory of each given feature across all group members,
    using LOWESS smoothing (as implemented in zplib.scalar_stats.moving_mean_std.moving_mean)

    Each produced meta "worm" will have the averaged timecourse features, the
    "name" of the group, the average of any specified summary features (such as
    "lifespan"), and attributes "worms" and "n" that list the underlying worms
    and the count of the same.

    Parameters:
        grouped_worms: dictionary mapping from group names to Worms objects of
            the worms in each group. The group name will be the "name" attribute
            of each "meta worm" produced. Such a grouped_worms dict can be
            produced from the Worms.bin or Worms.group_by functions.
        *features: one or more feature names or callable functions which provide
            features for which time averages should be calculated.
        age_feature: feature to use for the "age" axis of the average trajectories.
            Generally "age" is right, but trajectories could be centered on
            time of death using a "ghost_age" feature that counts down to zero
            at the time of death. This parameter may be a callable function or
            a feature name.
        summary_features: names of summary feature to average across each of the
            underlying worms.
        min_age, max_age: the beginning and end of the window to analyze.
            If not specified, features from the very beginning and/or to the
            very end of the timecourse will be retrieved.
        smooth: smoothing parameter passed to LOWESS moving_mean function.
            Represents the fraction of input data points that will be smoothed
            across at each output data point.

    Returns:
        Worms object, sorted by group name.

    Example:
    lifespan_bins = worms.bin('lifespan', nbins=5)
    averaged_worms = meta_worms(lifespan_bins, 'gfp_95th')
    averaged_worms.plot_timecourse('gfp_95th')

    """
    meta_worms = Worms()
    for group_name, worms in sorted(grouped_worms.items()):
        meta_worm = Worm(group_name)
        meta_worm.n = len(worms)
        meta_worm.worms = worms
        ages, data = worms.get_timecourse_features(*features, min_age=min_age, max_age=max_age, age_feature=age_feature, filter_valid=False)
        ages_out = numpy.linspace(ages.min(), ages.max(), 100)
        meta_worm.td.age = ages_out
        for feature, feature_data in zip(features, data.T):
            mask = numpy.isfinite(feature_data)
            feature_ages = ages[mask]
            feature_data = feature_data[mask]
            trend = moving_mean_std.moving_mean(feature_ages, feature_data, points_out=ages_out, smooth=smooth, iters=1)[1]
            setattr(meta_worm.td, feature, trend)
        for feature in summary_features:
            setattr(meta_worm, feature, worms.get_feature(feature).mean())
        setattr(meta_worm.td, age_feature, ages_out) # the same for all loops; just set once
        meta_worms.append(meta_worm)
    return meta_worms

def gaussian_filter(ages, age_to_smooth, sigma, window_size=numpy.inf):
    """Gaussian-weighted smoothing filter (Causal: does not use future data).

    The filter returns the weights for each entry in the ages input, for the
    given age_to_smooth.

    Parameters:
        ages: ages of the worm at each timepoint
        age_to_smooth: age for which to calculate the smoothing weights.
        sigma: temporal sigma for smoothing.
        window_size: if specified, the number of hours in the past that should
            be used in the weighting at all. (I.e. if specified, the result will
            be a truncated gaussian weighting.)
    """
    time_delta = ages - age_to_smooth
    weights = numpy.exp(-time_delta**2/(2*sigma))
    ignore = (time_delta > 0) | (time_delta < -window_size)
    weights[ignore] = 0
    return weights

def uniform_filter(ages, age_to_smooth, window_size):
    """Uniform-weighted smoothing filter (Causal: does not use future data).

    The filter returns a uniform (boxcar) filter over a specified range of past
    data. That is, the smoothed value will simply be the (unweighted) average of
    all the data point within the given window.

    Parameters:
        ages: ages of the worm at each timepoint
        age_to_smooth: age for which to calculate the smoothing weights.
        window_size: The number of hours in the past that should be averaged over.
    """
    time_delta = ages - age_to_smooth
    return (time_delta <= 0) & (time_delta >= -window_size)

class Worm(object):
    """Object for storing data pertaining to an indiviual worm's life.

    The object has an attribute, 'worm.td' (for "timecourse data") to contain
    measurements made at each timepoint. These include at a minimum "timepoint"
    (the string identifier for each timepoint, generally a timestamp) and "age".

    Other attributes, such as worm.lifespan and worm.name represent data or
    summary statistics valid over the worm's entire life.

    Convenience accessor functions for getting a range of timecourse measurements
    are provided.
    """
    def __init__(self, name, feature_names=[], timepoint_data=[], time_units='hours'):
        """It is generally preferable to construct worms from a factory function
        such as read_worms or meta_worms, rather than using the constructor.

        Parameters:
            name: identifier for this individual animal
            feature_names: names of the timecourse features measured for
                this animal
            timepoint_data: for each timepoint, each of the measured features.
            time_units: the unit of time for the values in the "age" column and
                any column ending in "_age". This unit will also be used for
                summary columns ending in "span". All time units will be
                converted into hours internally. (The rescale_time() method
                can be used to change units later, but this is at your own
                risk: all time-based analysis code can and should assume time
                units of hours.)
                Must be one of 'days', 'hours', 'minutes', or 'seconds'.

        """
        self.name = name
        self.td = _TimecourseData()
        vals_for_features = [[] for _ in feature_names]
        if time_units not in TIME_UNITS:
            raise ValueError(f"'time_units' must be one of: {list(TIME_UNITS)}")
        time_scale = TIME_UNITS[time_units]
        for timepoint in timepoint_data:
            # add each entry in the timepoint data to the corresponding list in
            # vals_for_features
            for feature_vals, item in zip(vals_for_features, timepoint):
                if item is None:
                    item = numpy.nan
                feature_vals.append(item)
        for feature_name, feature_vals in zip(feature_names, vals_for_features):
            arr = numpy.array(feature_vals)
            if feature_name == 'age' or feature_name.endswith('_age'):
                arr *= time_scale
            setattr(self.td, feature_name, arr)

    def rescale_time(self, time_scale):
        """Rescale all timecourse and summary features by a given factor.

        The "age" timecourse feature and all others ending in "_age", and any
        summary feature ending in "span" will be multiplied by the provided
        time_scale parameter.

        Use carefully: all functions that care about the absolute time scale
        will assume that time is scaled in hours.
        """
        for feature_name, feature_vals in self.td._items():
            if feature_name == 'age' or feature_name.endswith('_age'):
                setattr(self.td, feature_name, numpy.asarray(feature_vals) * time_scale)
        for feature_name, feature_val in self.__dict__.items():
            if feature_name.endswith('span'):
                setattr(self, feature_name, feature_val * time_scale)

    def calculate_lifespan_simple(self, last_timepoint_is_first_dead=True):
        """Calculate the lifespan of each animal using a simplistic method.

        Parameters:
            last_timepoint_is_first_dead: if True, the last timepoint in the
                data file is assumed to represent the first time the worm was
                annotated as dead. The death time is assumed to be halfway
                between the last-alive and first-dead time.
                Otherwise, the last timepoint is assumed to represent the last
                time the worm was known to be alive. In this case, the last
                sample interval is calculated, and the death time is assumed to
                be half a sample interval after the last-alive time.
        """
        if last_timepoint_is_first_dead:
            self.lifespan = self.td.age[-2:].mean() # midpoint between last-alive and first-dead
        else:
            self.lifespan = self.td.age[-1] + (self.td.age[-1] - self.td.age[-2])/2 # halfway through the next interval, assuming equal intervals

    def calculate_ages_and_spans(self):
        """Calculate ages and spans (in hours) from annotated stage data.

        Requires 'stage' and 'timestamp' timecourse data fields. Calculates the
        following timecourse data:
            age
            adult_age
            ghost_age

        Calculates the following summary data:
            lifespan
            [xxx]_span for each non-egg, non-dead stage
        """
        hours = (self.td.timestamp - self.td.timestamp[0]) / 3600
        stages, indices = numpy.unique(self.td.stage, return_index=True)
        order = indices.argsort()
        stages = stages[order]
        indices = indices[order]
        if stages[0] == 'egg':
            # had an egg-visible timepoint: assume hatch was halfway between the
            # last egg-seen time and first larva-seen time.
            hatched_i = indices[1]
            hatch_time = hours[hatched_i-1:hatched_i+1].mean()
            stages = stages[1:]
            indices = indices[1:]
        else:
            # no egg timepoint. Best guess for hatch time is the first larva-seen time.
            hatch_time = hours[0]
        self.td.age = hours - hatch_time
        transition_times = [hatch_time] + [hours[i-1:i+1].mean() for i in indices[1:]]
        transition_times = numpy.array(transition_times)
        spans = transition_times[1:] - transition_times[:-1]
        for stage, span in zip(stages[:-1], spans):
            setattr(self, f'{stage}span', span)
        if stages[-1] != 'dead':
            raise ValueError('No timepoint with "dead" label is present; cannot calculate lifespan.')
        death_time = transition_times[-1]
        self.td.ghost_age = hours - death_time
        self.lifespan = death_time - hatch_time
        try:
            adult_i = list(stages).index('adult')
        except ValueError:
            raise ValueError('No timepoint with "adult" label is present; cannot calculate adult_age.')
        adult_time = transition_times[adult_i]
        self.td.adult_age = hours - adult_time

    def __repr__(self):
        return f'Worm("{self.name}")'

    def get_time_range(self, feature, min_age=-numpy.inf, max_age=numpy.inf, age_feature='age', match_closest=False, filter_valid=True):
        """Get data from a timecourse feature in a specific time window.

        Parameters:
            feature: the name of a timecourse feature available in worm.td to
                retrieve (such as "gfp_95th", say), or a function that will be
                called as feature(worm) that will calculate a timecourse feature
                (see examples).
            min_age, max_age: the beginning and end of the window in which to
                retrieve features. If not specified, features from the very
                beginning and/or to the very end of the timecourse will be
                retrieved.
            age_feature: the name of the feature to use to determine the age
                window. Defaults to 'age', but other monotonic features, such as
                a time-left-alive 'ghost_age' could also be used. If this is a
                function, it will be called as age_feature(worm) to generate
                the ages to examine (see examples).
            match_closest: If False (default), the age window is strictly
                defined as age <= max_age and >= min_age. However, if
                match_closest is True, the window will begin at the nearest age
                to min_age, and end at the nearest age to max_age.
            filter_valid: If True (default), data with value nan, even in the age
                range requested, will not be returned.

        Returns:
            ages: the ages within the window
            data: the feature values corresponding to those ages

        Examples:
            # Assume that worm.td.gfp_95th and worm.td.gfp_background exist.

            #Basic usage:
            midlife_ages, midlife_gfp = worm.get_time_range('gfp_95th', 3, 7)
            old_ages, old_gfp = worm.get_time_range('gfp_95th', min_age=7)

            # Using a function as a feature:
            def bg_subtracted_gfp(worm):
                return worm.td.gfp_95th - worm.td.gfp_background
            ages, gfp_minus_bg = worm.get_time_range(bg_subtracted_gfp, 3, 7)

            # Alternately, if you want to store the feature for later use:
            worm.td.gfp_minus_bg = worm.td.gfp_95th - worm.td.gfp_background
            ages, gfp_minus_bg = worm.get_time_range('gfp_minus_bg', 3, 7)

            # Using a function as an age feature:
            def ghost_age(worm):
                # zero means dead, more negative means longer to live
                return worm.td.age - worm.lifespan
            ghost_ages, near_death_gfp = worm.get_time_range('gfp_95th', min_age=-1, age_feature=ghost_age)
        """
        ages = age_feature(self) if callable(age_feature) else getattr(self.td, age_feature)
        data = feature(self) if callable(feature) else getattr(self.td, feature)
        if match_closest:
            mask = self._get_closest_times_mask(ages, min_age, max_age)
        else:
            mask = (ages >= min_age) & (ages <= max_age)
        if filter_valid and numpy.issubdtype(data.dtype, numpy.floating):
            mask &= ~numpy.isnan(data)
        return ages[mask], data[mask]

    @staticmethod
    def _get_closest_times_mask(ages, min_age, max_age):
        li = numpy.argmin(numpy.absolute(ages - min_age))
        ui = numpy.argmin(numpy.absolute(ages - max_age))
        r = numpy.arange(len(ages))
        return (r >= li) & (r <= ui)

    def interpolate_feature(self, feature, age, age_feature='age'):
        """Estimate timecourse feature values at arbitrary times.

        Use linear interpolation to estimate the value of a given timecourse
        feature at one or more arbitrary times. Useful for calculating feature
        values across multiple worms on a uniform time-basis.

        Parameters:
            feature: the name of a timecourse feature available in worm.td to
                retrieve (such as "gfp_95th", say), or a function that will be
                called as feature(worm) that will calculate a timecourse feature
                (see documentation/examples for get_age_range).
            age: a single age or a list/array of ages at which to interpolate
                the feature values.
            age_feature: the name of the feature to use to determine the ages.
                Defaults to 'age', but other monotonic features, such as
                a time-left-alive 'ghost_age' could also be used. If this is a
                function, it will be called as age_feature(worm) to generate
                the ages to examine (see documentation/examples for
                get_age_range).

        Returns the level or levels of the feature at the specified age(s).

        Example:
            day7gfp = worm.interpolate_feature('gfp_95th', 7)
            daily_gfp = worm.interpolate_feature('gfp_95th', numpy.arange(3, 7))
        """
        ages, value = self.get_time_range(feature, age_feature=age_feature)
        return numpy.interp(age, ages, value)

    def smooth_feature(self, feature, filter=uniform_filter, age_feature='age',
                       min_age=-numpy.inf, max_age=numpy.inf, **filter_params):
        """Smooth a feature with a given filter-function.

        Data points within a given age range will be smooothed with a given
        filter function. The pre-defined gaussian and uniform filters will
        smooth the data with a gaussian- or un-weighted average (respectively)
        of the data points previous to each current point. (I.e. these will be
        "causal" filters that do not incorporate future data.)

        For each timepoint in the selected time range, the filter function will
        be called as filter(ages, age_to_smooth, **filter_params), where the
        ages parameter contains the age at which each data point was acquired,
        and the age_to_smooth parameter is the particular age at to get a
        smoothed value for. The smoothing filter then chooses how to weight
        each data point based on its temporal distance from the age under
        consideration. The smoothing filter returns the weights for each
        timepoint in the range, and a weighted average of the data in that
        range is calculated.

        The resulting smoothed data will be stored in a td attribute with the
        name "smoothed_{feature}". Data points outside the provided time range
        will be un-smoothed in this output.

        Parameters:
            feature: feature to filter over
            filter: a smoothing filter that will be called as
                filter(ages, age_to_smooth). Use of the pre-defined
                gaussian_filter and uniform_filter functions is encouraged.
            age_feature: the name of the feature to use to determine the age
                window (as used by get_time_range).
            min_age, max_age: Limits between which to perform filtering;
                outside this range, no filtering is performed.
            **filter_params: Additional parameters passed to the filter function.
                In particular, uniform_filter requires a "window_size"
                parameter, which defines the temporal window size that the filter
                averages over. The gaussian_filter function requires a "sigma"
                parameter giving the temporal standard deviation; it also takes
                an optional "window_size" parameter.
        """
        ages, data = self.get_time_range(feature, age_feature=age_feature,
            min_age=min_age, max_age=max_age, filter_valid=True) # No nans here
        smoothed = []
        for age_to_smooth in ages:
            weights = filter(ages, age_to_smooth, **filter_params)
            total_weight = weights.sum()
            if total_weight == 0:
                raise ValueError('Smoothing filter did not weight any data.')
            smoothed.append(numpy.dot(data, weights / total_weight))
        age_mask = numpy.isin(getattr(self.td, age_feature), ages)
        smoothed_feature = numpy.array(getattr(self.td, feature))
        smoothed_feature[age_mask] = smoothed
        setattr(self.td, 'smoothed_' + feature, smoothed_feature)

    def merge_with(self, other):
        """Merge summary and timecourse data with another worm.

        Merging of timecourse data is supported in all cases: when the
        timepoints completely or partially overlap, or are disjoint. The other
        worm must have a matching name. If any timepoints are in common those
        data must match. There is one exception: if one worm has
        nan/empty-string values at a timepoint and another has
        non-nan/empty-strings, the non-nan/empty values will be used.

        Note: only handles int/float/string data types. Any int data types will
        be converted to float in cases where timepoints do not fully overlap,
        and missing values will be filled with nan. For string data, missing
        values are represented by empty strings.
        """
        assert other.name == self.name
        # first merge timecourse data
        if set(self.td.timepoint) == set(other.td.timepoint):
            # timepoints are the same; can merge trivially
            for k, v in other.td._items():
                if hasattr(self.td, k):
                    # if both worms have the same named timecourse information, make sure it matches
                    assert numpy.all(v == getattr(self.td, k))
                else:
                    setattr(self.td, k, v)
        else:
            # timepoints are different, need to merge in more complex fashion
            self._unify_timecourses(other)
        # now merge summary data
        for k, v in other.__dict__.items():
            if k == 'td':
                continue
            if hasattr(self, k):
                assert numpy.all(v == getattr(self, k))
            else:
                setattr(self, k, v)

    def _unify_timecourses(self, other):
        new_timepoints = numpy.unique(numpy.concatenate([self.td.timepoint, other.td.timepoint]))
        len_new = len(new_timepoints)
        our_mask = numpy.in1d(new_timepoints, self.td.timepoint, assume_unique=True)
        other_mask = numpy.in1d(new_timepoints, other.td.timepoint, assume_unique=True)
        # the following holds:
        # assert (new_timepoints[our_mask] == self.td.timepoint).all()
        # assert (new_timepoints[other_mask] == other.td.timepoint).all()
        our_features = set(self.td._keys())
        our_features.remove('timepoint')
        other_features = set(other.td._keys())
        other_features.remove('timepoint')
        ours_only = our_features - other_features
        other_only = other_features - our_features
        both = our_features.intersection(other_features)
        for feature in ours_only:
            self._convert_values(self.td, feature, len_new, our_mask)
        for feature in other_only:
            self._convert_values(other.td, feature, len_new, other_mask)
        if len(both) > 0:
            ours_in_other = numpy.in1d(self.td.timepoint, other.td.timepoint, assume_unique=True)
            other_in_ours = numpy.in1d(other.td.timepoint, self.td.timepoint, assume_unique=True)
            for feature in both:
                our_v = numpy.asarray(getattr(self.td, feature))
                other_v = numpy.asarray(getattr(other.td, feature))
                ours_good = _valid_values(our_v)
                others_good = _valid_values(other_v)
                if numpy.any((our_v[ours_in_other] != other_v[other_in_ours]) &
                   ours_good[ours_in_other] & others_good[other_in_ours]):
                    # if there are any data values that (a) overlap, (b) compare as unequal and (c) are both non-nan,
                    # then we have a data conflict
                    raise ValueError(f'worms have different values of "{feature}" for one or more of the timepoints that are in common')
                new_values = numpy.empty(len_new, dtype=numpy.promote_types(our_v.dtype, other_v.dtype))
                if numpy.issubdtype(new_values.dtype, numpy.floating):
                    new_values.fill(numpy.nan)
                take_from_ours = our_mask.copy()
                take_from_ours[our_mask] = ours_good # set mask false where ours has nan
                new_values[take_from_ours] = our_v[ours_good]
                take_from_other = other_mask.copy()
                take_from_other[other_mask] = others_good # set mask false where other has nan
                new_values[take_from_other] = other_v[others_good]
                setattr(self.td, feature, new_values)
        self.td.timepoint = new_timepoints

    def _convert_values(self, src_td, feature, len_new, mask):
        values = numpy.asarray(getattr(src_td, feature))
        if values.dtype.kind in 'SU': # string dtype
            dtype = values.dtype
        else:
            dtype = float
        new_values = numpy.empty(len_new, dtype=dtype)
        new_values[mask] = values
        if dtype is float:
            new_values[~mask] = numpy.nan
        setattr(self.td, feature, new_values)

def _valid_values(array):
    if numpy.issubdtype(array.dtype, numpy.floating):
        return ~numpy.isnan(array)
    elif array.dtype.kind == 'S':
        return array != b''
    elif array.dtype.kind == 'U':
        return array != ''
    else:
        return numpy.ones(array.shape, dtype=bool)

class Worms(collections.UserList):
    """List-like collection of Worm objects with convenience functions.

    Construct as any other list:
        worms = Worms()
        worms.append(some_worm)

    Or use a factory function:
        worms = read_worms(...)
        worms = meta_worms(...)
        (see documentation for the above functions)

    Slicing, etc, will yield a new Worms list:
        every_fifth = worms[::5]

    See the documentation for worms-specific methods to read and write data
    files, and to gather and analyze data pertaining to groups of worms.
    """

    def __getitem__(self, i):
        # work around a bug in python 3.6 where UserList doesn't return the right type
        # for slices. TODO: Check later if this is fixed and can remove workaround
        if isinstance(i, slice):
            return self.__class__(self.data[i])
        else:
            return self.data[i]

    def read_summary_data(self, path, add_new=False, delimiter='\t', time_units='hours'):
        """Read in summary data (not timecourse data) for each worm.

        Summary statistics are read from columns in a delimited text file and
        associated with the matching worm. The first column of the file is
        assumed to contain the name of each worm, which must match the worm.name
        attribute of a worm in this Worms list. Entries in other columns will be
        added as attributes of the matching worm.

        Parameters:
            path: path to data file.
            add_new: if True, when a name is encountered in the datafile that
                does not match any existing worm names, create a new worm and
                add it to this Worms list. If False, print a warning and ignore.
            delimiter: delimiter for input data file.
            time_units: the unit of time for the values in any column with a
                name ending in 'span'. All time units will be converted into
                hours internally. (The rescale_time() method can be used to
                change units later, but this is at your own risk: all
                time-based analysis code can and should assume time units of
                hours.)
                Must be one of 'days', 'hours', 'minutes', or 'seconds'.
        """
        if time_units not in TIME_UNITS:
            raise ValueError(f"'time_units' must be one of: {list(TIME_UNITS)}")
        time_scale = TIME_UNITS[time_units]
        named_worms = {worm.name: worm for worm in self}
        header, data = datafile.read_delimited(path, delimiter=delimiter)
        header = [colname.replace(' ', '_') for colname in header]
        for name, *rest in data:
            if name not in named_worms:
                if add_new:
                    worm = Worm(name)
                    self.append(worm)
                else:
                    print(f'Ignoring record for unknown worm "{name}".')
                    continue
            else:
                worm = named_worms[name]
            for feature, val in zip(header[1:], rest):
                if feature.endswith('span'):
                    val *= time_scale
                setattr(worm, feature, val)

    def write_summary_data(self, path, features=None, delimiter='\t', error_on_missing=True):
        """Write out summary data (not timecourse data) for each worm.

        Summary data for each worm are stored as attributes of that worm, e.g.
        worm.lifespan. This function writes these data features to a text file
        indexed by worm.name.

        Parameters:
            path: path to data file to write.
            features: list of summary features to write out for each worm (e.g.
                ['name', 'lifespan']). If None (the default), write out all
                features. If 'name' is not the first element of the provided
                list, it will be added as such.
            delimiter: delimiter for output data file.
            error_on_missing: If True, if a worm is found that does not have a
                specified feature name as an attribute, an error will be raised.
                If False, then an empty value will be provided for the feature
                name in the output file.
        """
        if features is None:
            features = set()
            for worm in self:
                features.update(worm.__dict__)
            features.remove('name')
            features.remove('td')
            features = ['name'] + sorted(features)
        elif features[0] != 'name':
            if 'name' in features:
                features.remove('name')
            features.insert('name', 0)
        data = [features]
        for worm in self:
            row = []
            for feature in features:
                if feature not in worm.__dict__ and error_on_missing:
                    raise ValueError(f'Worm "{worm.name}" has no "{feature}" feature.')
                else:
                    row.append(getattr(worm, feature, ''))
            data.append(row)
        datafile.write_delimited(path, data, delimiter)

    def write_timecourse_data(self, path, multi_worm_file=False, features=None, suffix=None, delimiter='\t', error_on_missing=True):
        """Write out timecourse data for each worm.

        Write out one or more timecourse data features for each worm to either
        a single "multi-worm" text file indexed by worm.name and worm.td.timepoint,
        or to multiple single-worm text files named according to worm.name and
        indexed by worm.td.timepount.

        Parameters:
            path: path to data file(s) to write. If multi_worm_file=False,
                the path represents a parent directory in which to write the
                worm files. If multi_worm_file=True, the path represents the
                filename to write out.
            multi_worm_file: see above.
            features: list of timecourse features to write out for each worm (e.g.
                ['age', 'gfp_95th']). If None (the default), write out all
                features. If 'timepoint' is not the first element of the provided
                list, it will be added as such.
            suffix: if multi_worm_file=False, this controls the suffix of the
                generated data files. If None, then infer the suffix from the
                delimiter: 'csv', 'tsv', and 'dat' for comma, tab, and other
                delimiters, respectivey.
            delimiter: delimiter for output data file(s).
            error_on_missing: If True, if a worm is found that does not have a
                specified feature name in worm.td, an error will be raised.
                If False, then empty values will be provided for the feature
                name in the output file.
        """
        path = pathlib.Path(path)
        if not multi_worm_file:
            path.mkdir(parents=False, exist_ok=True)
            if suffix is None:
                if delimiter == ',':
                    suffix = '.csv'
                elif delimiter == '\t':
                    suffix = '.tsv'
                else:
                    suffix = '.dat'

        if features is None:
            features = set()
            for worm in self:
                features.update(worm.td.__dict__)
            features.remove('timepoint')
            features = ['timepoint'] + sorted(features)
        elif features[0] != 'timepoint':
            if 'timepoint' in features:
                features.remove('timepoint')
            features.insert('timepoint', 0)

        if multi_worm_file:
            data = [['name'] + features]
        for worm in self:
            n = len(worm.td.timepoint)
            missing = [''] * n
            cols = []
            for feature in features:
                if feature not in worm.td.__dict__:
                    if error_on_missing:
                        raise ValueError(f'Worm "{worm.name}" has no "{feature}" feature.')
                    else:
                        vals = missing
                else:
                    vals = getattr(worm.td, feature)
                    if vals.dtype == float:
                        good = ~numpy.isnan(vals)
                        vals = [str(v) if g else '' for v, g in zip(vals, good)]
                    elif vals.dtype == object:
                        vals = [str(v) if v is not None else '' for v in vals]
                cols.append(vals)
            assert all(len(c) == n for c in cols)
            rows = [[] for _ in range(n)]
            for col in cols:
                for row, colval in zip(rows, col):
                    row.append(colval)
            if multi_worm_file:
                for row in rows:
                    row.insert(0, worm.name)
                data.extend(rows)
            else:
                worm_path = path / (worm.name + suffix)
                datafile.write_delimited(worm_path, [features] + rows, delimiter)
        if multi_worm_file:
            datafile.write_delimited(path, data, delimiter)

    def merge_in(self, other_worms, add_new=False):
        """Merge timecourse and summary data from a second Worms list.

        This function is useful if you have several different text files
        containing timecourse data for the same worms. This can arise if primary
        and derived timecourse measures are stored in separate files, for example.

        In that case, it's most expiditious to load the timecourse files into
        separate Worms lists using read_worms(), and then merge the resulting
        lists together.

        Worms will be merged by matching the worm.name attribute. Worms in the
        other_worms list that do not match can be added or ignored (with awarning
        printed), depending on the add_new parameter.

        If both matching worms have the same timecourse or summary features,
        those features are checked to make sure they're equal.

        Example:
            worms = read_worms('basic_measures/*.csv')
            health_data = read_worms('health_measures/*.csv')
            worms.merge_in(health_data)
        """
        named_worms = {worm.name: worm for worm in self}
        for other_worm in other_worms:
            if other_worm.name not in named_worms:
                if add_new:
                    self.append(other_worm)
                else:
                    print("no match found for", other_worm.name)
                continue
            named_worms[other_worm.name].merge_with(other_worm)

    def sort(self, feature, reverse=False):
        """Sort Worms list in place, according to a summary feature value.

        Parameters:
            feature: the name of a summary feature (e.g. 'lifespan'). If a
                callable function, will be called as feature(worm) to generate
                the feature to sort on.
            reverse: if False, sort from lowest to highest value; if True, reverse.

        Examples:
            worms.sort('lifespan')

            # sort by highest value of worm.td.gfp_95th at any time, from high
            # to low values:
            def peak_gfp(worm):
                return worm.td.gfp_95th.max()
            worms.sort(peak_gfp, reverse=True)

            # sort by slope of worm.td.gfp_95th in the range of 3-7 days, if
            # worm lived that long:
            from scipy import stats
            def gfp_slope(worm):
                if worm.lifespan < 7:
                    return None
                else:
                    age, gfp = worm.get_time_range('gfp_95th', 3, 7)
                    return stats.linregress(age, gfp).slope
            worms.sort(gfp_slope)
        """
        if callable(feature):
            key = feature
        else:
            def key(worm):
                return getattr(worm, feature)
        super().sort(key=key, reverse=reverse)

    def filter(self, criterion):
        """Return a new Worms list where all worms meet the specified criterion.

        Evaluate the provided criterion as criterion(worm) for each worm in the
        list, and return a new Worms list for all those with a True-valued
        result, or a result that converts to True (i.e. not 0 or None, &c.)

        Example:
        def long_lived(worm):
            return worm.lifespan > 12
        long_lived_worms = worms.filter(long_lived)

        # or more tersely with a lambda expression:
        long_lived_worms = worms.filter(lambda worm: worm.lifespan > 12)
        """
        mask = self.get_feature(criterion).astype(bool)
        return self.__class__([worm for worm, keep in zip(self, mask) if keep])

    def get_time_range(self, feature, min_age=-numpy.inf, max_age=numpy.inf, age_feature='age', match_closest=False, filter_valid=True):
        """Get data within a given age range for each worm.

        Calls get_time_range(...) on each worm in the list and return the results.

        Parameters:
            feature: the name of a timecourse feature available in worm.td to
                retrieve (such as "gfp_95th", say), or a function that will be
                called as feature(worm) that will calculate a timecourse feature
                (see examples in Worm.get_time_range).
            min_age, max_age: the beginning and end of the window in which to
                retrieve features. If not specified, features from the very
                beginning and/or to the very end of the timecourse will be
                retrieved.
            age_feature: the name of the feature to use to determine the age
                window. Defaults to 'age', but other monotonic features, such as
                a time-left-alive 'ghost_age' could also be used. If this is a
                function, it will be called as age_feature(worm) to generate
                the ages to examine (see examples).
            match_closest: If False (default), the age window is strictly
                defined as age <= max_age and >= min_age. However, if
                match_closest is True, the window will begin at the nearest age
                to min_age, and end at the nearest age to max_age.
            filter_valid: If True (default), data with value nan, even in the age
                range requested, will not be returned.

        Returns: list of numpy arrays, one array per worm. Each array has shape
            (n, 2), where n is the number of timepoints in the specified age range.
            array[:, 0] is the ages at each timepoint, and array[:, 1] is the feature
            values for each timepoint.

        Example:
            # assuming each worm has a 'integrated_gfp' entry in worm.td,
            # the below will retrieve all the values between days 5 and 7
            # (assuming that the times are in days...).
            data = worms.get_time_range('integrated_gfp', min_age=5, max_age=7)
            # calculate the average relationship between age and GFP level:
            all_data = numpy.concatenate(data)
            all_ages = all_data[:,0]
            all_gfp = all_data[:, 1]
            fit = scipy.stats.linregress(all_ages, all_gfp)
            print(fit.slope, fit.intercept)
        """
        out = []
        for worm in self:
            ages, values = worm.get_time_range(feature, min_age, max_age, age_feature, match_closest, filter_valid)
            out.append(numpy.transpose([ages, values]))
        return out

    def get_feature(self, feature):
        """Get the specified feature for each worm in the list.

        Parameters:
            feature: the name of an attribute of each worm object to retrieve
                (such as "lifespan", say), or a function that will be called as
                feature(worm) that will calculate such a feature (see examples).

        Returns: list of feature values, one for each worm.

        Example:
            lifespans = worms.get_feature('lifespan')

            # find the age at which the timecourse feature 'gfp' peaks
            def peak_gfp_age(worm):
                peak_gfp_i = worm.td.gfp.argmax()
                return worm.td.age[peak_gfp_i]
            peak_gfp_ages = worms.get_feature(peak_gfp_age)
        """
        return numpy.array([feature(worm) if callable(feature) else getattr(worm, feature) for worm in self])

    def get_features(self, *features):
        """Get one or more specified features from each worm in the list.

        Parameter list: the names of attributes of each worm object to retrieve
                (such as "lifespan", say), or functions that will be called as
                feature(worm) that will calculate such a feature.

        Returns: numpy.array of shape (n_worms, n_features)
        """
        if len(features) == 1:
            out = self.get_feature(features[0])
            # the feature-function might return a vector, not a scalar, so we might not actually
            # need to add a new axis...
            if out.ndim == 1:
                out = out[:, numpy.newaxis]
            return out
        else:
            return numpy.transpose([self.get_feature(feature) for feature in features])

    def get_timecourse_features(self, *features, min_age=-numpy.inf, max_age=numpy.inf, age_feature='age', match_closest=False, filter_valid=True):
        """Get multiple timecourse features within a given age range for each worm.

        Parameters:
            features: names of timecourse features available in worm.td to
                retrieve (such as "gfp_95th", say), or functions that will be
                called as feature(worm) that will calculate a timecourse feature
                (see examples in Worm.get_time_range).
            min_age, max_age: the beginning and end of the window in which to
                retrieve features. If not specified, features from the very
                beginning and/or to the very end of the timecourse will be
                retrieved.
            age_feature: the name of the feature to use to determine the age
                window. Defaults to 'age', but other monotonic features, such as
                a time-left-alive 'ghost_age' could also be used. If this is a
                function, it will be called as age_feature(worm) to generate
                the ages to examine (see examples).
            match_closest: If False (default), the age window is strictly
                defined as age <= max_age and >= min_age. However, if
                match_closest is True, the window will begin at the nearest age
                to min_age, and end at the nearest age to max_age.
            filter_valid: If True (default), data with value nan, even in the
                age range requested, will not be returned.

        Returns: ages, data
            ages: 1d array of length total_timepoints, where total_timepoints
                is the total number of timepoints (across all the worms) within
                the age range.
            data: numpy.array of shape (total_timepoints, n_features), where
                n_features is the number of specified timepoint features.

        If there are 100 worms, each with 12 timepoints, then
        get_timecourse_features('gfp', 'texture') will return an array of shape
        (1200) containing the ages, and (1200, 2) containing the feature data.
        """
        data = []
        for feature in features:
            ages, feature_data = numpy.concatenate(self.get_time_range(feature, min_age, max_age, age_feature, match_closest, filter_valid=False)).T
            data.append(feature_data)
        data = numpy.transpose(data) # shape (n_timepoints, n_features)
        if filter_valid:
            mask = numpy.all(~numpy.isnan(data), axis=1)
            data = data[mask]
            ages = ages[mask]
        return ages, data

    def z_transform(self, feature, feature_out=None, recenter_only=False,
            min_age=-numpy.inf, max_age=numpy.inf, age_feature='age',
            match_closest=False, filter_valid=True, points_out=300, smooth=0.4):
        """Produce a z-score from timecourse data.

        Create a new timecourse feature that is a worm's z-score relative to the
        population, with respect to a given feature. LOWESS regression is used
        to estimate the population's moving mean and standard deviation. Then,
        each individual's z-scores are calculated relative to that mean/std.

        Parameters:
            feature: the name of a timecourse feature available in worm.td to
                retrieve (such as "gfp_95th", say), or a function that will be
                called as feature(worm) that will calculate a timecourse feature
                (see examples in Worm.get_time_range).
            feature_out: the timecourse feature name to store the resulting z
                scores as. If None, use '{feature}_z', assuming 'feature' is a
                string name.
            recenter_only: instead of z-transforming (data-mean)/std, just
                recenter the data by subtrating the mean.
            min_age, max_age: the beginning and end of the window in which to
                retrieve features. If not specified, features from the very
                beginning and/or to the very end of the timecourse will be
                retrieved.
            age_feature: the name of the feature to use to determine the age
                window. Defaults to 'age', but other monotonic features, such as
                a time-left-alive 'ghost_age' could also be used. If this is a
                function, it will be called as age_feature(worm) to generate
                the ages to examine (see examples).
            match_closest: If False (default), the age window is strictly
                defined as age <= max_age and >= min_age. However, if
                match_closest is True, the window will begin at the nearest age
                to min_age, and end at the nearest age to max_age.
            filter_valid: If True (default), data with value nan, even in the age
                range requested, will not be returned.
            points_out: number of points to evaluate the mean and std trendlines
                along, or a set of x-values at which to evaluate the trendlines.
                Points outside the range of the input x-values will evaluate to
                nan: no extrapolation will be attempted.
            smooth: smoothing parameter passed to LOWESS moving_mean_std function.
                Represents the fraction of input data points that will be smoothed
                across at each output data point.

        Returns trend_x, mean_trend, std_trend
            trend_x: 1-d array of length points_out containing the x-values at
                which the mean and std trends are evaluated.
            mean_trend, std_trend: y-values for the mean and std trendlines.
        """
        if feature_out is None:
            if callable(feature):
                raise ValueError('feature_out must be specified when the provided feature is a function, not a string')
            else:
                feature_out = f'{feature}_z'
        data = self.get_time_range(feature, min_age, max_age, age_feature, match_closest, filter_valid)
        all_ages, all_data = numpy.concatenate(data).T
        res = moving_mean_std.z_transform(all_ages, all_data, points_out, smooth=smooth, iters=1)
        mean_est, std_est, z_est, trend_x, mean_trend, std_trend = res
        if recenter_only:
            output = all_data - mean_est
        else:
            output = z_est
        lengths = list(map(len, data))
        worm_data = numpy.split(output, numpy.cumsum(lengths)[:-1])
        worm_ages = [d[:, 0] for d in data]
        for worm, ages, data in zip(self, worm_ages, worm_data):
            worm_ages = age_feature(worm) if callable(age_feature) else getattr(worm.td, age_feature)
            vals = numpy.empty(worm_ages.shape, dtype=float)
            vals.fill(numpy.nan)
            age_mask = numpy.in1d(worm_ages, ages, assume_unique=True)
            vals[age_mask] = data
            setattr(worm.td, feature_out, vals)
        return trend_x, mean_trend, std_trend

    def group_by(self, keys):
        """Given a list of keys (one for each worm), return a dictionary mapping
        each key to all the worms that have the same key.

        Example, assuming four worms:
        keys = ['good', 'bad', 'good', 'good']
        groups = worms.group_by(keys)
        groups['good'] # list containing worms 0, 2, and 3
        """
        assert len(keys) == len(self)
        worms = collections.defaultdict(self.__class__)
        for worm, key in zip(self, keys):
            worms[key].append(worm)
        return dict(worms)

    def bin(self, feature, nbins, equal_count=False):
        """Group worms into bins based on a feature value.

        Most useful for passing to the meta_worms factory function, to calculate
        average time trends for grouped worms.

        Parameters:
            feature: the name of an attribute of each worm object to retrieve
                (such as "lifespan", say), or a function that will be called as
                feature(worm) that will calculate such a feature.
            nbins: number of groups to bin worms into
            equal_count: if False (default), worms will be binned based on a
                splitting the total feature range into equal width bins. If True,
                worms will be grouped into bins with equal numbers of worms.

        Returns: dict mapping bin names (strings with either the bin number,
            if equal_count is True, or a description of the feature range for
            the bin) to worms in that bin.

        Example:
            lifespan_terciles = worms.bin('lifespan', nbins=3, equal_count=True)
            shortest_tercile = lifespan_terciles['0']

            age_at_death_cohorts = worms.bin('lifespan', nbins=7)
            for range, binned_worms in sorted(age_at_death_cohorts.items()):
                print(range, len(binned_worms))
        """
        data = self.get_feature(feature)
        if equal_count:
            ranks = data.argsort().argsort()
            bins = list(map(str, (nbins * ranks / (len(ranks))).astype(int)))
        else:
            bin_edges = numpy.linspace(data.min(), data.max(), nbins+1)
            bin_numbers = numpy.digitize(data, bin_edges[1:-1])
            bin_names = [f'[{bin_edges[i]:.1f}-{bin_edges[i]+1:.1f})' for i in range(len(bin_edges)-2)]
            bin_names.append(f'[{bin_edges[-2]:.1f}-{bin_edges[-1]:.1f}]')
            bins = [bin_names[n] for n in bin_numbers]
        return self.group_by(bins)

    def get_regression_data(self, *features, target='lifespan', control_features=None, filter_valid=True):
        """Get data matrices useful for predicting a target feature based on one or more other features,
        optionally controlling for a set of features.

        Parameters:
            features: one or more features to retrieve to serve as indepdendent
                variable(s). Passed to get_features()
            target: feature to serve as dependent variable. Passed to get_feature()
            control_features: if not None, a list of features to pass to
                get_features() to serve as variables to control for.
            filter_valid: if True, filter out entries of X, y, and C where any
                entries are nan.

        Returns: X, y, C
            X: array of shape (n_worms, n_features)
            y: array of shape (n_worms)
            C: array of shape (n_worms, n_control_features)
        """
        X = self.get_features(*features)
        y = self.get_feature(target)
        C = None if control_features is None else self.get_features(*control_features)
        if filter_valid:
            valid_mask = numpy.isfinite(X).all(axis=1) & numpy.isfinite(y)
            if C is not None:
                valid_mask &= numpy.isfinite(C).all(axis=1)
            X = X[valid_mask]
            y = y[valid_mask]
            if C is not None:
                C = C[valid_mask]
        return X, y, C

    def regress(self, *features, target='lifespan', control_features=None, regressor=None):
        """Use zplib.scalar_stats.regress to determine relationship between features.

        See zplib.scalar_stats.regress for more information.

        Parameters:
            features: one or more features to retrieve to serve as indepdendent
                variable(s). Passed to get_features()
            target: feature to serve as dependent variable. Passed to get_feature()
            control_features: if not None, a list of features to pass to
                get_features() to serve as variables to control for.
            regressor: if not None, a sklearn-style regressor to use. Othwerwise
                zplib.scalar_stats.regress will use linear regression.

        Returns: zplib.scalar_stats.RegressionResult, which is a named tuple
            with fields: 'y_est', 'resid', 'R2', 'regressor', and 'X':
                y_est: estimated target feature based on input features
                resid: residuals (y - y_est, where y is the target feature)
                R2: R-squared value.
                regressor: the regressor fit to the data.
                X: the input data matrix, possibly transformed by controlling
                    for the input parameters.

        Example:
            # determine relationship between peak GFP value and lifespan,
            # controlling for length at the time of that GFP peak.
            for worm in worms:
                peak_gfp_i = worm.td.gfp.argmax()
                worm.peak_gfp = worm.td.gfp[peak_gfp_i]
                worm.td.length_at_gfp_peak = worm.td.length[peak_gfp_i]

            result = worms.regress('peak_gfp', control_features=['length_at_gfp_peak'])
            print(result.R2)
        """
        X, y, C = self.get_regression_data(*features, target=target, control_features=control_features)
        return regress.regress(X, y, C, regressor)

    def get_regression_time_data(self, *features, target='age', min_age=-numpy.inf, max_age=numpy.inf, age_feature='age'):
        """Get timecourse data in a format suitable for predicting age
        (or some other age-varying target) from one or more timecourse features.

        Parameters:
            features: one or more features to serve as independent variables.
            min_age, max_age, age_feature: see get_timecourse_features()
            target: feature to serve as dependent variable.

        Returns: X, y
            X: array of shape (n_timepoints, n_features)
            y: array of shape (n_timepoints)
        """
        X = self.get_timecourse_features(*features, min_age=min_age, max_age=max_age, age_feature=age_feature)[1]
        y = self.get_timecourse_features(target, min_age=min_age, max_age=max_age, age_feature=age_feature)[1].squeeze()
        return X, y

    def _timecourse_plot_data(self, feature, min_age=-numpy.inf, max_age=numpy.inf, age_feature='age', color_by='lifespan'):
        time_ranges = self.get_time_range(feature, min_age, max_age, age_feature)
        color_vals = colorize.scale(self.get_feature(color_by), output_max=1)
        colors = colorize.color_map(color_vals, uint8=False)
        out = []
        for time_range, color in zip(time_ranges, colors):
            x, y = time_range.T
            out.append((x, y, color))
        return out

    def plot_timecourse(self, feature, min_age=-numpy.inf, max_age=numpy.inf,
        age_feature='age', time_units='hours', color_by='lifespan'):
        """Plot values of a given feature for each worm, colored by a given
        worm feature (defaults to lifespan).

        Parameters:
            feature: the name of a timecourse feature available in worm.td to
                retrieve (such as "gfp_95th", say), or a function that will be
                called as feature(worm) that will calculate a timecourse feature
                (see examples in Worm.get_time_range).
            min_age, max_age: the beginning and end of the window in which to
                retrieve features. If not specified, features from the very
                beginning and/or to the very end of the timecourse will be
                retrieved.
            age_feature: the name of the feature to use to determine the age
                window and the plot axes. Defaults to 'age', but other
                monotonic features, such as a time-left-alive 'ghost_age' could
                also be used. If this is a function, it will be called as
                age_feature(worm) to generate the ages to examine (see
                examples in Worm.get_time_range).
            time_units: one of "days", "hours", "minutes", or "seconds",
                representing the units in which the time scale should be plotted.
            color_by: worm feature to use for color scale of each timecourse.
        """
        import matplotlib.pyplot as plt
        if time_units not in TIME_UNITS:
            raise ValueError(f"'time_units' must be one of: {list(TIME_UNITS)}")
        time_scale = TIME_UNITS[time_units]
        plt.clf()
        for x, y, c in self._timecourse_plot_data(feature, min_age, max_age, age_feature, color_by):
            plt.plot(x/time_scale, y, color=c)

class _TimecourseData(object):
    def __repr__(self):
        return 'Timecourse Data:\n' + '\n'.join('    ' + k for k in sorted(self.__dict__) if not k.startswith('_'))

    def _items(self):
        return self.__dict__.items()

    def _keys(self):
        return self.__dict__.keys()

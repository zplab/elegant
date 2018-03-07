import numpy
import glob
import pathlib
import collections

from zplib.scalar_stats import moving_mean_std
from zplib.scalar_stats import regress
from zplib.image import colorize
from zplib import datafile

def read_worms(*path_globs, name_prefix='', delimiter='\t', last_timepoint_is_first_dead=True, age_scale=1):
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

    Parameters:
        *path_globs: all non-keyword arguments are treated as paths to worm data
            files, or as glob expressions matching multiple such data files.
        name_prefix: if a string, this will be used as the prefix (see above).
            If a function, it will be called with the pathlib.Path of the input
            file in question, so that the file name and/or name of the parent
            directories can be used to compute the prefix.
        delimiter: controls whether the data are assumed to be tab, comma, or
            otherwise delimited.
        last_timepoint_is_first_dead: if True, the last timepoint in the data
            file is assumed to represent the first time the worm was annotated
            as dead. Otherwise, the last timepoint is assumed to represent the
            last time the worm was known to be alive.
        age_scale: scale-factor for the ages read in. The values in the "age"
            column and any column ending in '_age' will be multiplied by this
            scalar. (Useful e.g. for converting hours to days.)

    Returns: Worms object

    Examples:
        worms = read_worms('/path/to/spe-9/datafiles/*.csv', name_prefix='spe-9 ', delimiter=',')

        def get_genotype(path):
            # assume that the grandparent path contains the genotype
            return path.parent.parent.name + ' '
        worms = read_worms('/path/to/*/datafiles/*.tsv', name_prefix=get_genotype)
    """
    worms = Worms()
    for path_glob in path_globs:
        for path in map(pathlib.Path, glob.iglob(str(path_glob), recursive=True)):
            if callable(name_prefix):
                prefix = name_prefix(path)
            else:
                prefix = name_prefix
            for name, header, data in _read_datafile(path, prefix, delimiter):
                worms.append(Worm(name, header, data, last_timepoint_is_first_dead, age_scale, file_path=path))
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
        smooth: smoothing parameter passed to LOWESS moving_mean function.
            Represents the fraction of input data points that will be smoothed
            across at each output data point.
        min_age, max_age: the beginning and end of the window to analyze.
            If not specified, features from the very beginning and/or to the
            very end of the timecourse will be retrieved.

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
        ages, data = worms.get_timecourse_features(*features, min_age=min_age, max_age=max_age, filter_valid=False)
        ages_out = numpy.linspace(ages.min(), ages.max(), 100)
        for feature_data in data.T:
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
    def __init__(self, name, feature_names=[], timepoint_data=[], last_timepoint_is_first_dead=True, age_scale=1, file_path=None):
        """It is generally preferable to construct worms from a factory function
        such as read_worms or meta_worms, rather than using the constructor.

        Parameters:
            name: identifier for this individual animal
            feature_names: names of the timecourse features measured for
                this animal
            timepoint_data: for each timepoint, each of the measured features.
            last_timepoint_is_first_dead: if True, the last timepoint in the
                data file is assumed to represent the first time the worm was
                annotated as dead. Otherwise, the last timepoint is assumed to
                represent the last time the worm was known to be alive.
            age_scale: scale-factor for the ages read in. The values in the
                "age" column and any column ending in '_age' will be multiplied
                by this scalar. (Useful e.g. for converting hours to days.)
            file_path: if not None, used for error reporting if the worm data
                is invalid.
        """
        self.name = name
        self.td = _TimecourseData()
        vals_for_features = [[] for _ in feature_names]
        if 'age' not in feature_names or 'timepoint' not in feature_names:
            raise ValueError(f'')
        for timepoint in timepoint_data:
            # add each entry in the timepoint data to the corresponding list in
            # vals_for_features
            for feature_vals, item in zip(vals_for_features, timepoint):
                feature_vals.append(item)
        for feature_name, feature_vals in zip(feature_names, vals_for_features):
            arr = numpy.array(feature_vals)
            if feature_name == 'age' or feature_name.endswith('_age'):
                arr *= age_scale
            setattr(self.td, feature_name, arr)
        if hasattr(self.td, 'age'):
            if last_timepoint_is_first_dead:
                self.lifespan = self.td.age[-2:].mean() # midpoint between last-alive and first-dead
            else:
                self.lifespan = self.td.age[-1] + (self.td.age[-1] - self.td.age[-2])/2 # halfway through the next interval, assumes equal intervals

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

    def read_summary_data(self, path, add_new=False, delimiter='\t'):
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
        """
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
            path.mkdir(parents=False, exist_ok=False)
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
                if feature not in worm.td.__dict__ and error_on_missing:
                    raise ValueError(f'Worm "{worm.name}" has no "{feature}" feature.')
                else:
                    cols.append(getattr(worm.td, feature, missing))
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
                worm_path = (path / worm.name).with_suffix(suffix)
                datafile.write_delimited(worm_path, [features] + rows, delimiter)
        if multi_worm_file:
            datafile.write_delimited(path, data, delimiter)

    def merge_in(self, other_worms):
        """Merge timecourse and summary data from a second Worms list.

        This function is useful if you have several different text files
        containing timecourse data for the same worms. This can arise if primary
        and derived timecourse measures are stored in separate files, for example.

        In that case, it's most expiditious to load the timecourse files into
        separate Worms lists using read_worms(), and then merge the resulting
        lists together.

        Worms will be merged by matching the worm.name attribute. Worms in the
        other_worms list that do not match will be ignored (and a warning will
        be printed). If both matching worms have the same timecourse or summary
        features, those features are checked to make sure they're equal.

        Example:
            worms = read_worms('basic_measures/*.csv')
            health_data = read_worms('health_measures/*.csv')
            worms.merge_in(health_data)
        """
        named_worms = {worm.name: worm for worm in self}
        for other_worm in other_worms:
            if other_worm.name not in named_worms:
                print("no match found for", other_worm.name)
                continue
            our_worm = named_worms[other_worm.name]
            # merge timecourse data
            for k, v in other_worm.td.__dict__.items():
                if hasattr(our_worm.td, k):
                    # if both worms have the same named timecourse information, make sure it matches
                    assert numpy.all(v == getattr(our_worm.td, k))
                else:
                    setattr(our_worm.td, k, v)
            # merge summary data
            for k, v in other_worm.__dict__.items():
                if k == 'td':
                    pass
                if hasattr(our_worm, k):
                    assert numpy.all(v == getattr(our_worm, k))
                else:
                    setattr(our_worm, k, v)

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

    def plot_timecourse(self, feature, min_age=-numpy.inf, max_age=numpy.inf, age_feature='age'):
        """Plot values of a given feature for each worm, colored by lifespan.

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
        """
        import matplotlib.pyplot as plt
        plt.clf()
        for x, y, c in self._timecourse_plot_data(feature, min_age, max_age, age_feature):
            plt.plot(x, y, color=c)

class _TimecourseData(object):
    def __repr__(self):
        return 'Timecourse Data:\n' + '\n'.join('    ' + k for k in sorted(self.__dict__) if not k.startswith('_'))


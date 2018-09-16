# This code is licensed under the MIT License (see LICENSE file for details)

import pkg_resources
import pickle
import functools
import numpy

from zplib.curve import interpolate

def default_estimator(pixels_per_micron, experiment_temperature=None, age_factor=1):
    """Load default widths vs. age data and PCA basis for width smoothing.

    Parameters:
        pixels_per_micron: conversion factor for objective used.
        experiment_temperature: used to (crudely) correct the width-vs-age data
            for the current experimental temperature.
        age_factor: further time-multiplier for age data, to use as necessary.
            If your animals are growing faster than the default width estimator
            expects (even after adjusting for temperature), pass a value < 1;
            if growing slower pass a value > 1.

    Returns a WidthEstimator instance with the width profile and PCA basis.
    """
    with pkg_resources.resource_stream('elegant', 'width_data/width_trends.pickle') as f:
        trend_data = pickle.load(f)

    if experiment_temperature is not None:
        # NB: width_trends.pickle currently is based on an experiment at 23.5C
        age_factor *= calculate_temp_factor(experiment_temperature, ref_temperature=23.5)

    with pkg_resources.resource_stream('elegant', 'width_data/width_pca.pickle') as f:
        pca_data = pickle.load(f)
    pca_basis = pca_data['pcs']

    # NB: width_trends.pickle currently has ages in days: convert to hours
    return WidthEstimator(trend_data['width_trends']*pixels_per_micron, trend_data['ages']*24*age_factor, pca_basis)

def calculate_temp_factor(experiment_temperature, ref_temperature):
    # Average developmental-timing factors from Table 2 of Byerly, Cassada and Russell 1976
    time_factors = [1.90, 1.37, 1]
    temps = [16, 19.5, 25]
    # assume time scaling factors are roughly linear in this range...
    time_out, time_in = numpy.interp([experiment_temperature, ref_temperature], temps, time_factors)
    return time_out / time_in

class WidthEstimator:
    def __init__(self, width_trends, ages=None, pca_basis=None):
        """Calculate the average width profiles based on age, and PCA smooth widths.

        Parameters:
            width_trends: array of shape (m, n), where m is the number of points
                along the width profile and n is the number of ages at which the
                profile was measured. I.e. entry width_trends[a, b] is the mean
                width at position a along the profile at age b. If a 1d array is
                provided, then that will be the age-independent mean width profile.
            ages: array/list of ages at which the mean width profile was measured
                (length n). May be None, if only a 1d array of widths is provided.
            pca_basis: numpy.ndarray of shape (n, m) providing an orthonormal
                basis for the widths, where the number of basis vectors is n and
                their dimensionality is m. Used to perform PCA-based smoothing of
                the width profile by projecting a given profile into the PCA basis.

        Example:
            estimator = default_width_data(pixels_per_micron=1)[0]
            width_profile = estimator.widths_for_age(36) # profile of average widths at 36 hours
        """
        width_trends = numpy.asarray(width_trends)
        if ages is None:
            ages = [0]
            assert width_trends.ndim == 1
            width_trends = numpy.transpose([width_trends])
        self.ages = numpy.asarray(ages)
        assert len(self.ages) == width_trends.shape[1]
        self.mean_age = self.ages.mean()
        self.width_trends = width_trends

        if pca_basis is not None:
            pca_basis = numpy.asarray(pca_basis)
            if not numpy.allclose((pca_basis**2).sum(axis=1), numpy.ones(len(pca_basis))):
                raise ValueError('a unit-length (non-normalized) PCA basis must be provided')
        self.pca_basis = pca_basis

    def width_profile_for_age(self, age):
        """Return the default width profile for a worm of the given age.

        Parameter:
            age: age in hours, or None for the default widths
        """
        if age is None:
            age = self.mean_age
        return numpy.array([numpy.interp(age, self.ages, wt) for wt in self.width_trends])

    @functools.lru_cache(maxsize=64) # cache commonly-used masks
    def width_tck_for_age(self, age):
        """Return the default width tck for a worm of the given age.

        Parameter:
            age: age in hours, or None for the default widths
        """
        return self._to_tck(self.width_profile_for_age(age))

    def pca_smooth_widths(self, width_tck, mean_widths):
        """Return PCA-smoothed worm widths.

        Parameters:
            width_tck: spline to be smoothed
            mean_widths: mean width profile (not tck) for the worm in question
                (can be obtained using width_profile_for_age method, e.g.)
        """
        if self.pca_basis is None:
            return None
        basis_shape = self.pca_basis.shape[1]
        x = numpy.linspace(0, 1, basis_shape)
        mean_shape = mean_widths.shape[0]
        if mean_shape != basis_shape:
            mean_widths = numpy.interp(x, numpy.linspace(0, 1, mean_shape), mean_widths)
        widths = interpolate.spline_evaluate(width_tck, x)
        pca_projection = numpy.dot(widths - mean_widths, self.pca_basis.T)
        pca_reconstruction = mean_widths + numpy.dot(pca_projection, self.pca_basis)
        return self._to_tck(pca_reconstruction)

    @staticmethod
    def _to_tck(widths):
        x = numpy.linspace(0, 1, len(widths))
        smoothing = 0.0625 * len(widths)
        return interpolate.fit_nonparametric_spline(x, widths, smoothing=smoothing)

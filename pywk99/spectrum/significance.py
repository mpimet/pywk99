"""Asses the spectral significance in the difference of two spectra."""
import random

import numpy as np
import xarray as xr

from pywk99.spectrum.background import _smooth_spectrum


def bootstrap_mean_difference_test(spectra_a,
                                   spectra_b,
                                   alpha=0.05,
                                   resamplings=1000):
    bootstrap_distances = _sample_with_replacement(spectra_a,
                                                   spectra_b,
                                                   resamplings)
    ci_lower, ci_upper = _compute_confidence_intervals(bootstrap_distances,
                                                       alpha)
    significant_regions = np.logical_or(ci_lower > 0,
                                        ci_upper < 0).rename("significant")
    bootstrap_results = xr.merge([ci_lower, ci_upper, significant_regions])
    return bootstrap_results


def get_log_distance_statistic(spectra_a, spectra_b):
    wk_spectrum_a = (sum(spectra_a) / len(spectra_a))
    wk_spectrum_b = (sum(spectra_b) / len(spectra_b))
    log_distance = np.log10(wk_spectrum_b/wk_spectrum_a)
    return log_distance


def _sample_with_replacement(spectra_a, spectra_b, resamplings):
    if isinstance(resamplings, int):
        resamplings = range(resamplings)
    log_distances = []
    for i in resamplings:
        sampled_spectra_a = random.choices(spectra_a, k=len(spectra_a))
        sampled_spectra_b = random.choices(spectra_b, k=len(spectra_b))
        log_distance_sampled = get_log_distance_statistic(sampled_spectra_a,
                                                          sampled_spectra_b)
        log_distance_sampled = log_distance_sampled.assign_coords(
            {"bootstrap_iteration": i}
            )
        log_distances.append(log_distance_sampled)
    bootstrap_distances = xr.concat(log_distances, dim="bootstrap_iteration")
    return bootstrap_distances


def _compute_confidence_intervals(bootstrap_distances, alpha):
    quantile_lower = alpha/2
    quantile_upper = (1 - alpha/2)
    ci_lower = bootstrap_distances.quantile(
        quantile_lower, dim="bootstrap_iteration"
        ).rename("lower").drop("quantile")
    ci_upper = bootstrap_distances.quantile(
        quantile_upper, dim="bootstrap_iteration"
        ).rename("upper").drop("quantile")
    ci_lower = _smooth_spectrum(ci_lower)
    ci_upper = _smooth_spectrum(ci_upper)
    return ci_lower, ci_upper
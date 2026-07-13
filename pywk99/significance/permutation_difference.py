
import random
from typing import Callable, Union

import numpy as np
import xarray as xr
import itertools
import operator

from pywk99.spectrum.background import smooth_spectrum
from pywk99.significance.statistics import log_distance_statistic
from pywk99.significance.statistics import coh2_distance_statistic

STATISTIC_FUNCTIONS = {"log_distance" : log_distance_statistic,
                       "coh2_distance" : coh2_distance_statistic}

def permutation_difference_test(
    spectra_a: list[Union[xr.DataArray, xr.Dataset]],
    spectra_b: list[Union[xr.DataArray, xr.Dataset]],
    statistic: str = "log_distance",
    alpha: float = 0.05,
    resamplings: int = None,
    all_permutations = False,
    smooth_passes: int = 0,
) -> xr.Dataset:
    """Test the significance in the difference between two spectra."""
    statistic_function = STATISTIC_FUNCTIONS[statistic]
    if smooth_passes > 0:
        spectra_a = [smooth_spectrum(s, passes=smooth_passes) for s in spectra_a]
        spectra_b = [smooth_spectrum(s, passes=smooth_passes) for s in spectra_b]
    if all_permutations:
        resamplings = 0
        permutation_diffs = _all_permutations(
            spectra_a, spectra_b, statistic_function
        )
    elif resamplings:
        permutation_diffs = _sample_permutation(
            spectra_a, spectra_b, statistic_function, resamplings
        )
    else:
        raise ValueError("Provide resamplings or make all_permutations = True")
    ci_lower, ci_upper = _compute_quantiles(
        permutation_diffs, alpha
    )
    mean_diff = statistic_function(spectra_a, spectra_b)
    mean_diff.name = statistic
    significant_regions = np.logical_or(mean_diff < ci_lower,
                                        mean_diff > ci_upper)
    significant_regions.name = "significant"
    bootstrap_results = xr.merge([ci_lower,
                                  ci_upper,
                                  mean_diff,
                                  significant_regions])
    bootstrap_results.attrs = permutation_diffs.attrs
    bootstrap_results.attrs["alpha"] = alpha
    bootstrap_results.attrs["resamplings"] = resamplings
    bootstrap_results.attrs["spectra_a_windows_num"] = len(spectra_a)
    bootstrap_results.attrs["spectra_b_windows_num"] = len(spectra_b)
    return bootstrap_results


def _sample_permutation(
    spectra_a, spectra_b, statistic_function, resamplings
):
    all_spectra = spectra_a + spectra_b
    log_distances = []
    for i in range(resamplings):
        random.shuffle(all_spectra)
        sampled_spectra_a = all_spectra[:len(spectra_a)]
        sampled_spectra_b = all_spectra[len(spectra_a):]
        log_distance_sampled = statistic_function(
            sampled_spectra_a, sampled_spectra_b
        )
        log_distance_sampled = log_distance_sampled.assign_coords(
            {"bootstrap_iteration": i}
        )
        log_distances.append(log_distance_sampled)
    bootstrap_distances = xr.concat(log_distances, dim="bootstrap_iteration")
    bootstrap_distances.attrs = dict(resamplings=i+1,
                                     method="random")
    return bootstrap_distances


def _all_permutations(
    spectra_a, spectra_b, statistic_function
):
    spectra = spectra_a + spectra_b
    len_b = len(spectra_b)
    len_t = len(spectra)
    index_t = range(len_t)
    distances = []
    for i, combination_b in enumerate(itertools.combinations(index_t, len_b)):
        combination_a = tuple(set(index_t) - set(combination_b))
        sampled_spectra_a = operator.itemgetter(*combination_a)(spectra)
        sampled_spectra_b = operator.itemgetter(*combination_b)(spectra)
        distance_sampled = statistic_function(
            sampled_spectra_a, sampled_spectra_b
        )
        distance_sampled = distance_sampled.assign_coords(
            {"bootstrap_iteration": i}
        )
        distances.append(distance_sampled)
    bootstrap_distances = xr.concat(distances, dim="bootstrap_iteration")
    bootstrap_distances.attrs = dict(resamplings=i+1,
                                     method="all_permutations")
    return bootstrap_distances


def _compute_quantiles(permutation_diffs, alpha):
    quantile_lower = alpha / 2
    quantile_upper = 1 - alpha / 2
    ci_lower = permutation_diffs.quantile(
        quantile_lower, dim="bootstrap_iteration"
    ).drop("quantile")
    ci_upper = permutation_diffs.quantile(
        quantile_upper, dim="bootstrap_iteration"
    ).drop("quantile")
    ci_lower.name = "lower"
    ci_upper.name = "upper"
    return ci_lower, ci_upper


def log_distance_statistic(spectra_a, spectra_b):
    wk_spectrum_a = sum(spectra_a) / len(spectra_a)
    wk_spectrum_b = sum(spectra_b) / len(spectra_b)
    log_distance = np.log10(wk_spectrum_b / wk_spectrum_a)
    return log_distance
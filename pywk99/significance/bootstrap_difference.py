"""Asses the significance in the difference of two spectra."""

import random
from typing import Callable, Union

import numpy as np
import xarray as xr

from pywk99.spectrum.background import smooth_spectrum
from pywk99.significance.statistics import log_distance_statistic

STATISTIC_CALLABLE = Callable[
    [Union[xr.DataArray, xr.Dataset], Union[xr.DataArray, xr.Dataset]],
    xr.Dataset,
]


def bootstrap_difference_test(
    spectra_a: list[Union[xr.DataArray, xr.Dataset]],
    spectra_b: list[Union[xr.DataArray, xr.Dataset]],
    statistic_function: STATISTIC_CALLABLE = None,
    alpha: float = 0.05,
    resamplings: int = 1000,
) -> xr.Dataset:
    """Test the significance in the difference between two spectra."""
    if statistic_function is None:
        statistic_function = log_distance_statistic
    bootstrap_distances = _sample_with_replacement(
        spectra_a, spectra_b, statistic_function, resamplings
    )
    ci_lower, ci_upper = _compute_confidence_intervals(
        bootstrap_distances, alpha
    )
    significant_regions = np.logical_or(ci_lower > 0, ci_upper < 0)
    significant_regions.name = "significant"
    bootstrap_results = xr.merge([ci_lower, ci_upper, significant_regions])
    return bootstrap_results


def _sample_with_replacement(
    spectra_a, spectra_b, statistic_function, resamplings
):
    if isinstance(resamplings, int):
        resamplings = range(resamplings)
    log_distances = []
    for i in resamplings:
        sampled_spectra_a = random.choices(spectra_a, k=len(spectra_a))
        sampled_spectra_b = random.choices(spectra_b, k=len(spectra_b))
        log_distance_sampled = statistic_function(
            sampled_spectra_a, sampled_spectra_b
        )
        log_distance_sampled = log_distance_sampled.assign_coords(
            {"bootstrap_iteration": i}
        )
        log_distances.append(log_distance_sampled)
    bootstrap_distances = xr.concat(log_distances, dim="bootstrap_iteration")
    return bootstrap_distances


def _compute_confidence_intervals(bootstrap_distances, alpha):
    quantile_lower = alpha / 2
    quantile_upper = 1 - alpha / 2
    ci_lower = bootstrap_distances.quantile(
        quantile_lower, dim="bootstrap_iteration"
    ).drop("quantile")
    ci_upper = bootstrap_distances.quantile(
        quantile_upper, dim="bootstrap_iteration"
    ).drop("quantile")
    ci_lower = smooth_spectrum(ci_lower)
    ci_upper = smooth_spectrum(ci_upper)
    ci_lower.name = "lower"
    ci_upper.name = "upper"
    return ci_lower, ci_upper

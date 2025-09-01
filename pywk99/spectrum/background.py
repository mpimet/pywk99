"""Smooth the Power Spectrum witha 121 filter as Wheeler and Kiladis, 1999."""

import numpy as np
import xarray as xr


def get_background_spectrum(symmetric_spectrum: xr.DataArray,
                            asymmetric_spectrum: xr.DataArray) -> xr.DataArray:
    """Get the background spectrum from wheeler and kiladis"""
    new_spectrum = (symmetric_spectrum + asymmetric_spectrum)/2
    new_spectrum = smooth_spectrum(new_spectrum)
    return new_spectrum


def smooth_spectrum(spectrum: xr.DataArray, passes: int = 10) -> xr.DataArray:
    """Smooth the Power Spectrum with a 121 filter."""
    new_spectrum = spectrum.copy()
    rows, columns = np.shape(spectrum)
    # looping over rows and vector as spectrum matrix size is small
    for _ in range(passes):
        for row in range(rows):
            new_spectrum[row, :] = _filter_121(new_spectrum[row, :])
        for column in range(columns):
            new_spectrum[:, column] = _filter_121(new_spectrum[:, column])
    return new_spectrum


def _filter_121(variable: np.ndarray) -> np.ndarray:
    new_variable = variable.copy()
    kernel = np.array([1, 2, 1])/4
    new_variable[0] = (3*new_variable[0] + new_variable[1])/4
    new_variable[-1] = (3*new_variable[-1] + new_variable[-2])/4
    new_variable[1:-1] = np.convolve(variable, kernel, 'valid')
    return new_variable

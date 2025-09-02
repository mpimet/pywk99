"""Smooth the Power Spectrum with a 121 filter as Wheeler and Kiladis, 1999."""

import numpy as np
import xarray as xr
from scipy.ndimage import convolve1d


def get_background_spectrum(symmetric_spectrum: xr.DataArray,
                            asymmetric_spectrum: xr.DataArray) -> xr.DataArray:
    """Get the background spectrum from wheeler and kiladis"""
    new_spectrum = (symmetric_spectrum + asymmetric_spectrum)/2
    new_spectrum = smooth_spectrum(new_spectrum)
    return new_spectrum


def smooth_spectrum(spectrum: xr.DataArray, passes: int = 10) -> xr.DataArray:
    """Smooth the Power Spectrum with a 121 filter."""
    kernel = np.array([1, 2, 1]) / 4
    arr = spectrum.values.copy()
    for _ in range(passes):
        arr = convolve1d(arr, kernel, axis=0, mode="nearest")
        arr = convolve1d(arr, kernel, axis=1, mode="nearest")
    return xr.DataArray(arr, coords=spectrum.coords, dims=spectrum.dims)


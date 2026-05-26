"""Take Discrete Fourier Transforms of Variables."""

from typing import Any
import xarray as xr
import numpy as np


def fourier_transform(variable: xr.DataArray) -> xr.DataArray:
    """Take the Discrete Fourier Transform of a variable."""
    variable = variable.transpose("time", "lon", ...)
    variable_array = variable.values
    spectrum_array = np.fft.fftn(variable_array, axes=[0, 1])
    spectrum_array = np.fft.fftshift(spectrum_array, axes=[0, 1])
    # construct dataarray
    frequency = _get_frequencies(variable)
    wavenumbers = _get_wavenumbers(variable)
    remaining_dims = list(variable.dims)[2:]
    new_dims = ["frequency", "wavenumber"] + remaining_dims
    new_coords = ([("frequency", frequency, {"units": "Cycles per Day"}),
                   ("wavenumber", wavenumbers, {"units": "Zonal Wavenumber"})] +
                  [(dim, variable[dim].data, variable[dim].attrs)
                   for dim in remaining_dims])
    spectrum = xr.DataArray(
        data=spectrum_array,
        dims=new_dims,
        coords=new_coords
    )
    return spectrum


def inverse_fourier_transform(spectrum: xr.DataArray,
                              xarray_coords) -> xr.DataArray:
    """Take the Inverse Discrete Fourier Transform of a spectrum."""
    if not 'lon' in xarray_coords and not 'time' in xarray_coords:
        raise ValueError("Either dimension 'lon' or dimension 'time' not " +
                         "present in coords!")
    spectrum_array = spectrum.transpose("frequency", "wavenumber", ...).values
    variable_array = np.fft.ifftshift(spectrum_array, axes=[0, 1])
    variable_array = np.real(np.fft.ifftn(variable_array, axes=[0, 1]))
    # construct dataarray
    remaining_dims = list(spectrum.dims)[2:]
    new_dims = ["time", "lon"] + remaining_dims
    new_coords = ([xarray_coords['time'], xarray_coords['lon']] +
                  [(dim, spectrum[dim].data, spectrum[dim].attrs)
                   for dim in remaining_dims])
    variable = xr.DataArray(data=variable_array,
                            coords=new_coords,
                            dims=new_dims
                            )
    variable = variable.transpose("time", "lon", ...)
    return variable


def _get_frequencies(variable: xr.DataArray) -> np.ndarray:
    """Get the frequency in cycles per day after a numpy fftshift."""
    sampling_rate = variable.time[1].values - variable.time[0].values
    sampling_rate = sampling_rate.astype("timedelta64[s]").astype(float)
    sampling_rate = sampling_rate / 86400 # in cycles per day
    sampling_fs = 1 / sampling_rate
    n_time = len(variable.time)
    fourier_sequence = np.fft.fftfreq(n_time, 1/n_time).astype(int)
    fourier_sequence = np.fft.fftshift(fourier_sequence)
    frequency = fourier_sequence * sampling_fs / n_time
    return frequency


def _get_wavenumbers(variable: xr.DataArray) -> np.ndarray:
    """Get the zonal wavenumbers of the fft spectrum after a numpy fftshift."""
    n_lon = len(variable.lon)
    wavenumber = np.fft.fftfreq(n_lon, 1/n_lon).astype(int)
    wavenumber = np.fft.fftshift(wavenumber)
    wavenumber = -wavenumber  # positive wave number is eastward in WK99
    return wavenumber


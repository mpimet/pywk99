"""Compute a Power Spectrum following Wheeler and Kiladis, 1999."""

from typing import Optional, Union
import numpy as np
import xarray as xr
import pandas as pd

from pywk99.timeseries.fourier import fourier_transform
from pywk99.timeseries.timeseries import check_for_exactly_two_variables
from pywk99.timeseries.timeseries import convert_to_dataset
from pywk99.timeseries.timeseries import check_for_one_max_two_variables
from pywk99.timeseries.timeseries import check_time_index_has_expected_frequency
from pywk99.timeseries.timeseries import check_variable_coordinates_are_sorted
from pywk99.timeseries.timeseries import taper_variable_time_ends
from pywk99.timeseries.timeseries import remove_linear_trend
from pywk99.grid import dataset_to_equatorial_latlon_grid

_VALID_SEASONS = ["DJF", "MAM", "JJA", "SON"]


def get_spectrum(spc_quantity: str,
                 variable: Union[xr.DataArray, xr.Dataset],
                 component_type: str,
                 data_frequency: Optional[str] = None,
                 window_length: Optional[str] = None,
                 overlap_length: Optional[str] = None,
                 season: Optional[str] = None,
                 min_periods_season: Optional[int] = None,
                 taper_alpha: Optional[float] = None,
                 grid_type: str = None,
                 grid_dict: Optional[dict] = None) -> xr.DataArray:
    """
    Get a Wheeler and Kiladis 1999 power, amplitude or cross spectrum.

    Parameters
    ----------
    spc_quantity: str
        The type of spectra to compute. Either "power", "amplitude", or
        "cross".
    variable : xr.DataArray or xr.Dataset
        An xarray DataArray or Dataset with "time", "lon", "lat" coordinates.
        In case a Dataset is provided, then it must have exactly 2 data
        variable, and requires to use spc_quantity = "cross".
    component_type : str
        Refers to symmetry about the equator. Either "symmetric", "asymmetric",
        or "full".
    data_frequency : str, optional
        Time difference between consecutive times. Estimated by default.
    window_length : str, optional, default "96D"
        Length of time segments to use for individual power spectrums.
    overlap_length : str, optional, default "60D"
        Overlap of time segments.
    season : str, optional
        Either "DJF", "MAM", "JJA", or "SON". Note that 'window_length' and
        'overlap_length' still control the length of the analysis window.
    min_periods_season : str, optional
        Defines the minimum number of periods, within the window, that must
        correspond to the selected season. To be used only when season is
        defined. For instance, for variable with a data frequency of a day,
        setting 'min_periods_season' to 30 would indicate that at least 30 days
        of the data within each window must be in the selected season. By
        default it is set to int(0.25*overlap_length/data_frequency).
    taper_alpha: float, optional
        Alpha value determining the shape of the Tukey window filter function.
    grid_type: str, optional, default "latlon"
        The type of grid of the dataarray. Either "latlon" or "healpix". If
        "healpix" then a grid_dict must be also provided.
    grid_dict: dict, optional
        A dictionary with grid metadata. Used when grid_type = "healpix". The
        dictionary must have keys for "nside", "nested" and "minmax_lat".

    Returns
    -------
    wk_spectrum : xr.DataArray
        Datarray with "frequency" and "wavenumber" coordinates. The zonal
        wavenumber-frequency power spectra for the variable, summed over all
        "lat" coordinates.

    Raises
    ------
    ValueError
        If the data frequency of the variable is not the same at all times.
        If it is not possible to construct overlapping data segments.
        If variable coordinates are not sorted.
        If the season is not recognized.
    """
    # process inputs
    check_for_one_max_two_variables(variable)
    variable = convert_to_dataset(variable)
    variable = dataset_to_equatorial_latlon_grid(variable,
                                                 grid_type,
                                                 grid_dict)
    check_variable_coordinates_are_sorted(variable)
    window_length_np = pd.Timedelta(window_length).to_numpy()
    overlap_length_np = pd.Timedelta(overlap_length).to_numpy()
    data_frequency_np = _get_data_frequency(variable, data_frequency)
    if min_periods_season is None:
        min_periods_season = int(0.25 * overlap_length_np / data_frequency_np)
    # define variable segments
    variable_segments = _get_successive_overlapping_segments(
        variable, window_length_np, overlap_length_np, data_frequency_np)
    if season is not None:
        variable_segments = _choose_segments_within_season(
            variable_segments, season, min_periods_season)
    # construct spectrum
    wk_spectrums = [
        _one_segment_spectrum(variable_segment, spc_quantity, component_type,
                              data_frequency_np, taper_alpha)
        for variable_segment in variable_segments
    ]
    number_of_spectrums = len(wk_spectrums)
    wk_spectrum = (sum(wk_spectrums) / number_of_spectrums).sum("lat")
    wk_spectrum = wk_spectrum[wk_spectrum.frequency > 0]
    wk_spectrum = wk_spectrum.sortby(["frequency", "wavenumber"])
    return wk_spectrum


def _get_data_frequency(
        variable: xr.Dataset,
        data_frequency: Optional[str] = None) -> np.timedelta64:
    """Get the data_frequency to consider for the analysis."""
    if data_frequency:
        data_frequency_np = pd.Timedelta(data_frequency).to_numpy()
    else:
        data_frequency_np = (variable.time[1] - variable.time[0]).values
    return data_frequency_np


def _one_segment_spectrum(variable_segment: xr.Dataset, spc_quantity: str,
                          component_type: str,
                          data_frequency: np.timedelta64,
                          taper_alpha: float) -> xr.DataArray:
    """Get the spectrum of a variable for one time segment."""
    check_time_index_has_expected_frequency(variable_segment, data_frequency)
    new_segment = remove_linear_trend(variable_segment)
    new_segment = taper_variable_time_ends(new_segment, taper_alpha)
    if component_type != 'full':
        new_segment = _get_symmetry_component(new_segment, component_type)
    wk_spectrum = _compute_hayashi_spectrum(new_segment, spc_quantity)
    return wk_spectrum


def _compute_hayashi_spectrum(variable: xr.Dataset,
                              spc_quantity: str) -> xr.DataArray:
    """Get the spectrum of the spectral quantity for all latitudes"""
    spectrum_functions = {
        "power": _compute_hayashi_power_spectrum,
        "amplitude": _compute_hayashi_amplitude_spectrum,
        "cross": _compute_hayashi_cross_spectrum
    }
    spectrum_function = spectrum_functions[spc_quantity]
    spectrum = spectrum_function(variable)
    spectrum = spectrum.where(spectrum.frequency >= 0, drop=True)
    return spectrum


def _compute_hayashi_power_spectrum(variable: xr.Dataset) -> xr.DataArray:
    n_time = len(variable.time)
    n_lon = len(variable.lon)
    varname = list(variable.keys())[0]
    variable_fft = fourier_transform(variable[varname])
    spectrum = np.abs(variable_fft / (n_time * n_lon))**2
    return spectrum


def _compute_hayashi_amplitude_spectrum(
        variable: xr.Dataset) -> xr.DataArray:
    n_time = len(variable.time)
    n_lon = len(variable.lon)
    varname = list(variable.keys())[0]
    variable_fft = fourier_transform(variable[varname])
    spectrum = np.abs(variable_fft / (n_time * n_lon))
    return spectrum


def _compute_hayashi_cross_spectrum(variables: xr.Dataset) -> xr.DataArray:
    check_for_exactly_two_variables(variables)
    n_time = len(variables.time)
    n_lon = len(variables.lon)
    varlist = list(variables.keys())
    variable1_fft = fourier_transform(variables[varlist[0]]) / (n_time * n_lon)
    variable2_fft = fourier_transform(variables[varlist[1]]) / (n_time * n_lon)
    cross_spectrum = variable1_fft * np.conj(variable2_fft)
    return cross_spectrum


def _get_symmetry_component(variable: xr.Dataset,
                            component_type: str = "symmetric") -> xr.Dataset:
    """Get the symmetric of asymmetric components of the variable."""
    new_variables_dict = dict()
    for variable_name in list(variable.keys()):
        new_var = _get_symmetry_component_datarray(variable[variable_name],
                                                   component_type)
        new_variables_dict[variable_name] = new_var
    new_variable = xr.Dataset(new_variables_dict)
    return new_variable


def _get_symmetry_component_datarray(variable: xr.DataArray,
                                     component_type) -> xr.DataArray:
    """Get the symmetric of asymmetric components of the variable."""
    new_variable = variable.copy().transpose("time", "lon", "lat")
    array_p = new_variable.isel(lat=slice(None, None, 1)).values
    array_m = new_variable.isel(lat=slice(None, None, -1)).values
    if component_type == "symmetric":
        new_variable.values = (array_p + array_m) / 2
    elif component_type == "asymmetric":
        new_variable.values = (array_p - array_m) / 2
    else:
        raise ValueError(f"Unknown component_type '{component_type}'")
    return new_variable


def _get_successive_overlapping_segments(
        variable: xr.Dataset, window_length: np.timedelta64,
        overlap_length: np.timedelta64,
        data_frequency: np.timedelta64) -> list[xr.DataArray]:
    """Get successive dataarrays for the provided variable."""
    start = variable.time[0].values
    end = variable.time[-1].values
    intervals = _get_successive_overlapping_time_intervals(
        start, end, window_length, overlap_length, data_frequency)
    da_segments = []
    for interval in intervals:
        time_slice = slice(interval[0], interval[1], None)
        da_segment = variable.sel(time=time_slice)
        da_segments.append(da_segment)
    return da_segments


def _get_successive_overlapping_time_intervals(
    start: np.datetime64, end: np.datetime64, window_length: np.timedelta64,
    overlap_length: np.timedelta64, data_frequency: np.timedelta64
) -> list[tuple[np.datetime64, np.datetime64]]:
    """Get successive overlapping segments between two dates."""
    if start > end:
        raise ValueError("Interval start must occur before end.")
    if overlap_length > window_length:
        raise ValueError("overlap_length must be smaller than window_length.")
    data_length = (end - start) + data_frequency
    if data_length < window_length:
        raise ValueError(
            "window_length must be smaller than length between dates.")
    step_length = window_length - overlap_length
    # define the number of step intervals
    maximum_number_of_steps = int(np.floor(data_length / step_length))
    number_steps_per_window_length = int(np.ceil(window_length / step_length))
    number_of_steps = (maximum_number_of_steps -
                       number_steps_per_window_length + 1)
    if (start + number_of_steps * step_length + window_length -
            data_frequency) <= end:
        number_of_steps = number_of_steps + 1
    # compute the intervals dates
    interval_times = [
        (start + n * step_length,
         start + n * step_length + window_length - data_frequency)
        for n in range(number_of_steps)
    ]
    return interval_times


def _choose_segments_within_season(
    variable_segments: list[xr.Dataset],
    season: str,
    min_periods_season: int,
) -> list[xr.Dataset]:
    """Choose segments for a specific season and remove the rest."""
    if season not in _VALID_SEASONS:
        raise ValueError(f"'season' must be one of {_VALID_SEASONS}")
    variable_segments = [
        segment for segment in variable_segments
        if np.sum(season == segment.time.dt.season) >= min_periods_season
    ]
    return variable_segments

"""Filter for equatorial wave bands following Wheeler and Kiladis 1999."""

from typing import Optional, Union, Tuple
import xarray as xr
import numpy as np

from pywk99.timeseries.fourier import fourier_transform
from pywk99.timeseries.fourier import inverse_fourier_transform
from pywk99.timeseries.timeseries import taper_variable_time_ends
from pywk99.timeseries.timeseries import remove_linear_trend
from pywk99.timeseries.timeseries import remove_seasonal_cycle
from pywk99.timeseries.timeseries import check_variable_coordinates_are_sorted
from pywk99.filter.window import FilterPoint, FilterWindow
from pywk99.grid.grid import dataarray_to_equatorial_latlon_grid


def filter_variable(
    variable: xr.DataArray,
    filter_windows: Union[FilterWindow, list[FilterWindow]],
    taper: bool = True,
    taper_alpha: float = 0.5,
    rm_seasonal_cycle: bool = True,
    rm_linear_trend: bool = True,
    grid_type: str = "latlon",
    grid_dict: Optional[dict] = None,
) -> xr.Dataset:
    new_variable = dataarray_to_equatorial_latlon_grid(
        variable, grid_type, grid_dict
    )
    new_variable = _preprocess(
        new_variable, taper, taper_alpha, rm_seasonal_cycle, rm_linear_trend
    )
    spectrum = fourier_transform(new_variable)
    if isinstance(filter_windows, FilterWindow):
        filter_windows = [filter_windows]
    data_vars = dict()
    seen_window_names = []
    for window in filter_windows:
        field_name = _set_field_name(window.name, seen_window_names)
        seen_window_names.append(window.name)
        data_vars[field_name] = _filter(spectrum, window, new_variable.coords)
    filtered_variables = xr.Dataset(data_vars=data_vars)
    return filtered_variables


def modify_spectrum(
    spectrum: xr.DataArray, filter_window: FilterWindow, action: str = "filter"
) -> xr.DataArray:
    """Filter the spectrum with the wave filter window."""
    mask = _get_window_mask(spectrum, filter_window, action)
    modified_spectrum = mask * spectrum
    return modified_spectrum


def _filter(
    spectrum: xr.DataArray, filter_window: FilterWindow, xarray_coords
) -> xr.DataArray:
    masked_spectrum = modify_spectrum(spectrum, filter_window, "filter")
    filtered_variable = inverse_fourier_transform(
        masked_spectrum, xarray_coords
    )
    return filtered_variable


def _preprocess(
    variable: Union[xr.DataArray, xr.Dataset],
    taper: bool,
    taper_alpha: float,
    rm_seasonal_cycle: bool,
    rm_linear_trend: bool,
) -> Union[xr.DataArray, xr.Dataset]:
    check_variable_coordinates_are_sorted(variable)
    modified_variable = variable.transpose("time", "lon", ...)
    if rm_linear_trend:
        modified_variable = remove_linear_trend(modified_variable)
    if rm_seasonal_cycle:
        modified_variable = remove_seasonal_cycle(modified_variable)
    if taper:
        modified_variable = taper_variable_time_ends(
            modified_variable, taper_alpha
        )
    return modified_variable


def _set_field_name(window_name: str, seen_window_names: list[str]) -> str:
    name_count = seen_window_names.count(window_name)
    if name_count == 0:
        return window_name
    else:
        return f"{window_name}{name_count + 1}"


def _get_window_mask(
    spectrum: xr.DataArray, filter_window: FilterWindow, action: str = "filter"
) -> xr.DataArray:
    """Get a wavenumber-frequency mask corresponding to the filter window."""
    bbox_wavenumbers, bbox_frequencies = (
        _window_bbox_wavenumber_and_frequencies(spectrum, filter_window)
    )
    mask = _get_mask_base(spectrum, action)
    include_fft_reflection = bool(np.any(spectrum.frequency.values < 0))
    for wavenumber in bbox_wavenumbers.values:
        for frequency in bbox_frequencies.values:
            mask = _modify_mask_value_at_point(
                mask,
                filter_window,
                action,
                wavenumber,
                frequency,
                include_fft_reflection,
            )
    return mask


def _window_bbox_wavenumber_and_frequencies(
    spectrum: xr.DataArray, filter_window: FilterWindow
) -> Tuple[np.ndarray, np.ndarray]:
    """Get the wavenumbers and frequencies of the window bounding box."""
    k_wmin, w_wmin, k_wmax, w_wmax = filter_window.bounds
    if not np.all(np.diff(spectrum.wavenumber.values) > 0):
        aux = k_wmin
        k_wmin = k_wmax
        k_wmax = aux
    wavenumbers = spectrum.wavenumber.sel(wavenumber=slice(k_wmin, k_wmax))
    frequencies = spectrum.frequency.sel(frequency=slice(w_wmin, w_wmax))
    return wavenumbers, frequencies


def _get_mask_base(spectrum: xr.DataArray, action: str) -> xr.DataArray:
    if action == "filter":
        return xr.zeros_like(spectrum, dtype=bool)
    elif action == "substract":
        return xr.ones_like(spectrum, dtype=bool)
    else:
        raise ValueError(f"Unrecognized action '{action}'")


def _modify_mask_value_at_point(
    mask: xr.DataArray,
    filter_window: FilterWindow,
    action: str,
    wavenumber: float,
    frequency: float,
    include_fft_reflection: bool = True,
) -> xr.DataArray:
    ACTION_VALUE = {"filter": True, "substract": False}
    point = FilterPoint(wavenumber, frequency)
    point_is_contained = filter_window.covers(point)
    point_loc_dict1 = dict(wavenumber=wavenumber, frequency=frequency)
    if point_is_contained:
        mask.loc[point_loc_dict1] = ACTION_VALUE[action]
        if include_fft_reflection:
            # rounding errors in the index make the following necessary
            approx_point_loc_dict2 = dict(
                wavenumber=-wavenumber, frequency=-frequency
            )
            reflection_point = mask.sel(
                approx_point_loc_dict2, method="nearest"
            )
            point_loc_dict2 = dict(
                wavenumber=reflection_point.wavenumber.values,
                frequency=reflection_point.frequency.values,
            )
            mask.loc[point_loc_dict2] = ACTION_VALUE[action]
    return mask

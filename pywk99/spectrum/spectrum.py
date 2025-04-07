"""
Get Spectra following Wheeler and Kiladis, 1999.

The Spectra correspond to the average of the multiple overlapping
windows available on the dataset and summed over all latitudes for either the
symmetric or antisymmetric component
"""

from typing import Optional
import numpy as np
import xarray as xr

from pywk99.spectrum._spectrum import get_spectrum

_VALID_SEASONS = ["DJF", "MAM", "JJA", "SON"]


def get_power_spectrum(
    variable: xr.DataArray,
    component_type: str,
    data_frequency: Optional[str] = None,
    window_length: str = "96D",
    overlap_length: str = "60D",
    season: Optional[str] = None,
    min_periods_season: Optional[int] = None,
    taper_alpha: Optional[float] = 0.5,
    grid_type: str = "latlon",
    grid_dict: Optional[dict] = None,
) -> xr.DataArray:
    """
    Get the Wheeler and Kiladis 1999 power spectrum of a variable.

    See pywk99.spectrum.get_spectrum for argument documentation.
    """
    power_spectrum = get_spectrum(
        "power",
        variable,
        component_type,
        data_frequency,
        window_length,
        overlap_length,
        season,
        min_periods_season,
        taper_alpha,
        grid_type,
        grid_dict
    )
    return power_spectrum


def get_amplitude_spectrum(
    variable: xr.DataArray,
    component_type: str,
    data_frequency: Optional[str] = None,
    window_length: str = "96D",
    overlap_length: str = "60D",
    season: Optional[str] = None,
    min_periods_season: Optional[int] = None,
    taper_alpha: Optional[float] = 0.5,
    grid_type: str = "latlon",
    grid_dict: Optional[dict] = None,
) -> xr.DataArray:
    """
    Get the Wheeler and Kiladis 1999 amplitude spectrum of a variable.

    See pywk99.spectrum.get_spectrum for argument documentation.
    """
    amplitude_spectrum = get_spectrum(
        "amplitude",
        variable,
        component_type,
        data_frequency,
        window_length,
        overlap_length,
        season,
        min_periods_season,
        taper_alpha,
        grid_type,
        grid_dict
    )
    return amplitude_spectrum


def get_cross_spectrum(
    variable: xr.Dataset,
    component_type: str,
    data_frequency: Optional[str] = None,
    window_length: str = "96D",
    overlap_length: str = "60D",
    season: Optional[str] = None,
    min_periods_season: Optional[int] = None,
    taper_alpha: Optional[float] = 0.5,
    grid_type: str = "latlon",
    grid_dict: Optional[dict] = None,
) -> xr.DataArray:
    """
    Get the Wheeler and Kiladis 1999 cross spectrum of two variables.

    See pywk99.spectrum.get_spectrum for argument documentation.
    """
    cross_spectrum = get_spectrum(
        "cross",
        variable,
        component_type,
        data_frequency,
        window_length,
        overlap_length,
        season,
        min_periods_season,
        taper_alpha,
        grid_type,
        grid_dict
    )
    return cross_spectrum


def get_co_spectrum(
    variable: xr.Dataset,
    component_type: str,
    data_frequency: Optional[str] = None,
    window_length: str = "96D",
    overlap_length: str = "60D",
    season: Optional[str] = None,
    min_periods_season: Optional[int] = None,
    taper_alpha: Optional[float] = 0.5,
    grid_type: str = "latlon",
    grid_dict: Optional[dict] = None,
) -> xr.DataArray:
    """
    Get the Wheeler and Kiladis 1999 co-spectrum of two variables.

    See pywk99.spectrum.get_spectrum for argument documentation.
    """
    cross_spectrum = get_spectrum(
        "cross",
        variable,
        component_type,
        data_frequency,
        window_length,
        overlap_length,
        season,
        min_periods_season,
        taper_alpha,
        grid_type,
        grid_dict
    )
    co_spectrum = np.real(cross_spectrum)
    return co_spectrum


def get_quadrature_spectrum(
    variable: xr.Dataset,
    component_type: str,
    data_frequency: Optional[str] = None,
    window_length: str = "96D",
    overlap_length: str = "60D",
    season: Optional[str] = None,
    min_periods_season: Optional[int] = None,
    taper_alpha: Optional[float] = 0.5,
    grid_type: str = "latlon",
    grid_dict: Optional[dict] = None,
) -> xr.DataArray:
    """
    Get the Wheeler and Kiladis 1999 quadrature-spectrum of two variables.

    See pywk99.spectrum.get_spectrum for argument documentation.
    """
    cross_spectrum = get_spectrum(
        "cross",
        variable,
        component_type,
        data_frequency,
        window_length,
        overlap_length,
        season,
        min_periods_season,
        taper_alpha,
        grid_type,
        grid_dict
    )
    quadrature_spectrum = np.imag(cross_spectrum)
    return quadrature_spectrum

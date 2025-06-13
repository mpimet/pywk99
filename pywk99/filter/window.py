"""Define filtering windows for various waves following Wheeler and Kiladis."""

from dataclasses import dataclass
from typing import Optional, Tuple, Union
import numpy as np
from shapely.geometry import Point, Polygon, MultiPolygon

from pywk99.waves import LinearWave

DISPERSION_CURVE_POINTS = 100


class FilterPoint(Point):
    """See shapely.geometry.Point."""


@dataclass
class FilterWindow:
    name: str
    polygon: Union[Polygon, MultiPolygon]

    @property
    def bounds(self) -> Tuple[float, float, float, float]:
        return self.polygon.bounds

    def union(self, other) -> "FilterWindow":
        new_polygon = self.polygon.union(other.polygon)
        new_name = f"{self.name}_{other.name}"
        return FilterWindow(new_name, new_polygon)

    def covers(self, point: FilterPoint):
        return self.polygon.covers(point)


def get_mjo_window() -> FilterWindow:
    """
    Get a window commonly used for isolating the MJO.

    Returns
    -------
    wave_filter : FilterWindow
        A window to filter in wavenumber-frequency space.
    """
    mjo_window = get_box_filter_window(1, 5, 0.001, 0.04, name="mjo")
    return mjo_window


def get_tropical_depression_window() -> FilterWindow:
    """
    Get a window commonly used for isolating tropical depressions.

    Returns
    -------
    wave_filter : FilterWindow
        A window to filter in wavenumber-frequency space.
    """
    name = "tropical_depression"
    polygon = Polygon([(-20.0, 0.3), (-20.0, 0.5), (-6, 0.33), (-6, 0.13)])
    td_window = FilterWindow(name, polygon)
    return td_window


def get_box_filter_window(k_min: float, k_max: float,
                          w_min: float, w_max: float,
                          name: Optional[str] = None) -> FilterWindow:
    """
    Get a box filter on the wavenumber-frequency space.

    Parameters
    ----------
    k_min : float
        Minimum wavenumber to include in the filter.
    k_max : float
        Maximum wavenumber to include in the filter.
    w_min : float
        Minimum frequency, in cycles per day, to include in the filter.
    w_max : float
        Maximum frequency, in cycles per day, to include in the filter.
    name : str, optional
        Name of the window. Default is "box".

    Returns
    -------
    wave_filter : FilterWindow
        A window to filter in wavenumber-frequency space.
    """
    if not name:
        name = "box"
    polygon = Polygon.from_bounds(k_min, w_min, k_max, w_max)
    box_window = FilterWindow(name, polygon)
    return box_window


def get_wave_filter_window(wave_type: str,
                           k_min: float, k_max: float,
                           w_min: float, w_max: float,
                           h_min: float, h_max: float) -> FilterWindow:
    """
    Get a polygon representing the filter in wavenumber-frequency space.

    Parameters
    ----------
    wave_type : str
        Either "mixed_rossby_gravity", "inertio_gravity", "kelvin",
        "equatorial_rossby", or "gravity".
    k_min : float
        Minimum wavenumber to include in the filter.
    k_max : float
        Maximum wavenumber to include in the filter.
    w_min : float
        Minimum frequency, in cycles per day, to include in the filter.
    w_max : float
        Maximum frequency, in cycles per day, to include in the filter.
    h_min : float
        Minimum equivalent height to include in the filter.
    h_max : float
        Maximum equivalent height to include in the filter.

    Returns
    -------
    wave_filter : FilterWindow
        A window to filter in wavenumber-frequency space.
    """
    wave_frequency_polygon = Polygon.from_bounds(k_min, w_min, k_max, w_max)
    dispersion_polygon = _get_dispersion_curves_polygon(
        wave_type, k_min, k_max, h_min, h_max
    )
    polygon = wave_frequency_polygon.intersection(dispersion_polygon)
    wave_window = FilterWindow(wave_type, polygon)
    return wave_window


def _get_dispersion_curves_polygon(wave_type: str,
                                   k_min: float, k_max: float,
                                   h_min: float, h_max: float) -> Polygon:
    """Get a polygon defined by two dispersions of the equivalent depths."""
    wavenumber = np.linspace(k_min, k_max, DISPERSION_CURVE_POINTS)
    min_omega = LinearWave(wave_type, h_min).frequency(wavenumber)
    max_omega = LinearWave(wave_type, h_max).frequency(wavenumber)
    valid_min = ~np.isnan(min_omega)
    valid_max = ~np.isnan(max_omega)
    coords = list(zip(wavenumber[valid_min], min_omega[valid_min]))
    coords = coords + list(zip(np.flip(wavenumber[valid_max]),
                               np.flip(max_omega[valid_max])))
    dispersion_polygon = Polygon(coords)
    return dispersion_polygon


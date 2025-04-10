"""Plot dispersion relations as seen in Wheeler and Kiladis, 1999."""
from typing import Optional, Tuple
from matplotlib import pyplot as plt
import numpy as np

from pywk99.waves.waves import wk_waves, individual_waves


def plot_dispersion_relations(
        component_type: str,
        ax: Optional[plt.Axes] = None,
        k_min: float = -14,
        k_max: Optional[float] = 14,
        w_min: Optional[float] = 0,
        w_max: Optional[float] = 0.8,
        equivalent_depths: Optional[list[float]] = None,
        color = 'grey',
        lw = 0.5,
        ls = '-') -> None:
    """Plot the dispersion crurves of the waves in WK diagrams"""
    if ax is None:
        ax = plt.gca()
    if not equivalent_depths:
        equivalent_depths = [8, 12, 25, 50, 90]
    wave_number = np.linspace(k_min, k_max, 100)
    waves = wk_waves(component_type, equivalent_depths)
    for wave in waves:
        omega = wave.frequency(wave_number)
        ax.plot(wave_number, omega, color=color, lw=lw, ls=ls)
    _set_axis_limits(ax, k_min, k_max, w_min, w_max)

def plot_individual_dispersion_relations(
        wave_list: list[Tuple[str, int]],
        ax: Optional[plt.Axes] = None,
        k_min: float = -14,
        k_max: Optional[float] = 14,
        w_min: Optional[float] = 0,
        w_max: Optional[float] = 0.8,
        equivalent_depths: Optional[list[float]] = None,
        color = 'grey',
        lw = 0.5,
        ls = '-') -> None:
    """Plot the dispersion crurves of individual waves in spectra"""
    if ax is None:
        ax = plt.gca()
    if not equivalent_depths:
        equivalent_depths = [8, 12, 25, 50, 90]
    wave_number = np.linspace(k_min, k_max, 100)
    waves = individual_waves(wave_list, equivalent_depths)
    for wave in waves:
        omega = wave.frequency(wave_number)
        ax.plot(wave_number, omega, color=color, lw=lw, ls=ls)
    _set_axis_limits(ax, k_min, k_max, w_min, w_max)


def _set_axis_limits(ax: plt.Axes, k_min: float, k_max: float, w_min: float,
              w_max: float) -> None:
    ax.set_xlim(k_min, k_max)
    ax.set_ylim(w_min, w_max)

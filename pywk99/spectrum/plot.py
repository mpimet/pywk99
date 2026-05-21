"""Create Spectrum plots as seen in Wheeler and Kiladis, 1999."""

from typing import Optional, Union

from matplotlib import pyplot as plt
import xarray as xr
import numpy as np

from pywk99.filter.window import FilterWindow
from pywk99.filter.filter import modify_spectrum


def plot_spectrum(spectrum: xr.DataArray,
                  ax: Optional[plt.Axes] = None,
                  k_min: Optional[float] = -14,
                  k_max: Optional[float] = 14,
                  w_min: Optional[float] = 0,
                  w_max: Optional[float] = 0.8,
                  flagged_windows: Optional[FilterWindow] = None,
                  use_log: bool = True,
                  cmap: str = "magma",
                  **kwargs) -> None:
    """Plot a longitude-time Power Spectrum."""
    plot_spectrum = spectrum.copy()
    if ax is None:
        ax = plt.gca()
    if flagged_windows:
        plot_spectrum = modify_spectrum(
            plot_spectrum, flagged_windows, 'substract'
            )
    if use_log:
        with np.errstate(divide='ignore'):
            plot_spectrum = np.log10(plot_spectrum)
    plot_spectrum.sel(
        wavenumber=slice(k_min, k_max)
    ).plot.contourf(levels=50,
                    cmap=cmap,
                    ax=ax,
                    **kwargs)
    _set_axis_limits(ax, k_min, k_max, w_min, w_max)


def plot_spectrum_peaks(spectrum: xr.DataArray,
                        background: xr.DataArray,
                        ax: Optional[plt.Axes] = None,
                        k_min: Optional[float] = -14,
                        k_max: Optional[float] = 14,
                        w_min: Optional[float] = 0,
                        w_max: Optional[float] = 0.8,
                        flagged_windows: Optional[FilterWindow] = None,
                        ) -> None:
    """Plot the statistically significant pearks of at lon-time Spectrum."""
    spectrum_peaks = spectrum/background
    if ax is None:
        ax = plt.gca()
    if flagged_windows:
        spectrum_peaks = modify_spectrum(
            plot_spectrum, flagged_windows, 'substract'
            )
    spectrum_peaks.sel(wavenumber=slice(k_min, k_max)
                       ).plot.contourf(levels=np.arange(1.1, 2.0, 0.1),
                                       cmap="afmhot_r",
                                       ax=ax)
    _set_axis_limits(ax, k_min, k_max, w_min, w_max)


def plot_coherence(cross_spectrum: xr.DataArray,
                   ax: Optional[plt.Axes] = None,
                   coherence_threshold: Optional[float] = 0.05,
                   k_min: Optional[float] = -14,
                   k_max: Optional[float] = 14,
                   w_min: Optional[float] = 0,
                   w_max: Optional[float] = 0.8,
                  flagged_windows: Optional[FilterWindow] = None,
                   cmap: str = "YlOrRd",
                   **kwargs) -> None:
    """Plot a longitude-time Power Spectrum."""
    plot_spectrum = cross_spectrum.copy()
    if ax is None:
        ax = plt.gca()
    if flagged_windows:
        plot_spectrum = modify_spectrum(
            plot_spectrum, flagged_windows, 'substract'
            )
    # compute coherence and phase arrows
    coh2 = plot_spectrum.coherence_squared
    coh2 = coh2.where(coh2 > coherence_threshold, drop=True)
    # plot
    coh2.sel(
        wavenumber=slice(k_min, k_max)
    ).plot.contourf(levels=50,
                    cmap=cmap,
                    ax=ax,
                    **kwargs)
    _set_axis_limits(ax, k_min, k_max, w_min, w_max)


def plot_phase_arrows(cross_spectrum: xr.DataArray,
                      ax: Optional[plt.Axes] = None,
                      coherence_threshold: Optional[float] = 0.05,
                      stride: Optional[float] = 1,
                      flagged_windows: Optional[FilterWindow] = None,
                      **kwargs) -> None:
    """Plot a longitude-time Power Spectrum."""
    plot_spectrum = cross_spectrum.copy()
    if ax is None:
        ax = plt.gca()
    if flagged_windows:
        plot_spectrum = modify_spectrum(
            plot_spectrum, flagged_windows, 'substract'
            )
    # compute coherence and phase arrows
    coh2 = plot_spectrum.coherence_squared
    coh2 = coh2.where(coh2 > coherence_threshold, drop=True)
    u = xr.apply_ufunc(np.real, 1j * np.conj(plot_spectrum.cross) )
    u = u / np.abs(plot_spectrum.cross)
    v = xr.apply_ufunc(np.imag, 1j * np.conj(plot_spectrum.cross) )
    v = v / np.abs(plot_spectrum.cross)
    u = u.where(coh2 > coherence_threshold, drop=True)
    v = v.where(coh2 > coherence_threshold, drop=True)
    # plot
    ax.quiver(u.wavenumber[::stride],
              u.frequency[::stride],
              u[::stride,::stride],
              v[::stride,::stride],
              angles="uv", pivot='mid', units='x', **kwargs)


def _set_axis_limits(ax: plt.Axes, k_min: float, k_max: float, w_min: float,
                     w_max: float) -> None:
    ax.set_xlim(k_min, k_max)
    ax.set_ylim(w_min, w_max)

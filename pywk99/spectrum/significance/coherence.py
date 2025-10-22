"""Assert the significance of spectra."""

import xarray as xr


def coherence_squared_significance(coherence_squared: xr.DataArray,
                                   alpha: float,
                                   dof: int) -> xr.DataArray:
    if alpha < 0 or alpha > 1.0:
        raise ValueError("alpha must be between 0 an 1")
    significant_level = 1 - alpha ** (1 / (dof - 1))
    significant = coherence_squared >= significant_level
    significant.attrs["alpha"] = alpha
    significant.attrs["dof"] = dof
    significant.attrs["significant_coherence_level"] = significant_level
    return significant
"""Convert datasets to latlon grid"""

from typing import Optional
import xarray as xr

from pywk99.grid.healpix import healpix_to_equatorial_latlon


def convert_to_equatorial_latlon_grid(
    dataset: xr.Dataset, grid_type: str, grid_dict: Optional[dict]
) -> xr.Dataset:
    if grid_type == "latlon":
        return dataset
    elif grid_type == "healpix":
        if grid_dict is None:
            raise ValueError("No grid_dict provided for healpix conversion.")
        return healpix_to_equatorial_latlon(dataset, **grid_dict)
    else:
        raise ValueError("Grid type not found.")

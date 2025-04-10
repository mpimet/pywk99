"""Convert datasets to latlon grid"""

from typing import Optional
import xarray as xr


from pywk99.grid.healpix import dataarray_healpix_to_equatorial_latlon
from pywk99.grid.healpix import dataset_healpix_to_equatorial_latlon


def dataset_to_equatorial_latlon_grid(
    dataset: xr.Dataset, grid_type: str, grid_dict: Optional[dict]
) -> xr.Dataset:
    if grid_type == "latlon":
        return dataset
    elif grid_type == "healpix":
        if grid_dict is None:
            raise ValueError("No grid_dict provided for healpix conversion.")
        return dataset_healpix_to_equatorial_latlon(dataset, **grid_dict)
    else:
        raise ValueError("Grid type not found.")


def dataarray_to_equatorial_latlon_grid(
    dataarray: xr.DataArray, grid_type: str, grid_dict: Optional[dict]
) -> xr.DataArray:
    if grid_type == "latlon":
        return dataarray
    elif grid_type == "healpix":
        if grid_dict is None:
            raise ValueError("No grid_dict provided for healpix conversion.")
        return dataarray_healpix_to_equatorial_latlon(dataarray, **grid_dict)
    else:
        raise ValueError("Grid type not found.")
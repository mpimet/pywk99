"""Test healpix transformation to latlongirds."""

import pytest

import healpy as hp
import xarray as xr
import numpy as np
import pandas as pd

from pywk99.grid.healpix import dataset_healpix_to_equatorial_latlon


@pytest.fixture
def variable():
    nside = 32  #  zoom 6
    npix = hp.nside2npix(nside)
    cell = hp.reorder(np.arange(npix), r2n=True)
    time = pd.date_range(
        start="2000-01-01 06:00", end="2000-01-02 06:00", freq="12h"
    )
    lon, lat = hp.pix2ang(nside, cell, lonlat=True)
    data = np.exp(-((lat) ** 2) / (5**2)) * (np.sin(5 * np.deg2rad(lon)))
    data = np.reshape(data, (-1, 1))
    data = np.tile(data, len(time)).T
    dataarray = xr.DataArray(
        data=data, dims=("time", "cell"), coords={"time": time, "cell": cell}
    )
    variable = xr.Dataset(
        {"olr": dataarray, "olr2": dataarray},
        coords={
            "crs": (
                "crs",
                [np.nan],
                {
                    "grid_mapping_name": "healpix",
                    "healpix_nside": 32,
                    "healpix_order": "nest",
                },
            )
        },
    )
    return variable


@pytest.fixture
def latlon_variable(variable):
    grid_dict = dict(nside=32, nest=True, minmax_lat=20)
    latlon_variable = dataset_healpix_to_equatorial_latlon(
        variable, **grid_dict
    )
    return latlon_variable


def test_transformation_has_4xnside_equatorial_points(latlon_variable) -> None:
    assert len(latlon_variable.lon) == 128


def test_transformation_keeps_variables(latlon_variable) -> None:
    assert len(latlon_variable.data_vars) == 2


def test_transformation_keeps_time(latlon_variable, variable) -> None:
    assert np.all(latlon_variable.time == variable.time)


def test_transformation_has_equatorial_lats(latlon_variable) -> None:
    assert np.all(latlon_variable.lat <= 20)
    assert np.all(latlon_variable.lat >= -20)

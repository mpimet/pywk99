import pandas as pd
import pytest

import numpy as np
import xarray as xr
import healpy as hp

from pywk99.filter.filter import filter_variable
from pywk99.filter.window import get_box_filter_window


@pytest.fixture
def variable():
    """Variable with frequency 15/360 CPD and wavenumber 5"""
    nside = 32  #  zoom 6
    npix = hp.nside2npix(nside)
    cell = hp.reorder(np.arange(npix), r2n=True)
    time = pd.date_range("2020-01-01", freq="1D", periods=365)
    lon, lat = hp.pix2ang(nside, cell, lonlat=True)
    data = np.ones(shape=(len(time), len(cell)))
    time_cycle = np.cos(15 * 2 * np.pi * np.arange(len(time)) / len(time))
    lon_cycle = np.cos(2 * np.deg2rad(lon)) * np.exp(-(lat**2) / 15**2)
    data = 10 + ((data * lon_cycle).T * time_cycle).T
    variable = xr.DataArray(
        data, dims=["time", "cell"], coords={"time": time, "cell": cell}
    )
    return variable


def test_filter_works_with_healpix_data(variable):
    filter_window = get_box_filter_window(0, 10, 0 / 360, 5 / 360)
    test_filtered_variable = filter_variable(
        variable,
        filter_window,
        grid_type="healpix",
        grid_dict={"nside": 32, "nest": True, "minmax_lat": 15},
    )
    assert (2 * test_filtered_variable).std() < 0.1 / 100

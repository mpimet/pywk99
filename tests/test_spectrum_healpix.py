"""Test power spectrum for healpix data fields."""
import pytest

import pandas as pd
import healpy as hp
import xarray as xr
import numpy as np

from pywk99.spectrum.spectrum import get_power_spectrum

@pytest.fixture
def variable():
    nside = 32 #  zoom 6
    npix = hp.nside2npix(nside)
    cell = hp.reorder(np.arange(npix), r2n=True)
    time = pd.date_range(start="2000-01-01 06:00",
                        end="2001-01-01 06:00",
                        freq="12h")
    lon, lat = hp.pix2ang(nside, cell, lonlat=True)
    data = np.exp(-((lat) ** 2) / (5 ** 2))*(np.sin(5 * np.deg2rad(lon)))
    data = np.reshape(data, (-1, 1))
    data = np.tile(data, len(time)).T
    dataarray = xr.DataArray(data=data,
                            dims=("time", "cell"),
                            coords={"time": time, "cell": cell})
    variable = xr.Dataset({"olr": dataarray},
                        coords ={"crs": ("crs", [np.nan],
                                            {'grid_mapping_name': 'healpix',
                                            'healpix_nside': 32,
                                            'healpix_order': 'nest'})})
    return variable


@pytest.fixture(params=["symmetric", "asymmetric"])
def spectrum(variable, request):
    component_type = request.param
    spectrum = get_power_spectrum(variable,
                                  component_type,
                                  window_length="30D",
                                  overlap_length="10D",
                                  grid_type="healpix",
                                  grid_dict={"nside": 32,
                                             "nest": True,
                                             "minmax_lat": 15})
    return spectrum


def test_power_spectrum_shape(spectrum):
    # from variable time segments ((time_points - 1)/2, 4*nside)
    assert np.shape(spectrum) == (29, 128)


def test_power_spectrum_frequency_between_zero_and_one(spectrum):
    assert np.all(spectrum.frequency.values == np.arange(1, 30)/29)


def test_power_spectrum_wavenumbers(spectrum):
    # note that positive wave number is eastward in WK99
    assert np.all(spectrum.wavenumber.values == np.arange(-63, 65))


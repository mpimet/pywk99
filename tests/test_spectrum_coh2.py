"""Test power spectrum for time-lon-lat fields."""

import numpy as np
import pytest
import xarray as xr

from pywk99.spectrum.spectrum import get_cross_spectrum


@pytest.fixture
def variables():
    olr1 = xr.open_dataarray("tests/olr.test.nc").transpose(
        "time", "lon", "lat"
    )
    olr2 = olr1.copy()
    variables = xr.Dataset({"olr1": olr1, "olr2": olr2})
    variables = variables.sortby(["lat"])
    return variables


@pytest.fixture(params=["symmetric", "asymmetric"])
def cross_spectrum(variables, request):
    component_type = request.param
    cross_spectrum = get_cross_spectrum(
        variables, component_type, window_length="30D", overlap_length="10D"
    )
    return cross_spectrum


def test_cross_spectrum_coherence_is_one(cross_spectrum) -> None:
    assert np.all(np.ravel(cross_spectrum.coh2.values) == pytest.approx(1.0))


def test_cross_spectrum_angle_is_zero(cross_spectrum) -> None:
    assert np.all(np.ravel(cross_spectrum.phase.values) == pytest.approx(0.0))

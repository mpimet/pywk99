"""Test fourier filter for time-lon-lat fields."""
import pytest

from pywk99.timeseries.fourier import fourier_transform
from pywk99.timeseries.fourier import inverse_fourier_transform
from pywk99.timeseries.timeseries import remove_linear_trend

import xarray as xr
import numpy as np




@pytest.fixture
def variable():
    variable = xr.open_dataarray("tests/olr.test.nc").transpose("time",
                                                                "lon",
                                                                "lat")
    return variable


@pytest.fixture
def variable_segment(variable):
    variable_segment = variable.isel(time=slice(0, 96*2+1))
    return variable_segment


def test_fourier_foward_and_backward(variable):
    variable = remove_linear_trend(variable)
    spectrum = fourier_transform(variable)
    test_variable = inverse_fourier_transform(spectrum,
                                              xarray_coords=variable.coords)
    difference = np.abs(test_variable - variable)
    assert difference.max() < 1E-12


def test_fourier_foward_and_backward_multiple_height(variable):
    variable = remove_linear_trend(variable)
    variable_twolayer = xr.concat([variable, variable], dim="height"
                                  ).assign_coords(height=[10., 20.])
    spectrum = fourier_transform(variable_twolayer)
    test_variable = inverse_fourier_transform(spectrum,
                                              xarray_coords=variable_twolayer.coords)
    difference = np.abs(test_variable - variable_twolayer)
    assert difference.max() < 1E-12


def test_fourier_assigned_frequency(variable_segment):
    ntime = len(variable_segment.time)
    spectrum = fourier_transform(variable_segment)
    sampling_frequency = 1/0.5 # CPD
    expected_positive_frequencies = [k*sampling_frequency/ntime
                                    for k in range(ntime//2+1)]
    expected_negative_frequencies = [k*sampling_frequency/ntime
                                    for k in range(-ntime//2+1,0,1)]
    expected_frequencies = (expected_negative_frequencies +
                            expected_positive_frequencies)
    expected_frequencies = np.array(expected_frequencies)
    assert np.all(expected_frequencies == spectrum.frequency)


def test_fourier_assigned_wavenumbers(variable_segment):
    spectrum = fourier_transform(variable_segment)
    expected_positive_wavenumbers = [val for val in range(72)]
    expected_negative_wavenumbers = [val for val in range(-72, 0, 1)]
    expected_wavenumbers = (expected_negative_wavenumbers +
                            expected_positive_wavenumbers)
    expected_wavenumbers = - np.array(expected_wavenumbers)
    assert np.all(expected_wavenumbers == spectrum.wavenumber)

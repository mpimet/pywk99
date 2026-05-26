"""Test power spectrum for time-lon-lat fields."""

import pandas as pd
import pytest

import xarray as xr
import numpy as np

from pywk99.spectrum.spectrum import get_power_spectrum
from pywk99.spectrum._spectrum import _choose_segments_within_season
from pywk99.spectrum._spectrum import _get_successive_overlapping_segments


@pytest.fixture
def variable():
    variable = xr.open_dataarray("tests/olr.test.nc").transpose("time",
                                                                "lon",
                                                                "lat")
    variable = variable.sortby(["lat"])
    return variable


@pytest.fixture(params=["symmetric", "asymmetric"])
def spectrum(variable, request):
    component_type = request.param
    spectrum = get_power_spectrum(variable, component_type,
                                  window_length="30D",
                                  overlap_length="10D")
    return spectrum


def test_successive_overlapping_segments_dates(variable):
    # define 90 days variable
    variable = variable.isel(time=slice(0, 90*2+1))
    # construct test segments
    window_length = np.timedelta64(30, "D")
    overlap_length = np.timedelta64(10, "D")
    data_frequency = np.timedelta64(12, "h")
    segments = _get_successive_overlapping_segments(variable,
                                                    window_length,
                                                    overlap_length,
                                                    data_frequency)
    # construct expected dates
    expected_intervals = [(np.datetime64("2000-01-01T06:00:00.000000000"),
                           np.datetime64("2000-01-30T18:00:00.000000000")),
                          (np.datetime64("2000-01-21T06:00:00.000000000"),
                           np.datetime64("2000-02-19T18:00:00.000000000")),
                          (np.datetime64("2000-02-10T06:00:00.000000000"),
                           np.datetime64("2000-03-10T18:00:00.000000000")),
                          (np.datetime64("2000-03-01T06:00:00.000000000"),
                           np.datetime64("2000-03-30T18:00:00.000000000"))]
    # test
    for i, segment in enumerate(segments):
        test_interval = (segment.time[0].values, segment.time[-1].values)
        expected_interval = expected_intervals[i]
        assert test_interval == expected_interval


def test_power_spectrum_shape(spectrum):
    # from variable time segments ((time_points - 1)/2, lon_points)
    assert np.shape(spectrum) == (29, 144)


def test_power_spectrum_frequency(spectrum):
    assert np.all(spectrum.frequency.values == np.arange(1, 30)/30)


def test_power_spectrum_wavenumbers(spectrum):
    # note that positive wave number is eastward in WK99
    assert np.all(spectrum.wavenumber.values == np.arange(-71, 73))


def test_passing_data_frequency_as_argument_works(variable) -> None:
    spectrum = get_power_spectrum(variable,
                                  "symmetric",
                                  data_frequency="12H",
                                  window_length="30D",
                                  overlap_length="10D")
    assert np.shape(spectrum) == (29, 144)
    assert np.all(spectrum.frequency.values == np.arange(1, 30)/30)
    assert np.all(spectrum.wavenumber.values == np.arange(-71, 73))


def test_spectrum_for_specific_one_season(variable) -> None:
    spectrum = get_power_spectrum(variable,
                                  "symmetric",
                                  data_frequency="12H",
                                  window_length="30D",
                                  overlap_length="10D",
                                  season="MAM")
    assert np.shape(spectrum) == (29, 144)
    assert np.all(spectrum.frequency.values == np.arange(1, 30)/30)
    assert np.all(spectrum.wavenumber.values == np.arange(-71, 73))


def test_spectrum_seasons_must_be_valid_ones(variable) -> None:
    with pytest.raises(ValueError):
        get_power_spectrum(variable,
                           "symmetric",
                           window_length="30D",
                           overlap_length="10D",
                           season="AAA")


def test_choose_segments_within_season(variable) -> None:
    season = "MAM"
    min_periods_season = 10
    window_length = np.timedelta64(96, "D")
    overlap_length = np.timedelta64(60, "D")
    data_frequency = np.timedelta64(12, "h")
    variable_segments = _get_successive_overlapping_segments(
            variable, window_length, overlap_length, data_frequency
        )
    variable_segments = _choose_segments_within_season(variable_segments,
                                                       season,
                                                       min_periods_season)
    for segment in variable_segments:
        assert np.sum(segment.time.dt.season == season) >= min_periods_season


def test_spectrum_has_dof_for_single_variate(spectrum) -> None:
    assert spectrum.attrs["dof_single_variate"]


def test_spectrum_has_number_of_windows(spectrum) -> None:
    assert spectrum.attrs["number_of_windows"]


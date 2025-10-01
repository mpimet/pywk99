"""Test the computation of the degrees of freedom"""

import pandas as pd
from pywk99.spectrum._dof import degrees_of_freedom_for_single_variate


def test_dof_for_wk99_case():
    latitudes = [-15, 15]
    start = pd.Timestamp("2001-01-01")
    end = pd.Timestamp("2018-01-01")
    window_length = pd.Timedelta("92D")
    season = None
    dof = degrees_of_freedom_for_single_variate(
        latitudes, start, end, window_length, season
    )
    assert dof == 1012 / 2


def test_dof_for_wk99_case_with_season():
    latitudes = [-15, 15]
    start = pd.Timestamp("2001-01-01")
    end = pd.Timestamp("2018-01-01")
    window_length = pd.Timedelta("92D")
    season = "JJA"
    dof = degrees_of_freedom_for_single_variate(
        latitudes, start, end, window_length, season
    )
    assert dof == int(253 / 2)
"""Conservatively estimate the degrees of freedom of a single variate."""

from typing import Optional, Sequence, Union
import pandas as pd


def degrees_of_freedom_for_single_variate(
    latitudes: Sequence[float],
    start: pd.Timestamp,
    end: pd.Timestamp,
    window_length: Union[str, pd.Timedelta],
    season: Optional[str] = None,
) -> Union[int, None]:
    """Conservatively estimate the degrees of freedom of a single variate."""
    independent_latitudes = _get_independent_latitudes(latitudes)
    period_length = _get_period_length(start, end, season)
    dof = _degrees_of_freedom_for_single_estimate(
        independent_latitudes, period_length, window_length
    )
    return dof


def _get_period_length(
    start: pd.Timestamp, end: pd.Timestamp, season: Optional[str] = None
) -> pd.Timedelta:
    """Get period length and assume is a forth for seasons."""
    period_length = pd.Timestamp(end) - pd.Timestamp(start)
    if season is not None:
        period_length = period_length / 4
    return period_length


def _get_independent_latitudes(latitudes: Sequence[float]) -> float:
    """Assign 0.25 DOF for each degree in the latitude range."""
    lat_range = max(latitudes) - min(latitudes)
    independent_latitudes = 0.25 * lat_range
    return independent_latitudes


def _degrees_of_freedom_for_single_estimate(
    independent_latitudes: float,
    period_length: pd.Timedelta,
    window_length: Union[str, pd.Timedelta],
) -> int:
    dof = int(
        independent_latitudes * period_length / pd.Timedelta(window_length)
    )
    return dof

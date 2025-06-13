"""Test waves windows"""

import pytest
from pywk99.filter.window import FilterPoint, FilterWindow
from pywk99.filter.window import get_wave_filter_window
from pywk99.filter.window import get_tropical_depression_window
from pywk99.filter.window import get_mjo_window

from shapely.geometry import Polygon


@pytest.fixture
def filter_window():
    shapely_polygon = Polygon.from_bounds(1, 1, 5, 5)
    filter_window = FilterWindow("test1", shapely_polygon)
    return filter_window


@pytest.fixture
def joined_filter_window(filter_window):
    polygon2 = Polygon.from_bounds(7, 7, 10, 10)
    filter_window2 = FilterWindow("test2", polygon2)
    joined_filter_window = filter_window.union(filter_window2)
    return joined_filter_window


def test_filter_window_has_name(filter_window):
    assert filter_window.name == "test1"


def test_filter_window_has_polygon(filter_window):
    assert filter_window.polygon == Polygon.from_bounds(1, 1, 5, 5)


def test_joining_multiple_windows_joins_polygons(joined_filter_window):
    polygon1 = Polygon.from_bounds(1, 1, 5, 5)
    polygon2 = Polygon.from_bounds(7, 7, 10, 10)
    expected_polygon = polygon1.union(polygon2)
    assert expected_polygon == joined_filter_window.polygon


def test_filter_window_bounds_method_relegate_to_shapely(joined_filter_window):
    bounds = joined_filter_window.bounds
    shapely_bounds = joined_filter_window.polygon.bounds
    assert bounds == shapely_bounds


def test_filter_window_cover_method_relegate_to_shapely(joined_filter_window):
    point1 = FilterPoint(100, 100)
    covers1 = joined_filter_window.covers(point1)
    shapely_covers1 = joined_filter_window.polygon.covers(point1)
    point2 = FilterPoint(3, 3)
    covers2 = joined_filter_window.covers(point2)
    shapely_covers2 = joined_filter_window.polygon.covers(point2)
    assert covers1 == shapely_covers1
    assert covers2 == shapely_covers2


def test_joining_filter_mixes_names(joined_filter_window):
    expected_filter_name = "test1_test2"
    assert expected_filter_name == joined_filter_window.name


@pytest.mark.parametrize("wave_type, wavenumber, frequency, is_inside",
    [("kelvin", 10, 0.1, False),
     ("kelvin", 10, 0.3, True),
     ("kelvin", 10, 0.5, False),
     ("kelvin", 16, 0.3, False)])
def test_wave_filter_window_bounds(wave_type, wavenumber, frequency, is_inside):
    window = get_wave_filter_window(wave_type, 1, 15, 0.0, 0.4, 8, 90)
    test_point = FilterPoint(wavenumber, frequency)
    assert window.covers(test_point) == is_inside


def test_mjo_window_function():
    mjo_window = get_mjo_window()
    assert isinstance(mjo_window, FilterWindow)


def test_tropical_depression_window_function():
    td_window = get_tropical_depression_window()
    assert isinstance(td_window, FilterWindow)


def test_window_name_can_be_changed(filter_window):
    filter_window.name = "new_name"
    assert filter_window.name == "new_name"

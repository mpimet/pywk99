"""Convert healpix data to a latlon grid"""

import xarray as xr
import numpy as np
import healpy as hp
from scipy import interpolate


MAXIMUM_LAT_RANGE = 25


def dataset_healpix_to_equatorial_latlon(
    dataset: xr.Dataset,
    nside: int,
    nest: str,
    minmax_lat: float
) -> xr.Dataset:
    """
    Extract a latlon dataarray from a healpix dataset.

    The latlon array extracted is for a band around the equator.
    """
    latlon_datarrays = []
    for variable_name in dataset.data_vars:
        dataarray = dataset[variable_name]
        latlon_datarray_aux = dataarray_healpix_to_equatorial_latlon(
            dataarray, nside, nest, minmax_lat
        )
        latlon_datarray_aux.name = variable_name
        latlon_datarrays.append(latlon_datarray_aux)
    return xr.merge(latlon_datarrays)


def dataarray_healpix_to_equatorial_latlon(
    healpix_dataarray: xr.DataArray,
    nside: int,
    nest: str,
    minmax_lat: float
) -> xr.DataArray:
    """
    Extract a latlon dataarray from a healpix dataset.

    The latlon array extracted is for a band around the equator.
    """
    if minmax_lat > MAXIMUM_LAT_RANGE:
        msg = (f"Selected latitudinal belt (minmax_lat = {minmax_lat}) is too "
               "wide for a meaningful analysis of equatorial waves.")
        raise ValueError(msg)
    # get data
    data = healpix_dataarray.values
    time = healpix_dataarray.time.values
    lat, lon = _get_pix_latlon(nside, nest)

    # get latitudes
    unique_lats = np.unique(np.round(lat, 10))
    unique_lats = unique_lats[unique_lats <= minmax_lat]
    unique_lats = unique_lats[unique_lats >= -minmax_lat]

    # get final longitudes
    final_lons = lon[np.where(np.round(lat, 10) == unique_lats[0])]
    final_lons = np.sort(final_lons)

    # sort and remap, if needed, to final longitudes
    ntime = len(time)
    nlon = len(final_lons)
    nlat = len(unique_lats)
    resampled_data = np.zeros((ntime, nlon, nlat))
    for i, unique_lat in enumerate(unique_lats):
        # get values
        ring_index = np.where(np.round(lat, 10) == unique_lat)[0]
        lons = lon[ring_index]
        data_ring = data[:, ring_index]
        # sort
        sorted_index = np.argsort(lons)
        sorted_lons = lons[sorted_index]
        sorted_data_ring = data_ring[:, sorted_index]
        # interpolate
        if i % 2 != 0:
            sorted_data_ring = _interp_array_along_first_axis(
                final_lons, sorted_lons, sorted_data_ring, period=360
            )
        resampled_data[:, :, i] = sorted_data_ring

    # save as latlon dataarray
    latlon_dataarray = xr.DataArray(
        data=resampled_data,
        dims=["time", "lon", "lat"],
        coords={"time": time, "lat": unique_lats, "lon": final_lons},
    )
    return latlon_dataarray


def _interp_array_along_first_axis(x, xp, fp, period):
    x = x % period
    xp = xp % period
    xp = np.concatenate((xp[-1:] - period, xp, xp[0:1] + period))
    fp = np.concatenate((fp[:, -1:], fp, fp[:, 0:1]), axis=1)
    interp_func = interpolate.interp1d(xp, fp)
    return interp_func(x)


def _get_pix_latlon(nside, nest):
    if nest is False:
        raise NotImplementedError("nest=False is not implemented.")
    npix = hp.nside2npix(nside)
    cell = hp.reorder(np.arange(npix), r2n=True)
    lon, lat = hp.pix2ang(nside, cell, lonlat=True)
    return lat, lon

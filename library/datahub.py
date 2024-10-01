import json
import os.path

import requests
import logging
from itertools import product

import math
import numpy as np
import pandas as pd
import xarray as xr
import meteostat as mt
from netCDF4 import Dataset
from scipy.interpolate import griddata

from tqdm import tqdm
from scipy import ndimage
from global_land_mask import globe

from library.decorators import retry_api_call

def get_station_data(domain, start_date, end_date):
    '''
    function to query weather stations data using meteostat

    :param domain:      list              - domain list (lon_min, lon_max, lat_min, lat_max)
    :param start_date:  datetime          - initial date
    :param end_date:    datetime          - end date
    '''

    logging.info(f"getting weather station into domain")

    stations = mt.Stations().bounds((domain[-1], domain[0]), (domain[2], domain[1])).fetch()

    station_df = []

    # loop into all stations
    for station_id, row in stations.iterrows():
        # get data for specified period
        data = mt.Daily(station_id, start=start_date, end=end_date).fetch()

        # include weather station info
        data['station'] = station_id
        data['latitude'] = row.latitude
        data['longitude'] = row.longitude
        data['elevation'] = row.elevation

        # append DataFrame to current weather station
        station_df.append(data)

    # concat all stations
    return pd.concat(station_df, axis=0)


def open_topo(topo_pathfile, grid_lat, grid_lon, nc_engine='netcdf4'):
    '''
    function to open topography netCDF file

    :param topo_pathfile:     str       - topography netCDF file
    :param grid_lat:          np.array  - latitude values of grid
    :param grid_lon:          np.array  - longitude values of grid
    :param nc_engine:         str       - netCDF engine, netcdf4 or xarray
    '''

    if not os.path.exists(topo_pathfile):
        raise FileNotFoundError(f"topography file not found @ {topo_pathfile}")

    if nc_engine in ['netcdf4']:

        # opening topography file
        nc = Dataset(topo_pathfile, 'r')

        # get coords
        lats_ = nc.variables['y'][:]
        lons_ = nc.variables['x'][:]

        needs_cut_and_interpol = np.any(lats_ != grid_lat) or np.any(lons_ != grid_lon)

        if needs_cut_and_interpol:
            # find closest index in topography file
            bounds_lon_idx = list(map(lambda coord: np.argmin(np.abs(coord - lons_)), [grid_lon[0], grid_lon[-1]]))
            bounds_lat_idx = list(map(lambda coord: np.argmin(np.abs(coord - lats_)), [grid_lat[0], grid_lat[-1]]))

            # lat and lon id bounds into a single list
            domain_idx = bounds_lon_idx + bounds_lat_idx
            # print(domain_idx)

            # ensure closed interval on right side
            for bound_id, max_idx in zip([1, 3], [len(lons_), len(lats_)]):
                if domain_idx[bound_id] < max_idx:
                    domain_idx[bound_id] += 1

            lats_ = lats_[domain_idx[2]:domain_idx[3]]
            lons_ = lons_[domain_idx[0]:domain_idx[1]]

            # open topo only in specified domain
            topo = nc.variables['z'][domain_idx[2]:domain_idx[3],
                   domain_idx[0]:domain_idx[1]]

        else:
            topo = nc.variables['z'][:]

        nc.close()

        if needs_cut_and_interpol:
            # interpolate topo array to expected resolution
            grid_lon_m, grid_lat_m = np.meshgrid(grid_lon, grid_lat)  # final grid coords
            lon_m, lat_m = np.meshgrid(lons_, lats_)  # topo coords

            # set topography points
            topo_points = np.array([lat_m.ravel(), lon_m.ravel()]).T

            # interpol topo to interpolated field resolution
            logging.info(f"interpolating topography file to correct grid")
            topo = griddata(topo_points, topo.ravel(), (grid_lat_m, grid_lon_m), method='nearest')

    elif nc_engine in ['xarray']:

        ds = xr.open_dataset(topo_pathfile).sel(y=slice(grid_lat[0], grid_lat[-1]),
                                                x=slice(grid_lon[0], grid_lon[-1]))

        lats_ = ds['y'].values
        lons_ = ds['x'].values
        needs_interpol = np.any(lats_ != grid_lat) or np.any(lons_ != grid_lon)

        if needs_interpol:
            logging.info(f"interpolating topography file to correct grid")
            ds = ds.interp(y=grid_lat, x=grid_lon, method='nearest')

            lats_ = ds['y'].values
            lons_ = ds['x'].values

        topo = ds['z'].values

    else:
        raise NotImplementedError(f"engine {nc_engine} not implemented")

    return topo, lats_, lons_


def gen_etopo(latitudes, longitudes, topo_pathfile='topography.nc', max_iterations=1000, save_output=True):
    '''
    function to create elevation netcdf file

    :param latitudes:      np.array  - grid latitudes array
    :param longitudes:     np.array  - grid longitude array
    :param topo_pathfile:  str       - output path name
    '''

    # create elevations array
    elevations = np.full((len(latitudes), len(longitudes)), np.nan)

    coordinates = list(product(latitudes, longitudes))

    # calculate the batch size to do open topo data requests
    total_coordinates = len(coordinates)
    batch_size = math.ceil(total_coordinates / max_iterations)

    # iteration by coords using batches
    for batch_start in tqdm(range(0, batch_size*((len(coordinates) + batch_size)//batch_size) + 1, batch_size), desc="Processing Coordinates"):

        if batch_start >= len(coordinates):
            continue

        batch_coords = coordinates[batch_start:min(batch_start + batch_size, len(coordinates))]
        elevations_batch = get_elevation_batch(batch_coords)

        # request response to grid
        for idx, (lat, lon) in enumerate(batch_coords):
            i = batch_start // len(longitudes)  # get latitude index
            j = (batch_start + idx) % len(longitudes)  # get longitude index
            elevations[i, j] = elevations_batch[idx]

    # avoiding nans
    mask = np.isnan(elevations)

    if np.any(mask):
        logging.info(f"nan values were found filling them @ fraction of nans {np.sum(mask)/elevations.size}")
        # applying nearest neighbor to fill nan
        elevations[mask] = ndimage.generic_filter(elevations, np.nanmean, size=3, mode='nearest')[mask]

    if save_output:
        # crete netcdf file
        dataset = Dataset(topo_pathfile, 'w', format='NETCDF4')

        # define dimensions
        dataset.createDimension('y', len(latitudes))
        dataset.createDimension('x', len(longitudes))

        # create variables
        latitudes_var = dataset.createVariable('y', np.float32, ('y',))
        longitudes_var = dataset.createVariable('x', np.float32, ('x',))
        elevations_var = dataset.createVariable('z', np.float32, ('y', 'x'),
                                                zlib=True, complevel=9, least_significant_digit=2)

        # update values
        latitudes_var[:] = latitudes
        longitudes_var[:] = longitudes
        elevations_var[:, :] = elevations

        logging.info(f"file {topo_pathfile} created")

        # close file
        dataset.close()

    else:
        return elevations


@retry_api_call(max_tries=5)
def get_elevation_batch(coordinates):
    '''
    Function to get elevation using OpenTopoData API, requesting values by batches

    :param coordinates:  list of tuples   - list of coordinates that represents the current batch
    '''

    if np.all([globe.is_ocean(lat, lon) for lat, lon in coordinates]):
        return [0 for _ in coordinates]

    locations = "|".join([f"{lat},{lon}" for lat, lon in coordinates])
    url = f"https://api.opentopodata.org/v1/etopo1?locations={locations}"
    response = requests.get(url)

    if response.status_code == 200:
        data = json.loads(response.text)
        return [result['elevation'] if result['elevation'] > 0 else 0 for result in data['results']]
    else:
        response.raise_for_status()  # this will trigger the retry mechanism if API fails


@retry_api_call(max_tries=5)
def get_elevation(lat, lon):
    '''
    Function to get elevation using OpenTopoData API, single coordinate

    :param lat:  float   - latitude value
    :param lon:  float   - longitude value
    '''

    if globe.is_ocean(lat, lon):
        return 0

    url = f"https://api.opentopodata.org/v1/etopo1?locations={lat},{lon}"
    response = requests.get(url)

    if response.status_code == 200:
        data = json.loads(response.text)
        return data['results'][0]['elevation']
    else:
        response.raise_for_status()  # this will trigger the retry mechanism if API fails
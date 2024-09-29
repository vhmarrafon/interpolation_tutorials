import json
import requests
import logging
from itertools import product

import math
import numpy as np
import netCDF4 as nc

from tqdm import tqdm
from scipy import ndimage
from global_land_mask import globe

from library.decorators import retry_api_call

# config logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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


    print((batch_start, batch_start+batch_size, len(coordinates)))
    # avoiding nans
    mask = np.isnan(elevations)

    if np.any(mask):
        logging.info(f"nan values were found filling then @ fraction of nans {elevations.size/np.sum(mask)}")
        # applying nearest neighbor to fill nan
        elevations[mask] = ndimage.generic_filter(elevations, np.nanmean, size=3, mode='nearest')[mask]

    if save_output:
        # crete netcdf file
        dataset = nc.Dataset(topo_pathfile, 'w', format='NETCDF4')

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
import time
import json
import requests
import logging
from itertools import product

import math
import numpy as np
import netCDF4 as nc

from tqdm import tqdm

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
    elevations = np.zeros((len(latitudes), len(longitudes)))

    coordinates = list(product(latitudes, longitudes))

    # calculate the batch size to do open topo data requests
    total_coordinates = len(coordinates)
    batch_size = math.ceil(total_coordinates / max_iterations)

    # iteration by coords using batches
    for batch_start in tqdm(range(0, batch_size*(len(coordinates) + batch_size)//batch_size, batch_size), desc="Processing Coordinates"):

        batch_coords = coordinates[batch_start:min(batch_start + batch_size, len(coordinates))]
        elevations_batch = get_elevation_batch(batch_coords)

        # request response to grid
        for idx, (lat, lon) in enumerate(batch_coords):
            i = batch_start // len(longitudes)  # get latitude index
            j = (batch_start + idx) % len(longitudes)  # get longitude index
            elevations[i, j] = elevations_batch[idx]

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



def get_elevation_batch(coordinates, current_try=0):
    '''
    function to get elevation using opentopodata API, requesting values by batches

    :param coordinates:  list of tuples   - list of coordinates that represents the current batch
    '''

    max_tries = 5
    locations = "|".join([f"{lat},{lon}" for lat, lon in coordinates])
    url = f"https://api.opentopodata.org/v1/etopo1?locations={locations}"
    response = requests.get(url)
    if response.status_code == 200:
        data = json.loads(response.text)
        return [result['elevation'] for result in data['results']]
    else:
        if current_try > max_tries:
            raise RuntimeError(f"max tries attempted in open topo data API, please try again")

        else:
            time.sleep(1)
            return get_elevation_batch(coordinates, current_try=current_try+1)


def get_elevation(lat, lon, current_try=0):
    '''
    function to get elevation using opentopodata API, single coordinate

    :param lat:  float   - latitude value
    :param lon:  float   - longitude value
    '''

    max_tries = 5
    url = f"https://api.opentopodata.org/v1/etopo1?locations={lat},{lon}"
    response = requests.get(url)
    if response.status_code == 200:
        data = json.loads(response.text)
        elevation = data['results'][0]['elevation']
        return elevation
    else:
        if current_try > max_tries:
            raise RuntimeError(f"max tries attempted in open topo data API, please try again")

        else:
            time.sleep(1)
            return get_elevation(lat, lon, current_try=current_try+1)



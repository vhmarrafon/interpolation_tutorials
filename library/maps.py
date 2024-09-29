import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import cartopy.feature as cfeature
import cartopy.crs as ccrs
from cartopy.mpl.gridliner import LATITUDE_FORMATTER, LONGITUDE_FORMATTER
import geopandas as gpd
from shapely.geometry import Point

from library.colors import ColorScale


# available lat/lon keys
LAT_KEYS = ['latitude', 'lat', 'latidutes', 'lats']
LON_KEYS = ['longitude', 'lon', 'longitudes', 'lons']



class PlotMap(ColorScale):

    def __init__(self, ds, verbose=False):
        '''
        :param ds xarray.Dataset:
        '''

        super().__init__()

        self.verbose = verbose

        if isinstance(ds, xr.Dataset):
            self.ds = ds

        else:
            raise ValueError(f'stdin error dataset must be a xarray Dataset @ {type(ds)} are passed')

        # renaming coords
        for lat_key, lon_key in zip(LAT_KEYS, LON_KEYS):
            if lat_key in self.ds.keys() and lat_key != 'latitude':
                self.ds = self.ds.rename({lat_key: 'latitude', lon_key: 'longitude'})

    def gen_mask(self, arr2D, gdf, lon_meshgrid, lat_meshgrid):

        coords = np.vstack((lon_meshgrid.flatten(), lat_meshgrid.flatten())).T

        points_gdf = gpd.GeoDataFrame(geometry=[Point(xy) for xy in coords])

        # Set the CRS for points_gdf to EPSG:4326
        points_gdf.crs = "EPSG:4326"

        # Perform a spatial join to identify points outside the polygons union
        points_within_polygons = gpd.sjoin(points_gdf, gdf, how='left', op='within')

        # Get the indices of the points within polygons
        indices_within_polygons = points_within_polygons.dropna().index.values

        # Create a mask to identify points outside the polygons union
        mask = np.zeros_like(arr2D, dtype=bool)
        mask[np.unravel_index(indices_within_polygons, mask.shape)] = True

        # Apply the mask to the array, replacing values outside the polygons union with np.nan
        arr2D[~mask] = np.nan

        return arr2D

    def __translator(self, **kwargs):

        # set cmap
        self.c = self.cmap(kwargs.get('cmap', 'c1'))

        # set gxout
        self.gxout = kwargs.get('gxout', 'shaded')

        # set time
        ds = self.ds.isel(time=kwargs.get("sett", 0))

        # set lats
        if kwargs.get('setlat'):
            lats = kwargs.get('setlat')
            ds = ds.sel(latitude=slice(lats[0], lats[1]))

        # set lons
        if kwargs.get('setlon'):
            lons = kwargs.get('setlon')
            ds = ds.sel(longitude=slice(lons[0], lons[1]))

        # set level
        if kwargs.get('setlev'):
            ds = ds.sel(level=kwargs.get('setlev'))

        # set color levels
        self.clevs = kwargs.get('clevs')

        # set display cbar
        self.cbar = kwargs.get('cbar', True)

        # color bar extension
        self.extend = kwargs.get('extend', 'both')

        # set cbar label
        self.clabel = kwargs.get('clabel', '')

        # set title
        self.title = kwargs.get('title', '')
        
        # set member
        self.member = kwargs.get('setmember', 0)

        # output dpi
        self.dpi = kwargs.get('dpi', 300)

        return ds

    def display(self, v, **kwargs):
        '''
        :param v str: variable name
        '''

        ds = self.__translator(**kwargs)

        # removing exists image attr
        if hasattr(self, 'im'):
            del self.im

        # applying command on array
        if kwargs.get('cmd') or kwargs.get('command'):
            cmd = kwargs.get('cmd') if kwargs.get('cmd') else kwargs.get('command')
            if cmd != '':
                arr = eval('ds[v].values' + cmd)

        else:
            arr = np.squeeze(ds[v].values)
            
            
        if len(arr.shape) > 2:
            arr = arr[self.member, :, :]


        # figure object
        fig = plt.figure(figsize=kwargs.get('figsize', (10, 8)))

        # Create axis
        ax = fig.add_subplot(111, projection=ccrs.PlateCarree())
        
        ax.set_extent([ds['longitude'].values[0], ds['longitude'].values[-1],
                       ds['latitude'].values[0], ds['latitude'].values[-1]])

        # griding
        lon, lat = np.meshgrid(ds.longitude.values, ds.latitude.values)

        if kwargs.get("shapefile") is not None:
            arr = self.gen_mask(arr, kwargs.get("shapefile"), lon, lat)

        # Ploting variable
        if self.gxout in ['shaded']:
            im = ax.contourf(lon, lat, arr,
                             levels=self.clevs, cmap=self.c,
                             transform=ccrs.PlateCarree(), extend=self.extend)

        elif self.gxout in ['contour']:
            im = ax.contour(lon, lat, arr,
                            levels=self.clevs, cmap=self.c,
                            transform=ccrs.PlateCarree(), extend=self.extend)

        if not kwargs.get("ocean", True):
            ax.add_feature(cfeature.OCEAN.with_scale("50m"), zorder=3, facecolor="white")


        # Add colorbar
        if self.cbar:
            cbar1_position = [0.91, 0.24, 0.05, 0.5]
            cax1 = fig.add_axes(cbar1_position)
            cb1 = plt.colorbar(im, cax=cax1, orientation='vertical') #, shrink=0.7, aspect=10, pad=0.03)
            # cb1.set_label('Accuracy (%)', fontsize=18)
            cb1.set_ticks(self.clevs)
            cb1.ax.tick_params(labelsize=15)


            # cbar = plt.colorbar(im, ax=ax, pad=0.06, fraction=0.023)
            # cbar.set_label(label=self.clabel, size=20)
            # cbar.ax.tick_params(labelsize=12)

        # coast
        ax.add_feature(cfeature.COASTLINE)

        # countries
        ax.add_feature(cfeature.BORDERS)
        
        if kwargs.get("BR", True):
            # brazilian states
            states = cfeature.NaturalEarthFeature(category='cultural',
                                                  name='admin_1_states_provinces',
                                                  scale='50m',
                                                  facecolor='none')

            ax.add_feature(states, edgecolor='k', linestyle='--')
            
        else:
            ax.add_feature(cfeature.STATES)

        # title
        ax.set_title(self.title, fontsize=20)

        # gridlines
        g1 = ax.gridlines(crs=ccrs.PlateCarree(), linestyle='--', color='gray', draw_labels=True, alpha=0.8)

        # drop right and top labels
        g1.right_labels = False
        g1.top_labels = False

        # lat and lon labels
        g1.yformatter = LATITUDE_FORMATTER
        g1.xformatter = LONGITUDE_FORMATTER

        self.im = im

        if kwargs.get('opath') is not None:
            print(f"creating output path @ {kwargs.get('opath')}")
            plt.savefig(kwargs.get('opath'), dpi=self.dpi, bbox_inches='tight')
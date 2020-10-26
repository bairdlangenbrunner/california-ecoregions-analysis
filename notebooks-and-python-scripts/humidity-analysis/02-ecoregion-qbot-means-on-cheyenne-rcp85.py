import geopandas
import matplotlib.pyplot as mp
import cartopy
import numpy
import xarray
import shapely
import itertools
import glob
import os

lon_lo, lon_hi, lat_lo, lat_hi = 234., 246.25, 32., 42.5

WHICH_ECOREGION = 4
ECOREGION_NAME = 'sierra_nevada'

# import shape file (twice because .crs doesn't work otherwise?)
ca_eco_l3_gs = geopandas.GeoSeries.from_file('../ca_eco_l3/ca_eco_l3.shp')
ca_eco_l3_gs = geopandas.GeoSeries.from_file('../ca_eco_l3/ca_eco_l3.shp')
ecoregion_series = ca_eco_l3_gs.loc[WHICH_ECOREGION]

# remap to Plate Carree projection for later intersection calculations
ca_eco_l3_gs_4326 = ca_eco_l3_gs.to_crs({'init': 'epsg:4326'})
ecoregion_series_geom_4326 = ca_eco_l3_gs_4326.loc[WHICH_ECOREGION]

#ca_eco_l3_gdf = geopandas.read_file('../ca_eco_l3/ca_eco_l3.shp')
#ecoregion_names = ca_eco_l3_gdf['US_L3NAME']
#print(ecoregion_names)

################################################################################
# open example

root_dir = '/glade/collections/cdg/data/cesmLE/CESM-CAM5-BGC-LE/atm/proc/tseries/daily/TS/'
file_name = 'b.e11.B20TRC5CNBDRD.f09_g16.002.cam.h1.TS.19200101-20051231.nc'
ncfile = xarray.open_dataset(root_dir + file_name)
#ts_data = ncfile['TS'].sel(lat=slice(lat_lo,lat_hi),lon=slice(lon_lo,lon_hi)).values
ts_lat = ncfile['lat'].sel(lat=slice(lat_lo,lat_hi)).values
ts_lon = ncfile['lon'].sel(lon=slice(lon_lo,lon_hi)).values

################################################################################

lon_pcolormesh = numpy.zeros(ts_lon.size+2)
lon_pcolormesh[1:-1] = ts_lon
lon_pcolormesh[0] = ts_lon[0]-numpy.diff(ts_lon)[0]
lon_pcolormesh[-1] = ts_lon[-1]+numpy.diff(ts_lon)[-1]
lon_pcolormesh_midpoints = lon_pcolormesh[:-1]+0.5*(numpy.diff(lon_pcolormesh))

lat_pcolormesh = numpy.zeros(ts_lat.size+2)
lat_pcolormesh[1:-1] = ts_lat
lat_pcolormesh[0] = ts_lat[0]-numpy.diff(ts_lat)[0]
lat_pcolormesh[-1] = ts_lat[-1]+numpy.diff(ts_lat)[-1]
lat_pcolormesh_midpoints = lat_pcolormesh[:-1]+0.5*(numpy.diff(lat_pcolormesh))

################################################################################

# not dependent on actual data
latlon_index_combos = numpy.array([i for i in itertools.product(range(ts_lat.size),range(ts_lon.size))])
lat_polygon_hi_list = []
lon_polygon_hi_list = []
lat_polygon_lo_list = []
lon_polygon_lo_list = []
lon_list = []
lat_list = []
for latlon in latlon_index_combos:

    lat_idx = latlon[0]
    lon_idx = latlon[1]
    
    lon_list.append(ts_lon[latlon[1]])
    lat_list.append(ts_lat[latlon[0]])

    lon_polygon_hi_list.append(-360.+lon_pcolormesh_midpoints[lon_idx+1])
    lon_polygon_lo_list.append(-360.+lon_pcolormesh_midpoints[lon_idx])

    lat_polygon_hi_list.append(lat_pcolormesh_midpoints[lat_idx+1])
    lat_polygon_lo_list.append(lat_pcolormesh_midpoints[lat_idx])

polygon_boxes = numpy.array([shapely.geometry.box(i[0],i[1],i[2],i[3]) for i in zip(lon_polygon_lo_list,lat_polygon_lo_list, lon_polygon_hi_list,lat_polygon_hi_list)])

################################################################################

# calculate True/False intersection list of these projected ecoregion polygons
intersects_TF_list = []
for box in polygon_boxes:
    intersects_TF_list.append(ecoregion_series_geom_4326.intersects(box))
latlon_index_combos_intersect = latlon_index_combos[intersects_TF_list]

# calculates percent area overlap
#for box in polygon_boxes[intersects_TF_list]:
#    print('{:.2f}'.format(ecoregion_series_geom_4326.intersection(box).area/box.area))

# create mask of NaNs and fractional values
mask_and_weights = numpy.zeros((ts_lat.size,ts_lon.size))*numpy.nan
for i,latlon in enumerate(latlon_index_combos_intersect):
    mask_and_weights[latlon[0],latlon[1]] = ecoregion_series_geom_4326.intersection(\
    polygon_boxes[intersects_TF_list][i]).area/polygon_boxes[intersects_TF_list][i].area

################################################################################

# loop through all TS files

file_name_list = [os.path.basename(f) for f in sorted(glob.glob('/glade/collections/cdg/data/cesmLE/CESM-CAM5-BGC-LE/atm/proc/tseries/daily/TS/b.e11.BRCP85C5CNBDRD.f09_g16.???.cam.h1.TS.*'))]
# for historical:  b.e11.B20TRC5CNBDRD.f09_g16.???.cam.h1.TS.*
# for future:  b.e11.BRCP85C5CNBDRD.f09_g16.???.cam.h1.TS.*

for file_name in file_name_list:

    print(file_name)
    ncfile_temporary = xarray.open_dataset(root_dir + file_name)
    ts_data = ncfile_temporary['TS'].sel(lat=slice(lat_lo,lat_hi),lon=slice(lon_lo,lon_hi)).values
    
    # now calculations weighting calculations
    weighted_ecoregion_ts_mean = numpy.nansum(mask_and_weights*ts_data, axis=(1,2))/numpy.nansum(mask_and_weights)
    time_data = ncfile_temporary['time']

    ts_data_array = xarray.DataArray(weighted_ecoregion_ts_mean, coords=[time_data], dims=['time'])

    run_type = file_name.split('.')[2]
    date_range = file_name.split('.')[8]
    simulation_index = file_name.split('.')[4]

    attr_dict = {}
    attr_dict['units'] = 'Kelvin'
    attr_dict['LENS ensemble number'] = simulation_index
    ts_data_array.attrs = attr_dict

    data_set = xarray.Dataset({'TS': (['time'], ts_data_array)}, coords={'time': (['time'], time_data)})
    data_set.to_netcdf('LENS_run_'+simulation_index+'_'+ECOREGION_NAME+'_'+run_type+'_'+date_range+'.nc', unlimited_dims='time')

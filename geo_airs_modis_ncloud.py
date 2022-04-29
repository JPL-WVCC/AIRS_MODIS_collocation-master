import time
import os
import glob
import geo_call
import geo
import numpy as np
import netCDF4 as nc4

start_time = time.time()

outdir='/raid15/qyue/VIIRS/MODIS/201601/Index/'
weight_file='/raid15/mschreie/DATABASE/COLLOCATION/MODIS_AIRS/MODIS_AIRS_FOOTPRINTS_AVER_1KM/ADAPT_FOOT_SMEAR_AVER_MODIS_AIRS_C0051_C1851_G.hdf'
airs_weightfunc = geo.read_airs_weightfunc(weight_file)

datadirs=''

for iday in range (20,32):
  dataDir1=datadirs+'/archive/AIRSOps/airs/gdaac/v5/2016/01/'+str(iday).zfill(2)+'/airibrad/'
  dataDir3=datadirs+'/scratch2/fwi/MODIS_DATA/MYD06/2016/'+str(iday).zfill(3)+'/'
  dataDir4=datadirs+'/scratch2/fwi/MODIS_DATA/MYD03_061/2016/'+str(iday).zfill(3)+'/'
 # dataDir4='/raid15/qyue/VIIRS/MODIS/MYD03_061/2016/001/'
  print(iday)

  for i in range(0, 24):

    print('ROUND', i, i*10, (i+1)*10, i*12, (i+1)*12+2)
    # get AIRS files
    airs_files = sorted(glob.glob(dataDir1+'AIRS*hdf'))[i*10:(i+1)*10]
    
    # get MODIS files 
    if i == 0:
        modis_geo_files = sorted(glob.glob(dataDir4+'MYD03*hdf'))[i*12:(i+1)*12+1]
    else:
        modis_geo_files = sorted(glob.glob(dataDir4+'MYD03*hdf'))[i*12-1:(i+1)*12+1]

    # read MODIS data 
    modis_lon, modis_lat, modis_satAzimuth, modis_satZenith = geo.read_modis_geo(modis_geo_files)
    #modis_cloud = geo.read_modis_cloud(modis_cloud_files)

    #flen=len(modis_geo_files)+len(modis_cloud_files)

    #print("Reading MODIS are done in --- %s seconds --- for %d files " % (time.time() - start_time, flen))

    #success=geo.convert_modis_cloudflag(modis_cloud)
    flen=len(modis_geo_files)
    print(airs_files,modis_geo_files)
    #print("MODIS byte conversion are done in --- %s seconds --- for %d files " % (time.time() - start_time, flen))

    # read AIRS data 
    airs_lon, airs_lat, airs_satAzimuth, airs_satRange, airs_satZenith = geo.read_airs_geo(airs_files)

    flen=len(airs_files)
    
    #print("Reading AIRS are done in --- %s seconds --- for %d files " % (time.time() - start_time, flen))

    #airs_satRange=np.zeros(airs_lat.shape)
    #airs_satRange=705000.

    modis_sdrQa=np.zeros(modis_lat.shape)
    
    dy,dx,my,mx=geo_call.geo_call(airs_satAzimuth, airs_satZenith, airs_lat, airs_lon, airs_satRange,
                                  modis_lat, modis_lon, modis_sdrQa)
    
    #print("collocation are done in --- %s seconds " % (time.time() - start_time))

    dy_flatten = np.array([item for lst in dy.reshape(-1) for item in lst])
    dy_size = np.array([len(lst) for lst in dy.reshape(-1)]).reshape(dy.shape)
    dx_flatten = np.array([item for lst in dx.reshape(-1) for item in lst])
    

    f = nc4.Dataset('/raid15/qyue/VIIRS/MODIS/201601/Index/IND_AIRS_MODISMOD_201601'+str(iday)+'_'+str(i)+'.nc','w', format='NETCDF4') #'w' stands for write

    f.createDimension('m',dy_flatten.size)
    f.createDimension('x', dy.shape[0])
    f.createDimension('y', dy.shape[1])

    y_flatten = f.createVariable('dy', 'i4', ('m',))
    y_size=f.createVariable('dy_size','i4',('x', 'y', ))
    x_flatten = f.createVariable('dx', 'i4', ('m',))

    y_size[:]=dy_size
    y_flatten[:]=dy_flatten
    x_flatten[:]=dx_flatten

    f.description="Demo Data for 2015 Jan"

    f.close()
        
  print("done in --- %s seconds --- for 2015 Jan %d files" % (time.time() - start_time, iday))
  start_time=time.time()
# collocation is done 


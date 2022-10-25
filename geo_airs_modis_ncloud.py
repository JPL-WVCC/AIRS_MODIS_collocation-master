import time
import os,sys
import glob
import geo_call
import geo
import numpy as np
import netCDF4 as nc4
import shutil
from datetime import datetime, timedelta
import logging

module_logger = logging.getLogger("airs_modis_parallel_run_matchup.geo_airs_modis_ncloud")

not_a_value = -9999.

# func to do the colocation
def call_match_airs_modis(airs_files, modis_geo_files, iday, i, output_dir):

    try:
      modis_lon, modis_lat, modis_satAzimuth, modis_satZenith = geo.read_modis_geo(modis_geo_files)
    except UnboundLocalError:
      sys.exit(0)

    #modis_cloud = geo.read_modis_cloud(modis_cloud_files)

    #flen=len(modis_geo_files)+len(modis_cloud_files)

    #print("Reading MODIS are done in --- %s seconds --- for %d files " % (time.time() - start_time, flen))

    #success=geo.convert_modis_cloudflag(modis_cloud)
    flen=len(modis_geo_files)
    print('num of modis_geo_files: ', flen)

    if flen == 0:
      sys.exit(0)

    print(airs_files, modis_geo_files)
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

    # example: AIRS.2016.01.30.240.L1B.AIRS_Rad.v5.0.23.0.G16031113544.hdf
    # use this name: IND_AIRS_MODIS.2016.01.30.240.L1B.AIRS_Rad.v5.0.23.0.G16031113544
    airs_file = os.path.basename(airs_files[0])

    output_basename = airs_file.replace('AIRS.', 'IND_AIRS_MODIS1km.')[0:-4]
    pos1 = output_basename.find('.L1B')
    output_basename = output_basename[:pos1]

    ### f = nc4.Dataset('/raid15/leipan/VIIRS/MODIS/201601/Index/IND_AIRS_MODISMOD_201601'+str(iday)+'_'+str(i)+'.nc','w', format='NETCDF4') #'w' stands for write

    # do not introduce this level of subdir
    """
    subdir1 = os.path.join(output_dir, output_basename)
    if os.path.exists(subdir1) == False:
      os.makedirs(subdir1)
    output_nc_file = os.path.join(output_dir, output_basename, output_basename+'.nc')
    """

    output_nc_file = os.path.join(output_dir, output_basename+'.nc')

    ### f = nc4.Dataset(output_dir+'IND_AIRS_MODISMOD_201601'+str(iday)+'_'+str(i)+'.nc','w', format='NETCDF4') #'w' stands for write
    f = nc4.Dataset(output_nc_file, 'w', format='NETCDF4') #'w' stands for write

    """
    f.createDimension('m',dy_flatten.size)
    f.createDimension('x', dy.shape[0])
    f.createDimension('y', dy.shape[1])
    """

    f.createDimension('GranuleCount_ImagerPixel', dy_flatten.size)
    f.createDimension('sounder_atrack', dy.shape[0])
    f.createDimension('sounder_xtrack', dy.shape[1])

    """
    y_flatten = f.createVariable('dy', 'i4', ('m',))
    y_size=f.createVariable('dy_size','i4',('x', 'y', ))
    x_flatten = f.createVariable('dx', 'i4', ('m',))
    """

    y_flatten = f.createVariable('number_of_pixels', 'i4', ('GranuleCount_ImagerPixel',), zlib=True)
    y_flatten.setncatts({'long_name':u'imager cross-track index', 'units':u'none', 'var_desc':u'imager cross-track index'})

    y_size=f.createVariable('FOVCount_ImagerPixel','i4',('sounder_atrack', 'sounder_xtrack', ), zlib=True)
    y_size.setncatts({'long_name':u'count of imager pixels per sounder FOV', 'units':u'none', 'var_desc':u'count of imager pixels per sounder FOV'})

    x_flatten = f.createVariable('number_of_lines', 'i4', ('GranuleCount_ImagerPixel',), zlib=True)
    x_flatten.setncatts({'long_name':u'imager along-track index after concatenating imager granules along track', 'units':u'none', 'var_desc':u'imager along-track index after concatenating imager granules along track'})

    y_size[:]=dy_size
    y_flatten[:]=dy_flatten
    x_flatten[:]=dx_flatten

    # add global attributes
        
    f.VERSION = '1'
    f.SHORT_NAME = "Aqua_AIRS_MODIS1km_IND"
    f.TITLE = "Aqua AIRS-MODIS 1-km Matchup Indexes V1"
    f.IDENTIFIER_PRODUCT_DOI_AUTHORITY = "http://dx.doi.org/"
    f.IDENTIFIER_PRODUCT_DOI = "10.5067/MEASURES/WVCC/DATA210"
        
    ct = datetime.now()
    f.PRODUCTIONDATE = ct.isoformat()

    f.description="Version-1 AIRS-MODIS 1km collocation index product by the project of Multidecadal Satellite Record of Water Vapor, Temperature, and Clouds (PI: Eric Fetzer) funded by NASA’s Making Earth System Data Records for Use in Research Environments (MEaSUREs) Program following Wang et al. (2016, https://doi.org/10.3390/rs8010076) and Yue et al. (2022, https://doi.org/10.5194/amt-15-2099-2022)."

    f.TIME_TOLERANCE = "900 seconds"
    f.DISTANCE_TOLERANCE = "AIRS FOV size angle 1.1 deg divided by 2"

    # get source granule info
    nf = 0
    try:
      f.MODIS_FILE1 = os.path.basename(modis_geo_files[0])
      nf += 1
    except IndexError:
      pass
        
    try:
      f.MODIS_FILE2 = os.path.basename(modis_geo_files[1])
      nf += 1
    except IndexError:
      pass
        
    try:
      f.MODIS_FILE3 = os.path.basename(modis_geo_files[2])
      nf += 1
    except IndexError:
      pass
        
    nf = np.int32(nf)
    f.MODIS_FILES_COUNT = nf

    output_filename = os.path.join('product_root_dir', 'hello')

    if nf < 3:
      module_logger.info('Warning: there are {0} input MODIS granules for {1}, not 3 as usual!'.format(nf, output_filename))

    airs_str = os.path.basename(airs_files[0])
    f.AIRS_FILE = airs_str

    print('calling hdf_attributes with ', airs_files[0])
    attr_name_list = ['start_Time', 'end_Time', 'start_Latitude', 'end_Latitude', 'start_Longitude', 'end_Longitude']
    attr_dict = geo.hdf_attributes(airs_files[0], attr_name_list)
    print('attr_dict: ', attr_dict)

    # get AIRS start/end time from attributes
    starttime = datetime(1993,1,1,0,0) + timedelta(seconds=attr_dict['start_Time'])
    endtime = datetime(1993,1,1,0,0) + timedelta(seconds=attr_dict['end_Time'])

    start_date = starttime.strftime('%Y-%m-%dT%H:%M:%SZ')
    end_date   = endtime.strftime('%Y-%m-%dT%H:%M:%SZ')

    ### print ('f.start_Latitude: ', f.start_Latitude, ' , f.end_Longitude: ', f.end_Longitude)

    f.RANGEBEGINNINGDATE = start_date.split('T')[0]
    print('f.RANGEBEGINNINGDATE: ', f.RANGEBEGINNINGDATE)


    f.RANGEBEGINNINGTIME = start_date.split('T')[1].replace('Z', '')
    f.RANGEENDINGDATE = end_date.split('T')[0]
    f.RANGEENDINGTIME = end_date.split('T')[1].replace('Z', '')

    # do not get lat lon box from attributes
    """
    lon_min = attr_dict['start_Longitude'] # AIRS_START_LONGITUDE
    lat_min = attr_dict['start_Latitude']
    lon_max = attr_dict['end_Longitude']
    lat_max = attr_dict['end_Latitude']

    ### f.SOUTHBOUNDINGCOORDINATE = lat_min
    f.AIRS_START_LATITUDE = lat_min
    ### f.WESTBOUNDINGCOORDINATE = lon_min
    f.AIRS_START_LONGITUDE = lon_min
    ### f.NORTHBOUNDINGCOORDINATE = lat_max
    f.AIRS_END_LATITUDE = lat_max
    ### f.EASTBOUNDINGCOORDINATE = lon_max
    f.AIRS_END_LONGITUDE = lon_max
    """

    ### print('1 type(airs_lon): ', type(airs_lon))
    # flatten the array
    airs_lon = airs_lon.ravel()
    airs_lat = airs_lat.ravel()

    # turn into a set with unique elements
    set1 = list(map(np.unique, airs_lon))
    set2 = list(map(np.unique, airs_lat))

    # remove not_a_value
    airs_lon = [ele for ele in set1 if ele != not_a_value]
    airs_lat = [ele for ele in set2 if ele != not_a_value]

    ### sys.exit(0)

    # rather, get lat lon box from lat lon data
    ### lon_min = min(map(min, airs_lon))
    lon_min = min(airs_lon)
    lat_min = min(airs_lat)
    lon_max = max(airs_lon)
    lat_max = max(airs_lat)

    # f.SOUTHBOUNDINGCOORDINATE = min(lat_min, lat_max) # suggested by Qing
    f.SOUTHBOUNDINGCOORDINATE = min(lat_min, lat_max)
    ### f.WESTBOUNDINGCOORDINATE = min(lon_min, lon_max) # suggested by Qing
    f.WESTBOUNDINGCOORDINATE = min(lon_min, lon_max)
    ### f.NORTHBOUNDINGCOORDINATE = max(lat_max, lat_min) # suggested by Qing
    f.NORTHBOUNDINGCOORDINATE = max(lat_max, lat_min)
    ### f.EASTBOUNDINGCOORDINATE = max(lon_max, lon_min) # suggested by Qing
    f.EASTBOUNDINGCOORDINATE = max(lon_max, lon_min)

    f.close()
        
# end of call_match_airs_modis()



if __name__ == '__main__':

  start_time = time.time()

  ### outdir='/raid15/leipan/VIIRS/MODIS/201601/Index/'
  outdir='/raid15/leipan/VIIRS/MODIS/201601/debug/'
  weight_file='/raid15/mschreie/DATABASE/COLLOCATION/MODIS_AIRS/MODIS_AIRS_FOOTPRINTS_AVER_1KM/ADAPT_FOOT_SMEAR_AVER_MODIS_AIRS_C0051_C1851_G.hdf'
  airs_weightfunc = geo.read_airs_weightfunc(weight_file)

  datadirs=''

  if os.path.exists(outdir):
    shutil.rmtree(outdir)
  os.makedirs(outdir)

  for iday in range (20,32):
    print('========================== ', iday)

    dataDir1=datadirs+'/archive/AIRSOps/airs/gdaac/v5/2016/01/'+str(iday).zfill(2)+'/airibrad/'
    ### dataDir3=datadirs+'/scratch2/fwi/MODIS_DATA/MYD06/2016/'+str(iday).zfill(3)+'/'
    ### dataDir4=datadirs+'/scratch2/fwi/MODIS_DATA/MYD03_061/2016/'+str(iday).zfill(3)+'/'
    dataDir4='/raid15/qyue/VIIRS/MODIS/MYD03_061/2016/001/'

    for i in range(0, 24):

      print('   ----------  ROUND', i, i*10, (i+1)*10, i*12, (i+1)*12+2)
      # get AIRS files
      airs_files = sorted(glob.glob(dataDir1+'AIRS*hdf'))[i*10:(i+1)*10]
      print('num of airs_files: ', len(airs_files))
    
      # get MODIS files 
      if i == 0:
          modis_geo_files = sorted(glob.glob(dataDir4+'MYD03*hdf'))[i*12:(i+1)*12+1]
          ### print('dataDir4: ', dataDir4)
          ### print('1 modis_geo_files: ', modis_geo_files)
      else:
          modis_geo_files = sorted(glob.glob(dataDir4+'MYD03*hdf'))[i*12-1:(i+1)*12+1]
          ### print('2 modis_geo_files: ', modis_geo_files)

      # read MODIS data 
      print('num of modis_geo_files: ', len(modis_geo_files))
      print('')

      print('IND_AIRS_MODISMOD_201601'+str(iday)+'_'+str(i)+'.nc')

      call_match_airs_modis(airs_files, modis_geo_files, iday, i, outdir)

    print("done in --- %s seconds --- for 2015 Jan %d files" % (time.time() - start_time, iday))
    start_time=time.time()
  # collocation is done 


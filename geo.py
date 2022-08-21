import numpy as np
from numpy import linalg as LA
from numpy import sqrt, sin, cos, deg2rad, arctan2, \
    arcsin, rad2deg
import xml.etree.ElementTree as etree
import h5py
import io
from pyhdf.SD import SD, SDC
from pyhdf.HDF import *
from pyhdf.VS import *
import netCDF4

### from pyhdf.SD import SD, SDC
### from pyhdf import HDF, VS, V

from pykdtree.kdtree import KDTree

class cloudclass:
    def __init__(self):
        self.placeholder=-1


WGS84_A = 6378137.0
WGS84_F = 1.0 / 298.257223563
WGS84_B = WGS84_A*(1.0 - WGS84_F)
WGS84_E2 = 2 * WGS84_F - WGS84_F ** 2

#Rotational angular velocity of Earth in radians/sec from IERS
#   Conventions (2003).
ANGVEL = 7.2921150e-5;

def LLA2ECEF(lonIn, latIn, altIn):
    """
	Transform lon,lat,alt (WGS84 degrees, meters) to  ECEF
	x,y,z (meters)
    """
    lonRad = deg2rad(np.asarray(lonIn, dtype=np.float64) ) 
    latRad = deg2rad(np.asarray(latIn, dtype=np.float64) )
    alt    = np.asarray(altIn, dtype=np.float64) 
    a, b, e2 = WGS84_A, WGS84_B, WGS84_E2

    ## N = Radius of Curvature (meters), defined as:
    N = a/sqrt(1.0-e2*(sin(latRad)**2.0))
            
    ##$ calcute X, Y, Z
    x=(N+alt)*cos(latRad)*cos(lonRad)
    y=(N+alt)*cos(latRad)*sin(lonRad)
    z=(b**2.0/a**2.0*N + altIn)*sin(latRad)

    return x, y, z 


def RAE2ENU(azimuthIn, zenithIn, rangeIn):
    """
    Transform azimuth, zenith, range to ENU x,y,z (meters)
    """
    azimuth = deg2rad(np.asarray(azimuthIn, dtype=np.float64))
    zenith  = deg2rad(np.asarray(zenithIn, dtype=np.float64))
    r       = np.asarray(rangeIn, dtype=np.float64)

    # up 
    up = r*cos(zenith)
  
    # projection on the x-y plane 
    p = r*sin(zenith)  
  
    # north 
    north = p*cos(azimuth)
 
    # east
    east = p*sin(azimuth)   

    return east, north, up


def ENU2ECEF (east, north, up, lon, lat):
    """
    Convert local East, North, Up (ENU) coordinates to the (x,y,z) Earth Centred Earth Fixed (ECEF) coordinates
    Reference is here:  
    http://www.navipedia.net/index.php/Transformations_between_ECEF_and_ENU_coordinates
    Note that laitutde should be geocentric latitude instead of geodetic latitude 
    Note: 

    On June 16 2015
    This note from https://en.wikipedia.org/wiki/Geodetic_datum 
    Note: \ \phi is the geodetic latitude. A prior version of this page showed use of the geocentric latitude (\ \phi^\prime).
    The geocentric latitude is not the appropriate up direction for the local tangent plane. If the
    original geodetic latitude is available it should be used, otherwise, the relationship between geodetic and geocentric
    latitude has an altitude dependency, and is captured by ...
    """    

    x0 = np.asarray(east, dtype=np.float64)
    y0 = np.asarray(north, dtype=np.float64)
    z0 = np.asarray(up, dtype=np.float64)

    lm = deg2rad(np.asarray(lon, dtype=np.float64))
    ph = deg2rad(np.asarray(lat, dtype=np.float64))

    x=-1.0*x0*sin(lm)-y0*cos(lm)*sin(ph)+z0*cos(lm)*cos(ph)
    y= x0*cos(lm) -y0*sin(lm)*sin(ph)+z0*sin(lm)*cos(ph)
    z= x0*0       +y0*cos(ph)        +z0*sin(ph)   

    return x, y, z
        
##################################################################################### 
def match_cris_viirs(crisLos, crisPos, viirsPos, viirsMask):
    """
    Match crisLos with viirsPos using the method by Wang et al. (2016)
    Wang, L., D. A. Tremblay, B. Zhang, and Y. Han, 2016: Fast and Accurate 
      Collocation of the Visible Infrared Imaging Radiometer Suite 
      Measurements and Cross-track Infrared Sounder Measurements. 
      Remote Sensing, 8, 76; doi:10.3390/rs8010076.     
    """
    
    # Derive Satellite Postion 
    crisSat = crisPos - crisLos 
        
        # using KD-tree to find best matched points 
    
    # build kdtree to find match index 
    pytree_los = KDTree(viirsPos.reshape(viirsPos.size//3, 3))
    dist_los, idx_los = pytree_los.query(crisPos.reshape(crisPos.size//3, 3) , sqr_dists=False)
    
    my, mx = np.unravel_index(idx_los, viirsPos.shape[0:2])
    
    
    idy, idx  = find_match_index(crisLos.reshape(crisLos.size//3, 3),\
                                     crisSat.reshape(crisSat.size//3, 3),\
                                     viirsPos, viirsMask, mx, my)
        
    idy = np.array(idy).reshape(crisLos.shape[0:crisLos.ndim-1])
    idx = np.array(idx).reshape(crisLos.shape[0:crisLos.ndim-1])

    return idy, idx

##############################################################################################



# Satellite data reader 
# read CrIS SDR files 
def read_cris_sdr (filelist, sdrFlag=False):

    """
    Read JPSS CrIS SDR and return LW, MW, SW Spectral. Note that this method
    is very fast but can't open too many files (<1024) simultaneously.  
    """

    sdrs = [h5py.File(filename) for filename in filelist]
    real_lw = np.concatenate([f['All_Data']['CrIS-SDR_All']['ES_RealLW'][:] for f in sdrs])
    real_mw = np.concatenate([f['All_Data']['CrIS-SDR_All']['ES_RealMW'][:] for f in sdrs])
    real_sw = np.concatenate([f['All_Data']['CrIS-SDR_All']['ES_RealSW'][:] for f in sdrs])
    
    if not sdrFlag: 
        return real_lw, real_mw, real_sw
    else:  
        QF3_CRISSDR = np.concatenate([f['All_Data']['CrIS-SDR_All']['QF3_CRISSDR'][:] for f in sdrs])
        QF4_CRISSDR = np.concatenate([f['All_Data']['CrIS-SDR_All']['QF4_CRISSDR'][:] for f in sdrs])

        #sdrQa = shift(shift(qf3,-6),6)
        sdrQa = QF3_CRISSDR & 0b00000011
    
        #GeoQa = shift(shift(shift(qf3, 2),-7), 7)
        geoQa = (QF3_CRISSDR & 0b00000100) >> 2

        # dayFlag = shift(shift(qf4, -7), 7)
        dayFlag = QF4_CRISSDR & 0b00000001
        return real_lw, real_mw, real_sw, sdrQa, geoQa, dayFlag
            
####################################################################################    
## read CrIS GOE files     
def read_cris_geo (filelist, ephemeris = False):
    
    """
    Read JPSS CrIS Geo files and return Longitude, Latitude, SatelliteAzimuthAngle, SatelliteRange, SatelliteZenithAngle.
        if ephemeris=True, then return forTime, midTime, satellite position, velocity, attitude 
    """
    
    geos = [h5py.File(filename) for filename in filelist]
    
    if ephemeris == False:  
        Latitude  = np.concatenate([f['All_Data']['CrIS-SDR-GEO_All']['Latitude'] [:] for f in geos])
        Longitude = np.concatenate([f['All_Data']['CrIS-SDR-GEO_All']['Longitude'][:] for f in geos])
        SatelliteAzimuthAngle = np.concatenate([f['All_Data']['CrIS-SDR-GEO_All']['SatelliteAzimuthAngle'][:] for f in geos])
        SatelliteRange = np.concatenate([f['All_Data']['CrIS-SDR-GEO_All']['SatelliteRange'][:] for f in geos])
        SatelliteZenithAngle = np.concatenate([f['All_Data']['CrIS-SDR-GEO_All']['SatelliteZenithAngle'][:] for f in geos])
        return Longitude, Latitude, SatelliteAzimuthAngle, SatelliteRange, SatelliteZenithAngle
    if ephemeris == True:
        FORTime  = np.concatenate([f['All_Data']['CrIS-SDR-GEO_All']['FORTime'] [:] for f in geos])
        MidTime  = np.concatenate([f['All_Data']['CrIS-SDR-GEO_All']['MidTime'] [:] for f in geos])
        SCPosition  = np.concatenate([f['All_Data']['CrIS-SDR-GEO_All']['SCPosition'] [:] for f in geos])
        SCVelocity  = np.concatenate([f['All_Data']['CrIS-SDR-GEO_All']['SCVelocity'] [:] for f in geos])
        SCAttitude  = np.concatenate([f['All_Data']['CrIS-SDR-GEO_All']['SCAttitude'] [:] for f in geos])
        return FORTime, MidTime, SCPosition, SCVelocity, SCAttitude

#################################################################
## READ VIIRS Geofiles
 
def read_viirs_geo (filelist, ephemeris=False, hgt=False):

    """
    Read JPSS VIIRS Geo files and return Longitude, Latitude, SatelliteAzimuthAngle, SatelliteRange, SatelliteZenithAngle.
        if ephemeris=True, then return midTime, satellite position, velocity, attitude 
    """

    if type(filelist) is str: filelist = [filelist]
    if len(filelist) ==0: return None
    
    # Open user block to read Collection_Short_Name
    with h5py.File(filelist[0], 'r') as fn:
            user_block_size = fn.userblock_size
 
    with io.open(filelist[0], 'rU',encoding="latin-1") as fs:
            ub_text = fs.read(user_block_size)
    ub_xml = etree.fromstring(ub_text.rstrip('\x00'))
    
    #print(ub_text)
    #print(etree.tostring(ub_xml))
    CollectionName = ub_xml.find('Data_Product/N_Collection_Short_Name').text+'_All'    
    #print(CollectionName)

    # read the data
    geos = [h5py.File(filename, 'r') for filename in filelist]
    
    if not ephemeris:
        Latitude  = np.concatenate([f['All_Data'][CollectionName]['Latitude'][:]  for f in geos])
        Longitude = np.concatenate([f['All_Data'][CollectionName]['Longitude'][:] for f in geos])
        SatelliteAzimuthAngle = np.concatenate([f['All_Data'][CollectionName]['SatelliteAzimuthAngle'][:] for f in geos])
        SatelliteRange = np.concatenate([f['All_Data'][CollectionName]['SatelliteRange'][:] for f in geos])
        SatelliteZenithAngle = np.concatenate([f['All_Data'][CollectionName]['SatelliteZenithAngle'][:] for f in geos])
        Height = np.concatenate([f['All_Data'][CollectionName]['Height'][:] for f in geos])
        if hgt: 
            return Longitude, Latitude, SatelliteAzimuthAngle, SatelliteRange, SatelliteZenithAngle, Height
        else: 
            return Longitude, Latitude, SatelliteAzimuthAngle, SatelliteRange, SatelliteZenithAngle
    
    if ephemeris: 
        MidTime  = np.concatenate([f['All_Data'][CollectionName]['MidTime'] [:] for f in geos])
        SCPosition  = np.concatenate([f['All_Data'][CollectionName]['SCPosition'][:] for f in geos])
        SCVelocity  = np.concatenate([f['All_Data'][CollectionName]['SCVelocity'][:] for f in geos])
        SCAttitude  = np.concatenate([f['All_Data'][CollectionName]['SCAttitude'][:] for f in geos])
        return MidTime, SCPosition, SCVelocity, SCAttitude 

####################################################################################        
## READ VIIRS SDR files
def read_viirs_sdr (filelist):
    
    """
    READ VIIRS SDR files
    """
        
    if type(filelist) is str: filelist = [filelist]
    if len(filelist) == 0: return None
    
    # Opne userbloack to read Collection_Short_Name
    with h5py.File(filelist[0], 'r') as fn:
            user_block_size = fn.userblock_size
 
    with io.open(filelist[0], 'rU',encoding="latin-1") as fn:
            ub_text = fn.read(user_block_size)

    ub_xml = etree.fromstring(ub_text.rstrip('\x00'))

    
    #print(etree.tostring(ub_xml, pretty_print=True))
    CollectionName = ub_xml.find('Data_Product/N_Collection_Short_Name').text+'_All'    
    #print(CollectionName)
    
    s='All_Data/'+CollectionName+'/'

    # Read datasets
    sdrs = [h5py.File(filename, 'r') for filename in filelist]
    
    if 'BrightnessTemperature' in sdrs[0][s].keys(): 
        BrightnessTemperature = np.concatenate([f[s+'BrightnessTemperature'] for f in sdrs])
        BT = BrightnessTemperature
        
        if 'BrightnessTemperatureFactors' in sdrs[0][s].keys(): 
            BrightnessTemperatureFactors=np.concatenate([f[s+'BrightnessTemperatureFactors'] for f in sdrs])
            BT = BrightnessTemperature * BrightnessTemperatureFactors[0] + BrightnessTemperatureFactors[1]
        
    if 'Reflectance' in sdrs[0][s].keys(): 
    
        Reflectance = np.concatenate([f[s+'Reflectance'] for f in sdrs])
        ReflectanceFactors=np.concatenate([f[s+'ReflectanceFactors'] for f in sdrs])
        BT = Reflectance * ReflectanceFactors[0] + ReflectanceFactors[1]    
    
    Radiance = np.concatenate([f[s+'Radiance'] for f in sdrs])
    
    if 'RadianceFactors' in sdrs[0][s].keys(): 
        RadianceFactors=np.concatenate([f[s+'RadianceFactors'] for f in sdrs])
        RAD = Radiance * RadianceFactors[0] + RadianceFactors[1]
    else: 
        RAD = Radiance
    
    if CollectionName.find('VIIRS-I') >= 0:
        qaStr = 'QF1_VIIRSIBANDSDR' 
    else:   qaStr = 'QF1_VIIRSMBANDSDR' 
    QF1_VIIRSBANDSDR = np.concatenate([f[s+qaStr] for f in sdrs])
        
    return BT, RAD, QF1_VIIRSBANDSDR
##############################################################################################


def find_match_index (cris_los, cris_sat, viirs_pos_in, viirs_sdrQa_in, \
                      mx, my, fovDia=0.963):


        nLine, nPixel = viirs_pos_in.shape[0:2]
        crisShape = cris_los.shape[0:cris_los.ndim]        
        
        # setup parameters
        cos_half_fov=cos(deg2rad(fovDia/2.0))
        if nPixel == 3200: nc = np.round(deg2rad(0.963/2)*833.0/0.75*4).astype(np.int)
        if nPixel == 6400: nc = np.round(deg2rad(0.963/2)*833.0/0.375*4).astype(np.int)

        if nPixel == 1354: nc = np.round(deg2rad(0.963/2)*833.0/0.8*4).astype(np.int)
        

        # return list
        index_x = []
        index_y = []

        for i in range(0, mx.size):

                xd = mx[i]
                yd = my[i]

                xb = 0        if xd-nc <0        else xd-nc
                xe = nPixel-1 if xd+nc >nPixel-1 else xd+nc

                yb = 0        if yd-nc <0        else yd-nc
                ye = nLine-1  if yd+nc >nLine-1  else yd+nc

                viirs_pos = viirs_pos_in[yb:ye, xb:xe, : ]
                viirs_Qa  = viirs_sdrQa_in[yb:ye, xb:xe]
                viirs_los = viirs_pos  - cris_sat[i, :]
                temp = np.dot(viirs_los, cris_los[i, :])
                temp = temp / LA.norm(viirs_los, axis=2)
                cos_angle = temp / LA.norm(cris_los[i, :])

                iy, ix = np.where ( (viirs_Qa == 0) & (cos_angle > cos_half_fov) )

                index_x.append(ix+xb)
                index_y.append(iy+yb)

        return index_y, index_x

    
##############################################################################################
##############################################################################################
##############################################################################################


def read_modis_time(infiles):


    n=0
    time1 = []
    for f in infiles:
      try:
        file = SD(f, SDC.READ)

        sds_obj = file.select('EV start time')        
        ### print('sds_obj: ', sds_obj)
        time1 = np.vstack((time1, sds_obj.get())) if n > 0 else sds_obj.get()
        ### print('time1.shape: ', time1.shape)

        n=n+1
      except HDF4Error:
        continue

    return time1


def read_modis_geo(infiles):


    n=0
    for f in infiles:
        file = SD(f, SDC.READ)

        sds_obj = file.select('Latitude')        
        print('sds_obj: ', sds_obj)
        lat = np.vstack((lat, sds_obj.get())) if n > 0 else sds_obj.get()
        print('lat.shape: ', lat.shape)
        sds_obj = file.select('Longitude')
        lon = np.vstack((lon, sds_obj.get())) if n > 0 else sds_obj.get()
        print('lon: ', lon)
        sds_obj = file.select('SensorAzimuth')
        satAzi = np.vstack((satAzi, sds_obj.get())) if n > 0 else sds_obj.get()

        sds_obj = file.select('SensorZenith')
        satZen = np.vstack((satZen, sds_obj.get())) if n > 0 else sds_obj.get()

        n=n+1

    return lon, lat, satAzi, satZen

def read_hdf_with_attr(file, sselec):

    sds_obj = file.select(sselec)
    data = sds_obj.get()
    nono=np.where(data < 0.)
    for key, value in sds_obj.attributes().iteritems():
        if key == 'add_offset':
            add_offset = value  
        if key == 'scale_factor':
            scale_factor = value
    data = (data - add_offset) * scale_factor
    data[nono] = -9999.

    return data
    
def read_modis_cloud(infiles):

    modis_cloud=cloudclass()
    n=0
    for f in infiles:
        file = SD(f, SDC.READ)

        data=read_hdf_with_attr(file, 'cloud_top_temperature_1km')
        modis_cloud.ctt1km = np.vstack((modis_cloud.ctt1km, data)) if n > 0 else data
        data=read_hdf_with_attr(file, 'Cloud_Effective_Radius')
        modis_cloud.reff = np.vstack((modis_cloud.reff, data)) if n > 0 else data
        data=read_hdf_with_attr(file, 'Cloud_Optical_Thickness')
        modis_cloud.cot = np.vstack((modis_cloud.cot, data)) if n > 0 else data
        data=read_hdf_with_attr(file, 'surface_temperature_1km')
        modis_cloud.surft = np.vstack((modis_cloud.surft, data)) if n > 0 else data
        data=read_hdf_with_attr(file, 'Cloud_Mask_1km')
        modis_cloud.cloud_mask = np.vstack((modis_cloud.cloud_mask, data)) if n > 0 else data
        
        n=n+1

    return modis_cloud

def convert_modis_cloudflag(modis_cloud):

    nx = modis_cloud.cloud_mask.shape[0]
    ny = modis_cloud.cloud_mask.shape[1]

    modis_cloud.cloudstatus_flag=np.zeros((nx,ny))
    modis_cloud.cloudmask_flag=np.zeros((nx,ny))
    modis_cloud.daynight_flag=np.zeros((nx,ny))
    modis_cloud.sunglint_flag=np.zeros((nx,ny))
    modis_cloud.snowice_flag=np.zeros((nx,ny))
    modis_cloud.surf_flag=np.zeros((nx,ny))

    modis_cloud.aerosol_flag=np.zeros((nx,ny))
    modis_cloud.cirrusthin_flag=np.zeros((nx,ny))
    modis_cloud.shadow_flag=np.zeros((nx,ny))

    bytearrmask=np.zeros((nx,ny,8))
    bytearrland=np.zeros((nx,ny,8))
       
    byt1=np.unpackbits(np.uint8(modis_cloud.cloud_mask[:,:,0]))
    byt2=np.unpackbits(np.uint8(modis_cloud.cloud_mask[:,:,1]))

    byt1=byt1.reshape(nx,ny,8)
    byt2=byt2.reshape(nx,ny,8)
            
    modis_cloud.cloudstatus_flag[:,:]=byt1[:,:,7]
    bytearrmask[:,:,6:8]=byt1[:,:,5:7]
    modis_cloud.cloudmask_flag[:,:]=np.packbits(np.uint8(bytearrmask), axis=2)[:,:,0]
    modis_cloud.daynight_flag[:,:]=byt1[:,:,4]
    modis_cloud.sunglint_flag[:,:]=byt1[:,:,3]
    modis_cloud.snowice_flag[:,:]=byt1[:,:,2]
    bytearrland[:,:,6:8]=byt1[:,:,0:2]
    modis_cloud.surf_flag[:,:]=np.packbits(np.uint8(bytearrland), axis=2)[:,:,0]
            
    modis_cloud.aerosol_flag[:,:]=byt2[:,:,7]
    modis_cloud.cirrusthin_flag[:,:]=byt2[:,:,6]
    modis_cloud.shadow_flag[:,:]=byt2[:,:,5]
            
    #modis_cloud.cloudmask_flag[:,:]=int(str(byt1[:,:,6])+str(byt1[:,:,5]), 2)
    #modis_cloud.surf_flag[:,:]=int(str(byt1[:,:,1])+str(byt1[:,:,0]), 2)

    return 1
            

def read_airs_time(infiles):

    n=0
    for f in infiles:
        file = SD(f, SDC.READ)

        sds_obj = file.select('Time')        
        time1 = np.vstack((time1, sds_obj.get())) if n > 0 else sds_obj.get()

        n=n+1
    return time1



def read_airs_geo(infiles):


    n=0
    for f in infiles:
        file = SD(f, SDC.READ)

        sds_obj = file.select('Latitude')        
        lat = np.vstack((lat, sds_obj.get())) if n > 0 else sds_obj.get()

        sds_obj = file.select('Longitude')
        lon = np.vstack((lon, sds_obj.get())) if n > 0 else sds_obj.get()

        sds_obj = file.select('satazi')
        satAzi = np.vstack((satAzi, sds_obj.get())) if n > 0 else sds_obj.get()

        sds_obj = file.select('satzen')
        satZen = np.vstack((satZen, sds_obj.get())) if n > 0 else sds_obj.get()

        sds_obj = file.select('scanang')
        scanang = np.vstack((scanang, sds_obj.get())) if n > 0 else sds_obj.get()
        
 
        filev = HDF(f,SDC.READ)
        vs_obj=filev.vstart()
        vdata_obj=vs_obj.attach('satheight')
        vdata_data = np.tile(np.asarray(vdata_obj[:]),(1,90))*1000
        vdata_obj.detach()
        vs_obj.end()
        filev.close()
        satheight = np.vstack((satheight, vdata_data)) if n > 0 else vdata_data
        
        n=n+1

    satRange=satheight/cos(deg2rad(scanang))

    return lon, lat, satAzi, satRange, satZen


def read_airs_weightfunc(infile):

    file = SD(infile, SDC.READ)
    
    sds_obj = file.select('Footprint')
    weightfunc =  sds_obj.get()
    
    return weightfunc


##################################################################################### 
def match_sounder_imager(sounderLos, sounderPos, imagerPos, imagerMask, dofootestimate=1):
    """
    Match sounderLos with imagerPos using the method by Wang et al. (2016)
    Wang, L., D. A. Tremblay, B. Zhang, and Y. Han, 2016: Fast and Accurate 
      Collocation of the Visible Infrared Imaging Radiometer Suite 
      Measurements and Cross-track Infrared Sounder Measurements. 
      Remote Sensing, 8, 76; doi:10.3390/rs8010076.     
    """
    
    # Derive Satellite Postion 
    sounderSat = sounderPos - sounderLos 
        
        # using KD-tree to find best matched points 
    
    # build kdtree to find match index 
    pytree_los = KDTree(imagerPos.reshape(imagerPos.size//3, 3))
    dist_los, idx_los = pytree_los.query(sounderPos.reshape(sounderPos.size//3, 3) , sqr_dists=False)
    
    my, mx = np.unravel_index(idx_los, imagerPos.shape[0:2])

    imy = np.array(my).reshape(sounderLos.shape[0:sounderLos.ndim-1])
    imx = np.array(mx).reshape(sounderLos.shape[0:sounderLos.ndim-1])
    
    if (dofootestimate != 0):
        idy, idx  = find_match_index_noweight(sounderLos.reshape(sounderLos.size//3, 3),\
                                              sounderSat.reshape(sounderSat.size//3, 3),\
                                              imagerPos, imagerMask, mx, my)
        
        idy = np.array(idy).reshape(sounderLos.shape[0:sounderLos.ndim-1])
        idx = np.array(idx).reshape(sounderLos.shape[0:sounderLos.ndim-1])

    else:
        idy=0
        idx=0
        
    return idy, idx, imy, imx


#############################################################################################


def weight_imager_by_sounderfoot(sx,sy, imx,imy, cmask, weighting=0, ctt=0, surft=0,
                                         cot=0, reff=0,
                                         cstatus=0,
                                         aerosol_flag=0, thinci_flag=0, snowice_flag=0):

    out=cloudclass()

    out.nx, out.ny = imx.shape
    out.cmask=np.zeros((sx,sy))
    
    out.ctt=-9999 if (np.isscalar(ctt)) else np.zeros((sx,sy))
    out.surft=-9999 if (np.isscalar(surft)) else np.zeros((sx,sy))
    out.cot=-9999 if (np.isscalar(cot)) else np.zeros((sx,sy))
    out.reff=-9999 if (np.isscalar(reff)) else np.zeros((sx,sy))
    out.cstatus=-9999 if (np.isscalar(cstatus)) else np.zeros((sx,sy))
    out.aerosol_flag=-9999 if (np.isscalar(aerosol_flag)) else np.zeros((sx,sy))
    out.thinci_flag=-9999 if (np.isscalar(thinci_flag)) else np.zeros((sx,sy))
    out.snowice_flag=-9999 if (np.isscalar(thinci_flag)) else np.zeros((sx,sy))

    notgood=np.where(weighting < 0.)
    weighting[notgood] = 0.
    
    nLine, nPixel = cmask.shape[0:2]
    nxt,nyt = weighting.shape[1:3]

    nx = nxt/2
    ny = nyt/2
    
    for i in range(0, imx.shape[0]):
        for j in range(0, imx.shape[1]):

            yd = imy[i,j]
            xd = imx[i,j]

            if yd-ny < 0:
                yb=0
                ybw=ny-yd
            else:
                yb=yd-ny
                ybw=0
                
            if yd+ny > nPixel-1:
                ye=nPixel-1
                yew=nPixel-yd+ny-1
            else:
                ye=yd+ny+1
                yew=nyt

            if xd-nx < 0:
                xb=0
                xbw=nx-xd
            else:
                xb=xd-nx
                xbw=0
                
            if xd+nx > nLine-1:
                xe=nLine-1
                xew=nLine-xd+nx-1
            else:
                xe=xd+nx+1
                xew=nxt

            weighting_cut=weighting[j,xbw:xew,ybw:yew]

            cmask_cut = cmask[xb:xe, yb:ye]
            
            out.cmask[i,j] = np.sum(cmask_cut*weighting_cut)/np.sum(weighting_cut)
            
            if not (np.isscalar(out.cstatus)) :
                cstatus_cut = cstatus[xb:xe, yb:ye]
                out.cstatus[i,j] = np.sum(cstatus_cut*weighting_cut)/np.sum(weighting_cut)
            if not (np.isscalar(out.aerosol_flag)) :
                aerosol_flag_cut = aerosol_flag[xb:xe, yb:ye]
                out.aerosol_flag[i,j] = np.sum(aerosol_flag_cut*weighting_cut)/np.sum(weighting_cut)
            if not (np.isscalar(out.thinci_flag)) :
                thinci_flag_cut = thinci_flag[xb:xe, yb:ye]
                out.thinci_flag[i,j] = np.sum(thinci_flag_cut*weighting_cut)/np.sum(weighting_cut)
            if not (np.isscalar(out.snowice_flag)) :
                snowice_flag_cut = snowice_flag[xb:xe, yb:ye]
                out.snowice_flag[i,j] = np.sum(snowice_flag_cut*weighting_cut)/np.sum(weighting_cut)

            if not (np.isscalar(out.ctt)) :
                ctt_cut=ctt[xb:xe,yb:ye]
                goodi=np.where(ctt_cut > 0.)
                ctt_cut=ctt_cut[goodi]
                weighting_here = weighting_cut[goodi]
                out.ctt[i,j] = np.sum(ctt_cut*weighting_here)/np.sum(weighting_here)
            if not (np.isscalar(out.surft)) :
                surft_cut = surft[xb:xe, yb:ye]
                goodi=np.where(surft_cut > 0.)
                surft_cut=surft_cut[goodi]
                weighting_here = weighting_cut[goodi]
                out.surft[i,j] = np.sum(surft_cut*weighting_here)/np.sum(weighting_here)
            if not (np.isscalar(out.cot)) :
                cot_cut = cot[xb:xe, yb:ye]
                goodi=np.where(cot_cut > 0.)
                cot_cut=cot_cut[goodi]
                weighting_here = weighting_cut[goodi]
                out.cot[i,j] = np.sum(cot_cut*weighting_here)/np.sum(weighting_here)
            if not (np.isscalar(out.reff)) :
                reff_cut = reff[xb:xe,yb:ye]
                goodi=np.where(reff_cut > 0.)
                reff_cut=reff_cut[goodi]
                weighting_here = weighting_cut[goodi]
                out.reff[i,j] = np.sum(reff_cut*weighting_here)/np.sum(weighting_here)

                
    return out
        

#############################################################################################

def write_cloud_adapt(cloud_adapt, savefile, nx1=-1, nx2=-1):
    
    fout = netCDF4.Dataset(savefile,'w', format='NETCDF4')
    fout.createDimension('numberOfAirsTracks', cloud_adapt.ny)
    if nx2 < 0:
        fout.createDimension('numberOfAirsXTracks', cloud_adapt.nx)
        n1=0
        n2=cloud_adapt.nx
    else:
        n1=nx1
        n2=nx2
        fout.createDimension('numberOfAirsXTracks', n2-n1)

    if not (np.isscalar(cloud_adapt.ctt)):
        ncin = fout.createVariable('cldTopTemp', 'f8', ('numberOfAirsXTracks','numberOfAirsTracks'))
        ncin[:] = np.array(cloud_adapt.ctt[n1:n2,:])
    if not (np.isscalar(cloud_adapt.surft)):
        ncin = fout.createVariable('surfaceTemp', 'f8', ('numberOfAirsXTracks','numberOfAirsTracks'))
        ncin[:] = np.array(cloud_adapt.surft[n1:n2,:])
    if not (np.isscalar(cloud_adapt.cot)):
        ncin = fout.createVariable('cldOpticalThick', 'f8', ('numberOfAirsXTracks','numberOfAirsTracks'))
        ncin[:] = np.array(cloud_adapt.cot[n1:n2,:])
    if not (np.isscalar(cloud_adapt.reff)):
        ncin = fout.createVariable('cldParticleSize', 'f8', ('numberOfAirsXTracks','numberOfAirsTracks'))
        ncin[:] = np.array(cloud_adapt.reff[n1:n2,:])
    if not (np.isscalar(cloud_adapt.cmask)):
        ncin = fout.createVariable('cloudMaskCloudiness', 'f8', ('numberOfAirsXTracks','numberOfAirsTracks'))
        ncin[:] = np.array(cloud_adapt.cmask[n1:n2,:])
    if not (np.isscalar(cloud_adapt.cstatus)):
        ncin = fout.createVariable('cloudMaskStatus', 'f8', ('numberOfAirsXTracks','numberOfAirsTracks'))
        ncin[:] = np.array(cloud_adapt.cstatus[n1:n2,:])
    if not (np.isscalar(cloud_adapt.aerosol_flag)):
        ncin = fout.createVariable('heavyAerosolFlag', 'f8', ('numberOfAirsXTracks','numberOfAirsTracks'))
        ncin[:] = np.array(cloud_adapt.aerosol_flag[n1:n2,:])
    if not (np.isscalar(cloud_adapt.thinci_flag)):
        ncin = fout.createVariable('thinCirrusFlag', 'f8', ('numberOfAirsXTracks','numberOfAirsTracks'))
        ncin[:] = np.array(cloud_adapt.thinci_flag[n1:n2,:])
    if not (np.isscalar(cloud_adapt.snowice_flag)):
        ncin = fout.createVariable('snowIceFlag', 'f8', ('numberOfAirsXTracks','numberOfAirsTracks'))
        ncin[:] = np.array(cloud_adapt.snowice_flag[n1:n2,:])
    if not (np.isscalar(cloud_adapt.lat)):
        ncin = fout.createVariable('latitude', 'f8', ('numberOfAirsXTracks','numberOfAirsTracks'))
        ncin[:] = np.array(cloud_adapt.lat[n1:n2,:])
    if not (np.isscalar(cloud_adapt.lon)):
        ncin = fout.createVariable('longitude', 'f8', ('numberOfAirsXTracks','numberOfAirsTracks'))
        ncin[:] = np.array(cloud_adapt.lon[n1:n2,:])

    fout.close()
        
    return 1




#############################################################################################

    
def find_match_index_noweight (sounder_los, sounder_sat, imager_pos_in, imager_sdrQa_in, \
                      mx, my, fovDia=0.963):


        nLine, nPixel = imager_pos_in.shape[0:2]
        sounderShape = sounder_los.shape[0:sounder_los.ndim]        
        
        # setup parameters
        cos_half_fov=cos(deg2rad(fovDia/2.0))
        if nPixel == 3200: nc = np.round(deg2rad(0.963/2)*833.0/0.75*4).astype(np.int)
        if nPixel == 6400: nc = np.round(deg2rad(0.963/2)*833.0/0.375*4).astype(np.int)

        if nPixel == 1354: nc = np.round(deg2rad(0.963/2)*833.0/0.8*4).astype(np.int)
        

        # return list
        index_x = []
        index_y = []

        for i in range(0, mx.size):

                xd = mx[i]
                yd = my[i]

                xb = 0        if xd-nc <0        else xd-nc
                xe = nPixel-1 if xd+nc >nPixel-1 else xd+nc

                yb = 0        if yd-nc <0        else yd-nc
                ye = nLine-1  if yd+nc >nLine-1  else yd+nc

                imager_pos = imager_pos_in[yb:ye, xb:xe, : ]
                imager_Qa  = imager_sdrQa_in[yb:ye, xb:xe]
                imager_los = imager_pos  - sounder_sat[i, :]
                temp = np.dot(imager_los, sounder_los[i, :])
                temp = temp / LA.norm(imager_los, axis=2)
                cos_angle = temp / LA.norm(sounder_los[i, :])

                iy, ix = np.where ( (imager_Qa == 0) & (cos_angle > cos_half_fov) )

                index_x.append(ix+xb)
                index_y.append(iy+yb)

        return index_y, index_x        
    

##############################################################################################

# func to get hdf attributes 
def hdf_attributes(file_name, list_names):

  list_values = []

  h = HDF(file_name)
  vs = h.vstart()

  for name1 in list_names:
    xid = vs.find(name1)
    stid = vs.attach(xid)
    nrecs, intmode, fields, size, name = stid.inquire()
    x = stid.read(nrecs)
    list_values.append(x[0][0])
    
  stid.detach()
  return list_values
    


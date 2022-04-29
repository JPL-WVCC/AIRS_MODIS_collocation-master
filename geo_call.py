import numpy as np
import geo
import pdb


def geo_call(sounder_satAzimuth, sounder_satZenith, sounder_lat, sounder_lon, sounder_satRange,
             imager_lat, imager_lon, imager_sdrQa):
 
 # compute Sounder Pos Vector in EFEC on the Earth Surface 
 sounder_pos= np.zeros(np.append(sounder_lat.shape, 3))

 if (sounder_pos.ndim == 3):
  sounder_pos[:, :, 0], sounder_pos[:, :, 1], sounder_pos[:, :, 2] \
   = geo.LLA2ECEF(sounder_lon, sounder_lat, np.zeros_like(sounder_lat))
 if (sounder_pos.ndim == 4):
  sounder_pos[:, :, :, 0], sounder_pos[:, :, :, 1], sounder_pos[:, :, :, 2] \
   = geo.LLA2ECEF(sounder_lon, sounder_lat, np.zeros_like(sounder_lat))
   
 # compute Sounder LOS Vector in ECEF 
 sounder_east, sounder_north, sounder_up = geo.RAE2ENU(sounder_satAzimuth, sounder_satZenith, sounder_satRange)

 sounder_los= np.zeros(np.append(sounder_lat.shape, 3))

 if (sounder_los.ndim == 3):
  sounder_los[:, :,  0], sounder_los[:, :, 1], sounder_los[:, :, 2] = \
   geo.ENU2ECEF(sounder_east, sounder_north, sounder_up, sounder_lon, sounder_lat)

 if (sounder_los.ndim == 4):
  sounder_los[:, :, :, 0], sounder_los[:, :, :, 1], sounder_los[:, :, :, 2] = \
   geo.ENU2ECEF(sounder_east, sounder_north, sounder_up, sounder_lon, sounder_lat)
 
 # compute imager POS vector in ECEF
 imager_pos= np.zeros(np.append(imager_lat.shape, 3))
 imager_pos[:, :, 0], imager_pos[:, :, 1], imager_pos[:, :, 2] = \
  geo.LLA2ECEF(imager_lon, imager_lat, np.zeros_like(imager_lat))

 # sounder_los is pointing from pixel to satellite, we need to
 #   change from satellite to pixel
 sounder_los = -1.0*sounder_los

 # using Kd-tree to find the closted pixel of IMAGER for each Sounder FOV
 dy, dx, mx, my = geo.match_sounder_imager(sounder_los, sounder_pos, imager_pos, imager_sdrQa)

 return dy,dx, my, mx


 

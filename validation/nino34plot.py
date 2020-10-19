#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct  3 21:22:19 2020

@author: lxy
"""

import numpy as np
import pandas as pd
#import os
from netCDF4 import Dataset
import scipy.io as sio
from scipy import stats
import scipy.interpolate

import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap, shiftgrid

#for LMR
def SSTA_cal1(SST):
    '''
    SST with (time,lat,lon)
    '''
    a=np.shape(SST)
    ssta=np.zeros(a);
    for i in range(0,a[1]):
        for j in range(0,a[2]):
            sst=pd.Series(SST[:,i,j])
            ssta[:,i,j]=sst-sst.rolling(window=30).mean()
    return ssta

#mask values not SST (continent part) 
def maskanom(SSTA):
    '''
    SSTA with (time,lat,lon)
    (the SSTA is calculated using 30-year running mean and may have nan)
    '''
    SSTa=np.ma.masked_invalid(SSTA)
    ssta=np.ma.masked_equal(SSTa,0)
    return ssta


def Nino34_lmr(SSTA,lat,lon):
    '''
    normalize Nino3.4 index
    El Nino and La Nina events intensity
    SST with (time,lat,lon)
    
    '''
    #ssta=np.ma.masked_equal(SSTA,0)
    #Nino3.4:170W-120W(95-120),5N-5S(43-47)
    ssta=SSTA[:,(lat<=5)&(lat>=-5),:]
    Nino34=np.ma.mean(ssta[:,:,(lon<=240)&(lon>=190)],axis=(1,2))
    nino34=pd.Series(Nino34)
    #normalization
    Nino34=nino34-nino34.rolling(window=30).mean()
#    Nino34=(nino34-nino34.rolling(window=30).mean())/nino34.rolling(window=30).std()

    return Nino34

#PHYDA
filepath=Dataset('/Users/lxy/Desktop/research/ENSO/data/PHYDA/da_hydro_DecFeb_r.1-2000_d.05-Jan-2018.nc')
SST=filepath.variables['tas_mn']
SST=np.ma.getdata(SST)
PDSI=filepath.variables['pdsi_mn']
PDSI=np.ma.getdata(PDSI)
lat=filepath.variables['lat']
lat_phy=np.ma.getdata(lat)
lon=filepath.variables['lon']
lon_phy=np.ma.getdata(lon)
year_phy=np.arange(1,2001)

ssta_phy=SSTA_cal1(SST)

# mask the continent part to nan
SSTA_PHY=ssta_phy.copy()
SSTA_PHY[~np.ma.getmask(PDSI)]=np.nan

nino34_phy=Nino34_lmr(SSTA_PHY,lat_phy,lon_phy)

#LMRver2.1
filepath1=Dataset('/Users/lxy/Desktop/research/ENSO/data/LMR/sst_LMRv2.1.nc')
SST_LMR=filepath1.variables['sst']
lat_lmr=np.load('/Users/lxy/Desktop/research/ENSO/data/LMR/LMR_lats_v2.npy')
lon_lmr=np.load('/Users/lxy/Desktop/research/ENSO/data/LMR/LMR_lons_v2.npy')
filepath2=Dataset('/Users/lxy/Desktop/research/ENSO/data/LMR/pdsi_MCruns_ensemble_mean_LMRv2.1.nc')
pdsi=filepath2.variables['pdsi']
year_lmr=np.arange(2001)


sst_lmr=np.mean(SST_LMR,1)
ssta_lmr=SSTA_cal1(sst_lmr)
sst_lmr[sst_lmr==-9.96921e+36]=np.nan
SSTA_LMR=maskanom(ssta_lmr)

nino34_lmr=Nino34_lmr(SSTA_LMR,lat_lmr,lon_lmr)


fig=plt.figure(figsize=(10,5))

plt.plot(year_phy[(year_phy>=1000)&(year_phy<=1999)], 
                  nino34_phy[(year_phy>=1000)&(year_phy<=1999)], label='PHYDA')
plt.plot(year_lmr[(year_lmr>=1000)&(year_lmr<=1999)], 
                  nino34_lmr[(year_lmr>=1000)&(year_lmr<=1999)], label='LMR')
#plt.ylim(-3,3)
plt.legend()
plt.axhline(y=0,color='k',ls='--')
#plt.title('Nino3.4 anomaly',fontsize=14)
plt.xlabel('Year')
plt.ylabel('Nino3.4 anomaly ($^oC$)')

#fig.savefig('/Users/lxy/Desktop/research/ENSO/cp_ep/validation/nino34.png',dpi=600)

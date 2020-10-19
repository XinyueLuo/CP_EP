#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 21 15:00:29 2020

@author: lxy
"""

"""
This file is to calculate NINO index of different dataset and make comparison with the HadISST
Nino1+2: (0-10S, 90W-80W)
Nino 3: (5N-5S, 150W-90W)
Nino 4: (5N-5S, 160E-150W)
Nino 3.4: (5N-5S, 170W-120W)
definition all come from NOAA (https://www.esrl.noaa.gov/psd/enso/dashboard.lanina.html)

"""
import numpy as np
import pandas as pd
#import os
from netCDF4 import Dataset
import scipy.io as sio
from scipy import stats
from eofs.standard import Eof
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap

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

#for HADISST
def SSTA_cal2(SST):
    '''
    SST with (time,lat,lon)
    '''
    a=np.shape(SST)
    ssta=np.zeros((a[2],a[0],a[1]));
    for i in range(0,a[0]):
        for j in range(0,a[1]):
            sst=pd.Series(SST[i,j,:])
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

def Nino_had(SSTA,lat,lon):
    '''
    normalize Nino index (1+2, 3, 4, 3.4)
    El Nino and La Nina events intensity
    SST with (time,lat,lon)
    
    '''
    #ssta=np.ma.masked_equal(SSTA,0)
    #Nino3.4:170W-120W(95-120),5N-5S(43-47)
    ssta=SSTA[:,(lat<=5)&(lat>=-5),:]
    Nino34=np.ma.mean(ssta[:,:,(lon<=240)&(lon>=190)],axis=(1,2))
    #normalization
    Nino34_std=Nino34/np.nanstd(Nino34)
    
    #Nino1+2:90W-80W,0-10S
    ssta=SSTA[:,(lat<=0)&(lat>=-10),:]
    Nino12=np.ma.mean(ssta[:,:,(lon<=280)&(lon>=270)],axis=(1,2))
    #normalization
    Nino12_std=Nino12/np.nanstd(Nino12)

    #Nino3:150W-90W,5N-5S
    ssta=SSTA[:,(lat<=5)&(lat>=-5),:]
    Nino3=np.ma.mean(ssta[:,:,(lon<=270)&(lon>=210)],axis=(1,2))
    #normalization
    Nino3_std=Nino3/np.nanstd(Nino3)

    #Nino4:160E-150W,5N-5S
    ssta=SSTA[:,(lat<=5)&(lat>=-5),:]
    Nino4=np.ma.mean(ssta[:,:,(lon<=210)&(lon>=160)],axis=(1,2))
    #normalization
    Nino4_std=Nino4/np.nanstd(Nino4)

    
    #Nino34_std=np.ma.anom(Nino34)
    return Nino12_std, Nino3_std, Nino34_std, Nino4_std

#function to calculate Nino index
def Nino(SSTA,lat,lon):
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
    Nino34_std=(nino34-nino34.rolling(window=30).mean())/nino34.rolling(window=30).std()
    
    #Nino1+2:90W-80W,0-10S
    ssta=SSTA[:,(lat<=0)&(lat>=-10),:]
    Nino12=np.ma.mean(ssta[:,:,(lon<=280)&(lon>=270)],axis=(1,2))
    nino12=pd.Series(Nino12)
    #normalization
    Nino12_std=(nino12-nino12.rolling(window=30).mean())/nino12.rolling(window=30).std()

    #Nino3:150W-90W,5N-5S
    ssta=SSTA[:,(lat<=5)&(lat>=-5),:]
    Nino3=np.ma.mean(ssta[:,:,(lon<=270)&(lon>=210)],axis=(1,2))
    nino3=pd.Series(Nino3)
    #normalization
    Nino3_std=(nino3-nino3.rolling(window=30).mean())/nino3.rolling(window=30).std()

    #Nino4:160E-150W,5N-5S
    ssta=SSTA[:,(lat<=5)&(lat>=-5),:]
    Nino4=np.ma.mean(ssta[:,:,(lon<=210)&(lon>=160)],axis=(1,2))
    nino4=pd.Series(Nino4)
    #normalization
    Nino4_std=(nino4-nino4.rolling(window=30).mean())/nino4.rolling(window=30).std()

    
    
    #Nino34_std=np.ma.anom(Nino34)
    return Nino12_std, Nino3_std, Nino34_std, Nino4_std


# =============================================================================
# #HadISST
# =============================================================================
data=sio.loadmat('/Users/lxy/Desktop/research/ENSO/data/HadiSST/HadISSTt(3)_fulltropics.mat')
SST_HAD=data['HadISST_annual']#1871-1999
lat_had=data['latHADISST'].squeeze()
lon_had=data['lonHADISST'].squeeze()
year_had=np.arange(1871,2000)

ssta_had=SSTA_cal2(SST_HAD)
SSTA_HAD=maskanom(ssta_had)

hadn12,hadn3,hadn34,hadn4=Nino_had(SSTA_HAD,lat_had,lon_had)

# =============================================================================
# #LMR2.0 LMR2.1
# =============================================================================
SST_LMR0=np.load('/Users/lxy/Desktop/research/ENSO/data/LMR/LMR_SST_v2.npy')
filepath1=Dataset('/Users/lxy/Desktop/research/ENSO/data/LMR/sst_LMRv2.1.nc')
SST_LMR1=filepath1.variables['sst']
year_lmr=np.arange(2001)
lat_lmr=np.load('/Users/lxy/Desktop/research/ENSO/data/LMR/LMR_lats_v2.npy')
lon_lmr=np.load('/Users/lxy/Desktop/research/ENSO/data/LMR/LMR_lons_v2.npy')
#ensemble mean
sst_lmr0=np.mean(SST_LMR0,1)
sst_lmr1=np.mean(SST_LMR1,1)

#running removal of 30-year
ssta_lmr0=SSTA_cal1(sst_lmr0)
SSTA_LMR0=maskanom(ssta_lmr0)
ssta_lmr1=SSTA_cal1(sst_lmr1)
SSTA_LMR1=maskanom(ssta_lmr1)

lmr0n12,lmr0n3,lmr0n34,lmr0n4=Nino(SSTA_LMR0,lat_lmr,lon_lmr)
lmr1n12,lmr1n3,lmr1n34,lmr1n4=Nino(SSTA_LMR1,lat_lmr,lon_lmr)


# =============================================================================
# #PHYDA
# =============================================================================
filepath2=Dataset('/Users/lxy/Desktop/research/ENSO/data/PHYDA/da_hydro_AprMar_r.1-2000_d.05-Jan-2018.nc')
SST_PHY=filepath2.variables['tas_mn']
lat_phy=filepath2.variables['lat']
lat_phy=np.ma.getdata(lat_phy)
lon_phy=filepath2.variables['lon']
lon_phy=np.ma.getdata(lon_phy)
year_phy=filepath2.variables['time']
year_phy=np.ma.getdata(year_phy)

SSTA_PHY=SSTA_cal1(SST_PHY)

phyn12,phyn3,phyn34,phyn4=Nino(SST_PHY,lat_phy,lon_phy)


# =============================================================================
# colralation coefficent
# =============================================================================

rlmr0n12,plmr0n12=stats.pearsonr(lmr0n12[(year_lmr>=1900)&(year_lmr<=1999)],
                                      hadn12[(year_had>=1900)&(year_had<=1999)])
rlmr0n3,plmr0n3=stats.pearsonr(lmr0n3[(year_lmr>=1900)&(year_lmr<=1999)],
                                      hadn3[(year_had>=1900)&(year_had<=1999)])
rlmr0n34,plmr0n34=stats.pearsonr(lmr0n34[(year_lmr>=1900)&(year_lmr<=1999)],
                                      hadn34[(year_had>=1900)&(year_had<=1999)])
rlmr0n4,plmr0n4=stats.pearsonr(lmr0n4[(year_lmr>=1900)&(year_lmr<=1999)],
                                      hadn4[(year_had>=1900)&(year_had<=1999)])

rlmr1n12,plmr1n12=stats.pearsonr(lmr1n12[(year_lmr>=1900)&(year_lmr<=1999)],
                                      hadn12[(year_had>=1900)&(year_had<=1999)])
rlmr1n3,plmr1n3=stats.pearsonr(lmr1n3[(year_lmr>=1900)&(year_lmr<=1999)],
                                      hadn3[(year_had>=1900)&(year_had<=1999)])
rlmr1n34,plmr1n34=stats.pearsonr(lmr1n34[(year_lmr>=1900)&(year_lmr<=1999)],
                                      hadn34[(year_had>=1900)&(year_had<=1999)])
rlmr1n4,plmr1n4=stats.pearsonr(lmr1n4[(year_lmr>=1900)&(year_lmr<=1999)],
                                      hadn4[(year_had>=1900)&(year_had<=1999)])

rphyn12,pphyn12=stats.pearsonr(phyn12[(year_phy>=1900)&(year_phy<=1999)],
                                      hadn12[(year_had>=1900)&(year_had<=1999)])
rphyn3,pphyn3=stats.pearsonr(phyn3[(year_phy>=1900)&(year_phy<=1999)],
                                      hadn3[(year_had>=1900)&(year_had<=1999)])
rphyn34,pphyn34=stats.pearsonr(phyn34[(year_phy>=1900)&(year_phy<=1999)],
                                      hadn34[(year_had>=1900)&(year_had<=1999)])
rphyn4,pphyn4=stats.pearsonr(phyn4[(year_phy>=1900)&(year_phy<=1999)],
                                      hadn4[(year_had>=1900)&(year_had<=1999)])




# =============================================================================
# plot
# =============================================================================
fig=plt.figure(figsize=(10,10))
ax1=plt.subplot(4, 1, 1)
plt.plot(year_had[(year_had>=1900)&(year_had<=1999)], 
                  hadn12[(year_had>=1900)&(year_had<=1999)],label='HADISST')
plt.plot(year_had[(year_had>=1900)&(year_had<=1999)], 
                  lmr0n12[(year_lmr>=1900)&(year_lmr<=1999)],ls='--',
                  label='LMR v2.0 (r='+str(round(rlmr0n12,2))+' p<0.005)')
plt.plot(year_had[(year_had>=1900)&(year_had<=1999)], 
                  lmr1n12[(year_lmr>=1900)&(year_lmr<=1999)],ls='--',
                  label='LMR v2.1 (r='+str(round(rlmr1n12,2))+' p<0.005)')
plt.plot(year_had[(year_had>=1900)&(year_had<=1999)], 
                  phyn12[(year_phy>=1900)&(year_phy<=1999)],ls='--',
                  label='PHYDA (r='+str(round(rphyn12,2))+' p<0.005)')

plt.ylim(-4,4)
plt.legend(loc='lower right',fontsize='small',ncol=2)
plt.axhline(y=0,color='k',ls='--')
plt.title('Nino1+2 index',fontsize=12)

ax2=plt.subplot(4, 1, 2)
plt.plot(year_had[(year_had>=1900)&(year_had<=1999)], 
                  hadn3[(year_had>=1900)&(year_had<=1999)],label='HADISST')
plt.plot(year_had[(year_had>=1900)&(year_had<=1999)], 
                  lmr0n3[(year_lmr>=1900)&(year_lmr<=1999)],ls='--',
                  label='LMR v2.0 (r='+str(round(rlmr0n3,2))+' p<0.005)')
plt.plot(year_had[(year_had>=1900)&(year_had<=1999)], 
                  lmr1n3[(year_lmr>=1900)&(year_lmr<=1999)],ls='--',
                  label='LMR v2.1 (r='+str(round(rlmr1n3,2))+' p<0.005)')
plt.plot(year_had[(year_had>=1900)&(year_had<=1999)], 
                  phyn3[(year_phy>=1900)&(year_phy<=1999)],ls='--',
                  label='PHYDA (r='+str(round(rphyn3,2))+' p<0.005)')

plt.ylim(-4,4)
plt.legend(loc='lower right',fontsize='small',ncol=2)
plt.axhline(y=0,color='k',ls='--')
plt.title('Nino3 index',fontsize=12)

ax3=plt.subplot(4, 1, 3)
plt.plot(year_had[(year_had>=1900)&(year_had<=1999)], 
                  hadn34[(year_had>=1900)&(year_had<=1999)],label='HADISST')
plt.plot(year_had[(year_had>=1900)&(year_had<=1999)], 
                  lmr0n34[(year_lmr>=1900)&(year_lmr<=1999)],ls='--',
                  label='LMR v2.0 (r='+str(round(rlmr0n34,2))+' p<0.005)')
plt.plot(year_had[(year_had>=1900)&(year_had<=1999)], 
                  lmr1n34[(year_lmr>=1900)&(year_lmr<=1999)],ls='--',
                  label='LMR v2.1 (r='+str(round(rlmr1n34,2))+' p<0.005)')
plt.plot(year_had[(year_had>=1900)&(year_had<=1999)], 
                  phyn34[(year_phy>=1900)&(year_phy<=1999)],ls='--',
                  label='PHYDA (r='+str(round(rphyn34,2))+' p<0.005)')

plt.ylim(-4,4)
plt.legend(loc='lower right',fontsize='small',ncol=2)
plt.axhline(y=0,color='k',ls='--')
plt.title('Nino3.4 index',fontsize=12)


ax4=plt.subplot(4, 1, 4)
plt.plot(year_had[(year_had>=1900)&(year_had<=1999)], 
                  hadn4[(year_had>=1900)&(year_had<=1999)],label='HADISST')
plt.plot(year_had[(year_had>=1900)&(year_had<=1999)], 
                  lmr0n4[(year_lmr>=1900)&(year_lmr<=1999)],ls='--',
                  label='LMR v2.0 (r='+str(round(rlmr0n4,2))+' p<0.005)')
plt.plot(year_had[(year_had>=1900)&(year_had<=1999)], 
                  lmr1n4[(year_lmr>=1900)&(year_lmr<=1999)],ls='--',
                  label='LMR v2.1 (r='+str(round(rlmr1n4,2))+' p<0.005)')
plt.plot(year_had[(year_had>=1900)&(year_had<=1999)], 
                  phyn4[(year_phy>=1900)&(year_phy<=1999)],ls='--',
                  label='PHYDA (r='+str(round(rphyn4,2))+' p<0.005)')

plt.ylim(-4,4)
plt.legend(loc='lower right',fontsize='small',ncol=2)
plt.axhline(y=0,color='k',ls='--')
plt.title('Nino4 index',fontsize=12)

plt.tight_layout()


#fig.savefig('/Users/lxy/Desktop/research/ENSO/cp_ep/validation/index.png',dpi=600)




#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 14:17:04 2019

@author: lxy
"""
"""
CP and EP index and SSTA patterns validation
(CP and EP El Nino events)
"""
import numpy as np
import pandas as pd
from netCDF4 import Dataset

#import os
import scipy.io as sio
from scipy import stats
from eofs.standard import Eof
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

#for NADA
def SSTA_cal2(SST):
    '''
    input with (lon,lat,time)
    output with (time, lat, lon)
    '''
    a=np.shape(SST)
    ssta=np.zeros((a[2],a[1],a[0]));
    for i in range(0,a[1]):
        for j in range(0,a[0]):
            sst=pd.Series(SST[j,i,:])
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

#function to calculate Nino3.4 index
def Nino34_nom(SSTA,lat,lon):
    '''
    normalize Nino3.4 index
    El Nino and La Nina events intensity
    SST with (time,lat,lon)
    
    '''
    #ssta=np.ma.masked_equal(SSTA,0)
    #Nino3.4:170W-120W(95-120),5N-5S(43-47)
    ssta=SSTA[:,(lat<=5)&(lat>=-5),:]
    Nino34=np.ma.mean(ssta[:,:,(lon<=240)&(lon>=190)],axis=(1,2))
     
    #normalization
    Nino34_std=Nino34/np.nanstd(Nino34)
    #Nino34_std=np.ma.anom(Nino34)
    return Nino34_std

#function for Nino3-4 index
def Nino3_4index(SSTA,lat,lon):
    """
    NIno3-4 index to define CP&EP
    Nino4 exceeds one standard deviation(1 sigma) and Nino4 exceeds SSTA average in Nino3---CP
    Nino3 exceeds one standard deviation(1 sigma) and Nino3 exceeds SSTA average in Nino4---EP
    SST with (time,lat,lon)
    """
    
# =============================================================================
#     #Nino3.4:170W-120W(95-120),5N-5S(43-47)
#     SSTA_34=np.ma.anom(SST[:,43:47,95:120],0)
#     Nino34=np.ma.average(SSTA_34,(1,2))
# =============================================================================
    #Nino3:150W-90W,5N-5S
    ssta=SSTA[:,(lat<=5)&(lat>=-5),:]
    Nino3=np.ma.average(ssta[:,:,(lon<=270)&(lon>=210)],axis=(1,2))
    
    #Nino4:160E-150W,5N-5S
    ssta=SSTA[:,(lat<=5)&(lat>=-5),:]
    Nino4=np.ma.mean(ssta[:,:,(lon<=210)&(lon>=160)],axis=(1,2))
    
# =============================================================================
#     nino3=pd.Series(Nino3)
#     nino4=pd.Series(Nino4)
#     
#     #moving Normalization
#     Nino3_std=(nino3-nino3.rolling(window=30).mean())/nino3.rolling(window=30).std()
#     Nino4_std=(nino4-nino4.rolling(window=30).mean())/nino4.rolling(window=30).std()
# =============================================================================
        
    return Nino4,Nino3 #cp,ep

def EP_CPindex(SSTA,lat,lon):
    """
    EP_CPindex method to define CP&EP
    FOR CP: regression of Nino1+2 SSTA associated with eastern warming is removed
    FOR EP: regression of Nino 4 SSTA associated with cetral warming is removed
    both EOF to find 1st PC time series (exceed on standard deviation-1 sigma)
    
    SSTA with (time,lat,lon) with masked values and nan
    """
    #Nino1+2:90W-80W(135-140),0-10S(45-50)
    ssta12=SSTA[:,(lat<=10)&(lat>=0),:]
    Nino12=np.ma.average(ssta12[:,:,(lon<=280)&(lon>=270)],(1,2))
    Nino12=np.ma.getdata(Nino12)
    
    #Nino4:160E-150W(80-105),5N-5S(43-47)
    ssta4=SSTA[:,(lat<=5)&(lat>=-5),:]
    Nino4=np.ma.average(ssta4[:,:,(lon<=210)&(lon>=160)],(1,2))
    Nino4=np.ma.getdata(Nino4)

    #tropical pacific:120E-80W(60-140),20S-20N(35-55)
    SSTA_TP=SSTA[:,(lat<=30)&(lat>=-30),:]
    SSTA_TP=SSTA_TP[:,:,(lon<=280)&(lon>=120)]
    
    lat1=lat[(lat<=30)&(lat>=-30)]
    lon1=lon[(lon<=280)&(lon>=120)]

    SSTA_TP12=np.zeros(SSTA_TP.shape)
    SSTA_TP4=np.zeros(SSTA_TP.shape)
    
    for i in range(0,SSTA_TP.shape[1]):
        for j in range(0,SSTA_TP.shape[2]):
            k12,_,_,_,_=stats.linregress(Nino12,SSTA_TP[:,i,j])
            SSTA_TP12[:,i,j]=SSTA_TP[:,i,j]-k12*Nino12
            k4,_,_,_,_=stats.linregress(Nino4,SSTA_TP[:,i,j])
            SSTA_TP4[:,i,j]=SSTA_TP[:,i,j]-k4*Nino4
    
    #EOF analysis  
    #coslat=np.cos(np.deg2rad(np.arange(-20,21,2)))
    #wgt=np.sqrt(coslat)[..., np.newaxis]
    solver12=Eof(SSTA_TP12)
    eof12=solver12.eofsAsCorrelation(neofs=1)
    PC12=solver12.pcs(npcs=1)
    PC12=PC12[:,0]
    a=eof12[:,(lat1<=5)&(lat1>=-5),:]
    if np.mean(a[:,:,(lon1<=240)&(lon1>=190)].squeeze(),(0,1))<0:
        PC12=-PC12
        
    solver4=Eof(SSTA_TP4)
    eof4=solver4.eofsAsCorrelation(neofs=1)
    PC4=solver4.pcs(npcs=1)
    PC4=PC4[:,0]
    b=eof4[:,(lat1<=5)&(lat1>=-5),:]

    if np.mean(b[:,:,(lon1<=240)&(lon1>=190)].squeeze(),(0,1))<0:
        PC4=-PC4
        
    #PC12 is for cp definition and PC4 is for EP
        
    #standardized
# =============================================================================
#     pc12_std=(PC12-np.mean(PC12))/np.std(PC12)
#     pc4_std=(PC4-np.mean(PC4))/np.std(PC4)
# =============================================================================
# =============================================================================
#     pc12=pd.Series(PC12[:,0])
#     pc4=pd.Series(PC4[:,0])
#     pc12_std=(pc12-pc12.rolling(window=30).mean())/pc12.rolling(window=30).std()
#     pc4_std=(pc4-pc4.rolling(window=30).mean())/pc4.rolling(window=30).std()
# =============================================================================


    return PC12,PC4 #CP, EP

def E_Cindex(SSTA,lat,lon):
    """
    E and C indices to define EP&CP
    two orthogonal axes are rotated 45° relative to the principal components of SSTA
    SSTA with (timme,lat,lon)
    """
    #tropical pacific:120E-80W(60-140),20S-20N(35-55)
    SSTA_TP=SSTA[:,(lat<=30)&(lat>=-30),:]
    SSTA_TP=SSTA_TP[:,:,(lon<=280)&(lon>=120)]    
    
    lat1=lat[(lat<=30)&(lat>=-30)]
    lon1=lon[(lon<=280)&(lon>=120)]
    #EOF analysis and to get the first 2 pcs
    #coslat=np.cos(np.deg2rad(np.arange(-20,21,2)))
    solver=Eof(SSTA_TP[29:,:,:])
    pcs=solver.pcs(npcs=2,pcscaling=1)
    eof=solver.eofsAsCorrelation(neofs=2)
    a=eof[0,(lat1<=5)&(lat1>=-5),:]
    b=eof[1,(lat1<=5)&(lat1>=-5),:]

    if np.mean(a[:,(lon1<=240)&(lon1>=190)],(0,1))<0:
        pcs[:,0]=-pcs[:,0]
        
    if np.mean(b[:,(lon1<=240)&(lon1>=190)],(0,1))>0:
        pcs[:,1]=-pcs[:,1]
    
    #do the 45rotation
    C_index=(pcs[:,0]+pcs[:,1])/np.sqrt(2)
    E_index=(pcs[:,0]-pcs[:,1])/np.sqrt(2)
    
    
    #find EP&CP years
# =============================================================================
#     CI_std=(C_index-np.mean(C_index))/np.std(C_index)
#     EI_std=(E_index-np.mean(E_index))/np.std(E_index)
# =============================================================================
    
# =============================================================================
#     cindex=pd.Series(C_index)
#     eindex=pd.Series(E_index)
#     
#     
#     #find EP&CP years
#     CI_std=(cindex-cindex.rolling(window=30).mean())/cindex.rolling(window=30).std()
#     EI_std=(eindex-eindex.rolling(window=30).mean())/eindex.rolling(window=30).std()
# =============================================================================
    
    return C_index,E_index
  
    
# =============================================================================
# NADA
# =============================================================================
filepath4=Dataset('/Users/lxy/Desktop/research/ENSO/data/nada_hd2_cl.nc')
pdsi_nada=filepath4.variables['pdsi']#(lon,lat,time)
lat_nada=filepath4.variables['lat']
lat_nada=np.ma.getdata(lat_nada)
lon_nada=filepath4.variables['lon']
lon_nada=np.ma.getdata(lon_nada)
lon_nada=lon_nada+360
yr_nada=filepath4.variables['time']
yr_nada=np.ma.getdata(yr_nada)#0-2005
year_nada=yr_nada[(yr_nada>=1870) & (yr_nada<=1999)]

# =============================================================================
# #match year flag
# pdsi_had=pdsi_nada[:,:,(yr_nada>=np.min(year_lmr))&(yr_nada<=np.max(year_lmr))]
# PDSIA_nada=SSTA_cal2(pdsi_had)
# =============================================================================


#match the year flag with HadISST
pdsi_had=pdsi_nada[:,:,(yr_nada>=1870) & (yr_nada<=1999)]
PDSIA_nada=SSTA_cal2(pdsi_had)

  
# =============================================================================
# HadISST
# =============================================================================
filepath3=Dataset('/Users/lxy/Desktop/research/ENSO/data/HadiSST/HadISST_sst.nc')
SST_HAD=filepath3.variables['sst']
lat_had=filepath3.variables['latitude']
lat_had=np.ma.getdata(lat_had)
lonhad=filepath3.variables['longitude']
lonhad=np.ma.getdata(lonhad)
year=np.arange(1870,2020)
year_had=year[(year>=1870) & (year<=1999)]

sst_had=np.zeros((150,180,360))

#annual mean (Jan-Dec)
for i in range(0,150):
    sst_had[i,:,:]=np.ma.average(SST_HAD[i*12:11+i*12,:,:],0)
    
ssthad,lon_had=shiftgrid(0.5,sst_had,lonhad,start=True)

#running removal of 30-year
ssta_had=SSTA_cal1(ssthad[(year>=842)&(year<=1999),:,:])
SSTA_HAD=maskanom(ssta_had)


cpindex_had,epindex_had=E_Cindex(SSTA_HAD,lat_had,lon_had)
#cpindex_had,epindex_had=EP_CPindex(SSTA_HAD[29:,:,:],lat_had,lon_had)
#cpindex_had,epindex_had=Nino3_4index(SSTA_HAD[29:,:,:],lat_had,lon_had)

# =============================================================================
# #LMR ver2.0
# SST_LMR=np.load('/Users/lxy/Desktop/research/ENSO/data/LMR/LMR_SST_v2.npy')
# 
# lat_lmr=np.load('/Users/lxy/Desktop/research/ENSO/data/LMR/LMR_lats_v2.npy')
# lon_lmr=np.load('/Users/lxy/Desktop/research/ENSO/data/LMR/LMR_lons_v2.npy')
# 
# #ensemble mean
# sst_lmr=np.mean(SST_LMR,1)
# 
# #sst_lmr=SST_LMR[:,6,:,:]
# 
# year=np.arange(2001)
# year_lmr=year[(year>=1842) & (year<=1999)]
# 
# #running removal of 30-year mean
# ssta_lmr=SSTA_cal1(sst_lmr[(year>=1842)&(year<=1999),:,:])
# SSTA=maskanom(ssta_lmr)
# =============================================================================



# =============================================================================
# # =============================================================================
# # #LMR ver2.1
# # =============================================================================
# filepath=Dataset('/Users/lxy/Desktop/research/ENSO/data/LMR/sst_LMRv2.1.nc')
# SST=filepath.variables['sst']
# lat_lmr=np.load('/Users/lxy/Desktop/research/ENSO/data/LMR/LMR_lats_v2.npy')
# lon_lmr=np.load('/Users/lxy/Desktop/research/ENSO/data/LMR/LMR_lons_v2.npy')
# year=np.arange(2001)
# year_lmr=year[(year>=842) & (year<=1999)]
# 
# #ensemble mean
# sst_lmr=np.mean(SST,1)
# 
# #running removal of 30-year
# ssta_lmr=SSTA_cal1(sst_lmr[(year>=842)&(year<=1999),:,:])
# SSTA_lmr=maskanom(ssta_lmr)
# 
# =============================================================================
# =============================================================================
# #LMRver2.1
# =============================================================================
filepath1=Dataset('/Users/lxy/Desktop/research/ENSO/data/LMR/sst_LMRv2.1.nc')
SST_LMR=filepath1.variables['sst']
lat_lmr=np.load('/Users/lxy/Desktop/research/ENSO/data/LMR/LMR_lats_v2.npy')
lon_lmr=np.load('/Users/lxy/Desktop/research/ENSO/data/LMR/LMR_lons_v2.npy')
filepath2=Dataset('/Users/lxy/Desktop/research/ENSO/data/LMR/pdsi_MCruns_ensemble_mean_LMRv2.1.nc')
pdsi=filepath2.variables['pdsi']
year=np.arange(2001)
year_lmr=year[(year>=842) & (year<=1999)]


sst_lmr=np.mean(SST_LMR,1)
PDSI=np.mean(pdsi,1)

ssta_lmr=SSTA_cal1(sst_lmr[(year>=842)&(year<=1999),:,:])
#sst_lmr[sst_lmr==-9.96921e+36]=np.nan

SSTA_LMR=maskanom(ssta_lmr)
pdsia_lmr=SSTA_cal1(PDSI)
PDSIA_LMR=maskanom(pdsia_lmr)

cpindex_lmr,epindex_lmr=E_Cindex(SSTA_LMR,lat_lmr,lon_lmr)
#cpindex_lmr,epindex_lmr=EP_CPindex(SSTA[29:,:,:],lat_lmr,lon_lmr)
#cpindex_lmr,epindex_lmr=Nino3_4index(SSTA[29:,:,:],lat_lmr,lon_lmr)

# =============================================================================
# PHYDA
# =============================================================================
filepath2=Dataset('/Users/lxy/Desktop/research/ENSO/data/PHYDA/da_hydro_DecFeb_r.1-2000_d.05-Jan-2018.nc')
SST=filepath2.variables['tas_mn']
SST=np.ma.getdata(SST)
PDSI=filepath2.variables['pdsi_mn']
PDSI=np.ma.getdata(PDSI)
lat=filepath2.variables['lat']
lat_phy=np.ma.getdata(lat)
lon=filepath2.variables['lon']
lon_phy=np.ma.getdata(lon)
year=np.arange(1,2001)
year_phy=year[(year>=1842) & (year<=1999)]

# mask the continent part to nan
SST[~np.ma.getmask(PDSI)]=np.nan

ssta_phy=SSTA_cal1(SST[(year>=842) & (year<=1999),:,:])
PDSIA_LMR=SSTA_cal1(PDSI)

SSTA=maskanom(ssta_lmr)

# =============================================================================
# #PHYDA
# filepath=Dataset('/Users/lxy/Desktop/research/ENSO/data/PHYDA/da_hydro_AprMar_r.1-2000_d.05-Jan-2018.nc')
# SST=filepath.variables['tas_mn']
# SST=np.ma.getdata(SST)
# PDSI=filepath.variables['pdsi_mn']
# PDSI=np.ma.getdata(PDSI)
# lat=filepath.variables['lat']
# lat_lmr=np.ma.getdata(lat)
# lon=filepath.variables['lon']
# lon_lmr=np.ma.getdata(lon)
# year=np.arange(1,2001)
# year_lmr=year[(year>=1842) & (year<=1999)]
# 
# 
# #mask the continent part to nan
# SST[~np.ma.getmask(PDSI)]=np.nan
# 
# ssta_lmr=SSTA_cal1(SST[(year>=1842) & (year<=1999),:,:])
# SSTA=maskanom(ssta_lmr)
# =============================================================================

# =============================================================================
# #CESM
# SST=np.load('/Users/lxy/Desktop/research/ENSO/data/CCSM4_SST_annual_anomalies_85001-200512_regridded.npy')
# lat_lmr=np.load('/Users/lxy/Desktop/research/ENSO/data/tropics_regrid_lats.npy')
# lon_lmr=np.load('/Users/lxy/Desktop/research/ENSO/data/tropics_regrid_lons.npy')
# year=np.arange(851,2006)
# year_lmr=year[(year>=1842) & (year<=1999)]
# 
# ssta_lmr=SSTA_cal1(SST[(year>=1842) & (year<=1999),:,:])
# SSTA=maskanom(ssta_lmr)
# =============================================================================



cpindex_lmr,epindex_lmr=E_Cindex(SSTA_LMR,lat_lmr,lon_lmr)
#cpindex_lmr,epindex_lmr=EP_CPindex(SSTA[29:,:,:],lat_lmr,lon_lmr)
#cpindex_lmr,epindex_lmr=Nino3_4index(SSTA[29:,:,:],lat_lmr,lon_lmr)

# =============================================================================
# subtract first 30 years
# =============================================================================
ssta_lmr=ssta_lmr[29:,:,:]
ssta_had=ssta_had[29:,:,:]


year_lmr=year_lmr[29:]
year_had=year_had[29:]

# standardized cp and ep index if HADISST
cphad=(cpindex_had-np.mean(cpindex_had))/np.std(cpindex_had)
ephad=(epindex_had-np.mean(epindex_had))/np.std(epindex_had)

# standardized cp and ep index if LMR
cpindex_lmr=pd.Series(cpindex_lmr)
epindex_lmr=pd.Series(epindex_lmr)
cplmr=(cpindex_lmr-cpindex_lmr.rolling(window=30).mean())/cpindex_lmr.rolling(window=30).std()
eplmr=(epindex_lmr-epindex_lmr.rolling(window=30).mean())/epindex_lmr.rolling(window=30).std()

# =============================================================================
# cplmr=cplmr.squeeze()
# eplmr=eplmr.squeeze()
# cphad=cphad.squeeze()
# ephad=ephad.squeeze()
# =============================================================================

# =============================================================================
# CP
# =============================================================================
sstacp_lmr=ssta_lmr[(cplmr>1) & (cplmr>eplmr) & (cplmr+eplmr>1),:,:]
yrcp_lmr=year_lmr[(cplmr>1) & (cplmr>eplmr) & (cplmr+eplmr>1)]
sstacp_had=ssta_had[(cphad>1) & (cphad>ephad) & (cphad+ephad>1),:,:]
yrcp_had=year_had[(cphad>1) & (cphad>ephad) & (cphad+ephad>1)]

#EP
sstaep_lmr=ssta_lmr[(eplmr>1) & (eplmr>cplmr) & (cplmr+eplmr>1),:,:]
yrep_lmr=year_lmr[(eplmr>1) & (eplmr>cplmr) & (cplmr+eplmr>1)]
sstaep_had=ssta_had[(ephad>1) & (ephad>cphad) & (cphad+ephad>1),:,:]
yrep_had=year_had[(ephad>1) & (ephad>cphad) & (cphad+ephad>1)]


sstaep_lmrmean=np.mean(sstaep_lmr[(yrep_lmr>=1900)&(yrep_lmr<=1999),:,:],0)
sstaep_lmrmean=sstaep_lmrmean[(lat_lmr<=30)&(lat_lmr>=-30),:]
sstaep_lmrmean=sstaep_lmrmean[:,(lon_lmr<=280)&(lon_lmr>=120)]
sstaep_lmrmean[sstaep_lmrmean==0]=np.nan

sstaep_hadmean=np.mean(sstaep_had[(yrep_had>=1900)&(yrep_had<=1999),:,:],0)
sstaep_hadmean=sstaep_hadmean[(lat_had<=30)&(lat_had>=-30),:]
sstaep_hadmean=sstaep_hadmean[:,(lon_had<=280)&(lon_had>=120)]
sstaep_hadmean[sstaep_hadmean==0]=np.nan

sstacp_lmrmean=np.mean(sstacp_lmr[(yrcp_lmr>=1900)&(yrcp_lmr<=1999),:,:],0)
sstacp_lmrmean=sstacp_lmrmean[(lat_lmr<=30)&(lat_lmr>=-30),:]
sstacp_lmrmean=sstacp_lmrmean[:,(lon_lmr<=280)&(lon_lmr>=120)]
sstacp_lmrmean[sstacp_lmrmean==0]=np.nan

sstacp_hadmean=np.mean(sstacp_had[(yrcp_had>=1900)&(yrcp_had<=1999),:,:],0)
sstacp_hadmean=sstacp_hadmean[(lat_had<=30)&(lat_had>=-30),:]
sstacp_hadmean=sstacp_hadmean[:,(lon_had<=280)&(lon_had>=120)]
sstacp_hadmean[sstacp_hadmean==0]=np.nan


# =============================================================================
# SIGNIFICANCE TEST
# =============================================================================
rc,pc=stats.pearsonr(cplmr[(year_lmr>=1900)&(year_lmr<=1999)],
                                      cphad[(year_had>=1900)&(year_had<=1999)])
re,pe=stats.pearsonr(eplmr[(year_lmr>=1900)&(year_lmr<=1999)],
                                      ephad[(year_had>=1900)&(year_had<=1999)])


# =============================================================================
# plot
# =============================================================================
clevs=np.linspace(-1.5, 1.5, 100)
fig=plt.figure(figsize=(8,6))
ax1=plt.subplot(3, 2, 1)
#plt.text(0.05, 0.95, 'a', transform=ax1.transAxes,fontsize=10, fontweight='bold', va='top')

m=Basemap(projection='cyl', llcrnrlon=120, llcrnrlat=-30, urcrnrlon=280, urcrnrlat=30)
m.drawcoastlines(linewidth=1)
m.fillcontinents(color='lightgray')
x, y = m(*np.meshgrid(lon_lmr[(lon_lmr<=280)&(lon_lmr>=120)], lat_lmr[(lat_lmr<=30)&(lat_lmr>=-30)]))
ax1 = m.contourf(x,y,sstacp_lmrmean,31,cmap=plt.cm.RdBu_r,levels=clevs,extend='both')
cbar = m.colorbar(ax1,shrink=0.8,ticks=[-1.5,-1,-0.5,0,0.5,1,1.5])
cbar.ax.tick_params(labelsize=7)
plt.title('PHYDA CP',fontsize=10)

ax2=plt.subplot(3, 2, 3)
#plt.text(0.05, 0.95, 'c', transform=ax2.transAxes,fontsize=10, fontweight='bold', va='top')

m=Basemap(projection='cyl', llcrnrlon=120, llcrnrlat=-30, urcrnrlon=280, urcrnrlat=30)
m.drawcoastlines(linewidth=1)
m.drawcountries(linewidth=1)
m.fillcontinents(color='lightgray')
x, y = m(*np.meshgrid(lon_had[(lon_had<=280)&(lon_had>=120)], lat_had[(lat_had<=30)&(lat_had>=-30)]))
ax2 = m.contourf(x,y,sstacp_hadmean,31,cmap=plt.cm.RdBu_r,levels=clevs,extend='both')
cbar = m.colorbar(ax2,shrink=0.8,ticks=[-1.5,-1,-0.5,0,0.5,1,1.5])
cbar.ax.tick_params(labelsize=7)
plt.title('HADISST CP',fontsize=10)


ax3=plt.subplot(3, 2, 2)
#plt.text(0.05, 0.95, 'b', transform=ax3.transAxes,fontsize=10, fontweight='bold', va='top')

m=Basemap(projection='cyl', llcrnrlon=120, llcrnrlat=-30, urcrnrlon=280, urcrnrlat=30)
m.drawcoastlines(linewidth=1)
m.drawcountries(linewidth=1)
m.fillcontinents(color='lightgray')
x, y = m(*np.meshgrid(lon_lmr[(lon_lmr<=280)&(lon_lmr>=120)], lat_lmr[(lat_lmr<=30)&(lat_lmr>=-30)]))
ax3 = m.contourf(x,y,sstaep_lmrmean,31,cmap=plt.cm.RdBu_r,levels=clevs,extend='both')
cbar = m.colorbar(ax3,shrink=0.8,ticks=[-1.5,-1,-0.5,0,0.5,1,1.5])
cbar.ax.tick_params(labelsize=7)
plt.title('PHYDA EP',fontsize=10)


ax4=plt.subplot(3, 2, 4)
#plt.text(0.05, 0.95, 'd', transform=ax4.transAxes,fontsize=10, fontweight='bold', va='top')

m=Basemap(projection='cyl', llcrnrlon=120, llcrnrlat=-30, urcrnrlon=280, urcrnrlat=30)
m.drawcoastlines(linewidth=1)
m.drawcountries(linewidth=1)
m.fillcontinents(color='lightgray')
x, y = m(*np.meshgrid(lon_had[(lon_had<=280)&(lon_had>=120)], lat_had[(lat_had<=30)&(lat_had>=-30)]))
ax4 = m.contourf(x,y,sstaep_hadmean,31,cmap=plt.cm.RdBu_r,levels=clevs,extend='both')
cbar = m.colorbar(ax4,shrink=0.8,ticks=[-1.5,-1,-0.5,0,0.5,1,1.5])
cbar.ax.tick_params(labelsize=7)
plt.title('HADISST EP',fontsize=10)


ax5=plt.subplot(3, 2, 5)
#plt.text(0.05, 0.95, 'e', transform=ax5.transAxes,fontsize=10, fontweight='bold', va='top')

plt.plot(year_had[(year_had>=1900)&(year_had<=1999)], cphad[(year_had>=1900)&(year_had<=1999)],
                  label='HADISST')
plt.plot(year_had[(year_had>=1900)&(year_had<=1999)], cplmr[(year_lmr>=1900)&(year_lmr<=1999)],
                  label='PHYDA (r='+str(round(rc,2))+' p<0.05)')#(p='+str(round(pc,3))+')')
plt.axhline(y=0,color='k',ls='--')
plt.ylim(-4,4)
plt.legend(loc='lower right',fontsize='xx-small')
plt.title('C index',fontsize=10)


ax6=plt.subplot(3, 2, 6)
#plt.text(0.05, 0.95, 'f', transform=ax6.transAxes,fontsize=10, fontweight='bold', va='top')

plt.plot(year_had[(year_had>=1900)&(year_had<=1999)], ephad[(year_had>=1900)&(year_had<=1999)],
                  label='HADISST')
plt.plot(year_had[(year_had>=1900)&(year_had<=1999)], eplmr[(year_lmr>=1900)&(year_lmr<=1999)],
                  label='PHYDA (r='+str(round(re,2))+' p<0.05)')#(p='+str(round(pe,3))+')')
plt.legend(loc='lower right',fontsize='xx-small')
plt.axhline(y=0,color='k',ls='--')
plt.ylim(-4,4)
plt.title('E index',fontsize=10)

plt.tight_layout()

#fig.savefig('/Users/lxy/Desktop/research/ENSO/cp_ep/validation/cpep/E_C/phyda.png',dpi=600)

# =============================================================================
# clevs=np.linspace(-3, 3, 40)
# 
# ax1=plt.subplot(1, 2, 1)
# m=Basemap(projection='cyl', llcrnrlon=120, llcrnrlat=-30, urcrnrlon=280, urcrnrlat=30)
# m.drawcoastlines(linewidth=1)
# m.drawcountries(linewidth=1)
# m.fillcontinents(color='lightgray')
# x, y = m(*np.meshgrid(lon_lmr[(lon_lmr<=280)&(lon_lmr>=120)], lat_lmr[(lat_lmr<=30)&(lat_lmr>=-30)]))
# ax1 = m.contourf(x,y,np.squeeze(ssta_lmr[year_lmr==1997,30:61,60:141]),31,cmap=plt.cm.RdBu_r,levels=clevs)
# cbar = m.colorbar(ax1,shrink=0.8,ticks=[-3,-2,-1,0,1,2,3])
# cbar.ax.tick_params(labelsize=7)
# #plt.title('LMR EP',fontsize=10)
# 
# ax2=plt.subplot(1, 2, 2)
# m=Basemap(projection='cyl', llcrnrlon=120, llcrnrlat=-30, urcrnrlon=280, urcrnrlat=30)
# m.drawcoastlines(linewidth=1)
# m.drawcountries(linewidth=1)
# m.fillcontinents(color='lightgray')
# x, y = m(*np.meshgrid(lon_had[(lon_had<=280)&(lon_had>=120)], lat_had[(lat_had<=30)&(lat_had>=-30)]))
# ax2 = m.contourf(x,y,np.squeeze(ssta_had[(year_had==1997),1:61,80:241]),31,cmap=plt.cm.RdBu_r,levels=clevs)
# cbar = m.colorbar(ax2,shrink=0.8,ticks=[-3,-2,-1,0,1,2,3])
# cbar.ax.tick_params(labelsize=7)
# #plt.title('LMR EP',fontsize=10)
# 
# =============================================================================

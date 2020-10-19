#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  7 19:38:14 2020

@author: lxy
"""

"""
for PHYDA
general NINO 3.4 index, SSTA patterns and PDSI pattern validation
(ENSO events)
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
    Nino34_std=(nino34-nino34.rolling(window=30).mean())/nino34.rolling(window=30).std()
    #Nino34_std=np.ma.anom(Nino34)
    return Nino34_std

def Nino34_had(SSTA,lat,lon):
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


# =============================================================================
# PHYDA
# =============================================================================
filepath=Dataset('/Users/lxy/Desktop/research/ENSO/data/PHYDA/da_hydro_DecFeb_r.1-2000_d.05-Jan-2018.nc')
SST=filepath.variables['tas_mn']
SST=np.ma.getdata(SST)
PDSI=filepath.variables['pdsi_mn']
PDSI=np.ma.getdata(PDSI)
lat=filepath.variables['lat']
lat_lmr=np.ma.getdata(lat)
lon=filepath.variables['lon']
lon_lmr=np.ma.getdata(lon)
year_lmr=np.arange(1,2001)

ssta_lmr=SSTA_cal1(SST)

# mask the continent part to nan
SSTA_LMR=ssta_lmr.copy()
SSTA_LMR[~np.ma.getmask(PDSI)]=np.nan


# =============================================================================
# HadISST
# =============================================================================
# =============================================================================
# filepath3=Dataset('/Users/lxy/Desktop/research/ENSO/data/HadiSST/HadISST_sst.nc')
# SST_HAD=filepath3.variables['sst']
# lat_had=filepath3.variables['latitude']
# lat_had=np.ma.getdata(lat_had)
# lonhad=filepath3.variables['longitude']
# lonhad=np.ma.getdata(lonhad)
# year_had=np.arange(1870,2019)
# 
# sst_had=np.zeros((149,180,360))
# 
# #annual mean (Apr-Mar)
# for i in range(0,149):
#     sst_had[i,:,:]=np.ma.average(SST_HAD[i*12+3:2+(i+1)*12,:,:],0)
#     
# ssthad,lon_had=shiftgrid(0.5,sst_had,lonhad,start=True)
# 
# #running removal of 30-year
# ssta_had=SSTA_cal1(ssthad)
# SSTA_HAD=maskanom(ssta_had)
# =============================================================================

# =============================================================================
# ERSSTv5
# =============================================================================
# =============================================================================
# filepath3=Dataset('/Users/lxy/Desktop/research/ENSO/data/ERSSTv5/sst.mnmean.nc')
# SST_HAD=filepath3.variables['sst']#start from 1854 Jan
# lat_had=filepath3.variables['lat']
# lat_had=np.ma.getdata(lat_had)
# lonhad=filepath3.variables['lon']
# lon_had=np.ma.getdata(lonhad)
# year_had=np.arange(1854,2019)
# 
# sst_had=np.zeros((165,89,180))
# 
# #annual mean (Apr-Mar)
# for i in range(0,165):
#     sst_had[i,:,:]=np.ma.average(SST_HAD[i*12+3:2+(i+1)*12,:,:],0)
#     
# #ssthad,lon_had=shiftgrid(0.5,sst_had,lonhad,start=True)
# 
# #running removal of 30-year
# ssta_had=SSTA_cal1(sst_had)
# SSTA_HAD=maskanom(ssta_had)
# =============================================================================

# =============================================================================
# HadSST4
# =============================================================================
filepath3=Dataset('/Users/lxy/Desktop/research/ENSO/data/HadSSTv4/HadSST.4.0.0.0_median.nc')
SST_HAD=filepath3.variables['tos']#start from 1850 Jan
lat_had=filepath3.variables['latitude']
lat_had=np.ma.getdata(lat_had)
lonhad=filepath3.variables['longitude']
lonhad=np.ma.getdata(lonhad)
year_had=np.arange(1850,2018)

sst_had=np.zeros((168,36,72))

#annual mean (Apr-Mar)
for i in range(0,168):
    sst_had[i,:,:]=np.ma.average(SST_HAD[i*12+3:2+(i+1)*12,:,:],0)
    
ssthad,lon_had=shiftgrid(2.5,sst_had,lonhad,start=True)

#running removal of 30-year
ssta_had=SSTA_cal1(ssthad)
SSTA_HAD=maskanom(ssta_had)


#find El Nino and La Nina 
nino34_std_lmr=Nino34_lmr(SSTA_LMR,lat_lmr,lon_lmr)
nino34_std_had=Nino34_had(SSTA_HAD,lat_had,lon_had)

#El Nino lmr&hadisst
sstael_lmr=ssta_lmr[nino34_std_lmr>1,:,:]
yrel_lmr=year_lmr[nino34_std_lmr>1]

sstael_had=ssta_had[nino34_std_had>1,:,:]
yrel_had=year_had[nino34_std_had>1]

#La Nina lmr&hadisst
sstala_lmr=ssta_lmr[nino34_std_lmr<-1,:,:]
yrla_lmr=year_lmr[nino34_std_lmr<-1]

sstala_had=ssta_had[nino34_std_had<-1,:,:]
yrla_had=year_had[nino34_std_had<-1]

#EL mean& LA mean ssta
sstael_lmrmean=np.mean(sstael_lmr[(yrel_lmr>=1900)&(yrel_lmr<=1999),:,:],0)
sstael_lmrmean=sstael_lmrmean[(lat_lmr<=30)&(lat_lmr>=-30),:]
sstael_lmrmean=sstael_lmrmean[:,(lon_lmr<=280)&(lon_lmr>=120)]
#sstael_lmrmean[sstael_lmrmean==0]=np.nan

sstael_hadmean=np.mean(sstael_had[(yrel_had>=1900)&(yrel_had<=1999),:,:],0)
sstael_hadmean=sstael_hadmean[(lat_had<=30)&(lat_had>=-30),:]
sstael_hadmean=sstael_hadmean[:,(lon_had<=280)&(lon_had>=120)]
#sstael_hadmean[sstael_hadmean==0]=np.nan

sstala_lmrmean=np.mean(sstala_lmr[(yrla_lmr>=1900)&(yrla_lmr<=1999),:,:],0)
sstala_lmrmean=sstala_lmrmean[(lat_lmr<=30)&(lat_lmr>=-30),:]
sstala_lmrmean=sstala_lmrmean[:,(lon_lmr<=280)&(lon_lmr>=120)]
#sstala_lmrmean[sstala_lmrmean==0]=np.nan

sstala_hadmean=np.mean(sstala_had[(yrla_had>=1900)&(yrla_had<=1999),:,:],0)
sstala_hadmean=sstala_hadmean[(lat_had<=30)&(lat_had>=-30),:]
sstala_hadmean=sstala_hadmean[:,(lon_had<=280)&(lon_had>=120)]
#sstala_hadmean[sstala_hadmean==0]=np.nan

# =============================================================================
# SIGNIFICANCE TEST
# =============================================================================

# =============================================================================
# t, p = stats.ttest_ind(nino34_std_lmr[(year_lmr>=1900)&(year_lmr<=1999)],
#                                       nino34_std_had[(year_had>=1900)&(year_had<=1999)],equal_var=False)
# =============================================================================
a,b=stats.pearsonr(nino34_std_lmr[(year_lmr>=1900)&(year_lmr<=1999)],
                                      nino34_std_had[(year_had>=1900)&(year_had<=1999)])
resi=nino34_std_lmr[(year_lmr>=1900)&(year_lmr<=1999)]-nino34_std_had[(year_had>=1900)&(year_had<=1999)]
#plt.plot(resi)



# =============================================================================
# regrid
# =============================================================================
X, Y = np.meshgrid(lon_had[(lon_had<=280)&(lon_had>=120)],lat_had[(lat_had<=30)&(lat_had>=-30)])
XI, YI = np.meshgrid(lon_lmr[(lon_lmr<=280)&(lon_lmr>=120)],lat_lmr[(lat_lmr<=30)&(lat_lmr>=-30)])

nwsstael_lmr=scipy.interpolate.griddata((XI.flatten(),YI.flatten()),
                                        sstael_lmrmean.flatten(), (X,Y), method='cubic')
nwsstala_lmr=scipy.interpolate.griddata((XI.flatten(),YI.flatten()),
                                        sstala_lmrmean.flatten(), (X,Y), method='cubic')

#spatial bias 
had_lmrel=sstael_hadmean-nwsstael_lmr
had_lmrla=sstala_hadmean-nwsstala_lmr

rmse_el=np.sqrt(np.ma.mean(np.ma.masked_invalid(had_lmrel**2)))
rmse_la=np.sqrt(np.ma.mean(np.ma.masked_invalid(had_lmrla**2)))


# =============================================================================
# plot EL&LA
# =============================================================================
import seaborn as sns
sns.set(style="ticks")

clevs=np.linspace(-2, 2, 200)
fig=plt.figure(figsize=(12,5))
ax1=plt.subplot2grid((6, 3), (0, 0), rowspan=2)
plt.text(0.05, 0.95, 'a', transform=ax1.transAxes,
      fontsize=10, fontweight='bold', va='top')
m=Basemap(projection='cyl', llcrnrlon=120, llcrnrlat=-30, urcrnrlon=280, urcrnrlat=30)
m.drawcoastlines(linewidth=1)
m.fillcontinents(color='lightgray')
x, y = m(*np.meshgrid(lon_lmr[(lon_lmr<=280)&(lon_lmr>=120)], lat_lmr[(lat_lmr<=30)&(lat_lmr>=-30)]))
ax1 = m.contourf(x,y,sstael_lmrmean,31,cmap=plt.cm.RdBu_r,levels=clevs,extend='both')
#cbar = m.colorbar(ax1,shrink=0.8,ticks=[-1.5,-1,-0.5,0,0.5,1,1.5])
#cbar.ax.tick_params(labelsize=5)
plt.title('PHYDA El Nino',fontsize=8)

ax2=plt.subplot2grid((6, 3), (0, 1), rowspan=2)
plt.text(0.05, 0.95, 'b', transform=ax2.transAxes,
      fontsize=10, fontweight='bold', va='top')

m=Basemap(projection='cyl', llcrnrlon=120, llcrnrlat=-30, urcrnrlon=280, urcrnrlat=30)
m.drawcoastlines(linewidth=1)
m.drawcountries(linewidth=1)
m.fillcontinents(color='lightgray')
x, y = m(*np.meshgrid(lon_had[(lon_had<=280)&(lon_had>=120)], lat_had[(lat_had<=30)&(lat_had>=-30)]))
ax2 = m.contourf(x,y,sstael_hadmean,31,cmap=plt.cm.RdBu_r,levels=clevs,extend='both')
#cbar = m.colorbar(ax2,shrink=0.8,ticks=[-1.5,-1,-0.5,0,0.5,1,1.5])
#cbar.ax.tick_params(labelsize=5)
plt.title('HadSST4 El Nino',fontsize=8)


ax3=plt.subplot2grid((6, 3), (0, 2), rowspan=2)
plt.text(0.05, 0.95, 'c', transform=ax3.transAxes,
      fontsize=10, fontweight='bold', va='top')

m=Basemap(projection='cyl', llcrnrlon=120, llcrnrlat=-30, urcrnrlon=280, urcrnrlat=30)
m.drawcoastlines(linewidth=1)
m.drawcountries(linewidth=1)
m.fillcontinents(color='lightgray')
x, y = m(*np.meshgrid(lon_had[(lon_had<=280)&(lon_had>=120)], lat_had[(lat_had<=30)&(lat_had>=-30)]))
ax3 = m.contourf(x,y,had_lmrel,31,cmap=plt.cm.RdBu_r,levels=clevs,extend='both')
cbar = m.colorbar(ax3,shrink=0.8,ticks=[-1.5,-1,-0.5,0,0.5,1,1.5])
cbar.ax.tick_params(labelsize=5)
plt.title('HadSST4-PHYDA El Nino',fontsize=8)



ax4=plt.subplot2grid((6, 3), (2, 0), rowspan=2)
plt.text(0.05, 0.95, 'd', transform=ax4.transAxes,
      fontsize=10, fontweight='bold', va='top')

m=Basemap(projection='cyl', llcrnrlon=120, llcrnrlat=-30, urcrnrlon=280, urcrnrlat=30)
m.drawcoastlines(linewidth=1)
m.drawcountries(linewidth=1)
m.fillcontinents(color='lightgray')
x, y = m(*np.meshgrid(lon_lmr[(lon_lmr<=280)&(lon_lmr>=120)], lat_lmr[(lat_lmr<=30)&(lat_lmr>=-30)]))
ax4 = m.contourf(x,y,sstala_lmrmean,31,cmap=plt.cm.RdBu_r,levels=clevs,extend='both')
#cbar = m.colorbar(ax4,shrink=0.8,ticks=[-1.5,-1,-0.5,0,0.5,1,1.5])
#cbar.ax.tick_params(labelsize=5)
plt.title('PHYDA La Nina',fontsize=8)


ax5=plt.subplot2grid((6, 3), (2, 1), rowspan=2)
plt.text(0.05, 0.95, 'e', transform=ax5.transAxes,
      fontsize=10, fontweight='bold', va='top')

m=Basemap(projection='cyl', llcrnrlon=120, llcrnrlat=-30, urcrnrlon=280, urcrnrlat=30)
m.drawcoastlines(linewidth=1)
m.drawcountries(linewidth=1)
m.fillcontinents(color='lightgray')
x, y = m(*np.meshgrid(lon_had[(lon_had<=280)&(lon_had>=120)], lat_had[(lat_had<=30)&(lat_had>=-30)]))
ax5 = m.contourf(x,y,sstala_hadmean,31,cmap=plt.cm.RdBu_r,levels=clevs,extend='both')
#cbar = m.colorbar(ax5,shrink=0.8,ticks=[-1.5,-1,-0.5,0,0.5,1,1.5])
#cbar.ax.tick_params(labelsize=5)
plt.title('HadSST4 La Nina',fontsize=8)


ax6=plt.subplot2grid((6, 3), (2, 2), rowspan=2)
plt.text(0.05, 0.95, 'f', transform=ax6.transAxes,
      fontsize=10, fontweight='bold', va='top')

m=Basemap(projection='cyl', llcrnrlon=120, llcrnrlat=-30, urcrnrlon=280, urcrnrlat=30)
m.drawcoastlines(linewidth=1)
m.drawcountries(linewidth=1)
m.fillcontinents(color='lightgray')
x, y = m(*np.meshgrid(lon_had[(lon_had<=280)&(lon_had>=120)], lat_had[(lat_had<=30)&(lat_had>=-30)]))
ax6 = m.contourf(x,y,had_lmrla,31,cmap=plt.cm.RdBu_r,levels=clevs,extend='both')
cbar = m.colorbar(ax6,shrink=0.8,ticks=[-1.5,-1,-0.5,0,0.5,1,1.5])
cbar.ax.tick_params(labelsize=5)
plt.title('HadSST4-PHYDA La Nina',fontsize=8)


ax7=plt.subplot2grid((6, 6), (4, 1), colspan=4,rowspan=2)
plt.text(0.02, 0.95, 'g',transform=ax7.transAxes,
         fontsize=10, fontweight='bold', va='top')
plt.plot(year_had[(year_had>=1900)&(year_had<=1999)], 
                  nino34_std_lmr[(year_lmr>=1900)&(year_lmr<=1999)],
                  label='PHYDA (r='+str(round(a,2))+' p<0.005)')
plt.plot(year_had[(year_had>=1900)&(year_had<=1999)], 
                  nino34_std_had[(year_had>=1900)&(year_had<=1999)],label='HadSST4')
plt.ylim(-3,3)
plt.legend(loc='lower right',fontsize='xx-small',ncol=2)
plt.axhline(y=0,color='k',ls='--')
plt.title('Nino3.4 index',fontsize=8)


plt.tight_layout()

#fig.savefig('/Users/lxy/Desktop/research/ENSO/cp_ep/validation/ENSO/HadSST4/El&Laphy.png',dpi=600)

'''
validation on PDSI
LMR & NADA
NADA's PDSI composite based on HadISST ENSO year
'''
#PHYDA
PDSIA_LMR=SSTA_cal1(PDSI)


#NADA
filepath4=Dataset('/Users/lxy/Desktop/research/ENSO/data/nada_hd2_cl.nc')
pdsi_nada=filepath4.variables['pdsi']#(lon,lat,time)
lat_nada=filepath4.variables['lat']
lat_nada=np.ma.getdata(lat_nada)
lon_nada=filepath4.variables['lon']
lon_nada=np.ma.getdata(lon_nada)
lon_nada=lon_nada+360
yr_nada=filepath4.variables['time']
yr_nada=np.ma.getdata(yr_nada)#0-2005
#year_nada=yr_nada[(yr_nada>=np.min(year_lmr))&(yr_nada<=np.max(year_lmr))]
year_nada=yr_nada[(yr_nada>=np.min(year_had))]


# =============================================================================
# pdsi_had=pdsi_nada[:,:,(yr_nada>=np.min(year_lmr))&(yr_nada<=np.max(year_lmr))]
# PDSIA_nada=SSTA_cal2(pdsi_had)
# =============================================================================


#match the year flag with HadISST
pdsi_had=pdsi_nada[:,:,(yr_nada<=np.max(year_had))&(yr_nada>=np.min(year_had))]
PDSIA_nada=SSTA_cal2(pdsi_had)



#match Nino3.4 index of HadISST with PDSIA_nada
nino34_std_nada=nino34_std_had[year_had<=np.max(yr_nada)]

#extract all ENSO events
pdsiel_lmr=PDSIA_LMR[nino34_std_lmr>1,:,:]
pdsila_lmr=PDSIA_LMR[nino34_std_lmr<-1,:,:]


pdsiel_nada=PDSIA_nada[nino34_std_nada>1,:,:]
yrel_nada=year_nada[nino34_std_nada>1]
pdsila_nada=PDSIA_nada[nino34_std_nada<-1,:,:]
yrla_nada=year_nada[nino34_std_nada<-1]



#lat lon range
lat1=np.min(lat_nada)
lat2=65
lon1=190
lon2=295


#EL mean& LA mean ssta
pdsiel_lmrmean=np.mean(pdsiel_lmr[(yrel_lmr>=1900)&(yrel_lmr<=1999),:,:],0)
pdsiel_lmrmean=pdsiel_lmrmean[(lat_lmr<=lat2)&(lat_lmr>=lat1),:]
pdsiel_lmrmean=pdsiel_lmrmean[:,(lon_lmr<=lon2)&(lon_lmr>=lon1)]
#sstael_lmrmean[sstael_lmrmean==np.nan]=0

pdsila_lmrmean=np.mean(pdsila_lmr[(yrla_lmr>=1900)&(yrla_lmr<=1999),:,:],0)
pdsila_lmrmean=pdsila_lmrmean[(lat_lmr<=lat2)&(lat_lmr>=lat1),:]
pdsila_lmrmean=pdsila_lmrmean[:,(lon_lmr<=lon2)&(lon_lmr>=lon1)]
#sstala_lmrmean[sstala_lmrmean==np.nan]=0

pdsiel_nadamean=np.mean(pdsiel_nada[(yrel_nada>=1900)&(yrel_nada<=1999),:,:],0)
pdsiel_nadamean[np.isnan(pdsiel_nadamean)] = 0
pdsila_nadamean=np.mean(pdsila_nada[(yrla_nada>=1900)&(yrla_nada<=1999),:,:],0)
pdsila_nadamean[np.isnan(pdsila_nadamean)]=0

# =============================================================================
# regrid
# =============================================================================
X_nada, Y_nada = np.meshgrid(lon_nada,lat_nada)
X_lmr, Y_lmr = np.meshgrid(lon_lmr[(lon_lmr<=lon2)&(lon_lmr>=lon1)],
                             lat_lmr[(lat_lmr<=lat2)&(lat_lmr>=lat1)])

nwpdsiel_nada=scipy.interpolate.griddata((X_nada.flatten(),Y_nada.flatten()),
                                        pdsiel_nadamean.flatten(), (X_lmr,Y_lmr), method='cubic')
nwpdsila_nada=scipy.interpolate.griddata((X_nada.flatten(),Y_nada.flatten()),
                                        pdsila_nadamean.flatten(), (X_lmr,Y_lmr), method='cubic')

#spatial bias 
nada_lmrel=nwpdsiel_nada-pdsiel_lmrmean
nada_lmrla=nwpdsila_nada-pdsila_lmrmean

re_el=np.sqrt(np.ma.mean(np.ma.masked_invalid(nada_lmrel**2)))
re_la=np.sqrt(np.ma.mean(np.ma.masked_invalid(nada_lmrla**2)))

pdsiel_nadamean[pdsiel_nadamean==0] = np.nan
pdsila_nadamean[pdsila_nadamean==0]=np.nan

# =============================================================================
# plot EL&LA pdsi validation
# =============================================================================
clevs=np.linspace(-3, 3, 200)
fig=plt.figure(figsize=(9,4))
ax1=plt.subplot2grid((4, 3), (0, 0), rowspan=2)
plt.text(0.02, 0.9, 'a', transform=ax1.transAxes,
      fontsize=10, fontweight='bold', va='top')
m=Basemap(projection='cyl', llcrnrlon=lon1, llcrnrlat=lat1,urcrnrlon=lon2, urcrnrlat=lat2)
m.drawcoastlines(linewidth=1)
#m.fillcontinents(color='lightgray')
x, y = m(*np.meshgrid(lon_lmr[(lon_lmr<=lon2)&(lon_lmr>=lon1)],lat_lmr[(lat_lmr<=lat2)&(lat_lmr>=lat1)]))
ax1 = m.contourf(x,y,pdsiel_lmrmean,31,cmap=plt.cm.BrBG,levels=clevs)
#cbar = m.colorbar(ax1,shrink=0.8,ticks=[-1.5,-1,-0.5,0,0.5,1,1.5])
#cbar.ax.tick_params(labelsize=5)
plt.title('PHYDA El Nino',fontsize=8)

ax2=plt.subplot2grid((4, 3), (0, 1), rowspan=2)
plt.text(0.02, 0.9, 'b', transform=ax2.transAxes,
      fontsize=10, fontweight='bold', va='top')

m=Basemap(projection='cyl', llcrnrlon=lon1, llcrnrlat=lat1,urcrnrlon=lon2, urcrnrlat=lat2)
m.drawcoastlines(linewidth=1)
#m.drawcountries(linewidth=1)
#m.fillcontinents(color='lightgray')
x, y = m(*np.meshgrid(lon_nada, lat_nada))
ax2 = m.contourf(x,y,pdsiel_nadamean,31,cmap=plt.cm.BrBG,levels=clevs)
#cbar = m.colorbar(ax2,shrink=0.8,ticks=[-1.5,-1,-0.5,0,0.5,1,1.5])
#cbar.ax.tick_params(labelsize=5)
plt.title('NADA El Nino',fontsize=8)


ax3=plt.subplot2grid((4, 3), (0, 2), rowspan=2)
plt.text(0.02, 0.9, 'c', transform=ax3.transAxes,
      fontsize=10, fontweight='bold', va='top')

m=Basemap(projection='cyl', llcrnrlon=lon1, llcrnrlat=lat1,urcrnrlon=lon2, urcrnrlat=lat2)
m.drawcoastlines(linewidth=1)
#m.drawcountries(linewidth=1)
#m.fillcontinents(color='lightgray')
x, y = m(*np.meshgrid(lon_lmr[(lon_lmr<=lon2)&(lon_lmr>=lon1)],lat_lmr[(lat_lmr<=lat2)&(lat_lmr>=lat1)]))
ax3 = m.contourf(x,y,nada_lmrel,31,cmap=plt.cm.BrBG,levels=clevs)
cbar = m.colorbar(ax3,shrink=0.8,ticks=[-3,-2,-1,0,1,2,3])
cbar.ax.tick_params(labelsize=5)
plt.title('NADA-PHYDA El Nino',fontsize=8)



ax4=plt.subplot2grid((4, 3), (2, 0), rowspan=2)
plt.text(0.02, 0.9, 'd', transform=ax4.transAxes,
      fontsize=10, fontweight='bold', va='top')

m=Basemap(projection='cyl', llcrnrlon=lon1, llcrnrlat=lat1,urcrnrlon=lon2, urcrnrlat=lat2)
m.drawcoastlines(linewidth=1)
#m.drawcountries(linewidth=1)
#m.fillcontinents(color='lightgray')
x, y = m(*np.meshgrid(lon_lmr[(lon_lmr<=lon2)&(lon_lmr>=lon1)],lat_lmr[(lat_lmr<=lat2)&(lat_lmr>=lat1)]))
ax4 = m.contourf(x,y,pdsila_lmrmean,31,cmap=plt.cm.BrBG,levels=clevs)
#cbar = m.colorbar(ax4,shrink=0.8,ticks=[-1.5,-1,-0.5,0,0.5,1,1.5])
#cbar.ax.tick_params(labelsize=5)
plt.title('PHYDA La Nina',fontsize=8)


ax5=plt.subplot2grid((4, 3), (2, 1), rowspan=2)
plt.text(0.02, 0.9, 'e', transform=ax5.transAxes,
      fontsize=10, fontweight='bold', va='top')

m=Basemap(projection='cyl', llcrnrlon=lon1, llcrnrlat=lat1,urcrnrlon=lon2, urcrnrlat=lat2)
m.drawcoastlines(linewidth=1)
#m.drawcountries(linewidth=1)
#m.fillcontinents(color='lightgray')
x, y = m(*np.meshgrid(lon_nada, lat_nada))
ax5 = m.contourf(x,y,pdsila_nadamean,31,cmap=plt.cm.BrBG,levels=clevs)
#cbar = m.colorbar(ax5,shrink=0.8,ticks=[-1.5,-1,-0.5,0,0.5,1,1.5])
#cbar.ax.tick_params(labelsize=5)
plt.title('NADA La Nina',fontsize=8)


ax6=plt.subplot2grid((4, 3), (2, 2), rowspan=2)
plt.text(0.02, 0.9, 'f', transform=ax6.transAxes,
      fontsize=10, fontweight='bold', va='top')

m=Basemap(projection='cyl', llcrnrlon=lon1, llcrnrlat=lat1,urcrnrlon=lon2, urcrnrlat=lat2)
m.drawcoastlines(linewidth=1)
#m.drawcountries(linewidth=1)
#m.fillcontinents(color='lightgray')
x, y = m(*np.meshgrid(lon_lmr[(lon_lmr<=lon2)&(lon_lmr>=lon1)],lat_lmr[(lat_lmr<=lat2)&(lat_lmr>=lat1)]))
ax6 = m.contourf(x,y,nada_lmrla,31,cmap=plt.cm.BrBG,levels=clevs)
cbar = m.colorbar(ax3,shrink=0.8,ticks=[-3,-2,-1,0,1,2,3])
cbar.ax.tick_params(labelsize=5)
plt.title('NADA-PHYDA La Nina',fontsize=8)


plt.tight_layout()

#fig.savefig('/Users/lxy/Desktop/research/ENSO/cp_ep/validation/ENSO/newpdsi_phy.png',dpi=600)


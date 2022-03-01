#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  4 21:07:19 2020

@author: lxy
"""

import numpy as np
import pandas as pd
from netCDF4 import Dataset
import seaborn as sns

#import os
import scipy.io as sio
from scipy import stats, signal
from eofs.standard import Eof
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from mpl_toolkits.basemap import Basemap, shiftgrid

#for LMR
def SSTA_cal1(SST):
    '''
    detrend then get the anomaly
    SST with (time,lat,lon)
    '''
    a=np.shape(SST)
    ssta=np.zeros(a);
    for i in range(0,a[1]):
        for j in range(0,a[2]):
            sst=signal.detrend(SST[:,i,j])
            ssta[:,i,j]=sst-np.mean(sst)
    return ssta

##for NADA
#def SSTA_cal2(SST):
#    '''
#    input with (lon,lat,time)
#    output with (time, lat, lon)
#    '''
#    a=np.shape(SST)
#    ssta=np.zeros((a[2],a[1],a[0]));
#    for i in range(0,a[1]):
#        for j in range(0,a[0]):
#            sst=pd.Series(SST[j,i,:])
#            ssta[:,i,j]=sst-sst.rolling(window=30).mean()
#    return ssta

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
    solver=Eof(SSTA_TP)
    pcs=solver.pcs(npcs=2,pcscaling=1)
    eof=solver.eofsAsCorrelation(neofs=2)
    a=eof[0,(lat1<=5)&(lat1>=-5),:]
    b=eof[1,(lat1<=5)&(lat1>=-5),:]

    if np.mean(a[:,(lon1<=240)&(lon1>=190)],(0,1))<0:
        pcs[:,0]=-pcs[:,0]
        
    if np.mean(b[:,(lon1<=240)&(lon1>=190)],(0,1))<0:
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
# HadISST
# =============================================================================
filepath3=Dataset('/Users/lxy/Desktop/research/ENSO/data/HadiSST/HadISST_sst.nc')
SST_HAD=filepath3.variables['sst']
lat_had=filepath3.variables['latitude']
lat_had=np.ma.getdata(lat_had)
lonhad=filepath3.variables['longitude']
lonhad=np.ma.getdata(lonhad)
year=np.arange(1870,2020)
year_had=year[(year>=1948) & (year<=2008)]#PDSI data starts in year 1948

sst_had=np.zeros((150,180,360))

#annual mean (Jan-Dec)
for i in range(0,150):
    sst_had[i,:,:]=np.ma.average(SST_HAD[i*12:11+i*12,:,:],0)
    
ssthad,lon_had=shiftgrid(0.5,sst_had,lonhad,start=True)

#anomaly after detrend
ssta_had=SSTA_cal1(ssthad[(year>=1948)&(year<=2008),:,:])
SSTA_HAD=maskanom(ssta_had)

#C&E method
Cin_had,Ein_had=E_Cindex(SSTA_HAD,lat_had,lon_had)
Cinhad_std=(Cin_had-np.mean(Cin_had))/np.std(Cin_had)
Einhad_std=(Ein_had-np.mean(Ein_had))/np.std(Ein_had)


#CP&EP method
CPin_had,EPin_had=EP_CPindex(SSTA_HAD,lat_had,lon_had)
CPinhad_std=(CPin_had-np.mean(CPin_had))/np.std(CPin_had)
EPinhad_std=(EPin_had-np.mean(EPin_had))/np.std(EPin_had)

#Nino 3-4 method
cpindex_had,epindex_had=Nino3_4index(SSTA_HAD,lat_had,lon_had)
cphad=(cpindex_had-np.mean(cpindex_had))/np.std(cpindex_had)
ephad=(epindex_had-np.mean(epindex_had))/np.std(epindex_had)


filepath3=Dataset('/Users/lxy/Desktop/research/ENSO/data/PDSI/pdsi_pm_prec_cruV3.10_rnet_pgf_monthly_1948-2008.nc')
PDSI=filepath3.variables['pdsi_pm']
PDSI=np.ma.getdata(PDSI)
lat=filepath3.variables['latitude']
lat=np.ma.getdata(lat)
lon=filepath3.variables['longitude']
lon=np.ma.getdata(lon)
yr=np.arange(1948,2009)

PDSI=np.squeeze(PDSI, axis=1)
pdsi=np.zeros((61,180,360))

for i in range(0,61):
    pdsi[i,:,:]=np.ma.average(PDSI[i*12:11+i*12,:,:],0)

pdsia=SSTA_cal1(pdsi)

# =============================================================================
# C and E method
# =============================================================================
#CP
sstaC=np.mean(ssta_had[(Cinhad_std>1) & (Cinhad_std>Einhad_std) &(Cinhad_std+Einhad_std>0),:,:],0)
sstaC[sstaC==0]=np.nan

pdsiaC=np.mean(pdsia[(Cinhad_std>1) & (Cinhad_std>Einhad_std) &(Cinhad_std+Einhad_std>0),:,:],0)
pdsiaC[pdsiaC==0]=np.nan

yrC=yr[(Cinhad_std>1) & (Cinhad_std>Einhad_std) &(Cinhad_std+Einhad_std>0)]

#EP
sstaE=np.mean(ssta_had[(Einhad_std>1) & (Einhad_std>Cinhad_std) &(Cinhad_std+Einhad_std>0),:,:],0)
sstaE[sstaE==0]=np.nan

pdsiaE=np.mean(pdsia[(Einhad_std>1) & (Einhad_std>Cinhad_std) &(Cinhad_std+Einhad_std>0),:,:],0)
pdsiaE[pdsiaE==0]=np.nan

yrE=yr[(Einhad_std>1) & (Einhad_std>Cinhad_std) &(Cinhad_std+Einhad_std>0)]

# =============================================================================
# CP and EP method
# =============================================================================
#CP
sstaCP=np.mean(ssta_had[(CPinhad_std>1) & (CPinhad_std>EPinhad_std) &(CPinhad_std+EPinhad_std>0),:,:],0)
sstaCP[sstaCP==0]=np.nan

pdsiaCP=np.mean(pdsia[(CPinhad_std>1) & (CPinhad_std>EPinhad_std) &(CPinhad_std+EPinhad_std>0),:,:],0)
pdsiaCP[pdsiaCP==0]=np.nan

yrCP=yr[(CPinhad_std>1) & (CPinhad_std>EPinhad_std) &(CPinhad_std+EPinhad_std>0)]

#EP
sstaEP=np.mean(ssta_had[(EPinhad_std>1) & (EPinhad_std>CPinhad_std) &(CPinhad_std+EPinhad_std>0),:,:],0)
sstaEP[sstaEP==0]=np.nan

pdsiaEP=np.mean(pdsia[(EPinhad_std>1) & (EPinhad_std>CPinhad_std) &(CPinhad_std+EPinhad_std>0),:,:],0)
pdsiaEP[pdsiaEP==0]=np.nan

yrEP=yr[(EPinhad_std>1) & (EPinhad_std>CPinhad_std) &(CPinhad_std+EPinhad_std>0)]

# =============================================================================
# nino 3-4 method
# =============================================================================
#CP
sstacp_had=np.mean(ssta_had[(cphad>1) & (cphad>ephad) & (cphad+ephad>0),:,:],0)
sstacp_had[sstacp_had==0]=np.nan

pdsiacp_had=np.mean(pdsia[(cphad>1) & (cphad>ephad) & (cphad+ephad>0),:,:],0)
pdsiacp_had[pdsiacp_had==0]=np.nan

yrcp_had=year_had[(cphad>1) & (cphad>ephad) & (cphad+ephad>0)]

#EP
sstaep_had=np.mean(ssta_had[(ephad>1) & (ephad>cphad) & (cphad+ephad>0),:,:],0)
sstaep_had[sstaep_had==0]=np.nan

pdsiaep_had=np.mean(pdsia[(ephad>1) & (ephad>cphad) & (cphad+ephad>0),:,:],0)
pdsiaep_had[pdsiaep_had==0]=np.nan

yrep_had=yr[(ephad>1) & (ephad>cphad) & (cphad+ephad>0)]


lat1=-30
lat2=65
lon1=120
lon2=295

# =============================================================================
# plot
# =============================================================================
# sns.set(style="ticks")

clevs1=np.linspace(-2, 2, 400)
clevs2=np.linspace(-3, 3, 400)

cl1='magenta'
cl2='black'
cl3='blue'

fig=plt.figure(figsize=(9,11))

ax=plt.subplot2grid((7, 2), (0, 0), rowspan=2)

m=Basemap(projection='cyl', llcrnrlon=lon1, llcrnrlat=lat1, urcrnrlon=lon2, urcrnrlat=lat2)
m.drawcoastlines(linewidth=1)
#m.drawcountries(linewidth=1)
#m.fillcontinents(color='lightgray')
m.drawmeridians(np.arange(120,295,50),labels=[0,0,0,1],color='DimGray',fontsize=10)
m.drawparallels(np.arange(-25,65,25),labels=[1,0,0,0],color='DimGray',fontsize=10)
x2, y2 = m(*np.meshgrid(lon, lat))
ax10 = m.contourf(x2,y2,pdsiacp_had,31,cmap=plt.cm.BrBG,levels=clevs2,extend='both')

x1, y1 = m(*np.meshgrid(lon_had, lat_had))
ax9= m.contourf(x1,y1,sstacp_had,31,cmap=plt.cm.RdBu_r,levels=clevs1,extend='both')
plt.text(0.05, 0.95, 'a', transform=ax.transAxes,
      fontsize=14, fontweight='bold', va='top')
# =============================================================================
# add regions
# =============================================================================
# SW= 30-43N, 245E-253e (Herweijer et al 2007)

x1,y1 = m(245,30)
x2,y2 = m(245,43)
x3,y3 = m(253,43)
x4,y4 = m(253,30)
poly1 = Polygon([(x1,y1),(x2,y2),(x3,y3),(x4,y4)],
                facecolor='None',edgecolor = cl1,linewidth=2.5)
plt.gca().add_patch(poly1)

# NW= 42-50N, 235-245 

x1,y1 = m(234,43)
x2,y2 = m(234,51)
x3,y3 = m(245,51)
x4,y4 = m(245,43)
poly2 = Polygon([(x1,y1),(x2,y2),(x3,y3),(x4,y4)],
                facecolor='None',edgecolor = cl2,linewidth=2.5)
plt.gca().add_patch(poly2)

# NE-central= 30-50N, 260-290 

x1,y1 = m(260,30)
x2,y2 = m(260,50)
x3,y3 = m(290,50)
x4,y4 = m(290,30)
poly3 = Polygon([(x1,y1),(x2,y2),(x3,y3),(x4,y4)],
                facecolor='None',edgecolor = cl3,linewidth=2.5)
plt.gca().add_patch(poly3)


plt.title('(a) CP El Nino (Nino 3-4)',fontsize=16,fontweight='bold')


ax=plt.subplot2grid((7, 2), (0, 1), rowspan=2)

m=Basemap(projection='cyl', llcrnrlon=lon1, llcrnrlat=lat1, urcrnrlon=lon2, urcrnrlat=lat2)
m.drawcoastlines(linewidth=1)
#m.drawcountries(linewidth=1)
#m.fillcontinents(color='lightgray')
m.drawmeridians(np.arange(120,295,50),labels=[0,0,0,1],color='DimGray',fontsize=10)
m.drawparallels(np.arange(-25,65,25),labels=[1,0,0,0],color='DimGray',fontsize=10)
x2, y2 = m(*np.meshgrid(lon, lat))
ax12 = m.contourf(x2,y2,pdsiaep_had,31,cmap=plt.cm.BrBG,levels=clevs2,extend='both')

x1, y1 = m(*np.meshgrid(lon_had, lat_had))
ax11 = m.contourf(x1,y1,sstaep_had,31,cmap=plt.cm.RdBu_r,levels=clevs1,extend='both')

plt.text(0.05, 0.95, 'b', transform=ax.transAxes,
      fontsize=14, fontweight='bold', va='top')
# =============================================================================
# add regions
# =============================================================================
# SW= 30-43N, 245E-253e (Herweijer et al 2007)

x1,y1 = m(245,30)
x2,y2 = m(245,43)
x3,y3 = m(253,43)
x4,y4 = m(253,30)
poly1 = Polygon([(x1,y1),(x2,y2),(x3,y3),(x4,y4)],
                facecolor='None',edgecolor = cl1,linewidth=2.5)
plt.gca().add_patch(poly1)

# NW= 42-50N, 235-245 

x1,y1 = m(234,43)
x2,y2 = m(234,51)
x3,y3 = m(245,51)
x4,y4 = m(245,43)
poly2 = Polygon([(x1,y1),(x2,y2),(x3,y3),(x4,y4)],
                facecolor='None',edgecolor = cl2,linewidth=2.5)
plt.gca().add_patch(poly2)

# NE-central= 30-50N, 260-290 

x1,y1 = m(260,30)
x2,y2 = m(260,50)
x3,y3 = m(290,50)
x4,y4 = m(290,30)
poly3 = Polygon([(x1,y1),(x2,y2),(x3,y3),(x4,y4)],
                facecolor='None',edgecolor = cl3,linewidth=2.5)
plt.gca().add_patch(poly3)

plt.title('(b) EP El Nino (Nino 3-4)',fontsize=16,fontweight='bold')


ax=plt.subplot2grid((7, 2), (2, 0), rowspan=2)

m=Basemap(projection='cyl', llcrnrlon=lon1, llcrnrlat=lat1, urcrnrlon=lon2, urcrnrlat=lat2)
m.drawcoastlines(linewidth=1)
#m.drawcountries(linewidth=1)
#m.fillcontinents(color='lightgray')
m.drawmeridians(np.arange(120,295,50),labels=[0,0,0,1],color='DimGray',fontsize=10)
m.drawparallels(np.arange(-25,65,25),labels=[1,0,0,0],color='DimGray',fontsize=10)
x2, y2 = m(*np.meshgrid(lon, lat))
ax6 = m.contourf(x2,y2,pdsiaCP,31,cmap=plt.cm.BrBG,levels=clevs2,extend='both')

x1, y1 = m(*np.meshgrid(lon_had, lat_had))
ax5 = m.contourf(x1,y1,sstaCP,31,cmap=plt.cm.RdBu_r,levels=clevs1,extend='both')
plt.text(0.05, 0.95, 'c', transform=ax.transAxes,
      fontsize=14, fontweight='bold', va='top')
# =============================================================================
# add regions
# =============================================================================
# SW= 30-43N, 245E-253e (Herweijer et al 2007)

x1,y1 = m(245,30)
x2,y2 = m(245,43)
x3,y3 = m(253,43)
x4,y4 = m(253,30)
poly1 = Polygon([(x1,y1),(x2,y2),(x3,y3),(x4,y4)],
                facecolor='None',edgecolor = cl1,linewidth=2.5)
plt.gca().add_patch(poly1)

# NW= 42-50N, 235-245 

x1,y1 = m(234,43)
x2,y2 = m(234,51)
x3,y3 = m(245,51)
x4,y4 = m(245,43)
poly2 = Polygon([(x1,y1),(x2,y2),(x3,y3),(x4,y4)],
                facecolor='None',edgecolor = cl2,linewidth=2.5)
plt.gca().add_patch(poly2)

# NE-central= 30-50N, 260-290 

x1,y1 = m(260,30)
x2,y2 = m(260,50)
x3,y3 = m(290,50)
x4,y4 = m(290,30)
poly3 = Polygon([(x1,y1),(x2,y2),(x3,y3),(x4,y4)],
                facecolor='None',edgecolor = cl3,linewidth=2.5)
plt.gca().add_patch(poly3)

plt.title('(c) CP El Nino (CP-EP)',fontsize=16,fontweight='bold')


ax=plt.subplot2grid((7, 2), (2, 1), rowspan=2)

m=Basemap(projection='cyl', llcrnrlon=lon1, llcrnrlat=lat1, urcrnrlon=lon2, urcrnrlat=lat2)
m.drawcoastlines(linewidth=1)
#m.drawcountries(linewidth=1)
#m.fillcontinents(color='lightgray')
m.drawmeridians(np.arange(120,295,50),labels=[0,0,0,1],color='DimGray',fontsize=10)
m.drawparallels(np.arange(-25,65,25),labels=[1,0,0,0],color='DimGray',fontsize=10)
x2, y2 = m(*np.meshgrid(lon, lat))
ax8 = m.contourf(x2,y2,pdsiaEP,31,cmap=plt.cm.BrBG,levels=clevs2,extend='both')

x1, y1 = m(*np.meshgrid(lon_had, lat_had))
ax7 = m.contourf(x1,y1,sstaEP,31,cmap=plt.cm.RdBu_r,levels=clevs1,extend='both')
plt.text(0.05, 0.95, 'd', transform=ax.transAxes,
      fontsize=14, fontweight='bold', va='top')
# =============================================================================
# add regions
# =============================================================================
# SW= 30-43N, 245E-253e (Herweijer et al 2007)

x1,y1 = m(245,30)
x2,y2 = m(245,43)
x3,y3 = m(253,43)
x4,y4 = m(253,30)
poly1 = Polygon([(x1,y1),(x2,y2),(x3,y3),(x4,y4)],
                facecolor='None',edgecolor = cl1,linewidth=2.5)
plt.gca().add_patch(poly1)

# NW= 42-50N, 235-245 

x1,y1 = m(234,43)
x2,y2 = m(234,51)
x3,y3 = m(245,51)
x4,y4 = m(245,43)
poly2 = Polygon([(x1,y1),(x2,y2),(x3,y3),(x4,y4)],
                facecolor='None',edgecolor = cl2,linewidth=2.5)
plt.gca().add_patch(poly2)

# NE-central= 30-50N, 260-290 

x1,y1 = m(260,30)
x2,y2 = m(260,50)
x3,y3 = m(290,50)
x4,y4 = m(290,30)
poly3 = Polygon([(x1,y1),(x2,y2),(x3,y3),(x4,y4)],
                facecolor='None',edgecolor = cl3,linewidth=2.5)
plt.gca().add_patch(poly3)

plt.title('(d) EP El Nino (CP-EP)',fontsize=16,fontweight='bold')



ax=plt.subplot2grid((7, 2), (4, 0), rowspan=2)
m=Basemap(projection='cyl', llcrnrlon=lon1, llcrnrlat=lat1, urcrnrlon=lon2, urcrnrlat=lat2)
m.drawcoastlines(linewidth=1)
#m.fillcontinents(color='lightgray',zorder=0)
m.drawmeridians(np.arange(120,295,50),labels=[0,0,0,1],color='DimGray',fontsize=10)
m.drawparallels(np.arange(-25,65,25),labels=[1,0,0,0],color='DimGray',fontsize=10)
x2, y2 = m(*np.meshgrid(lon, lat))
ax2 = m.contourf(x2,y2,pdsiaC,31,cmap=plt.cm.BrBG,levels=clevs2,extend='both')

x1, y1 = m(*np.meshgrid(lon_had, lat_had))
ax1 = m.contourf(x1,y1,sstaC,31,cmap=plt.cm.RdBu_r,levels=clevs1,extend='both')
plt.text(0.05, 0.95, 'e', transform=ax.transAxes,
      fontsize=14, fontweight='bold', va='top')
# =============================================================================
# add regions
# =============================================================================
# SW= 30-43N, 245E-253e (Herweijer et al 2007)

x1,y1 = m(245,30)
x2,y2 = m(245,43)
x3,y3 = m(253,43)
x4,y4 = m(253,30)
poly1 = Polygon([(x1,y1),(x2,y2),(x3,y3),(x4,y4)],
                facecolor='None',edgecolor = cl1,linewidth=2.5)
plt.gca().add_patch(poly1)

# NW= 42-50N, 235-245 

x1,y1 = m(234,43)
x2,y2 = m(234,51)
x3,y3 = m(245,51)
x4,y4 = m(245,43)
poly2 = Polygon([(x1,y1),(x2,y2),(x3,y3),(x4,y4)],
                facecolor='None',edgecolor = cl2,linewidth=2.5)
plt.gca().add_patch(poly2)

# NE-central= 30-50N, 260-290 

x1,y1 = m(260,30)
x2,y2 = m(260,50)
x3,y3 = m(290,50)
x4,y4 = m(290,30)
poly3 = Polygon([(x1,y1),(x2,y2),(x3,y3),(x4,y4)],
                facecolor='None',edgecolor = cl3,linewidth=2.5)
plt.gca().add_patch(poly3)

plt.title('(e) CP El Nino (C and E)',fontsize=16,fontweight='bold')


ax=plt.subplot2grid((7, 2), (4, 1), rowspan=2)

m=Basemap(projection='cyl', llcrnrlon=lon1, llcrnrlat=lat1, urcrnrlon=lon2, urcrnrlat=lat2)
m.drawcoastlines(linewidth=1)
#m.drawcountries(linewidth=1)
#m.fillcontinents(color='lightgray')
m.drawmeridians(np.arange(120,295,50),labels=[0,0,0,1],color='DimGray',fontsize=10)
m.drawparallels(np.arange(-25,65,25),labels=[1,0,0,0],color='DimGray',fontsize=10)
x2, y2 = m(*np.meshgrid(lon, lat))
ax4 = m.contourf(x2,y2,pdsiaE,31,cmap=plt.cm.BrBG,levels=clevs2,extend='both')

x1, y1 = m(*np.meshgrid(lon_had, lat_had))
ax3 = m.contourf(x1,y1,sstaE,31,cmap=plt.cm.RdBu_r,levels=clevs1,extend='both')
plt.text(0.05, 0.95, 'f', transform=ax.transAxes,
      fontsize=14, fontweight='bold', va='top')
# =============================================================================
# add regions
# =============================================================================
# SW= 30-43N, 245E-253e (Herweijer et al 2007)

x1,y1 = m(245,30)
x2,y2 = m(245,43)
x3,y3 = m(253,43)
x4,y4 = m(253,30)
poly1 = Polygon([(x1,y1),(x2,y2),(x3,y3),(x4,y4)],
                facecolor='None',edgecolor = cl1,linewidth=2.5)
plt.gca().add_patch(poly1)

# NW= 42-50N, 235-245 

x1,y1 = m(234,43)
x2,y2 = m(234,51)
x3,y3 = m(245,51)
x4,y4 = m(245,43)
poly2 = Polygon([(x1,y1),(x2,y2),(x3,y3),(x4,y4)],
                facecolor='None',edgecolor = cl2,linewidth=2.5)
plt.gca().add_patch(poly2)

# NE-central= 30-50N, 260-290 

x1,y1 = m(260,30)
x2,y2 = m(260,50)
x3,y3 = m(290,50)
x4,y4 = m(290,30)
poly3 = Polygon([(x1,y1),(x2,y2),(x3,y3),(x4,y4)],
                facecolor='None',edgecolor = cl3,linewidth=2.5)
plt.gca().add_patch(poly3)

plt.title('(f) EP El Nino (C and E)',fontsize=16,fontweight='bold')


cb_ax = fig.add_axes([0.15, 0.12, 0.7, 0.013])
cbar1 = fig.colorbar(ax1, cax=cb_ax, orientation='horizontal',ticks=[-2,-1.5,-1,-0.5,0,0.5,1,1.5,2])
plt.title('SST anomaly',fontsize=12)

cb_ax = fig.add_axes([0.15, 0.05, 0.7, 0.013])
cbar1 = fig.colorbar(ax2, cax=cb_ax, orientation='horizontal',ticks=[-3,-2,-1,0,1,2,3])
plt.title('PDSI anomaly',fontsize=12)

plt.tight_layout()

# fig.savefig('/Users/lxy/Desktop/research/ENSO/cp_ep/validation/observation/observ_PDSIbxc1.png',dpi=600)


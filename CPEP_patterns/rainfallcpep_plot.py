#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 29 16:05:55 2020

@author: lxy
"""
import numpy as np
from netCDF4 import Dataset

#import os
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from mpl_toolkits.basemap import Basemap, shiftgrid
from scipy import stats
import scipy.interpolate

def pattern_cor(a,b):
    '''
    a and b are maps with same dimension
    '''
    A=a.flatten()
    B=b.flatten()
#    remove nan
    A_nonan=A[np.logical_not(np.isnan(A))]
    B_nonan=B[np.logical_not(np.isnan(B))]

    
    r, p=stats.pearsonr(A_nonan,B_nonan)

    return r, p


# =============================================================================
# PHYDA
# =============================================================================
# C and E method
SSTACE_phy=np.load('/Users/lxy/Desktop/research/ENSO/cp_ep/validation/rainfall/data/SSTA_phy(CE).npy')
PDSICE_phy=np.load('/Users/lxy/Desktop/research/ENSO/cp_ep/validation/rainfall/data/PDSI_phy(CE)_std.npy')

# Nino 3-4 method
SSTANI_phy=np.load('/Users/lxy/Desktop/research/ENSO/cp_ep/validation/rainfall/data/SSTA_phy(NINO).npy')
PDSINI_phy=np.load('/Users/lxy/Desktop/research/ENSO/cp_ep/validation/rainfall/data/PDSI_phy(NINO)_std.npy')

filepath=Dataset('/Users/lxy/Desktop/research/ENSO/data/PHYDA/da_hydro_AprMar_r.1-2000_d.05-Jan-2018.nc')
lat=filepath.variables['lat']
lat_phy=np.ma.getdata(lat)
lon=filepath.variables['lon']
lon_phy=np.ma.getdata(lon)


# =============================================================================
# pattern correlation
# =============================================================================
latmin=15
latmax=65
lonmin=190
lonmax=295

PDSICE=PDSICE_phy[:,:,(lon_phy<=lonmax)&(lon_phy>=lonmin)]
PDSICE=PDSICE[:, (lat_phy<=latmax)&(lat_phy>=latmin),:]
PDSINI=PDSINI_phy[:,:,(lon_phy<=lonmax)&(lon_phy>=lonmin)]
PDSINI=PDSINI[:, (lat_phy<=latmax)&(lat_phy>=latmin),:]

r1,p1=pattern_cor(PDSICE[0,:,:],PDSINI[0,:,:])
r2,p2=pattern_cor(PDSICE[1,:,:],PDSINI[1,:,:])



# =============================================================================
# LMR
# =============================================================================
# C and E method
SSTACE_lmr=np.load('/Users/lxy/Desktop/research/ENSO/cp_ep/validation/rainfall/data/SSTA_lmr(CE).npy')
PDSICE_lmr=np.load('/Users/lxy/Desktop/research/ENSO/cp_ep/validation/rainfall/data/PDSI_lmr(CE)_std.npy')

# Nino 3-4 method
SSTANI_lmr=np.load('/Users/lxy/Desktop/research/ENSO/cp_ep/validation/rainfall/data/SSTA_lmr(NINO).npy')
PDSINI_lmr=np.load('/Users/lxy/Desktop/research/ENSO/cp_ep/validation/rainfall/data/PDSI_lmr(NINO)_std.npy')

lat_lmr=np.load('/Users/lxy/Desktop/research/ENSO/data/LMR/LMR_lats_v2.npy')
lon_lmr=np.load('/Users/lxy/Desktop/research/ENSO/data/LMR/LMR_lons_v2.npy')


# =============================================================================
# pattern correlation
# =============================================================================
latmin=15
latmax=65
lonmin=190
lonmax=295

PDSICE=PDSICE_lmr[:,:,(lon_lmr<=lonmax)&(lon_lmr>=lonmin)]
PDSICE=PDSICE[:, (lat_lmr<=latmax)&(lat_lmr>=latmin),:]
PDSINI=PDSINI_lmr[:,:,(lon_lmr<=lonmax)&(lon_lmr>=lonmin)]
PDSINI=PDSINI[:, (lat_lmr<=latmax)&(lat_lmr>=latmin),:]

r3,p3=pattern_cor(PDSICE[0,:,:],PDSINI[0,:,:])
r4,p4=pattern_cor(PDSICE[1,:,:],PDSINI[1,:,:])



# =============================================================================
# pattern_cor between LMR and PHYDA
# =============================================================================
XI, YI = np.meshgrid(lon_phy,lat_phy)
X, Y = np.meshgrid(lon_lmr,lat_lmr)

# CE
PDSIce_phy=np.nan_to_num(PDSICE_phy)

PDSICE_phynw1=scipy.interpolate.griddata((XI.flatten(),YI.flatten()),
                                        PDSIce_phy[0,:,:].flatten(), (X,Y), method='cubic')
PDSICE_phynw2=scipy.interpolate.griddata((XI.flatten(),YI.flatten()),
                                        PDSIce_phy[1,:,:].flatten(), (X,Y), method='cubic')

PDSICE_phynw=np.array((PDSICE_phynw1,PDSICE_phynw2))

PDSICEnw=PDSICE_phynw[:,:,(lon_lmr<=lonmax)&(lon_lmr>=lonmin)]
PDSICEnw=PDSICEnw[:, (lat_lmr<=latmax)&(lat_lmr>=latmin),:]

a=np.ma.masked_invalid(PDSICE)

PDSICEnw[np.ma.getmask(a)]=np.nan

r5,p5=pattern_cor(PDSICE[0,:,:],PDSICEnw[0,:,:])
r6,p6=pattern_cor(PDSICE[1,:,:],PDSICEnw[1,:,:])


# NINO
PDSIni_phy=np.nan_to_num(PDSINI_phy)

PDSINI_phynw1=scipy.interpolate.griddata((XI.flatten(),YI.flatten()),
                                        PDSIni_phy[0,:,:].flatten(), (X,Y), method='cubic')
PDSINI_phynw2=scipy.interpolate.griddata((XI.flatten(),YI.flatten()),
                                        PDSIni_phy[1,:,:].flatten(), (X,Y), method='cubic')

PDSINI_phynw=np.array((PDSINI_phynw1,PDSINI_phynw2))

PDSINInw=PDSINI_phynw[:,:,(lon_lmr<=lonmax)&(lon_lmr>=lonmin)]
PDSINInw=PDSINInw[:, (lat_lmr<=latmax)&(lat_lmr>=latmin),:]

PDSINInw[np.ma.getmask(a)]=np.nan

r7,p7=pattern_cor(PDSINI[0,:,:],PDSINInw[0,:,:])
r8,p8=pattern_cor(PDSINI[1,:,:],PDSINInw[1,:,:])



# =============================================================================
# plot
# =============================================================================

lat1=-30
lat2=65
lon1=120
lon2=295

cl1='magenta'
cl2='black'
cl3='blue'


clevs1=np.linspace(-1, 1, 400)
clevs2=np.linspace(-3, 3, 400)

fig=plt.figure(figsize=(10,12))

ax=plt.subplot2grid((9, 2), (0, 0), rowspan=2)
plt.text(0.03, 0.95, 'a', transform=ax.transAxes,
      fontsize=16, fontweight='bold', va='top')
m=Basemap(projection='cyl', llcrnrlon=lon1, llcrnrlat=lat1, urcrnrlon=lon2, urcrnrlat=lat2)
m.drawcoastlines(linewidth=1)
# m.drawcountries(linewidth=1)
#m.fillcontinents(color='lightgray',zorder=0)
m.drawmeridians(np.arange(120,295,50),labels=[0,0,0,1],color='lightgrey',fontsize=10)
m.drawparallels(np.arange(-25,65,25),labels=[1,0,0,0],color='lightgrey',fontsize=10)
x, y = m(*np.meshgrid(lon_lmr, lat_lmr))
ax1 = m.contourf(x,y,SSTANI_lmr[0,:,:],31,cmap=plt.cm.RdBu_r,levels=clevs1,extend='both')
ax2 = m.contourf(x,y,PDSINI_lmr[0,:,:],31,cmap=plt.cm.BrBG,levels=clevs2,extend='both')

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


plt.title('LMR CP (Nino 3-4)',fontsize=16,fontweight='bold')

# =============================================================================
# 
# =============================================================================
ax=plt.subplot2grid((9, 2), (0, 1), rowspan=2)
plt.text(0.03, 0.95, 'b', transform=ax.transAxes,
      fontsize=16, fontweight='bold', va='top')

m=Basemap(projection='cyl', llcrnrlon=lon1, llcrnrlat=lat1, urcrnrlon=lon2, urcrnrlat=lat2)
m.drawcoastlines(linewidth=1)
# m.drawcountries(linewidth=1)
#m.fillcontinents(color='lightgray')
m.drawmeridians(np.arange(120,295,50),labels=[0,0,0,1],color='lightgrey',fontsize=10)
m.drawparallels(np.arange(-25,65,25),labels=[1,0,0,0],color='lightgrey',fontsize=10)
x, y = m(*np.meshgrid(lon_lmr, lat_lmr))
ax3 = m.contourf(x,y,SSTANI_lmr[1,:,:],31,cmap=plt.cm.RdBu_r,levels=clevs1,extend='both')
ax4 = m.contourf(x,y,PDSINI_lmr[1,:,:],31,cmap=plt.cm.BrBG,levels=clevs2,extend='both')

# =============================================================================
# add regions
# =============================================================================
# SW= 30-43N, 245E-253e (Herweijer et al 2007)

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


plt.title('LMR EP (Nino 3-4)',fontsize=16,fontweight='bold')


# =============================================================================
# 
# =============================================================================
ax=plt.subplot2grid((9, 2), (2, 0), rowspan=2)
plt.text(0.03, 0.95, 'c', transform=ax.transAxes,
      fontsize=16, fontweight='bold', va='top')

m=Basemap(projection='cyl', llcrnrlon=lon1, llcrnrlat=lat1, urcrnrlon=lon2, urcrnrlat=lat2)
m.drawcoastlines(linewidth=1)
#m.drawcountries(linewidth=1)
#m.fillcontinents(color='lightgray')
m.drawmeridians(np.arange(120,295,50),labels=[0,0,0,1],color='lightgrey',fontsize=10)
m.drawparallels(np.arange(-25,65,25),labels=[1,0,0,0],color='lightgrey',fontsize=10)
x, y = m(*np.meshgrid(lon_phy, lat_phy))
ax5 = m.contourf(x,y,SSTANI_phy[0,:,:],31,cmap=plt.cm.RdBu_r,levels=clevs1,extend='both')
ax6 = m.contourf(x,y,PDSINI_phy[0,:,:],31,cmap=plt.cm.BrBG,levels=clevs2,extend='both')

# =============================================================================
# add regions
# =============================================================================
# SW= 30-43N, 245E-253e (Herweijer et al 2007)

# SW= 30-43N, 245E-253e (Herweijer et al 2007)

x1,y1 = m(245,30)
x2,y2 = m(245,43)
x3,y3 = m(253,43)
x4,y4 = m(253,30)
poly1 = Polygon([(x1,y1),(x2,y2),(x3,y3),(x4,y4)],
                facecolor='None',edgecolor=cl1,linewidth=2.5)
plt.gca().add_patch(poly1)

# NW= 42-50N, 235-245 

x1,y1 = m(234,43)
x2,y2 = m(234,51)
x3,y3 = m(245,51)
x4,y4 = m(245,43)
poly2 = Polygon([(x1,y1),(x2,y2),(x3,y3),(x4,y4)],
                facecolor='None',edgecolor=cl2,linewidth=2.5)
plt.gca().add_patch(poly2)

# NE-central= 30-50N, 260-290 

x1,y1 = m(260,30)
x2,y2 = m(260,50)
x3,y3 = m(290,50)
x4,y4 = m(290,30)
poly3 = Polygon([(x1,y1),(x2,y2),(x3,y3),(x4,y4)],
                facecolor='None',edgecolor=cl3,linewidth=2.5)
plt.gca().add_patch(poly3)


plt.title('PHYDA CP (Nino 3-4)',fontsize=16,fontweight='bold')

# =============================================================================
# 
# =============================================================================
ax=plt.subplot2grid((9, 2), (2, 1), rowspan=2)
plt.text(0.03, 0.95, 'd', transform=ax.transAxes,
      fontsize=16, fontweight='bold', va='top')

m=Basemap(projection='cyl', llcrnrlon=lon1, llcrnrlat=lat1, urcrnrlon=lon2, urcrnrlat=lat2)
m.drawcoastlines(linewidth=1)
#m.drawcountries(linewidth=1)
#m.fillcontinents(color='lightgray')
m.drawmeridians(np.arange(120,295,50),labels=[0,0,0,1],color='lightgrey',fontsize=10)
m.drawparallels(np.arange(-25,65,25),labels=[1,0,0,0],color='lightgrey',fontsize=10)
x, y = m(*np.meshgrid(lon_phy, lat_phy))
ax7 = m.contourf(x,y,SSTANI_phy[1,:,:],31,cmap=plt.cm.RdBu_r,levels=clevs1,extend='both')
ax8 = m.contourf(x,y,PDSINI_phy[1,:,:],31,cmap=plt.cm.BrBG,levels=clevs2,extend='both')

# =============================================================================
# add regions
# =============================================================================
# SW= 30-43N, 245E-253e (Herweijer et al 2007)


x1,y1 = m(245,30)
x2,y2 = m(245,43)
x3,y3 = m(253,43)
x4,y4 = m(253,30)
poly1 = Polygon([(x1,y1),(x2,y2),(x3,y3),(x4,y4)],
                facecolor='None',edgecolor=cl1,linewidth=2.5)
plt.gca().add_patch(poly1)

# NW= 42-50N, 235-245 

x1,y1 = m(234,43)
x2,y2 = m(234,51)
x3,y3 = m(245,51)
x4,y4 = m(245,43)
poly2 = Polygon([(x1,y1),(x2,y2),(x3,y3),(x4,y4)],
                facecolor='None',edgecolor=cl2,linewidth=2.5)
plt.gca().add_patch(poly2)

# NE-central= 30-50N, 260-290 

x1,y1 = m(260,30)
x2,y2 = m(260,50)
x3,y3 = m(290,50)
x4,y4 = m(290,30)
poly3 = Polygon([(x1,y1),(x2,y2),(x3,y3),(x4,y4)],
                facecolor='None',edgecolor=cl3,linewidth=2.5)
plt.gca().add_patch(poly3)


plt.title('PHYDA EP (Nino 3-4)',fontsize=16,fontweight='bold')

# =============================================================================
# 
# =============================================================================
ax=plt.subplot2grid((9, 2), (4, 0), rowspan=2)
plt.text(0.03, 0.95, 'e', transform=ax.transAxes,
      fontsize=16, fontweight='bold', va='top')
m=Basemap(projection='cyl', llcrnrlon=lon1, llcrnrlat=lat1, urcrnrlon=lon2, urcrnrlat=lat2)
m.drawcoastlines(linewidth=1)
#m.fillcontinents(color='lightgray',zorder=0)
m.drawmeridians(np.arange(120,295,50),labels=[0,0,0,1],color='lightgrey',fontsize=10)
m.drawparallels(np.arange(-25,65,25),labels=[1,0,0,0],color='lightgrey',fontsize=10)
x, y = m(*np.meshgrid(lon_lmr, lat_lmr))
ax1 = m.contourf(x,y,SSTACE_lmr[0,:,:],31,cmap=plt.cm.RdBu_r,levels=clevs1,extend='both')
ax2 = m.contourf(x,y,PDSICE_lmr[0,:,:],31,cmap=plt.cm.BrBG,levels=clevs2,extend='both')

# =============================================================================
# add regions
# =============================================================================
# SW= 30-43N, 245E-253e (Herweijer et al 2007)


x1,y1 = m(245,30)
x2,y2 = m(245,43)
x3,y3 = m(253,43)
x4,y4 = m(253,30)
poly1 = Polygon([(x1,y1),(x2,y2),(x3,y3),(x4,y4)],
                facecolor='None',edgecolor=cl1,linewidth=2.5)
plt.gca().add_patch(poly1)

# NW= 42-50N, 235-245 

x1,y1 = m(234,43)
x2,y2 = m(234,51)
x3,y3 = m(245,51)
x4,y4 = m(245,43)
poly2 = Polygon([(x1,y1),(x2,y2),(x3,y3),(x4,y4)],
                facecolor='None',edgecolor=cl2,linewidth=2.5)
plt.gca().add_patch(poly2)

# NE-central= 30-50N, 260-290 

x1,y1 = m(260,30)
x2,y2 = m(260,50)
x3,y3 = m(290,50)
x4,y4 = m(290,30)
poly3 = Polygon([(x1,y1),(x2,y2),(x3,y3),(x4,y4)],
                facecolor='None',edgecolor=cl3,linewidth=2.5)
plt.gca().add_patch(poly3)


plt.title('LMR CP (C and E)',fontsize=16,fontweight='bold')

# =============================================================================
# 
# =============================================================================
ax=plt.subplot2grid((9, 2), (4, 1), rowspan=2)
plt.text(0.03, 0.95, 'f', transform=ax.transAxes,
      fontsize=16, fontweight='bold', va='top')

m=Basemap(projection='cyl', llcrnrlon=lon1, llcrnrlat=lat1, urcrnrlon=lon2, urcrnrlat=lat2)
m.drawcoastlines(linewidth=1)
#m.drawcountries(linewidth=1)
#m.fillcontinents(color='lightgray')
m.drawmeridians(np.arange(120,295,50),labels=[0,0,0,1],color='lightgrey',fontsize=10)
m.drawparallels(np.arange(-25,65,25),labels=[1,0,0,0],color='lightgrey',fontsize=10)
x, y = m(*np.meshgrid(lon_lmr, lat_lmr))
ax3 = m.contourf(x,y,SSTACE_lmr[1,:,:],31,cmap=plt.cm.RdBu_r,levels=clevs1,extend='both')
ax4 = m.contourf(x,y,PDSICE_lmr[1,:,:],31,cmap=plt.cm.BrBG,levels=clevs2,extend='both')

# =============================================================================
# add regions
# =============================================================================
# SW= 30-43N, 245E-253e (Herweijer et al 2007)


x1,y1 = m(245,30)
x2,y2 = m(245,43)
x3,y3 = m(253,43)
x4,y4 = m(253,30)
poly1 = Polygon([(x1,y1),(x2,y2),(x3,y3),(x4,y4)],
                facecolor='None',edgecolor=cl1,linewidth=2.5)
plt.gca().add_patch(poly1)

# NW= 42-50N, 235-245 

x1,y1 = m(234,43)
x2,y2 = m(234,51)
x3,y3 = m(245,51)
x4,y4 = m(245,43)
poly2 = Polygon([(x1,y1),(x2,y2),(x3,y3),(x4,y4)],
                facecolor='None',edgecolor=cl2,linewidth=2.5)
plt.gca().add_patch(poly2)

# NE-central= 30-50N, 260-290 

x1,y1 = m(260,30)
x2,y2 = m(260,50)
x3,y3 = m(290,50)
x4,y4 = m(290,30)
poly3 = Polygon([(x1,y1),(x2,y2),(x3,y3),(x4,y4)],
                facecolor='None',edgecolor=cl3,linewidth=2.5)
plt.gca().add_patch(poly3)


plt.title('LMR EP (C and E)',fontsize=16,fontweight='bold')

# =============================================================================
# 
# =============================================================================
ax=plt.subplot2grid((9, 2), (6, 0), rowspan=2)
plt.text(0.03, 0.95, 'g', transform=ax.transAxes,
      fontsize=16, fontweight='bold', va='top')

m=Basemap(projection='cyl', llcrnrlon=lon1, llcrnrlat=lat1, urcrnrlon=lon2, urcrnrlat=lat2)
m.drawcoastlines(linewidth=1)
#m.drawcountries(linewidth=1)
#m.fillcontinents(color='lightgray')
m.drawmeridians(np.arange(120,295,50),labels=[0,0,0,1],color='lightgrey',fontsize=10)
m.drawparallels(np.arange(-25,65,25),labels=[1,0,0,0],color='lightgrey',fontsize=10)
x, y = m(*np.meshgrid(lon_phy, lat_phy))
ax5 = m.contourf(x,y,SSTACE_phy[0,:,:],31,cmap=plt.cm.RdBu_r,levels=clevs1,extend='both')
ax6 = m.contourf(x,y,PDSICE_phy[0,:,:],31,cmap=plt.cm.BrBG,levels=clevs2,extend='both')

# =============================================================================
# add regions
# =============================================================================
# SW= 30-43N, 245E-253e (Herweijer et al 2007)

x1,y1 = m(245,30)
x2,y2 = m(245,43)
x3,y3 = m(253,43)
x4,y4 = m(253,30)
poly1 = Polygon([(x1,y1),(x2,y2),(x3,y3),(x4,y4)],
                facecolor='None',edgecolor=cl1,linewidth=2.5)
plt.gca().add_patch(poly1)

# NW= 42-50N, 235-245 

x1,y1 = m(234,43)
x2,y2 = m(234,51)
x3,y3 = m(245,51)
x4,y4 = m(245,43)
poly2 = Polygon([(x1,y1),(x2,y2),(x3,y3),(x4,y4)],
                facecolor='None',edgecolor=cl2,linewidth=2.5)
plt.gca().add_patch(poly2)

# NE-central= 30-50N, 260-290 

x1,y1 = m(260,30)
x2,y2 = m(260,50)
x3,y3 = m(290,50)
x4,y4 = m(290,30)
poly3 = Polygon([(x1,y1),(x2,y2),(x3,y3),(x4,y4)],
                facecolor='None',edgecolor=cl3,linewidth=2.5)
plt.gca().add_patch(poly3)

plt.title('PHYDA CP (C and E)',fontsize=16,fontweight='bold')


# =============================================================================
# 
# =============================================================================
ax=plt.subplot2grid((9, 2), (6, 1), rowspan=2)
plt.text(0.03, 0.95, 'h', transform=ax.transAxes,
      fontsize=16, fontweight='bold', va='top')

m=Basemap(projection='cyl', llcrnrlon=lon1, llcrnrlat=lat1, urcrnrlon=lon2, urcrnrlat=lat2)
m.drawcoastlines(linewidth=1)
#m.drawcountries(linewidth=1)
#m.fillcontinents(color='lightgray')
m.drawmeridians(np.arange(120,295,50),labels=[0,0,0,1],color='lightgrey',fontsize=10)
m.drawparallels(np.arange(-25,65,25),labels=[1,0,0,0],color='lightgrey',fontsize=10)
x, y = m(*np.meshgrid(lon_phy, lat_phy))
ax7 = m.contourf(x,y,SSTACE_phy[1,:,:],31,cmap=plt.cm.RdBu_r,levels=clevs1,extend='both')
ax8 = m.contourf(x,y,PDSICE_phy[1,:,:],31,cmap=plt.cm.BrBG,levels=clevs2,extend='both')

# =============================================================================
# add regions
# =============================================================================
# SW= 30-43N, 245E-253e (Herweijer et al 2007)

x1,y1 = m(245,30)
x2,y2 = m(245,43)
x3,y3 = m(253,43)
x4,y4 = m(253,30)
poly1 = Polygon([(x1,y1),(x2,y2),(x3,y3),(x4,y4)],
                facecolor='None',edgecolor=cl1,linewidth=2.5)
plt.gca().add_patch(poly1)

# NW= 42-50N, 235-245 

x1,y1 = m(234,43)
x2,y2 = m(234,51)
x3,y3 = m(245,51)
x4,y4 = m(245,43)
poly2 = Polygon([(x1,y1),(x2,y2),(x3,y3),(x4,y4)],
                facecolor='None',edgecolor=cl2,linewidth=2.5)
plt.gca().add_patch(poly2)

# NE-central= 30-50N, 260-290 

x1,y1 = m(260,30)
x2,y2 = m(260,50)
x3,y3 = m(290,50)
x4,y4 = m(290,30)
poly3 = Polygon([(x1,y1),(x2,y2),(x3,y3),(x4,y4)],
                facecolor='None',edgecolor=cl3,linewidth=2.5)
plt.gca().add_patch(poly3)


plt.title('PHYDA EP (C and E)',fontsize=16,fontweight='bold')


cb_ax = fig.add_axes([0.2, 0.07, 0.6, 0.008])
cbar1 = fig.colorbar(ax1, cax=cb_ax, orientation='horizontal',ticks=[-1,-0.75,-0.5,-0.25,0,0.25,0.5,0.75,1])
plt.title('SST anomaly',fontsize=12)

cb_ax = fig.add_axes([0.2, 0.02, 0.6, 0.008])
cbar1 = fig.colorbar(ax2, cax=cb_ax, orientation='horizontal',ticks=[-3,-2,-1,0,1,2,3])
plt.title('PDSI anomaly',fontsize=12)

plt.tight_layout()
plt.show()

# fig.savefig('/Users/lxy/Desktop/research/ENSO/cp_ep/validation/rainfall/lmr&phy_bxc1.png',dpi=300)
fig.savefig('/Users/lxy/Desktop/research/ENSO/cp_ep/validation/rainfall/lmr&phy_bxc1.eps', dpi=300)







# =============================================================================
# plot Nino 3-4 only
# =============================================================================

# =============================================================================
# lat1=-30
# lat2=65
# lon1=120
# lon2=295
# 
# 
# clevs1=np.linspace(-1, 1, 400)
# clevs2=np.linspace(-2, 2, 400)
# 
# fig=plt.figure(figsize=(12,10))
# 
# ax=plt.subplot2grid((5, 2), (0, 0), rowspan=2)
# plt.text(0.03, 0.95, 'a', transform=ax.transAxes,
#       fontsize=20, fontweight='bold', va='top')
# m=Basemap(projection='cyl', llcrnrlon=lon1, llcrnrlat=lat1, urcrnrlon=lon2, urcrnrlat=lat2)
# m.drawcoastlines(linewidth=1)
# #m.fillcontinents(color='lightgray',zorder=0)
# m.drawmeridians(np.arange(120,295,50),labels=[0,0,0,1],color='DimGray',fontsize=12)
# m.drawparallels(np.arange(-25,65,25),labels=[1,0,0,0],color='DimGray',fontsize=12)
# x, y = m(*np.meshgrid(lon_lmr, lat_lmr))
# ax1 = m.contourf(x,y,SSTANI_lmr[0,:,:],31,cmap=plt.cm.RdBu_r,levels=clevs1,extend='both')
# ax2 = m.contourf(x,y,PDSINI_lmr[0,:,:],31,cmap=plt.cm.BrBG,levels=clevs2,extend='both')
# 
# # =============================================================================
# # add regions
# # =============================================================================
# # SW= 30-43N, 245E-253e (Herweijer et al 2007)
# 
# x1,y1 = m(245,30)
# x2,y2 = m(245,43)
# x3,y3 = m(253,43)
# x4,y4 = m(253,30)
# poly1 = Polygon([(x1,y1),(x2,y2),(x3,y3),(x4,y4)],
#                 facecolor='None',edgecolor='firebrick',linewidth=3)
# plt.gca().add_patch(poly1)
# 
# # NW= 42-50N, 235-245 
# 
# x1,y1 = m(235,42)
# x2,y2 = m(235,50)
# x3,y3 = m(245,50)
# x4,y4 = m(245,42)
# poly2 = Polygon([(x1,y1),(x2,y2),(x3,y3),(x4,y4)],
#                 facecolor='None',edgecolor='navy',linewidth=3)
# plt.gca().add_patch(poly2)
# 
# # NE-central= 30-50N, 260-290 
# 
# x1,y1 = m(260,30)
# x2,y2 = m(260,50)
# x3,y3 = m(290,50)
# x4,y4 = m(290,30)
# poly3 = Polygon([(x1,y1),(x2,y2),(x3,y3),(x4,y4)],
#                 facecolor='None',edgecolor='darkviolet',linewidth=3)
# plt.gca().add_patch(poly3)
# 
# 
# plt.title('LMR CP (Nino 3-4)',fontsize=20,fontweight='bold')
# 
# 
# ax=plt.subplot2grid((5, 2), (0, 1), rowspan=2)
# plt.text(0.03, 0.95, 'b', transform=ax.transAxes,
#       fontsize=20, fontweight='bold', va='top')
# 
# m=Basemap(projection='cyl', llcrnrlon=lon1, llcrnrlat=lat1, urcrnrlon=lon2, urcrnrlat=lat2)
# m.drawcoastlines(linewidth=1)
# #m.drawcountries(linewidth=1)
# #m.fillcontinents(color='lightgray')
# m.drawmeridians(np.arange(120,295,50),labels=[0,0,0,1],color='DimGray',fontsize=12)
# m.drawparallels(np.arange(-25,65,25),labels=[0,0,0,0],color='DimGray',fontsize=12)
# x, y = m(*np.meshgrid(lon_lmr, lat_lmr))
# ax3 = m.contourf(x,y,SSTANI_lmr[1,:,:],31,cmap=plt.cm.RdBu_r,levels=clevs1,extend='both')
# ax4 = m.contourf(x,y,PDSINI_lmr[1,:,:],31,cmap=plt.cm.BrBG,levels=clevs2,extend='both')
# # =============================================================================
# # add regions
# # =============================================================================
# # SW= 30-43N, 245E-253e (Herweijer et al 2007)
# 
# x1,y1 = m(245,30)
# x2,y2 = m(245,43)
# x3,y3 = m(253,43)
# x4,y4 = m(253,30)
# poly1 = Polygon([(x1,y1),(x2,y2),(x3,y3),(x4,y4)],facecolor='None',edgecolor='firebrick',linewidth=3)
# plt.gca().add_patch(poly1)
# 
# # NW= 42-50N, 235-245 
# 
# x1,y1 = m(235,42)
# x2,y2 = m(235,50)
# x3,y3 = m(245,50)
# x4,y4 = m(245,42)
# poly2 = Polygon([(x1,y1),(x2,y2),(x3,y3),(x4,y4)],facecolor='None',edgecolor='navy',linewidth=3)
# plt.gca().add_patch(poly2)
# 
# # NE-central= 30-50N, 260-290 
# 
# x1,y1 = m(260,30)
# x2,y2 = m(260,50)
# x3,y3 = m(290,50)
# x4,y4 = m(290,30)
# poly3 = Polygon([(x1,y1),(x2,y2),(x3,y3),(x4,y4)],facecolor='None',edgecolor='darkviolet',linewidth=3)
# plt.gca().add_patch(poly3)
# 
# plt.title('LMR EP (Nino 3-4)',fontsize=20,fontweight='bold')
# 
# 
# ax=plt.subplot2grid((5, 2), (2, 0), rowspan=2)
# plt.text(0.03, 0.95, 'c', transform=ax.transAxes,
#       fontsize=20, fontweight='bold', va='top')
# 
# m=Basemap(projection='cyl', llcrnrlon=lon1, llcrnrlat=lat1, urcrnrlon=lon2, urcrnrlat=lat2)
# m.drawcoastlines(linewidth=1)
# #m.drawcountries(linewidth=1)
# #m.fillcontinents(color='lightgray')
# m.drawmeridians(np.arange(120,295,50),labels=[0,0,0,1],color='DimGray',fontsize=12)
# m.drawparallels(np.arange(-25,65,25),labels=[1,0,0,0],color='DimGray',fontsize=12)
# x, y = m(*np.meshgrid(lon_phy, lat_phy))
# ax5 = m.contourf(x,y,SSTANI_phy[0,:,:],31,cmap=plt.cm.RdBu_r,levels=clevs1,extend='both')
# ax6 = m.contourf(x,y,PDSINI_phy[0,:,:],31,cmap=plt.cm.BrBG,levels=clevs2,extend='both')
# # =============================================================================
# # add regions
# # =============================================================================
# # SW= 30-43N, 245E-253e (Herweijer et al 2007)
# 
# x1,y1 = m(245,30)
# x2,y2 = m(245,43)
# x3,y3 = m(253,43)
# x4,y4 = m(253,30)
# poly1 = Polygon([(x1,y1),(x2,y2),(x3,y3),(x4,y4)],facecolor='None',edgecolor='firebrick',linewidth=3)
# plt.gca().add_patch(poly1)
# 
# # NW= 42-50N, 235-245 
# 
# x1,y1 = m(235,42)
# x2,y2 = m(235,50)
# x3,y3 = m(245,50)
# x4,y4 = m(245,42)
# poly2 = Polygon([(x1,y1),(x2,y2),(x3,y3),(x4,y4)],facecolor='None',edgecolor='navy',linewidth=3)
# plt.gca().add_patch(poly2)
# 
# # NE-central= 30-50N, 260-290 
# 
# x1,y1 = m(260,30)
# x2,y2 = m(260,50)
# x3,y3 = m(290,50)
# x4,y4 = m(290,30)
# poly3 = Polygon([(x1,y1),(x2,y2),(x3,y3),(x4,y4)],facecolor='None',edgecolor='darkviolet',linewidth=3)
# plt.gca().add_patch(poly3)
# 
# plt.title('PHYDA CP (Nino 3-4)',fontsize=20,fontweight='bold')
# 
# 
# 
# ax=plt.subplot2grid((5, 2), (2, 1), rowspan=2)
# plt.text(0.03, 0.95, 'd', transform=ax.transAxes,
#       fontsize=20, fontweight='bold', va='top')
# 
# m=Basemap(projection='cyl', llcrnrlon=lon1, llcrnrlat=lat1, urcrnrlon=lon2, urcrnrlat=lat2)
# m.drawcoastlines(linewidth=1)
# #m.drawcountries(linewidth=1)
# #m.fillcontinents(color='lightgray')
# m.drawmeridians(np.arange(120,295,50),labels=[0,0,0,1],color='DimGray',fontsize=12)
# m.drawparallels(np.arange(-25,65,25),labels=[0,0,0,0],color='DimGray',fontsize=12)
# x, y = m(*np.meshgrid(lon_phy, lat_phy))
# ax7 = m.contourf(x,y,SSTANI_phy[1,:,:],31,cmap=plt.cm.RdBu_r,levels=clevs1,extend='both')
# ax8 = m.contourf(x,y,PDSINI_phy[1,:,:],31,cmap=plt.cm.BrBG,levels=clevs2,extend='both')
# # =============================================================================
# # add regions
# # =============================================================================
# # SW= 30-43N, 245E-253e (Herweijer et al 2007)
# 
# x1,y1 = m(245,30)
# x2,y2 = m(245,43)
# x3,y3 = m(253,43)
# x4,y4 = m(253,30)
# poly1 = Polygon([(x1,y1),(x2,y2),(x3,y3),(x4,y4)],
#                 facecolor='None',edgecolor='firebrick',linewidth=3,label='SW U.S.')
# plt.gca().add_patch(poly1)
# 
# # NW= 42-50N, 235-245 
# 
# x1,y1 = m(235,42)
# x2,y2 = m(235,50)
# x3,y3 = m(245,50)
# x4,y4 = m(245,42)
# poly2 = Polygon([(x1,y1),(x2,y2),(x3,y3),(x4,y4)],
#                 facecolor='None',edgecolor='navy',linewidth=3,label='NW U.S.')
# plt.gca().add_patch(poly2)
# 
# # NE-central= 30-50N, 260-290 
# 
# x1,y1 = m(260,30)
# x2,y2 = m(260,50)
# x3,y3 = m(290,50)
# x4,y4 = m(290,30)
# poly3 = Polygon([(x1,y1),(x2,y2),(x3,y3),(x4,y4)],
#                 facecolor='None',edgecolor='darkviolet',linewidth=3,label='E-Central U.S.')
# plt.gca().add_patch(poly3)
# 
# plt.title('PHYDA EP (Nino 3-4)',fontsize=20,fontweight='bold')
# 
# 
# cb_ax = fig.add_axes([0.2, 0.16, 0.6, 0.01])
# cbar1 = fig.colorbar(ax1, cax=cb_ax, orientation='horizontal',ticks=[-1,-0.75,-0.5,-0.25,0,0.25,0.5,0.75,1])
# plt.title('SST anomaly',fontsize=16)
# 
# cb_ax = fig.add_axes([0.2, 0.07, 0.6, 0.01])
# cbar1 = fig.colorbar(ax2, cax=cb_ax, orientation='horizontal',ticks=[-2,-1.5,-1,-0.5,0,0.5,1,1.5,2])
# plt.title('PDSI anomaly',fontsize=16)
# 
# # fig.legend(loc='center right', bbox_to_anchor=(0.8, 0.1),mode="expand",frameon=False,ncol=1,fontsize='large')
# 
# plt.tight_layout()
# 
# plt.show()
# # fig.savefig('/Users/lxy/Desktop/research/ENSO/cp_ep/validation/rainfall/lmr&phy(nino)box.png',dpi=300)
# 
# =============================================================================


# =============================================================================
# plot C and E only
# =============================================================================

# =============================================================================
# lat1=-30
# lat2=65
# lon1=120
# lon2=295
# 
# 
# clevs1=np.linspace(-1, 1, 400)
# clevs2=np.linspace(-2, 2, 400)
# 
# fig=plt.figure(figsize=(12,10))
# 
# ax=plt.subplot2grid((5, 2), (0, 0), rowspan=2)
# plt.text(0.03, 0.95, 'a', transform=ax.transAxes,
#       fontsize=20, fontweight='bold', va='top')
# m=Basemap(projection='cyl', llcrnrlon=lon1, llcrnrlat=lat1, urcrnrlon=lon2, urcrnrlat=lat2)
# m.drawcoastlines(linewidth=1)
# #m.fillcontinents(color='lightgray',zorder=0)
# m.drawmeridians(np.arange(120,295,50),labels=[0,0,0,1],color='DimGray',fontsize=12)
# m.drawparallels(np.arange(-25,65,25),labels=[1,0,0,0],color='DimGray',fontsize=12)
# x, y = m(*np.meshgrid(lon_lmr, lat_lmr))
# ax1 = m.contourf(x,y,SSTACE_lmr[0,:,:],31,cmap=plt.cm.RdBu_r,levels=clevs1,extend='both')
# ax2 = m.contourf(x,y,PDSICE_lmr[0,:,:],31,cmap=plt.cm.BrBG,levels=clevs2,extend='both')
# 
# # =============================================================================
# # add regions
# # =============================================================================
# # SW= 30-43N, 245E-253e (Herweijer et al 2007)
# 
# x1,y1 = m(245,30)
# x2,y2 = m(245,43)
# x3,y3 = m(253,43)
# x4,y4 = m(253,30)
# poly1 = Polygon([(x1,y1),(x2,y2),(x3,y3),(x4,y4)],
#                 facecolor='None',edgecolor='firebrick',linewidth=3)
# plt.gca().add_patch(poly1)
# 
# # NW= 42-50N, 235-245 
# 
# x1,y1 = m(235,42)
# x2,y2 = m(235,50)
# x3,y3 = m(245,50)
# x4,y4 = m(245,42)
# poly2 = Polygon([(x1,y1),(x2,y2),(x3,y3),(x4,y4)],
#                 facecolor='None',edgecolor='navy',linewidth=3)
# plt.gca().add_patch(poly2)
# 
# # NE-central= 30-50N, 260-290 
# 
# x1,y1 = m(260,30)
# x2,y2 = m(260,50)
# x3,y3 = m(290,50)
# x4,y4 = m(290,30)
# poly3 = Polygon([(x1,y1),(x2,y2),(x3,y3),(x4,y4)],
#                 facecolor='None',edgecolor='darkviolet',linewidth=3)
# plt.gca().add_patch(poly3)
# 
# 
# plt.title('LMR CP (C and E)',fontsize=20,fontweight='bold')
# 
# 
# ax=plt.subplot2grid((5, 2), (0, 1), rowspan=2)
# plt.text(0.03, 0.95, 'b', transform=ax.transAxes,
#       fontsize=20, fontweight='bold', va='top')
# 
# m=Basemap(projection='cyl', llcrnrlon=lon1, llcrnrlat=lat1, urcrnrlon=lon2, urcrnrlat=lat2)
# m.drawcoastlines(linewidth=1)
# #m.drawcountries(linewidth=1)
# #m.fillcontinents(color='lightgray')
# m.drawmeridians(np.arange(120,295,50),labels=[0,0,0,1],color='DimGray',fontsize=12)
# m.drawparallels(np.arange(-25,65,25),labels=[0,0,0,0],color='DimGray',fontsize=12)
# x, y = m(*np.meshgrid(lon_lmr, lat_lmr))
# ax3 = m.contourf(x,y,SSTACE_lmr[1,:,:],31,cmap=plt.cm.RdBu_r,levels=clevs1,extend='both')
# ax4 = m.contourf(x,y,PDSICE_lmr[1,:,:],31,cmap=plt.cm.BrBG,levels=clevs2,extend='both')
# # =============================================================================
# # add regions
# # =============================================================================
# # SW= 30-43N, 245E-253e (Herweijer et al 2007)
# 
# x1,y1 = m(245,30)
# x2,y2 = m(245,43)
# x3,y3 = m(253,43)
# x4,y4 = m(253,30)
# poly1 = Polygon([(x1,y1),(x2,y2),(x3,y3),(x4,y4)],facecolor='None',edgecolor='firebrick',linewidth=3)
# plt.gca().add_patch(poly1)
# 
# # NW= 42-50N, 235-245 
# 
# x1,y1 = m(235,42)
# x2,y2 = m(235,50)
# x3,y3 = m(245,50)
# x4,y4 = m(245,42)
# poly2 = Polygon([(x1,y1),(x2,y2),(x3,y3),(x4,y4)],facecolor='None',edgecolor='navy',linewidth=3)
# plt.gca().add_patch(poly2)
# 
# # NE-central= 30-50N, 260-290 
# 
# x1,y1 = m(260,30)
# x2,y2 = m(260,50)
# x3,y3 = m(290,50)
# x4,y4 = m(290,30)
# poly3 = Polygon([(x1,y1),(x2,y2),(x3,y3),(x4,y4)],facecolor='None',edgecolor='darkviolet',linewidth=3)
# plt.gca().add_patch(poly3)
# 
# plt.title('LMR EP (C and E)',fontsize=20,fontweight='bold')
# 
# 
# ax=plt.subplot2grid((5, 2), (2, 0), rowspan=2)
# plt.text(0.03, 0.95, 'c', transform=ax.transAxes,
#       fontsize=20, fontweight='bold', va='top')
# 
# m=Basemap(projection='cyl', llcrnrlon=lon1, llcrnrlat=lat1, urcrnrlon=lon2, urcrnrlat=lat2)
# m.drawcoastlines(linewidth=1)
# #m.drawcountries(linewidth=1)
# #m.fillcontinents(color='lightgray')
# m.drawmeridians(np.arange(120,295,50),labels=[0,0,0,1],color='DimGray',fontsize=12)
# m.drawparallels(np.arange(-25,65,25),labels=[1,0,0,0],color='DimGray',fontsize=12)
# x, y = m(*np.meshgrid(lon_phy, lat_phy))
# ax5 = m.contourf(x,y,SSTACE_phy[0,:,:],31,cmap=plt.cm.RdBu_r,levels=clevs1,extend='both')
# ax6 = m.contourf(x,y,PDSICE_phy[0,:,:],31,cmap=plt.cm.BrBG,levels=clevs2,extend='both')
# # =============================================================================
# # add regions
# # =============================================================================
# # SW= 30-43N, 245E-253e (Herweijer et al 2007)
# 
# x1,y1 = m(245,30)
# x2,y2 = m(245,43)
# x3,y3 = m(253,43)
# x4,y4 = m(253,30)
# poly1 = Polygon([(x1,y1),(x2,y2),(x3,y3),(x4,y4)],facecolor='None',edgecolor='firebrick',linewidth=3)
# plt.gca().add_patch(poly1)
# 
# # NW= 42-50N, 235-245 
# 
# x1,y1 = m(235,42)
# x2,y2 = m(235,50)
# x3,y3 = m(245,50)
# x4,y4 = m(245,42)
# poly2 = Polygon([(x1,y1),(x2,y2),(x3,y3),(x4,y4)],facecolor='None',edgecolor='navy',linewidth=3)
# plt.gca().add_patch(poly2)
# 
# # NE-central= 30-50N, 260-290 
# 
# x1,y1 = m(260,30)
# x2,y2 = m(260,50)
# x3,y3 = m(290,50)
# x4,y4 = m(290,30)
# poly3 = Polygon([(x1,y1),(x2,y2),(x3,y3),(x4,y4)],facecolor='None',edgecolor='darkviolet',linewidth=3)
# plt.gca().add_patch(poly3)
# 
# plt.title('PHYDA CP (C and E)',fontsize=20,fontweight='bold')
# 
# 
# 
# ax=plt.subplot2grid((5, 2), (2, 1), rowspan=2)
# plt.text(0.03, 0.95, 'd', transform=ax.transAxes,
#       fontsize=20, fontweight='bold', va='top')
# 
# m=Basemap(projection='cyl', llcrnrlon=lon1, llcrnrlat=lat1, urcrnrlon=lon2, urcrnrlat=lat2)
# m.drawcoastlines(linewidth=1)
# #m.drawcountries(linewidth=1)
# #m.fillcontinents(color='lightgray')
# m.drawmeridians(np.arange(120,295,50),labels=[0,0,0,1],color='DimGray',fontsize=12)
# m.drawparallels(np.arange(-25,65,25),labels=[0,0,0,0],color='DimGray',fontsize=12)
# x, y = m(*np.meshgrid(lon_phy, lat_phy))
# ax7 = m.contourf(x,y,SSTACE_phy[1,:,:],31,cmap=plt.cm.RdBu_r,levels=clevs1,extend='both')
# ax8 = m.contourf(x,y,PDSICE_phy[1,:,:],31,cmap=plt.cm.BrBG,levels=clevs2,extend='both')
# # =============================================================================
# # add regions
# # =============================================================================
# # SW= 30-43N, 245E-253e (Herweijer et al 2007)
# 
# x1,y1 = m(245,30)
# x2,y2 = m(245,43)
# x3,y3 = m(253,43)
# x4,y4 = m(253,30)
# poly1 = Polygon([(x1,y1),(x2,y2),(x3,y3),(x4,y4)],
#                 facecolor='None',edgecolor='firebrick',linewidth=3,label='SW U.S.')
# plt.gca().add_patch(poly1)
# 
# # NW= 42-50N, 235-245 
# 
# x1,y1 = m(235,42)
# x2,y2 = m(235,50)
# x3,y3 = m(245,50)
# x4,y4 = m(245,42)
# poly2 = Polygon([(x1,y1),(x2,y2),(x3,y3),(x4,y4)],
#                 facecolor='None',edgecolor='navy',linewidth=3,label='NW U.S.')
# plt.gca().add_patch(poly2)
# 
# # NE-central= 30-50N, 260-290 
# 
# x1,y1 = m(260,30)
# x2,y2 = m(260,50)
# x3,y3 = m(290,50)
# x4,y4 = m(290,30)
# poly3 = Polygon([(x1,y1),(x2,y2),(x3,y3),(x4,y4)],
#                 facecolor='None',edgecolor='darkviolet',linewidth=3,label='E-Central U.S.')
# plt.gca().add_patch(poly3)
# 
# plt.title('PHYDA EP (C and E)',fontsize=20,fontweight='bold')
# 
# 
# cb_ax = fig.add_axes([0.2, 0.16, 0.6, 0.01])
# cbar1 = fig.colorbar(ax1, cax=cb_ax, orientation='horizontal',ticks=[-1,-0.75,-0.5,-0.25,0,0.25,0.5,0.75,1])
# plt.title('SST anomaly',fontsize=16)
# 
# cb_ax = fig.add_axes([0.2, 0.07, 0.6, 0.01])
# cbar1 = fig.colorbar(ax2, cax=cb_ax, orientation='horizontal',ticks=[-2,-1.5,-1,-0.5,0,0.5,1,1.5,2])
# plt.title('PDSI anomaly',fontsize=16)
# 
# # fig.legend(loc='center right', bbox_to_anchor=(0.8, 0.1),mode="expand",frameon=False,ncol=1,fontsize='large')
# 
# plt.tight_layout()
# 
# plt.show()
# # fig.savefig('/Users/lxy/Desktop/research/ENSO/cp_ep/validation/rainfall/lmr&phy(CE)box.png',dpi=300)
# 
# =============================================================================



















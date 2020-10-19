#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 15 13:48:35 2019

@author: lxy
"""

import numpy as np
import pandas as pd
from netCDF4 import Dataset
import matplotlib.pyplot as plt
from scipy import stats
from eofs.standard import Eof
import seaborn as sns
#
#for LMR
def SSTA_cal(SST):
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

#function for EMI
def EMindex(SSTA,lat,lon):
    """
    EMI(El Nino Modoki Index) method to define CP(EMI>0.7 sigma)
    SST with (time,lat,lon)
    """
    #t_lat_lon=SST.shape
    #box-A:165E-140W(82-110),10S-10N(40-50)
    ssta_a=SSTA[:,(lat<=10)&(lat>=-10),:]
    SSTA_a=np.ma.average(ssta_a[:,:,(lon<=220)&(lon>=165)],axis=(1,2))

    #box-B:110W-70W(125-145),15S-5N(38-47)
    ssta_b=SSTA[:,(lat<=5)&(lat>=-15),:]
    SSTA_b=np.ma.average(ssta_b[:,:,(lon<=290)&(lon>=250)],axis=(1,2))
    #box-c:125E-145E(68-78),10S-20N(40-55)
    ssta_c=SSTA[:,(lat<=20)&(lat>=-10),:]
    SSTA_c=np.ma.average(ssta_c[:,:,(lon<=145)&(lon>=125)],axis=(1,2))
    
    #calculat EMI
    EMI=SSTA_a-0.5*SSTA_b-0.5*SSTA_c
    
    return EMI

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
    PC12=solver12.pcs(npcs=1,pcscaling=1)
    PC12=PC12[:,0]
    a=eof12[:,(lat1<=5)&(lat1>=-5),:]
    if np.mean(a[:,:,(lon1<=240)&(lon1>=190)].squeeze(),(0,1))<0:
        PC12=-PC12
        
    solver4=Eof(SSTA_TP4)
    eof4=solver4.eofsAsCorrelation(neofs=1)
    PC4=solver4.pcs(npcs=1,pcscaling=1)
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
# #LMR ver2.0
# sst=np.load('/Users/lxy/Desktop/research/ENSO/data/LMR_SST_v2.npy')
# lat_lmr=np.load('/Users/lxy/Desktop/research/ENSO/data/LMR/LMR_lats_v2.npy')
# lon_lmr=np.load('/Users/lxy/Desktop/research/ENSO/data/LMR/LMR_lons_v2.npy')
# =============================================================================

#LMR ver2.1
filepath=Dataset('/Users/lxy/Desktop/research/ENSO/data/LMR/sst_LMRv2.1.nc')
sst=filepath.variables['sst']
lat=np.load('/Users/lxy/Desktop/research/ENSO/data/LMR/LMR_lats_v2.npy')
lon=np.load('/Users/lxy/Desktop/research/ENSO/data/LMR/LMR_lons_v2.npy')
year=np.arange(2001)
yr=year[(year>=842) & (year<=1999)]


SST=np.mean(sst,1)
SSTA=SSTA_cal(SST[(year>=842) & (year<=1999),:,:])
SSTA=maskanom(SSTA)

yr=yr[29:]

# =============================================================================
# #PHYDA
# filepath=Dataset('/Users/lxy/Desktop/research/ENSO/data/PHYDA/da_hydro_AprMar_r.1-2000_d.05-Jan-2018.nc')
# SST=filepath.variables['tas_mn']
# sst=np.ma.getdata(SST)
# lat=filepath.variables['lat']
# lat=np.ma.getdata(lat)
# lon=filepath.variables['lon']
# lon=np.ma.getdata(lon)
# year=np.arange(1,2001)
# yr=year[(year>=842) & (year<=1999)]
# 
# SSTA=SSTA_cal(SST[(year>=842) & (year<=1999),:,:])
# SSTA=maskanom(SSTA)
# 
# 
# yr=yr[29:]
# =============================================================================

# =============================================================================
# EMI method (Ashok et al. 2007) to find CP
# =============================================================================

emi=EMindex(SSTA[29:,:,:],lat,lon)
emi=pd.Series(emi)

Emi=emi-emi.rolling(window=30).mean()
EMI=(emi-emi.rolling(window=30).mean())/emi.rolling(window=30).std()
yrcp_emi=yr[EMI>1]

#plt.plot(yr,Emi)

MCA_cpemi=Emi[(EMI>1)&(yr>=950)&(yr<=1350)]
LIA_cpemi=Emi[(EMI>1)&(yr>=1400)&(yr<=1800)]
PRE_cpemi=Emi[(EMI>1)&(yr>=1900)&(yr<=1999)]





# =============================================================================
# Nino3_4 index (Yeh et al. 2009) to define CP&EP
# =============================================================================
nino_cp,nino_ep=Nino3_4index(SSTA[29:,:,:],lat,lon)
nino_cp=pd.Series(nino_cp)
nino_ep=pd.Series(nino_ep)

Nino_cp=nino_cp-nino_cp.rolling(window=30).mean()
Nino_ep=nino_ep-nino_ep.rolling(window=30).mean()

NINO_cp=(nino_cp-nino_cp.rolling(window=30).mean())/nino_cp.rolling(window=30).std()
NINO_ep=(nino_ep-nino_ep.rolling(window=30).mean())/nino_ep.rolling(window=30).std()

yrcp_ni=yr[(NINO_cp>1) & (NINO_cp>NINO_ep) & (NINO_cp+NINO_ep>1)]
yrep_ni=yr[(NINO_ep>1) & (NINO_ep>NINO_cp) & (NINO_cp+NINO_ep>1)]


MCA_cpni=Nino_cp[(NINO_cp>1) & (NINO_cp>NINO_ep) & (NINO_cp+NINO_ep>1)&(yr>=950)&(yr<=1350)]
LIA_cpni=Nino_cp[(NINO_cp>1) & (NINO_cp>NINO_ep) & (NINO_cp+NINO_ep>1)&(yr>=1400)&(yr<=1800)]
PRE_cpni=Nino_cp[(NINO_cp>1) & (NINO_cp>NINO_ep) & (NINO_cp+NINO_ep>1)&(yr>=1900)&(yr<=1999)]

MCA_epni=Nino_ep[(NINO_ep>1) & (NINO_ep>NINO_cp) & (NINO_cp+NINO_ep>1)&(yr>=950)&(yr<=1350)]
LIA_epni=Nino_ep[(NINO_ep>1) & (NINO_ep>NINO_cp) & (NINO_cp+NINO_ep>1)&(yr>=1400)&(yr<=1800)]
PRE_epni=Nino_ep[(NINO_ep>1) & (NINO_ep>NINO_cp) & (NINO_cp+NINO_ep>1)&(yr>=1900)&(yr<=1999)]

mca_cpni=Nino_cp[(NINO_cp>1) & (NINO_cp>NINO_ep) & (NINO_cp+NINO_ep>1)&(yr>=1100)&(yr<1200)]
lia_cpni=Nino_cp[(NINO_cp>1) & (NINO_cp>NINO_ep) & (NINO_cp+NINO_ep>1)&(yr>=1550)&(yr<1650)]

mca_epni=Nino_ep[(NINO_ep>1) & (NINO_ep>NINO_cp) & (NINO_cp+NINO_ep>1)&(yr>=1100)&(yr<1200)]
lia_epni=Nino_ep[(NINO_ep>1) & (NINO_ep>NINO_cp) & (NINO_cp+NINO_ep>1)&(yr>=1550)&(yr<1650)]


# =============================================================================
# EP_CP index (Kao and Yu 2009) to find CP&EP
# =============================================================================
cpindex,epindex=EP_CPindex(SSTA[29:,:,:],lat,lon)
cpindex=pd.Series(cpindex)
epindex=pd.Series(epindex)

Cpindex=cpindex-cpindex.rolling(window=30).mean()
Epindex=epindex-epindex.rolling(window=30).mean()

CPindex=(cpindex-cpindex.rolling(window=30).mean())/cpindex.rolling(window=30).std()
EPindex=(epindex-epindex.rolling(window=30).mean())/epindex.rolling(window=30).std()

yrcp_p=yr[(CPindex>1) & (CPindex>EPindex) & (CPindex+EPindex>1)]
yrep_p=yr[(EPindex>1) & (EPindex>CPindex) & (CPindex+EPindex>1)]


MCA_cpp=Cpindex[(CPindex>1) & (CPindex>EPindex) & (CPindex+EPindex>1)&(yr>=950)&(yr<=1350)]
LIA_cpp=Cpindex[(CPindex>1) & (CPindex>EPindex) & (CPindex+EPindex>1)&(yr>=1400)&(yr<=1800)]
PRE_cpp=Cpindex[(CPindex>1) & (CPindex>EPindex) & (CPindex+EPindex>1)&(yr>=1900)&(yr<=1999)]

MCA_epp=Epindex[(EPindex>1) & (EPindex>CPindex) & (CPindex+EPindex>1)&(yr>=950)&(yr<=1350)]
LIA_epp=Epindex[(EPindex>1) & (EPindex>CPindex) & (CPindex+EPindex>1)&(yr>=1400)&(yr<=1800)]
PRE_epp=Epindex[(EPindex>1) & (EPindex>CPindex) & (CPindex+EPindex>1)&(yr>=1900)&(yr<=1999)]

mca_cpp=Cpindex[(CPindex>1) & (CPindex>EPindex) & (CPindex+EPindex>1)&(yr>=1100)&(yr<1200)]
lia_cpp=Cpindex[(CPindex>1) & (CPindex>EPindex) & (CPindex+EPindex>1)&(yr>=1550)&(yr<1650)]

mca_epp=Epindex[(EPindex>1) & (EPindex>CPindex) & (CPindex+EPindex>1)&(yr>=1100)&(yr<1200)]
lia_epp=Epindex[(EPindex>1) & (EPindex>CPindex) & (CPindex+EPindex>1)&(yr>=1550)&(yr<1650)]


# =============================================================================
# E_Cindex (Takahashi et al. 2011) to find CP&EP
# =============================================================================
cindex,eindex=E_Cindex(SSTA,lat,lon)
cindex=pd.Series(cindex)
eindex=pd.Series(eindex)

Cin=cindex-cindex.rolling(window=30).mean()
Ein=eindex-eindex.rolling(window=30).mean()

Cindex=(cindex-cindex.rolling(window=30).mean())/cindex.rolling(window=30).std()
Eindex=(eindex-eindex.rolling(window=30).mean())/eindex.rolling(window=30).std()

yrcp_ce=yr[(Cindex>1) & (Cindex>Eindex) & (Cindex+Eindex>1)]
yrep_ce=yr[(Eindex>1) & (Eindex>Cindex) & (Cindex+Eindex>1)]

# =============================================================================
# MCA_cpce=Cindex[(Cindex>1) & (Cindex>Eindex) & (Cindex+Eindex>1)&(yr>=950)&(yr<=1350)]
# LIA_cpce=Cindex[(Cindex>1) & (Cindex>Eindex) & (Cindex+Eindex>1)&(yr>=1400)&(yr<=1800)]
# PRE_cpce=Cindex[(Cindex>1) & (Cindex>Eindex) & (Cindex+Eindex>1)&(yr>=1900)&(yr<=1999)]
# 
# MCA_epce=Eindex[(Eindex>1) & (Eindex>Cindex) & (Cindex+Eindex>1)&(yr>=950)&(yr<=1350)]
# LIA_epce=Eindex[(Eindex>1) & (Eindex>Cindex) & (Cindex+Eindex>1)&(yr>=1400)&(yr<=1800)]
# PRE_epce=Eindex[(Eindex>1) & (Eindex>Cindex) & (Cindex+Eindex>1)&(yr>=1900)&(yr<=1999)]
# 
# mca_cpce=Cindex[(Cindex>1) & (Cindex>Eindex) & (Cindex+Eindex>1)&(yr>=1100)&(yr<1200)]
# lia_cpce=Cindex[(Cindex>1) & (Cindex>Eindex) & (Cindex+Eindex>1)&(yr>=1550)&(yr<1650)]
# 
# mca_epce=Eindex[(Eindex>1) & (Eindex>Cindex) & (Cindex+Eindex>1)&(yr>=1100)&(yr<1200)]
# lia_epce=Eindex[(Eindex>1) & (Eindex>Cindex) & (Cindex+Eindex>1)&(yr>=1550)&(yr<1650)]
# =============================================================================


MCA_cpce=Cin[(Cindex>1) & (Cindex>Eindex) & (Cindex+Eindex>1)&(yr>=950)&(yr<=1350)]
LIA_cpce=Cin[(Cindex>1) & (Cindex>Eindex) & (Cindex+Eindex>1)&(yr>=1400)&(yr<=1800)]
PRE_cpce=Cin[(Cindex>1) & (Cindex>Eindex) & (Cindex+Eindex>1)&(yr>=1900)&(yr<=1999)]

MCA_epce=Ein[(Eindex>1) & (Eindex>Cindex) & (Cindex+Eindex>1)&(yr>=950)&(yr<=1350)]
LIA_epce=Ein[(Eindex>1) & (Eindex>Cindex) & (Cindex+Eindex>1)&(yr>=1400)&(yr<=1800)]
PRE_epce=Ein[(Eindex>1) & (Eindex>Cindex) & (Cindex+Eindex>1)&(yr>=1900)&(yr<=1999)]

mca_cpce=Cin[(Cindex>1) & (Cindex>Eindex) & (Cindex+Eindex>1)&(yr>=1100)&(yr<1200)]
lia_cpce=Cin[(Cindex>1) & (Cindex>Eindex) & (Cindex+Eindex>1)&(yr>=1550)&(yr<1650)]

mca_epce=Ein[(Eindex>1) & (Eindex>Cindex) & (Cindex+Eindex>1)&(yr>=1100)&(yr<1200)]
lia_epce=Ein[(Eindex>1) & (Eindex>Cindex) & (Cindex+Eindex>1)&(yr>=1550)&(yr<1650)]





# =============================================================================
# plot PDF
# =============================================================================
sns.set(style="whitegrid")

fig=plt.figure(figsize=(8,16))

#ax1=plt.subplot2grid((5, 2), (0, 0), rowspan=2)
ax1=plt.subplot(4, 2, 1)

plt.text(0.05, 0.95, 'a', transform=ax1.transAxes,
      fontsize=12, fontweight='bold', va='top')

sns.distplot(MCA_cpemi,color="crimson", hist=False, kde = True,
             kde_kws = {'shade': True, 'linewidth': 1}, label = 'MCA')

plt.axvline(x=np.mean(MCA_cpemi), ls='--', color="crimson",label = 'MCA mean')

sns.distplot(LIA_cpemi,color="royalblue", hist=False, kde = True,
             kde_kws = {'shade': True, 'linewidth': 1}, label="LIA")

plt.axvline(x=np.mean(LIA_cpemi), ls='--', color="royalblue", label="LIA mean")

sns.distplot(PRE_cpemi,color="green", hist=False, kde = True,
             kde_kws = {'shade': True, 'linewidth': 1}, label = "20th Century")

plt.axvline(x=np.mean(PRE_cpemi), ls='--', color="green", label = "20th Century mean")

plt.legend(bbox_to_anchor=(1.3, 0.5), loc='center left',fontsize='medium')#
plt.xlabel('EMI anomaly spread')
bot,up=plt.ylim()
plt.title('EMI method',fontsize=15)

#plt.yticks(np.arange(0,up,1))
plt.ylabel('Density')



ax2=plt.subplot(4, 2, 3)
plt.text(0.05, 0.95, 'b', transform=ax2.transAxes,
      fontsize=12, fontweight='bold', va='top')
plt.text(0.75, 1.1,'Nino3-4 index method',transform=ax2.transAxes,fontsize=15)


sns.distplot(MCA_cpni,color="crimson", hist=False, kde = True,
             kde_kws = {'shade': True, 'linewidth': 1})

plt.axvline(x=np.mean(MCA_cpni),ls='--', color="crimson")

sns.distplot(LIA_cpni,color="royalblue", hist=False, kde = True,
             kde_kws = {'shade': True, 'linewidth': 1})

plt.axvline(x=np.mean(LIA_cpni), ls='--', color="royalblue")

sns.distplot(PRE_cpni,color="green", hist=False, kde = True,
             kde_kws = {'shade': True, 'linewidth': 1})

plt.axvline(x=np.mean(PRE_cpni),ls='--', color="green")

#plt.legend(loc='upper right',fontsize='x-small')
plt.xlabel('Nino4 index anoamly spread')
#plt.xticks(np.arange(0,2.5,0.5))
bot,up=plt.ylim()
plt.xlim([-0.5,2])
plt.ylim([0,4.4])
plt.ylabel('Density')


ax3=plt.subplot(4,2,4)
plt.text(0.05, 0.95, 'c', transform=ax3.transAxes,
      fontsize=12, fontweight='bold', va='top')

sns.distplot(MCA_epni,color="crimson", hist=False, kde = True,
             kde_kws = {'shade': True, 'linewidth': 1})

plt.axvline(x=np.mean(MCA_epni),ls='--', color="crimson")

sns.distplot(LIA_epni,color="royalblue", hist=False, kde = True,
             kde_kws = {'shade': True, 'linewidth': 1})

plt.axvline(x=np.mean(LIA_epni), ls='--', color="royalblue")

sns.distplot(PRE_epni,color="green", hist=False, kde = True,
             kde_kws = {'shade': True, 'linewidth': 1})

plt.axvline(x=np.mean(PRE_epni),ls='--', color="green")
#plt.legend(loc='upper right',fontsize='x-small')
plt.xlabel('Nino3 index anoamly spread')
plt.xlim([-0.5,2])
plt.ylim([0,4.4])

#fig.savefig('/Users/lxy/Desktop/research/ENSO/cp_ep/frequency/Nino3_4.png',dpi=600)

ax4=plt.subplot(4, 2, 5)
plt.text(0.05, 0.95, 'd', transform=ax4.transAxes,
      fontsize=12, fontweight='bold', va='top')
plt.text(0.77, 1.1,'CP-EP index method',transform=ax4.transAxes,fontsize=15)

sns.distplot(MCA_cpp,color="crimson", hist=False, kde = True,
             kde_kws = {'shade': True, 'linewidth': 1})

plt.axvline(x=np.mean(MCA_cpp),ls='--', color="crimson")

sns.distplot(LIA_cpp,color="royalblue", hist=False, kde = True,
             kde_kws = {'shade': True, 'linewidth': 1})

plt.axvline(x=np.mean(LIA_cpp), ls='--', color="royalblue")

sns.distplot(PRE_cpp,color="green", hist=False, kde = True,
             kde_kws = {'shade': True, 'linewidth': 1})

plt.axvline(x=np.mean(PRE_cpp),ls='--', color="green")

plt.xlabel('CPindex anoamly spread')
#plt.xticks(np.arange(0,2.5,0.5))
#bot,up=plt.ylim()
plt.ylim([0,1.2])
plt.xlim([-1, 6])
#plt.yticks(np.arange(0,up,1))
plt.ylabel('Density')


ax5=plt.subplot(4, 2, 6)
plt.text(0.05, 0.95, 'e', transform=ax5.transAxes,
      fontsize=12, fontweight='bold', va='top')

sns.distplot(MCA_epp,color="crimson", hist=False, kde = True,
             kde_kws = {'shade': True, 'linewidth': 1})

plt.axvline(x=np.mean(MCA_epp),ls='--', color="crimson")

sns.distplot(LIA_epp,color="royalblue", hist=False, kde = True,
             kde_kws = {'shade': True, 'linewidth': 1})

plt.axvline(x=np.mean(LIA_epp), ls='--', color="royalblue")

sns.distplot(PRE_epp,color="green", hist=False, kde = True,
             kde_kws = {'shade': True, 'linewidth': 1})

plt.axvline(x=np.mean(PRE_epp),ls='--', color="green")
plt.xlabel('EPindex anoamly spread')
#plt.xticks(np.arange(0,2.5,0.5))
plt.ylim([0,1.2])
plt.xlim([-1, 6])



ax6=plt.subplot(4, 2, 7)
plt.text(0.05, 0.95, 'f', transform=ax6.transAxes,
      fontsize=12, fontweight='bold', va='top')
plt.text(0.73, 1.1,'C and E index method',transform=ax6.transAxes,fontsize=15)

sns.distplot(MCA_cpce,color="crimson", hist=False, kde = True,
             kde_kws = {'shade': True, 'linewidth': 1})

plt.axvline(x=np.mean(MCA_cpce),ls='--', color="crimson")

sns.distplot(LIA_cpce,color="royalblue", hist=False, kde = True,
             kde_kws = {'shade': True, 'linewidth': 1})

plt.axvline(x=np.mean(LIA_cpce), ls='--', color="royalblue")

sns.distplot(PRE_cpce,color="green", hist=False, kde = True,
             kde_kws = {'shade': True, 'linewidth': 1})

plt.axvline(x=np.mean(PRE_cpce),ls='--', color="green")

plt.xlabel('Cindex anoamly spread')
#plt.xticks(np.arange(0,2.5,0.5))
bot,up=plt.ylim()
plt.ylim([0,1])
plt.xlim([-1,7])
#plt.xlim([-0.5,4.2])

#plt.yticks(np.arange(0,up,1))
plt.ylabel('Density')



ax7=plt.subplot(4, 2, 8)
plt.text(0.05, 0.95, 'g', transform=ax7.transAxes,
      fontsize=12, fontweight='bold', va='top')

sns.distplot(MCA_epce,color="crimson", hist=False, kde = True,
             kde_kws = {'shade': True, 'linewidth': 1})

plt.axvline(x=np.mean(MCA_epce),ls='--', color="crimson")

sns.distplot(LIA_epce,color="royalblue", hist=False, kde = True,
             kde_kws = {'shade': True, 'linewidth': 1})

plt.axvline(x=np.mean(LIA_epce), ls='--', color="royalblue")

sns.distplot(PRE_epce,color="green", hist=False, kde = True,
             kde_kws = {'shade': True, 'linewidth': 1})

plt.axvline(x=np.mean(PRE_epce),ls='--', color="green")

plt.xlabel('Eindex anoamly spread')
#plt.xticks(np.arange(0,2.5,0.5))
plt.ylim([0,1])
plt.xlim([-1,7])

plt.subplots_adjust(hspace=0.5,wspace=0.3)
#plt.tight_layout()

#plt.suptitle('C_Eindex (Takahashi et al. 2011)',fontsize=13)

#fig.savefig('/Users/lxy/Desktop/research/ENSO/cp_ep/frequency/anomaly/amplitude_lmr1.png',dpi=600)



# =============================================================================
# # =============================================================================
# # find cp&ep during 1617-1920
# # =============================================================================
# yrsp_ce=np.zeros(6).reshape(3,2)
# #Nino3_4 index
# yrsp_ce[0,0],=yrnino_cp[(yrnino_cp<1920) & (yrnino_cp>1617)].shape
# yrsp_ce[0,1],=yrnino_ep[(yrnino_ep<1920) & (yrnino_ep>1617)].shape
# #EP_CP index
# yrsp_ce[1,0],=yr_cp[(yr_cp<1920) & (yr_cp>1617)].shape
# yrsp_ce[1,1],=yr_ep[(yr_ep<1920) & (yr_ep>1617)].shape
# #E_C index
# yrsp_ce[2,0],=yrec_cp[(yrec_cp<1920) & (yrec_cp>1617)].shape
# yrsp_ce[2,1],=yrec_ep[(yrec_ep<1920) & (yrec_ep>1617)].shape
# =============================================================================




# =============================================================================

# =============================================================================
# number of cp&ep&cp/ep
# =============================================================================
# =============================================================================
# n=21
# ncp_ep=np.zeros((7,n))
# #ncp_ni=np.zeros(n)
# #nep_ni=np.zeros(n)
# #ncp_p=np.zeros(n)
# #nep_p=np.zeros(n)
# #ncp_ce=np.zeros(n)
# #nep_ce=np.zeros(n)
# 
# 
# for i in range(0,n):
#     ncp_ep[0,i],=yrcp_emi[(yrcp_emi<=(i+1)*50+950) & (yrcp_emi>i*50+950)].shape #EMI
#     ncp_ep[1,i],=yrcp_ni[(yrcp_ni<=(i+1)*50+950) & (yrcp_ni>i*50+950)].shape #Nino4
#     ncp_ep[2,i],=yrep_ni[(yrep_ni<=(i+1)*50+950) & (yrep_ni>i*50+950)].shape # Nino3
#     ncp_ep[3,i],=yrcp_p[(yrcp_p<=(i+1)*50+950) & (yrcp_p>i*50+950)].shape #CPindex
#     ncp_ep[4,i],=yrep_p[(yrep_p<=(i+1)*50+950) & (yrep_p>i*50+950)].shape # EPindex
#     ncp_ep[5,i],=yrcp_ce[(yrcp_ce<=(i+1)*50+950) & (yrcp_ce>i*50+950)].shape#Cindex
#     ncp_ep[6,i],=yrep_ce[(yrep_ce<=(i+1)*50+950) & (yrep_ce>i*50+950)].shape# Eindex
# 
# =============================================================================

#np.save('/Users/lxy/Desktop/research/ENSO/cp_ep/frequency/data/lmr_cpep.npy',ncp_ep)



'''
step plot of frequency change (per 50 years)

'''
ncpep_lmr=np.load('/Users/lxy/Desktop/research/ENSO/cp_ep/frequency/data/lmr_cpep.npy')
ncpep_phyda=np.load('/Users/lxy/Desktop/research/ENSO/cp_ep/frequency/data/phyda_cpep.npy')

# =============================================================================
# CP/EP ratio
# =============================================================================

threshold=1
cp_epni_lmr=ncpep_lmr[1,:]/ncpep_lmr[2,:]
cp_epp_lmr=ncpep_lmr[3,:]/ncpep_lmr[4,:]
cp_epce_lmr=ncpep_lmr[5,:]/ncpep_lmr[6,:]
a=(cp_epni_lmr+cp_epp_lmr+cp_epce_lmr)/3

cp_epni_phyda=ncpep_phyda[1,:]/ncpep_phyda[2,:]
cp_epp_phyda=ncpep_phyda[3,:]/ncpep_phyda[4,:]
cp_epce_phyda=ncpep_phyda[5,:]/ncpep_phyda[6,:]
b=(cp_epni_phyda+cp_epp_phyda+cp_epce_phyda)/3

lmr=np.zeros(21)
phyda=np.zeros(21)

#lmr[cp_epni_lmr>threshold]=1
#lmr[cp_epp_lmr>threshold]=1
#lmr[cp_epce_lmr>threshold]=1
lmr[a>threshold]=1

#phyda[cp_epni_phyda>threshold]=1
#phyda[cp_epp_phyda>threshold]=1
#phyda[cp_epce_phyda>threshold]=1
phyda[b>threshold]=1

# ====================================================================x=========
# plot--2
# =============================================================================
sns.set(style="ticks")



fig,(ax1, ax2, ax3, ax4, ax5, ax6)=plt.subplots(6,1,sharex=True,figsize=(10,10))

#ax1=plt.subplot(6, 1, 1)
ax1.text(0.02, 0.95, 'a', transform=ax1.transAxes,
      fontsize=12, fontweight='bold', va='top')

ax1.step(year[950:2000:50],ncpep_lmr[0,:],'grey', where='mid', label='EMI')
ax1.plot(year[950:2000:50],ncpep_lmr[0,:], 'o', color='grey', alpha=0.3)

ax1.step(year[950:2000:50],ncpep_lmr[1,:],'tab:blue', where='mid',label='Nino3_4 index')
ax1.plot(year[950:2000:50],ncpep_lmr[1,:], 'o', color='tab:blue', alpha=0.3)

ax1.step(year[950:2000:50],ncpep_lmr[3,:],'orange', where='mid',label='CP_EP index')
ax1.plot(year[950:2000:50],ncpep_lmr[3,:], 'o', color='orange', alpha=0.3)

ax1.step(year[950:2000:50],ncpep_lmr[5,:],'darkseagreen', where='mid', label='C_E index')
ax1.plot(year[950:2000:50],ncpep_lmr[5,:], 'o', color='darkseagreen', alpha=0.3)
ax1.legend(loc='upper right',fontsize='small',ncol=4)
ax1.tick_params(direction='in')
ax1.set_ylabel('LMR CP\n(number/50 years)',fontsize=8)
#plt.title('CP')
ax1.set_ylim([-0.5,20])

#ax2=plt.subplot(6, 1, 2,sharex = ax1)
ax2.text(0.02, 0.95, 'b', transform=ax2.transAxes,
      fontsize=12, fontweight='bold', va='top')

ax2.step(year[950:2000:50],ncpep_phyda[0,:],'grey', where='mid')
ax2.plot(year[950:2000:50],ncpep_phyda[0,:], 'o', color='grey', alpha=0.3)

ax2.step(year[950:2000:50],ncpep_phyda[1,:],'tab:blue', where='mid')
ax2.plot(year[950:2000:50],ncpep_phyda[1,:], 'o', color='tab:blue', alpha=0.3)

ax2.step(year[950:2000:50],ncpep_phyda[3,:],'orange', where='mid')
ax2.plot(year[950:2000:50],ncpep_phyda[3,:], 'o', color='orange', alpha=0.3)

ax2.step(year[950:2000:50],ncpep_phyda[5,:],'darkseagreen', where='mid')
ax2.plot(year[950:2000:50],ncpep_phyda[5,:], 'o', color='darkseagreen', alpha=0.3)

ax2.tick_params(direction='in')

#plt.legend(loc='upper right',fontsize='small',ncol=3)
ax2.set_ylabel('PHYDA CP\n(number/50 years)',fontsize=8)
#plt.title('EP')
ax2.set_ylim([-1,20])
ax2.set_yticks(np.arange(0,20,5))


#ax3=plt.subplot(6, 1, 3)
ax3.text(0.02, 0.95, 'c', transform=ax3.transAxes,
      fontsize=12, fontweight='bold', va='top')

ax3.step(year[950:2000:50],ncpep_lmr[2,:],'tab:blue', where='mid')
ax3.plot(year[950:2000:50],ncpep_lmr[2,:], 'o', color='tab:blue', alpha=0.3)

ax3.step(year[950:2000:50],ncpep_lmr[4,:],'orange', where='mid')
ax3.plot(year[950:2000:50],ncpep_lmr[4,:], 'o', color='orange', alpha=0.3)

ax3.step(year[950:2000:50],ncpep_lmr[6,:],'darkseagreen', where='mid')
ax3.plot(year[950:2000:50],ncpep_lmr[6,:], 'o', color='darkseagreen', alpha=0.3)

ax3.tick_params(direction='in')
#plt.legend(loc='upper right',fontsize='small',ncol=3)
ax3.set_ylabel('LMR EP\n(number/50 years)',fontsize=8)
#plt.title('CP/EP')
ax3.set_ylim([-1,15])
ax3.set_yticks(np.arange(0,15,5))


#ax4=plt.subplot(6, 1, 4)
ax4.text(0.02, 0.95, 'd', transform=ax4.transAxes,
      fontsize=12, fontweight='bold', va='top')

ax4.step(year[950:2000:50],ncpep_phyda[2,:],'tab:blue', where='mid')
ax4.plot(year[950:2000:50],ncpep_phyda[2,:], 'o', color='tab:blue', alpha=0.3)

ax4.step(year[950:2000:50],ncpep_phyda[4,:],'orange', where='mid')
ax4.plot(year[950:2000:50],ncpep_phyda[4,:], 'o', color='orange', alpha=0.3)

ax4.step(year[950:2000:50],ncpep_phyda[6,:],'darkseagreen', where='mid')
ax4.plot(year[950:2000:50],ncpep_phyda[6,:], 'o', color='darkseagreen', alpha=0.3)

ax4.tick_params(direction='in')

#plt.legend(loc='upper right',fontsize='small',ncol=3)
ax4.set_ylabel('PHYDA EP\n(number/50 years)',fontsize=8)
#plt.title('CP/EP')
ax4.set_ylim([-1,15])
ax4.set_yticks(np.arange(0,15,5))




#ax5=plt.subplot(6, 1, 5)
ax5.text(0.02, 0.95, 'e', transform=ax5.transAxes,
      fontsize=12, fontweight='bold', va='top')

ax5.step(year[950:2000:50],cp_epni_lmr,'tab:blue', where='mid')
ax5.plot(year[950:2000:50],cp_epni_lmr, 'o', color='tab:blue', alpha=0.3)

ax5.step(year[950:2000:50],cp_epp_lmr,'orange', where='mid')
ax5.plot(year[950:2000:50],cp_epp_lmr, 'o', color='orange', alpha=0.3)

ax5.step(year[950:2000:50],cp_epce_lmr,'darkseagreen', where='mid')
ax5.plot(year[950:2000:50],cp_epce_lmr, 'o', color='darkseagreen', alpha=0.3)

#import matplotlib.transforms as mtransforms
#trans = mtransforms.blended_transform_factory(ax5.transData, ax5.transAxes)
ax5.fill_between(year[950:2000:50], 0, 1, where= lmr==1, step='mid',
                 color='grey', alpha=0.5, transform=ax5.get_xaxis_transform())
#ax5.fill_between(year[950:2000:50], 0, 1, where= cp_epp_lmr>1.5, step='mid',
#                 color='mistyrose', alpha=0.5, transform=ax5.get_xaxis_transform())
#ax5.fill_between(year[950:2000:50], 0, 1, where= cp_epce_lmr>1.5,step='mid',
#                 color='mistyrose', alpha=0.5, transform=ax5.get_xaxis_transform())
#plt.legend(loc='upper right',fontsize='small',ncol=4)
ax5.tick_params(direction='in')
ax5.set_ylabel('LMR CP/EP\n(number/50 years)',fontsize=8)
#plt.title('CP')
ax5.set_ylim([-1,10])
ax5.set_yticks(np.arange(0,10,2))


#ax6=plt.subplot(6, 1, 6)
ax6.text(0.02, 0.95, 'f', transform=ax6.transAxes,
      fontsize=12, fontweight='bold', va='top')

ax6.step(year[950:2000:50],cp_epni_phyda,'tab:blue', where='mid')
ax6.plot(year[950:2000:50],cp_epni_phyda, 'o', color='tab:blue', alpha=0.3)

ax6.step(year[950:2000:50],cp_epp_phyda,'orange', where='mid')
ax6.plot(year[950:2000:50],cp_epp_phyda, 'o', color='orange', alpha=0.3)

ax6.step(year[950:2000:50],cp_epce_phyda,'darkseagreen', where='mid')
ax6.plot(year[950:2000:50],cp_epce_phyda, 'o', color='darkseagreen', alpha=0.3)

#trans = mtransforms.blended_transform_factory(ax6.transData, ax6.transAxes)
ax6.fill_between(year[950:2000:50], 0, 1, where= phyda==1, step='mid',
                 color='grey', alpha=0.5,transform=ax6.get_xaxis_transform())
#ax6.fill_between(year[950:2000:50], 0, 1, where= cp_epp_phyda>1.5, step='mid',
#                 color='coral', alpha=0.5,transform=ax6.get_xaxis_transform())
#ax6.fill_between(year[950:2000:50], 0, 1, where= cp_epce_phyda>1.5, step='mid',
#                 color='grey', alpha=0.5,transform=ax6.get_xaxis_transform())

#plt.legend(loc='upper right',fontsize='small',ncol=4), transform=ax6.get_xaxis_transform()
ax6.set_ylabel('LMR CP/EP\n(number/50 years)',fontsize=8)
#plt.title('CP')
ax6.set_ylim([-0.5,10])
ax6.set_xlabel('Year')

ax6.tick_params(direction='in')
ax6.set_yticks(np.arange(0,10,2))

plt.tight_layout()
plt.subplots_adjust(hspace=0)

#fig.savefig('/Users/lxy/Desktop/research/ENSO/cp_ep/frequency/frequencynew.png',dpi=600)

# =============================================================================
# significant test of distribution (MCA, LIA and 20th century)
# =============================================================================

# =============================================================================
# EMI method (Ashok et al. 2007) to find CP
# =============================================================================

tML_emi, pML_emi = stats.ttest_ind(MCA_cpemi,LIA_cpemi,equal_var=False)
tMP_emi, pMP_emi = stats.ttest_ind(MCA_cpemi,PRE_cpemi,equal_var=False)
tLP_emi, pLP_emi = stats.ttest_ind(LIA_cpemi,PRE_cpemi,equal_var=False)


# =============================================================================
# Nino3_4 index (Yeh et al. 2009) to define CP&EP
# =============================================================================


tML_cpni, pML_cpni = stats.ttest_ind(MCA_cpni,LIA_cpni,equal_var=False)
tMP_cpni, pMP_cpni = stats.ttest_ind(MCA_cpni,PRE_cpni,equal_var=False)
tLP_cpni, pLP_cpni = stats.ttest_ind(LIA_cpni,PRE_cpni,equal_var=False)
tml_cpni, pml_cpni = stats.ttest_ind(mca_cpni,lia_cpni,equal_var=False)
tmp_cpni, pmp_cpni = stats.ttest_ind(mca_cpni,PRE_cpni,equal_var=False)
tlp_cpni, plp_cpni = stats.ttest_ind(lia_cpni,PRE_cpni,equal_var=False)

tML_epni, pML_epni = stats.ttest_ind(MCA_epni,LIA_epni,equal_var=False)
tMP_epni, pMP_epni = stats.ttest_ind(MCA_cpni,PRE_cpni,equal_var=False)
tLP_epni, pLP_epni = stats.ttest_ind(LIA_cpni,PRE_cpni,equal_var=False)
tml_epni, pml_epni = stats.ttest_ind(mca_epni,lia_epni,equal_var=False)
tmp_epni, pmp_epni = stats.ttest_ind(mca_epni,PRE_epni,equal_var=False)
tlp_epni, plp_epni = stats.ttest_ind(lia_epni,PRE_epni,equal_var=False)


# =============================================================================
# EP_CP index (Kao and Yu 2009) to find CP&EP
# =============================================================================

tML_cpp, pML_cpp = stats.ttest_ind(MCA_cpp,LIA_cpp,equal_var=False)
tMP_cpp, pMP_cpp = stats.ttest_ind(MCA_cpp,PRE_cpp,equal_var=False)
tLP_cpp, pLP_cpp = stats.ttest_ind(LIA_cpp,PRE_cpp,equal_var=False)
tml_cpp, pml_cpp = stats.ttest_ind(mca_cpp,lia_cpp,equal_var=False)
tmp_cpp, pmp_cpp = stats.ttest_ind(mca_cpp,PRE_cpp,equal_var=False)
tlp_cpp, plp_cpp = stats.ttest_ind(lia_cpp,PRE_cpp,equal_var=False)

tML_epp, pML_epp = stats.ttest_ind(MCA_epp,LIA_epp,equal_var=False)
tMP_epp, pMP_epp = stats.ttest_ind(MCA_epp,PRE_epp,equal_var=False)
tLP_epp, pLP_epp = stats.ttest_ind(LIA_epp,PRE_epp,equal_var=False)
tml_epp, pml_epp = stats.ttest_ind(mca_epp,lia_epp,equal_var=False)
tmp_epp, pmp_epp = stats.ttest_ind(mca_epp,PRE_epp,equal_var=False)
tlp_epp, plp_epp = stats.ttest_ind(lia_epp,PRE_epp,equal_var=False)


# =============================================================================
# E_Cindex (Takahashi et al. 2011) to find CP&EP
# =============================================================================


tML_cpce, pML_cpce = stats.ttest_ind(MCA_cpce,LIA_cpce,equal_var=False)
tMP_cpce, pMP_cpce = stats.ttest_ind(MCA_cpce,PRE_cpce,equal_var=False)
tLP_cpce, pLP_cpce = stats.ttest_ind(LIA_cpce,PRE_cpce,equal_var=False)
tml_cpce, pml_cpce = stats.ttest_ind(mca_cpce,lia_cpce,equal_var=False)
tmp_cpce, pmp_cpce = stats.ttest_ind(mca_cpce,PRE_cpce,equal_var=False)
tlp_cpce, plp_cpce = stats.ttest_ind(lia_cpce,PRE_cpce,equal_var=False)

tML_epce, pML_epce = stats.ttest_ind(MCA_epce,LIA_epce,equal_var=False)
tMP_epce, pMP_epce = stats.ttest_ind(MCA_epce,PRE_epce,equal_var=False)
tLP_epce, pLP_epce = stats.ttest_ind(LIA_epce,PRE_epce,equal_var=False)
tml_epce, pml_epce = stats.ttest_ind(mca_epce,lia_epce,equal_var=False)
tmp_epce, pmp_epce = stats.ttest_ind(mca_epce,PRE_epce,equal_var=False)
tlp_epce, plp_epce = stats.ttest_ind(lia_epce,PRE_epce,equal_var=False)


# =============================================================================
# number of different period
# =============================================================================

#EMI
yrMCA_cpemi=yr[(EMI>1)&(yr>=950)&(yr<1350)]
yrLIA_cpemi=yr[(EMI>1)&(yr>=1400)&(yr<1800)]
yrPRE_cpemi=yr[(EMI>1)&(yr>=1900)&(yr<=1999)]

#Nino3-4
yrMCA_cpni=yr[(NINO_cp>1) & (NINO_cp>NINO_ep) & (NINO_cp+NINO_ep>1)&(yr>=950)&(yr<1350)]
yrLIA_cpni=yr[(NINO_cp>1) & (NINO_cp>NINO_ep) & (NINO_cp+NINO_ep>1)&(yr>=1400)&(yr<1800)]
yrPRE_cpni=yr[(NINO_cp>1) & (NINO_cp>NINO_ep) & (NINO_cp+NINO_ep>1)&(yr>=1900)&(yr<=1999)]

yrMCA_epni=yr[(NINO_ep>1) & (NINO_ep>NINO_cp) & (NINO_cp+NINO_ep>1)&(yr>=950)&(yr<1350)]
yrLIA_epni=yr[(NINO_ep>1) & (NINO_ep>NINO_cp) & (NINO_cp+NINO_ep>1)&(yr>=1400)&(yr<1800)]
yrPRE_epni=yr[(NINO_ep>1) & (NINO_ep>NINO_cp) & (NINO_cp+NINO_ep>1)&(yr>=1900)&(yr<=1999)]

#CP-EP index
yrMCA_cpp=yr[(CPindex>1) & (CPindex>EPindex) & (CPindex+EPindex>1)&(yr>=950)&(yr<1350)]
yrLIA_cpp=yr[(CPindex>1) & (CPindex>EPindex) & (CPindex+EPindex>1)&(yr>=1400)&(yr<1800)]
yrPRE_cpp=yr[(CPindex>1) & (CPindex>EPindex) & (CPindex+EPindex>1)&(yr>=1900)&(yr<=1999)]

yrMCA_epp=yr[(EPindex>1) & (EPindex>CPindex) & (CPindex+EPindex>1)&(yr>=950)&(yr<1350)]
yrLIA_epp=yr[(EPindex>1) & (EPindex>CPindex) & (CPindex+EPindex>1)&(yr>=1400)&(yr<1800)]
yrPRE_epp=yr[(EPindex>1) & (EPindex>CPindex) & (CPindex+EPindex>1)&(yr>=1900)&(yr<=1999)]

#C-E index
yrMCA_cpce=yr[(Cindex>1) & (Cindex>Eindex) & (Cindex+Eindex>1)&(yr>=950)&(yr<1350)]
yrLIA_cpce=yr[(Cindex>1) & (Cindex>Eindex) & (Cindex+Eindex>1)&(yr>=1400)&(yr<1800)]
yrPRE_cpce=yr[(Cindex>1) & (Cindex>Eindex) & (Cindex+Eindex>1)&(yr>=1900)&(yr<=1999)]

yrMCA_epce=yr[(Eindex>1) & (Eindex>Cindex) & (Cindex+Eindex>1)&(yr>=950)&(yr<1350)]
yrLIA_epce=yr[(Eindex>1) & (Eindex>Cindex) & (Cindex+Eindex>1)&(yr>=1400)&(yr<1800)]
yrPRE_epce=yr[(Eindex>1) & (Eindex>Cindex) & (Cindex+Eindex>1)&(yr>=1900)&(yr<=1999)]



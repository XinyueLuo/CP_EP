#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  7 16:43:01 2020

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
# #LMR ver2.1
# filepath=Dataset('/Users/lxy/Desktop/research/ENSO/data/LMR/sst_LMRv2.1.nc')
# sst=filepath.variables['sst']
# lat=np.load('/Users/lxy/Desktop/research/ENSO/data/LMR/LMR_lats_v2.npy')
# lon=np.load('/Users/lxy/Desktop/research/ENSO/data/LMR/LMR_lons_v2.npy')
# year=np.arange(2001)
# yr=year[(year>=842) & (year<=1999)]
# 
# yr=yr[29:]
# EMI_fre=np.zeros((3,20)) #MCA, LIA, 20th century
# CPni_fre=np.zeros((3,20))
# EPni_fre=np.zeros((3,20))
# CPp_fre=np.zeros((3,20))
# EPp_fre=np.zeros((3,20))
# CPce_fre=np.zeros((3,20))
# EPce_fre=np.zeros((3,20))
# =============================================================================

#PHYDA
filepath=Dataset('/Users/lxy/Desktop/research/ENSO/data/PHYDA/phyda_ens_tas_AprMar_r.1-2000_d.21-Nov-2018.nc')
SST=filepath.variables['tas_ens']#100 members
sst=np.ma.getdata(SST)
lat=filepath.variables['lat']
lat=np.ma.getdata(lat)
lon=filepath.variables['lon']
lon=np.ma.getdata(lon)
year=np.arange(1,2001)
yr=year[(year>=842) & (year<=1999)]


yr=yr[29:]


EMI_fre=np.zeros((3,100)) #MCA, LIA, 20th century
CPni_fre=np.zeros((3,100))
EPni_fre=np.zeros((3,100))
CPp_fre=np.zeros((3,100))
EPp_fre=np.zeros((3,100))
CPce_fre=np.zeros((3,100))
EPce_fre=np.zeros((3,100))

#SST=np.mean(sst,1)
for i in range(0,100):
    SST=sst[i,:,:,:]
    SSTA=SSTA_cal(SST[(year>=842) & (year<=1999),:,:])
    SSTA=maskanom(SSTA)


# =============================================================================
# EMI method (Ashok et al. 2007) to find CP
# =============================================================================

    emi=EMindex(SSTA[29:,:,:],lat,lon)
    emi=pd.Series(emi)
    
    Emi=emi-emi.rolling(window=30).mean()
    EMI=(emi-emi.rolling(window=30).mean())/emi.rolling(window=30).std()
    yrcp_emi=yr[EMI>1]
    
    #EMI
    EMI_fre[0,i],=yrcp_emi[(yrcp_emi>=950)&(yrcp_emi<1350)].shape
    EMI_fre[1,i],=yrcp_emi[(yrcp_emi>=1400)&(yrcp_emi<1800)].shape
    EMI_fre[2,i],=yrcp_emi[(yrcp_emi>=1900)&(yrcp_emi<1999)].shape


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

    CPni_fre[0,i],=yrcp_ni[(yrcp_ni>=950)&(yrcp_ni<1350)].shape
    CPni_fre[1,i],=yrcp_ni[(yrcp_ni>=1400)&(yrcp_ni<1800)].shape
    CPni_fre[2,i],=yrcp_ni[(yrcp_ni>=1900)&(yrcp_ni<1999)].shape

    EPni_fre[0,i],=yrep_ni[(yrep_ni>=950)&(yrep_ni<1350)].shape
    EPni_fre[1,i],=yrep_ni[(yrep_ni>=1400)&(yrep_ni<1800)].shape
    EPni_fre[2,i],=yrep_ni[(yrep_ni>=1900)&(yrep_ni<1999)].shape



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

    CPp_fre[0,i],=yrcp_p[(yrcp_p>=950)&(yrcp_p<1350)].shape
    CPp_fre[1,i],=yrcp_p[(yrcp_p>=1400)&(yrcp_p<1800)].shape
    CPp_fre[2,i],=yrcp_p[(yrcp_p>=1900)&(yrcp_p<1999)].shape

    EPp_fre[0,i],=yrep_p[(yrep_p>=950)&(yrep_p<1350)].shape
    EPp_fre[1,i],=yrep_p[(yrep_p>=1400)&(yrep_p<1800)].shape
    EPp_fre[2,i],=yrep_p[(yrep_p>=1900)&(yrep_p<1999)].shape




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

    CPce_fre[0,i],=yrcp_ce[(yrcp_ce>=950)&(yrcp_ce<1350)].shape
    CPce_fre[1,i],=yrcp_ce[(yrcp_ce>=1400)&(yrcp_ce<1800)].shape
    CPce_fre[2,i],=yrcp_ce[(yrcp_ce>=1900)&(yrcp_ce<1999)].shape

    EPce_fre[0,i],=yrep_ce[(yrep_ce>=950)&(yrep_ce<1350)].shape
    EPce_fre[1,i],=yrep_ce[(yrep_ce>=1400)&(yrep_ce<1800)].shape
    EPce_fre[2,i],=yrep_ce[(yrep_ce>=1900)&(yrep_ce<1999)].shape


#np.save('/Users/lxy/Desktop/research/ENSO/cp_ep/frequency/Ensemble/LMR/EMI.npy',EMI_fre)
#np.save('/Users/lxy/Desktop/research/ENSO/cp_ep/frequency/Ensemble/LMR/Nino34_cp.npy',CPni_fre)
#np.save('/Users/lxy/Desktop/research/ENSO/cp_ep/frequency/Ensemble/LMR/Nino34_ep.npy',EPni_fre)
#np.save('/Users/lxy/Desktop/research/ENSO/cp_ep/frequency/Ensemble/LMR/CPEP_cp.npy',CPp_fre)
#np.save('/Users/lxy/Desktop/research/ENSO/cp_ep/frequency/Ensemble/LMR/CPEP_ep.npy',EPni_fre)
#np.save('/Users/lxy/Desktop/research/ENSO/cp_ep/frequency/Ensemble/LMR/CE_cp.npy',CPce_fre)
#np.save('/Users/lxy/Desktop/research/ENSO/cp_ep/frequency/Ensemble/LMR/CE_ep.npy',EPce_fre)


EMI_fre=np.load('/Users/lxy/Desktop/research/ENSO/cp_ep/frequency/Ensemble/PHYDA/EMI.npy')
CPni_fre=np.load('/Users/lxy/Desktop/research/ENSO/cp_ep/frequency/Ensemble/PHYDA/Nino34_cp.npy')
EPni_fre=np.load('/Users/lxy/Desktop/research/ENSO/cp_ep/frequency/Ensemble/PHYDA/Nino34_ep.npy')
CPp_fre=np.load('/Users/lxy/Desktop/research/ENSO/cp_ep/frequency/Ensemble/PHYDA/CPEP_cp.npy')
EPp_fre=np.load('/Users/lxy/Desktop/research/ENSO/cp_ep/frequency/Ensemble/PHYDA/CPEP_ep.npy')
CPce_fre=np.load('/Users/lxy/Desktop/research/ENSO/cp_ep/frequency/Ensemble/LMR/CE_cp.npy')
EPce_fre=np.load('/Users/lxy/Desktop/research/ENSO/cp_ep/frequency/Ensemble/LMR/CE_ep.npy')


# =============================================================================
# EMI
# =============================================================================

#MCA vs.LIA
te1,pe1=stats.ttest_ind(EMI_fre[0,:]/8,EMI_fre[1,:]/8,equal_var=False) 
#MCA vs. 20th century   
te2,pe2=stats.ttest_ind(EMI_fre[0,:]/8,EMI_fre[2,:]/2,equal_var=False)    
#LIA vs. 20th century   
te3,pe3=stats.ttest_ind(EMI_fre[1,:]/8,EMI_fre[2,:]/2,equal_var=False)    


# =============================================================================
# Nino3-4
# =============================================================================

#CP
#MCA vs.LIA
tcni1,pcni1=stats.ttest_ind(CPni_fre[0,:]/8,CPni_fre[1,:]/8,equal_var=False) 
#MCA vs. 20th century   
tcni2,pcni2=stats.ttest_ind(CPni_fre[0,:]/8,CPni_fre[2,:]/2,equal_var=False)    
#LIA vs. 20th century   
tcni3,pcni3=stats.ttest_ind(CPni_fre[1,:]/8,CPni_fre[2,:]/2,equal_var=False)    

#EP
#MCA vs.LIA
teni1,peni1=stats.ttest_ind(EPni_fre[0,:]/8,EPni_fre[1,:]/8,equal_var=False) 
#MCA vs. 20th century   
teni2,peni2=stats.ttest_ind(EPni_fre[0,:]/8,EPni_fre[2,:]/2,equal_var=False)    
#LIA vs. 20th century   
teni3,peni3=stats.ttest_ind(EPni_fre[1,:]/8,EPni_fre[2,:]/2,equal_var=False)    



# =============================================================================
# CP-EP
# =============================================================================

#CP
#MCA vs.LIA
tcp1,pcp1=stats.ttest_ind(CPp_fre[0,:]/8,CPp_fre[1,:]/8,equal_var=False) 
#MCA vs. 20th century   
tcp2,pcp2=stats.ttest_ind(CPp_fre[0,:]/8,CPp_fre[2,:]/2,equal_var=False)    
#LIA vs. 20th century   
tcp3,pcp3=stats.ttest_ind(CPp_fre[1,:]/8,CPp_fre[2,:]/2,equal_var=False)    

#EP
#MCA vs.LIA
tep1,pep1=stats.ttest_ind(EPp_fre[0,:]/8,EPp_fre[1,:]/8,equal_var=False) 
#MCA vs. 20th century   
tep2,pep2=stats.ttest_ind(EPp_fre[0,:]/8,EPp_fre[2,:]/2,equal_var=False)    
#LIA vs. 20th century   
tep3,pep3=stats.ttest_ind(EPp_fre[1,:]/8,EPp_fre[2,:]/2,equal_var=False)    


# =============================================================================
# C and E
# =============================================================================

#CP
#MCA vs.LIA
tcce1,pcce1=stats.ttest_ind(CPce_fre[0,:]/8,CPce_fre[1,:]/8,equal_var=False) 
#MCA vs. 20th century   
tcce2,pcce2=stats.ttest_ind(CPce_fre[0,:]/8,CPce_fre[2,:]/2,equal_var=False)    
#LIA vs. 20th century   
tcce3,pcce3=stats.ttest_ind(CPce_fre[1,:]/8,CPce_fre[2,:]/2,equal_var=False)    

#EP
#MCA vs.LIA
tece1,pece1=stats.ttest_ind(EPce_fre[0,:]/8,EPce_fre[1,:]/8,equal_var=False) 
#MCA vs. 20th century   
tece2,pece2=stats.ttest_ind(EPce_fre[0,:]/8,EPce_fre[2,:]/2,equal_var=False)    
#LIA vs. 20th century   
tece3,pece3=stats.ttest_ind(EPce_fre[1,:]/8,EPce_fre[2,:]/2,equal_var=False)    






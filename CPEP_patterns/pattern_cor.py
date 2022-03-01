#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  4 14:38:49 2020

@author: lxy
"""

import numpy as np
import pandas as pd
from netCDF4 import Dataset
#import os
import scipy.io as sio
from scipy import stats
from eofs.standard import Eof
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap

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


Nino34_phy=np.load('/Users/lxy/Desktop/research/ENSO/cp_ep/validation/rainfall/Nino3-4_phy.npy')
Nino34_lmr=np.load('/Users/lxy/Desktop/research/ENSO/cp_ep/validation/rainfall/Nino3-4_lmr.npy')
E_C_phy=np.load('/Users/lxy/Desktop/research/ENSO/cp_ep/validation/rainfall/E_C_phy.npy')
E_C_lmr=np.load('/Users/lxy/Desktop/research/ENSO/cp_ep/validation/rainfall/E_C_lmr.npy')
EP_CP_phy=np.load('/Users/lxy/Desktop/research/ENSO/cp_ep/validation/rainfall/EP_CP_phy.npy')
EP_CP_lmr=np.load('/Users/lxy/Desktop/research/ENSO/cp_ep/validation/rainfall/EP_CP_lmr.npy')

# =============================================================================
# PHYDA
# =============================================================================

#E_C and EP_CP
recp_cpp,pecp_cpp=pattern_cor(E_C_phy[0,:,:],EP_CP_phy[0,:,:])
recp_epp,pecp_epp=pattern_cor(E_C_phy[1,:,:],EP_CP_phy[1,:,:])

#E_C and Nino3-4
recn_cpp,pecn_cpp=pattern_cor(E_C_phy[0,:,:],Nino34_phy[0,:,:])
recn_epp,pecn_epp=pattern_cor(E_C_phy[1,:,:],Nino34_phy[1,:,:])

#EP_CP and Nino34
rpn_cpp,ppn_cpp=pattern_cor(EP_CP_phy[0,:,:],Nino34_phy[0,:,:])
rpn_epp,ppn_epp=pattern_cor(EP_CP_phy[1,:,:],Nino34_phy[1,:,:])


# =============================================================================
# LMR
# =============================================================================

#E_C and EP_CP
recp_cpl,pecp_cpl=pattern_cor(E_C_lmr[0,:,:],EP_CP_lmr[0,:,:])
recp_epl,pecp_epl=pattern_cor(E_C_lmr[1,:,:],EP_CP_lmr[1,:,:])

#E_C and Nino3-4
recn_cpl,pecn_cpl=pattern_cor(E_C_lmr[0,:,:],Nino34_lmr[0,:,:])
recn_epl,pecn_epl=pattern_cor(E_C_lmr[1,:,:],Nino34_lmr[1,:,:])

#EP_CP and Nino34
rpn_cpl,ppn_cpl=pattern_cor(EP_CP_lmr[0,:,:],Nino34_lmr[0,:,:])
rpn_epl,ppn_epl=pattern_cor(EP_CP_lmr[1,:,:],Nino34_lmr[1,:,:])

# =============================================================================
# LMR and PHYDA
# =============================================================================

r1,p1=pattern_cor(E_C_lmr[0,:,:],E_C_phy[0,:,:])
r2,p2=pattern_cor(E_C_lmr[1,:,:],E_C_phy[1,:,:])


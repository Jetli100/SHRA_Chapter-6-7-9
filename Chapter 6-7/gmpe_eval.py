# Note: The codes were originally created by Prof. Jack Baker in the MATLAB

import numpy as np
from scipy.stats import norm
from scipy.interpolate import interp1d
from gmpe_bjf97 import gmpe_bjf97
from gmpe_CY_2014 import gmpe_CY_2014

# Master function to call a relevant GMPE and get median Sa and log standard deviation

##############
### Inputs ###
##############
# T         IM period of interest
# M         rupture magnitude
# rup           data structure with rupture parameters
# gmpeFlag      =1 for BJF97, =2 for CY14

###############
### Outputs ###
###############
# sa        median spectral acceleration, given rup
# sigma     log standard deviation, given rup


def gmpe_eval(T, M, rup, gmpeFlag):
    if gmpeFlag == 1: # BJF 1997 model
        sa, sigma = gmpe_bjf97(M, rup["R"], T, rup["Fault_Type"], rup["Vs30"])
    elif gmpeFlag == 2: # CY 2014 model
        sa, sigma, _ = gmpe_CY_2014(M, T, rup["R"], rup["R"], rup["R"], rup["Ztor"], rup["delta"], rup["rupLambda"], rup["Z10"], rup["Vs30"], rup["Fhw"], rup["FVS30"], rup["region"]) 
    else:
        print('Invalid gmpeFlag')

    return sa, sigma
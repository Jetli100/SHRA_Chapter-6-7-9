# Note: The codes were originally created by Prof. Jack Baker in the MATLAB

import numpy as np
import math
from gmpe_prob_bjf97 import gmpe_prob_bjf97
from scipy.interpolate import interp1d

# same as fn_GR_PSHA but with Mmax as an input parameter (this is a legacy format)

##############
### Inputs ###
##############

# lambda_M      rate of M>5
# M_vals        values of M corresponding to lambda_M
# x             PGA values of interest
# rup           data structure with rupture parameters
# R             distance of point source
# M_min         minimum M

# Output is a list of rate of exceeding x

def fn_GR_PSHA_sigma(lambda_M, M_vals, T, x, rup, sigmaFactor):
    lambda_occur = -np.diff(lambda_M) # find occurence rates from exceedance rates
    lambda_occur = np.append(lambda_occur, lambda_M[-1])
    
    # p(exceeding each x threshold value | M)
    lambda_x = []
    for j in range(len(x)):
        p_given_M = []
        for i in range(len(M_vals)):
            p_given_M.append(gmpe_prob_bjf97(x[j], M_vals[i], rup["R"], T, rup["Fault_Type"], rup["Vs30"], sigmaFactor))
        
        lambda_x.append(np.sum(lambda_occur * np.array(p_given_M)))
    
    return lambda_x
# Note: The codes were originally created by Prof. Jack Baker in the MATLAB
import numpy as np
import math
import os
import scipy.io as sio
from scipy.stats import norm
from gmpe_bjf97 import gmpe_bjf97


###################
### Description ###
###################

# The probability of exceeding a given PGA level x, using the BJF GMPE.


##########################
### Inputs and Outputs ###
##########################

# INPUT
#
#   x = a vector of amplitudes of interest
#   All other inputs are defined in gmpe_bjf97
#
# OUTPUT
#
#   p = probabilities of exceeding each x value

def gmpe_prob_bjf97(x, M, R, T, Fault_Type, Vs, *sigmaFactor):
    arg = list(sigmaFactor)
    sa, sigma = gmpe_bjf97(M, R, T, Fault_Type, Vs)
    if len(arg) > 0:
        sigmaFactor = max(arg[0], 0.0001) # allows  sigmaFactor=0 without an error
        sigma = sigma * sigmaFactor
    p = 1 - norm.cdf(np.log(x),np.log(sa),sigma)
        
    return p

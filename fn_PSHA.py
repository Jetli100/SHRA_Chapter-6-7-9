# Note: The codes were originally created by Prof. Jack Baker in the MATLAB

import numpy as np
import math
from scipy.stats import norm
from scipy.interpolate import interp1d
from gmpe_prob_bjf97 import gmpe_prob_bjf97
from gmpe_BSSA_2014 import gmpe_BSSA_2014
from gmpe_CY_2014 import gmpe_CY_2014

# General function to do PSHA calculations under a variety of Rup and GMM parameters

def fn_PSHA(rup, M_vals, lambda_M, T, x, x_example):
    
    # use BJF97 if not specified (for backwards compatibility)
    if "gmm" not in rup:
        rup["gmm"] = 1
    
    lambda_occur = -np.diff(lambda_M) # find occurence rates from exceedance rates
    lambda_occur = np.append(lambda_occur, lambda_M[-1])
    
    # p(exceeding each x threshold value | M)
    lambda_x = []
    M_disagg = []
    for j in range(len(x)):
        p_given_M = []
        for i in range(len(M_vals)):
            if rup["gmm"] == 1: # BJF 1997 model
                p_given_M.append(gmpe_prob_bjf97(x[j], M_vals[i], rup["R"], T, rup["Fault_Type"], rup["Vs30"]))
            elif rup["gmm"] == 2: # BSSA 2014 model
                sa, sigma, _ = gmpe_BSSA_2014(M_vals[i], T, rup["R"],rup["Fault_Type"], rup["region"], rup["Z10"], rup["Vs30"])
                p_given_M.append(1 - norm.cdf(np.log(x[j]), np.log(sa), sigma))            
            else: #CY 2014 model
                sa, sigma, _ = gmpe_CY_2014(M_vals[i], T, rup["R"], rup["R"], rup["R"], rup["Ztor"], rup["delta"], rup["lambda0"], rup["Z10"], rup["Vs30"], rup["Fhw"], rup["FVS30"], rup["region"])
                p_given_M.append(1 - norm.cdf(np.log(x[j]), np.log(sa), sigma))            

        lambda_x.append(np.sum(lambda_occur * np.array(p_given_M)))  # rate of exceeding x
        M_disagg.append((lambda_occur * np.array(p_given_M)) / lambda_x[j])
  

    # calcs for example IM case
    p_ex = []
    for i in range(len(M_vals)):
        if rup["gmm"] == 1: # BJF 1997 model
            p_ex.append(gmpe_prob_bjf97(x_example, M_vals[i], rup["R"], T, rup["Fault_Type"], rup["Vs30"]))
        elif rup["gmm"] == 2: # BSSA 2014 model
            sa, sigma, _ = gmpe_BSSA_2014(M_vals[i], T, rup["R"],rup["Fault_Type"], rup["region"], rup["Z10"], rup["Vs30"])
            p_ex.append(1 - norm.cdf(np.log(x_example), np.log(sa), sigma))
        else: # CY 2014 model
            sa, sigma, _ = gmpe_CY_2014(M_vals[i], T, rup["R"], rup["R"], rup["R"], rup["Ztor"], rup["delta"], rup["lambda0"], rup["Z10"], rup["Vs30"], rup["Fhw"], rup["FVS30"], rup["region"])
            p_ex.append(1 - norm.cdf(np.log(x_example), np.log(sa), sigma))
 
    example_output = [[i for i in range(len(M_vals))], list(M_vals), list(lambda_occur), p_ex, list(lambda_occur*np.array(p_ex))]
    lambda_example = np.sum(lambda_occur * np.array(p_ex))
    
    return lambda_x, lambda_example, example_output, M_disagg
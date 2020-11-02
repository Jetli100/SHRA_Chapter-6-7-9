# Note: The codes were originally created by Prof. Jack Baker in the MATLAB

import numpy as np
import math
from scipy.stats import norm
from scipy.interpolate import interp1d
from gmpe_eval import gmpe_eval

# Compute PSHA, with rupture rates for each M precomputed

##############
### Inputs ###
##############

# lambda_M      exceedance rate of EQs for each M
# M_vals        values of M corresponding to lambda_M
# x             IM values of interest
# x_example     example IM value for table
# rup           data structure with rupture parameters
# gmpeFlag      =1 for BJF97, =2 for CY14


def fn_PSHA_given_M_lambda(lambda_M, M_vals, T, x, x_example, rup, gmpeFlag):
    lambda_occur = -np.diff(lambda_M) # find occurence rates from exceedance rates
    lambda_occur = np.append(lambda_occur, lambda_M[-1])
    
    # p(exceeding each x threshold value | M)
    lambda0 = {}
    disagg = {}
    lambda0["x"] = []
    disagg["allv"] = []
    for j in range(len(x)):
        p_given_M = []
        for i in range(len(M_vals)):
            sa, sigma = gmpe_eval(T, M_vals[i], rup, gmpeFlag)
            p_given_M.append(1 - norm.cdf(np.log(x[j]),np.log(sa),sigma))
        p_given_M = np.array(p_given_M)
        lambda0["x"].append(np.sum(lambda_occur * p_given_M)) # rate of exceeding x
        disagg["allv"].append((lambda_occur * p_given_M) / lambda0["x"][j])
    
    # calcs for example IM case
    p_ex = []
    for i in range(len(M_vals)):
        sa, sigma = gmpe_eval(T, M_vals[i], rup, gmpeFlag)
        p_ex.append(1-norm.cdf(np.log(x_example),np.log(sa),sigma))

    example_output = [[i for i in range(len(M_vals))], list(M_vals), list(lambda_occur), p_ex, list(lambda_occur*np.array(p_ex))]
    p_ex = np.asarray(p_ex)
    M_vals = np.asarray(M_vals)
    lambda0["example"] = np.sum(lambda_occur * p_ex)
    disagg["example"] = (lambda_occur * p_ex) / lambda0["example"]
    disagg["Mbar"] = np.sum(M_vals * disagg["example"])
    
    # disagg conditional on occurence for example IM case
    xInc = x_example * 1.02 # do computations at an increment on x
    pInc = []
    for i in range(len(M_vals)):    
        sa, sigma = gmpe_eval(T, M_vals[i], rup, gmpeFlag)
        pInc.append(1 - norm.cdf(np.log(xInc),np.log(sa),sigma))
    pInc = np.array(pInc)
    lambdaInc = np.sum(lambda_occur * pInc)
    disagg["equal"] = ((lambda_occur * p_ex) - (lambda_occur * pInc)) / (lambda0["example"]-lambdaInc)
    disagg["equalMbar"] = np.sum(M_vals * disagg["equal"])
    
    # disaggs with epsilon
    deltaEps = 1  # final binning
    epsVals = np.arange(-3, 4, deltaEps) # epsilon bins

    deltaEpsFine = 0.1  # initial finer binning
    epsValsFine = np.arange(-35, 36, int(deltaEpsFine*10)) / 10  # midpoints of bins
    p_eps = norm.pdf(epsValsFine) * deltaEpsFine  # estimate PDF using a PMF with discrete epsilon increments
    lambda_M_and_eps = np.outer(lambda_occur, p_eps)  # rate of events with a given magnitude and epsilon  

    Ind = []
    for i in range(len(M_vals)):
        sa, sigma = gmpe_eval(T, M_vals[i], rup, gmpeFlag)    
        Ind.append((np.log(sa) + epsValsFine*sigma) > np.log(x_example)) # indicator that the M/epsilon value causes IM > x_example  

    exceedRatesFine = np.asarray(Ind) * lambda_M_and_eps # rates of given M/epsilon values exceeding IM
    lambdaExceed = np.sum(exceedRatesFine) # this is close to lambda.example, but may differ by a few percent due to epsilon discretization
    
    #  compute mean epsilon
    epsDeagg = np.sum(exceedRatesFine, axis=0) / np.sum(exceedRatesFine)
    disagg["epsBar"] = np.sum(epsValsFine * epsDeagg)

    #  aggregate results to coarser epsilon bins
    exceedRates = []
    for j in range(len(epsVals)):
        idx = np.nonzero((epsValsFine >= (epsVals[j]-deltaEps/2)) & (epsValsFine < (epsVals[j]+deltaEps/2)))[0]
        exceedRates.append(np.sum(exceedRatesFine[:,idx], axis=1))
    exceedRates = np.transpose(np.asarray(exceedRates))
    
    disagg["epsVals"] = epsVals  # return bin midpoints
    disagg["M_Eps"] = exceedRates / lambdaExceed  # magnitude and epsilon disaggregation
    disagg["eps"] = np.sum(exceedRates, axis=0) / lambdaExceed  #  epsilon disaggregation

    return lambda0, example_output, disagg
import numpy as np
import h5py
from scipy.stats import norm

############
### Note ###
############

# These are the "Equivalent PGA Structural Fragility Curves from
# section 5.4.4 of Hazus, not the more precise fragilities based on
# capacity curves. They are similar to the capacity curve results, but not
# equivalent.

#######################
### Input variables ###
#######################

### analysisCase.codeLevel ###  flag for code level:
#                                   1 --> High Code
#                                   2 --> Moderate Code
#                                   3 --> Low Code
#                                   4 --> Pre-Code

### analysisCase.buildingType ###  2- or 3-letter code for construction type. Allowable options:
#                                     W1 	Wood, Light Frame (< 5,000 sq. ft.)
#                                     W2 	Wood, Commercial and Industrial (> 5,000 sq. ft.)
#                                     S1L 	Steel Moment Frame
#                                     S1M 	Steel Moment Frame
#                                     S1H 	Steel Moment Frame
#                                     S2L 	Steel Braced Frame
#                                     S2M 	Steel Braced Frame
#                                     S2H 	Steel Braced Frame
#                                     S3 	Steel Light Frame
#                                     S4L 	Steel Frame with Cast?in?Place Concrete Shear Walls
#                                     S4M 	Steel Frame with Cast?in?Place Concrete Shear Walls
#                                     S4H 	Steel Frame with Cast?in?Place Concrete Shear Walls
#                                     S5L 	Steel Frame with Unreinforced Masonry Infill Walls
#                                     S5M 	Steel Frame with Unreinforced Masonry Infill Walls
#                                     S5H 	Steel Frame with Unreinforced Masonry Infill Walls
#                                     C1L 	Concrete Moment Frame
#                                     C1M 	Concrete Moment Frame
#                                     C1H 	Concrete Moment Frame
#                                     C2L 	Concrete Shear Walls
#                                     C2M 	Concrete Shear Walls
#                                     C2H 	Concrete Shear Walls
#                                     C3L 	Concrete Frame with Unreinforced Masonry Infill Walls
#                                     C3M 	Concrete Frame with Unreinforced Masonry Infill Walls
#                                     C3H 	Concrete Frame with Unreinforced Masonry Infill Walls
#                                     PC1 	Precast Concrete Tilt?Up Walls
#                                     PC2L 	Precast Concrete Frames with Concrete Shear Walls
#                                     PC2M 	Precast Concrete Frames with Concrete Shear Walls
#                                     PC2H 	Precast Concrete Frames with Concrete Shear Walls
#                                     RM1L 	Reinforced Masonry Bearing Walls with Wood or Metal Deck Diaphragms
#                                     RM1M 	Reinforced Masonry Bearing Walls with Wood or Metal Deck Diaphragms
#                                     RM2L 	Reinforced Masonry Bearing Walls with Precast Concrete Diaphragms
#                                     RM2M 	Reinforced Masonry Bearing Walls with Precast Concrete Diaphragms
#                                     RM2H 	Reinforced Masonry Bearing Walls with Precast Concrete Diaphragms
#                                     URML 	Unreinforced Masonry Bearing Walls
#                                     URMM 	Unreinforced Masonry Bearing Walls
#                                     MH 	Mobile Homes

### analysisCase.occType###  4- or 5-letter code for construction type. Allowable options:
#                                     RES1	Single Family Dwelling
#                                     RES2	Mobile Home
#                                     RES3	Multi Family Dwelling
#                                     RES4	Temporary Lodging
#                                     RES5	Institutional Dormitory
#                                     RES6	Nursing Home
#                                     COM1	Retail Trade
#                                     COM2	Wholesale Trade
#                                     COM3	Personal and Repair Services
#                                     COM4	Professional/Technical/ Business  Services
#                                     COM5	Banks/Financial Institutions
#                                     COM6	Hospital
#                                     COM7	Medical Office/Clinic
#                                     COM8	Entertainment & Recreation
#                                     COM9	Theaters
#                                     COM10	Parking
#                                     IND1	Heavy
#                                     IND2	Light
#                                     IND3	Food/Drugs/Chemicals
#                                     IND4	Metals/Minerals Processing
#                                     IND5	High Technology
#                                     IND6	Construction
#                                     AGR1	Agriculture
#                                     REL1	Church/Membership/Organization
#                                     GOV1	General Services
#                                     GOV2	Emergency Response
#                                     EDU1	Schools/Libraries
#                                     EDU2	Colleges/Universities

### pgaVals ###                   PGA values for which to compute loss ratios

#######################
### Output Variables###
#######################

### lossRatio ###                  loss ratio (total loss) for each PGA

### caseLabel ###                  text label describing the analysis case

### structLossRatio ###            loss ratio (structural) for each PGA

### nonStructAccLossRatio ###      loss ratio (nonstructural acceleration
#                                  sensitive) for each PGA

### nonStructDriftLossRatio ###    loss ratio (nonstructural drift
#                                  sensitive) for each PGA

def fn_HAZUS_loss(analysisCase, pgaVals):
    # Load data created by import_HAZUS_data.m

    # Make sure pgaVals is a list
    if isinstance(pgaVals, list) == False:
        pgaVals = [pgaVals]
    
    # find index for building type
    buildingTypeCode = ['W1', 'W2', 'S1L', 'S1M', 'S1H', 'S2L', 'S2M', 'S2H', 'S3', 'S4L', 'S4M', 'S4H', 'S5L', 'S5M', 'S5H', 'C1L', 'C1M', 'C1H', 'C2L', 'C2M', 'C2H', 'C3L', 'C3M', 'C3H', 'PC1', 'PC2L', 'PC2M', 'PC2H', 'RM1L', 'RM1M', 'RM2L', 'RM2M', 'RM2H', 'URML', 'URMM', 'MH']
    idxBldg = buildingTypeCode.index(analysisCase['buildingType'])
    # Make sure an appropriate building type was specified
    assert ~np.isnan(idxBldg), 'Error with specified building type'

    # find index of occupancy type
    occCode = ['RES1', 'RES2', 'RES3', 'RES4', 'RES5', 'RES6', 'COM1', 'COM2', 'COM3', 'COM4', 'COM5', 'COM6', 'COM7', 'COM8', 'COM9', 'COM10', 'IND1', 'IND2', 'IND3', 'IND4', 'IND5', 'IND6', 'AGR1', 'REL1', 'GOV1', 'GOV2', 'EDU1', 'EDU2']
    idxOcc = occCode.index(analysisCase['occType'])
    # Make sure an appropriate occupancy type was specified
    assert ~np.isnan(idxOcc), 'Error with specified building type'

    # Get fragility parameters for the given structure type and code level
    hf = h5py.File('hazusData.h5', 'r')
    medians = hf.get('medians')
    betas = hf.get('betas')
    medianDS = medians[analysisCase['codeLevel']].tolist()[idxBldg]
    betaDS = betas[analysisCase['codeLevel']].tolist()[idxBldg]
    
    # Make sure an appropriate occupancy type was specified
    assert ~np.isnan(medianDS[0]), 'Error, this building type and code level is not allowed'

    # Get loss ratio for the given occupancy type
    lossStruct = hf.get('lossStruct')[idxOcc]
    lossAccNS = hf.get('lossAccNS')[idxOcc]
    lossDriftNS = hf.get('lossDriftNS')[idxOcc]

    # Damage state exceedance probability per pgaVals
    pDsExceed = []
    length = len(pgaVals)
    for i in range(length):
        pDsExceed.append(norm.cdf(np.log(pgaVals[i]), np.log(medianDS), betaDS).tolist())
    # Pad with zeros for probability of exceeding DS5 (add another column with zeros), to help with differentiation
    # in the next step
    pDsExceed = np.c_[pDsExceed, np.zeros(length)]

    pDsEqual = np.empty((length, 4))
    for i in range(4):
        pDsEqual[:, i] = pDsExceed[:, i] - pDsExceed[:, i + 1]

    # Compute loss ratios per PGA
    structLossRatio = np.sum(pDsEqual * lossStruct, axis=1) / 100
    nonStructAccLossRatio = np.sum(pDsEqual * lossAccNS, axis=1) / 100
    nonStructDriftLossRatio = np.sum(pDsEqual * lossDriftNS, axis=1) / 100

    lossRatio = structLossRatio + nonStructAccLossRatio + nonStructDriftLossRatio

    # Make a label for the analysis case
    codeLevel = ['High', 'Moderate', 'Low', 'Pre']
    caseLabel = analysisCase['buildingType'] + ', ' + analysisCase['occType'] + ', ' + \
                codeLevel[analysisCase['codeLevel']] + '-code'

    return lossRatio, caseLabel, structLossRatio, nonStructAccLossRatio, nonStructDriftLossRatio


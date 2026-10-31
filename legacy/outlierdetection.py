# -*- coding: utf-8 -*-
"""
Created on Sat Jun 16 10:18:18 2018

@author: Berdakh
"""
#%% bad channel identification 
#%%
#X = ep.get_data()
#data = np.rollaxis(X, 0, 2)
#X1 = data.reshape(59,-1)
#A = outlierdetection(X1, dim=1, threshold=(None,1), maxIter=1, feat="var")

import scipy.signal 
import numpy
from math import ceil
from functools import reduce
    
def idOutliers(X, dim=1, threshold=(None,2), maxIter=2, feat="var"):
    '''Removes outliers from X. Based on idOutliers.m
    
    Removes outliers from X based on robust coviarance variance/mean 
    computation.
    
    Parameters
    ----------
    data : list of datapoints (numpy arrays) or a single numpy array.
    dim : the dimension along with outliers need to be detected.
    threshold : the upper and lower bound of the thershold in a tuple expressed
    in standard deviations.  
    maxIter : number of iterations that need to be performed.
    feat : feature on which outliers need to be based. "var" for variance, "mu"
    , for mean or a numpy array of the same shape as X.
    
    Returns
    -------
    out : a tuple of a list of inlier indices and a list of outlier indices.
    
    Examples
    --------
    >>> data, events = ftc.getData(0,100)
    >>> inliers, outliers = preproc.outlierdetection(data)
    '''   
#%     
    if feat=="var":
        feat = numpy.sqrt(numpy.abs(numpy.var(X,dim)))        
    elif feat =="mu":
        feat = numpy.mean(X,dim)              
    elif isinstance(feat,numpy.array):
        if not (all([isinstance(x,(int , float)) for x in feat]) and feat.shape == X.shape):
            raise Exception("Unrecognised feature type.")
    else:        
        raise Exception("Unrecognised feature type.")    
#%        
    outliers = []
    inliers = numpy.array(list(range(0,max(feat.shape))))   

    mufeat = numpy.zeros(maxIter)
    stdfeat = numpy.zeros(maxIter)
    
    
    for i in range(0,maxIter):
        mufeat = numpy.mean(feat[inliers]) # median  
        stdfeat = numpy.std(feat[inliers]) # std across channels 

        if threshold[0] is None:
            high = mufeat + threshold[1]*stdfeat # median + 3* standard deviation 
            bad = (feat > high)
            print(">>> Threshold: ", threshold[1])
            
        elif threshold[1] is None:
            low = mufeat+threshold[0]*stdfeat
            bad = (feat < low)
            print(">>> Threshold: ", threshold[0])

            
        else:
            high = mufeat+threshold[1]*stdfeat
            low = mufeat+threshold[0]*stdfeat
            bad = (feat > high) * (feat < low)

        if not any(bad):
            break
        else:
            outliers = outliers + list(inliers[bad])
            inliers = inliers[[ not x for x in bad]]
            
    print("== OUTLIERS FOUND ===", outliers)
    
    return (list(inliers), outliers)


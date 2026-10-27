# -*- coding: utf-8 -*-
"""
Created on Wed Jun 13 18:24:23 2018

@author: Berdakh
"""
import numpy

X = ep.get_data()
ch = np.rollaxis(X, 0, 2)
trial = X.reshape(17,-1)

A = outlierdetection(trial, dim=1, threshold =(None,1), maxIter=1, feat="var")

#%%
bad = list(A[1])

ch_names = ep.info['ch_names']
b=[]

for bads in bad:
    b.append(ch_names[bads])
    
print(b)
#%%
ep.info['bads'] = b
ep.interpolate_bads(reset_bads='True', mode = 'accurate')



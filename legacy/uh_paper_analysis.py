import numpy as np
from numpy.random import randn
from scipy import stats as stats
import os 
import mne
import matplotlib.pyplot as plt 
from mne.minimum_norm import (apply_inverse,apply_inverse_epochs,make_inverse_operator)
from mne.connectivity import seed_target_indices, spectral_connectivity
import glob 

"""
The goal of the analysis is to make a statistical appraisal of the neural
activation evoked from the BMI triggered exoskeleton at movement onset detected by Robot event.

1) The question is whether evidence can be found against the hull hypothesis that 
neural activations (plasticity) on (the somatosensory and motor) cortex 
    a) does not depend on whether or not the when BMI is inactive (subject is idle)
    b) does not change across sessions 

2) Can we find any evidence against hull hypothesis that neural activations patterns 
on the cortex does remain the same across sessions ?

* In other words, there is no evidence of cortical reorganization.  
""" 
#%%##############################################################################
# Read the EEG data
from uhClass import MRCP

mainDir = 'C:\\uhdata\\freesurfer'
subject = 'S9011'

os.chdir(os.path.join(mainDir, subject))
eegfiles = glob.glob('*-epoall.pickle')[0]


#%%
import pickle 

with open(eegfiles, 'rb') as pickle_file:
     eeg = pickle.load(pickle_file)  
     
#%% Plot EEG 
#from uhClass import MRCP
#filename = eegfilename.split('_s')[0]
ev =[]
for ep in eeg:    
    print(ep)    
    evoked = ep.average() 
    ev.append(evoked)
    
#%% STEP 1: COMPUTER THE SOURCE SPACE (SOURCE GRID ON MRI)
""" 
The source space defines the position of the candidate source locations.
The following code compute such a cortical source space with an
OCT-5 resolution.
"""
# this is a freesurfer folder name 
#src = mne.setup_source_space(subject, spacing='oct5', subjects_dir = mainDir) 

#%%
#src.plot(head=True, brain=True, skull = True, subjects_dir = mainDir)
#%% READ FORWARD SOLUTION
fname_fwd = os.path.join(mainDir, subject) + '\\' + subject + '-fwd.fif'
fwd = mne.read_forward_solution(fname_fwd)

#%%
noise_cov  = mne.compute_covariance(eeg[1], keep_sample_mean=True, tmin=-0.5, tmax=-0.0)

stcs = []
for ii, epoch in enumerate(ev):    
    # tmin, tmax = epoch.time_as_index([-1, -0.5])
    # calculate noise covariance matrix
    # make inverse operator
    
    info = epoch.info
    inverse_operator = make_inverse_operator(info, fwd, noise_cov, loose=0.2, depth=None)
    epoch.set_eeg_reference(ref_channels = "average", projection=True)
    
    # apply inverse solution
    method = 'MNE'
    snr = 3.
    lambda2 = 1. / snr ** 2
    stc = apply_inverse(epoch, inverse_operator, lambda2,method= method, pick_ori="normal",)
    stcs.append(stc)    
    
    
#%%
#e1 = copy.deepcopy(evoked)
#e2 = copy.deepcopy(evoked)
#
#e1.crop(-0.5, 0)
#e2.crop(0, 0.5)
#%% 
import copy 
# condition 1 -- > baseline 
cond1 = []
# condition 2 ---> Movement detected by Robot 
cond2 = [] 

for source in stcs:    
    cond1.append(copy.deepcopy(source).crop(-0.5, 0))
    cond2.append(copy.deepcopy(source).crop(0, 0.5))

#%%
n_vertices_sample, n_times = cond1[2].data.shape
n_subjects = len(cond2)

#%%
try:
    del X
except Exception:
    pass 

X = np.zeros([n_vertices_sample, n_times, n_subjects, 2])

for ii, c1 in enumerate(cond1):
    X[:,:,ii,0] = c1.data 

for ii, c2 in enumerate(cond2):
    X[:,:,ii,1] = c2.data

""""
X = np.zeros([n_vertices_sample, n_times, n_subjects])
for ii, c in enumerate(cond2):
    X[:,:, ii] = c.data 
"""

#%%############################################################################
# Finally, we want to compare the overall activity levels in each condition,
# the diff is taken along the last axis (condition). The negative sign makes
# it so condition1 > condition2 shows up as "red blobs" (instead of blue).

X = np.abs(X)  # only magnitude
X = X[:, :, :, 0] - X[:, :, :, 1]  # make paired contrast

#%%############################################################################
src_fname = os.path.join(mainDir, subject) + '\\'+ subject + '-src.fif'
# Read the source space we are morphing to
src = mne.read_source_spaces(src_fname)

fsave_vertices = [s['vertno'] for s in src]

#%%############################################################################
# Compute statistic
# -----------------
# To use an algorithm optimized for spatio-temporal clustering, we
# just pass the spatial connectivity matrix (instead of spatio-temporal)
print('Computing connectivity.')
connectivity = mne.spatial_src_connectivity(src)

#X1 = X[:,:,:,1]
#    Note that X needs to be a multi-dimensional array of shape
#    samples (subjects) x time x space, so we permute dimensions
X1 = np.transpose(X, [2, 1, 0])

# Now let's actually do the clustering. This can take a long time...

#%%    Here we set the threshold quite high to reduce computation.

p_threshold = 0.05
#t_threshold = -stats.distributions.t.ppf(p_threshold / 2., n_subjects - 1)
t_threshold = -stats.distributions.t.ppf(p_threshold / 2., n_subjects - 1)

#%%
tstep = cond2[0].tstep
from mne.stats import (spatio_temporal_cluster_1samp_test, summarize_clusters_stc)

print('Clustering.')

T_obs, clusters, cluster_p_values, H0 = clu = \
    spatio_temporal_cluster_1samp_test(X1, connectivity=None, n_jobs=1,
                                       threshold=t_threshold)
#    Now select the clusters that are sig. at p < 0.05 (note that this value
#    is multiple-comparisons corrected).
good_cluster_inds = np.where(cluster_p_values < 0.05)[0]

#%%
#from mne.stats import spatio_temporal_cluster_test
#
#a = spatio_temporal_cluster_test(
#            X, threshold=None, n_permutations=1024, tail=0, stat_fun=None,
#            connectivity=None, verbose=None, n_jobs=1, seed=None, max_step=1,
#            spatial_exclude=None, step_down_p=0, t_power=1, out_type='indices',
#            check_disjoint=False, buffer_size=1000)

#%%############################################################################
# Visualize the clusters
# ----------------------
print('Visualizing clusters.')
#    Now let's build a convenient representation of each cluster, where each
#    cluster becomes a "time point" in the SourceEstimate
#fsave_vertices = [np.arange(X.shape[0]), np.arange(X.shape[0])]

stc_all_cluster_vis = summarize_clusters_stc(clu, p_thresh=0.05, tstep=tstep,
                                             vertices=fsave_vertices, subject=subject)

#%%    Let's actually plot the first "time point" in the SourceEstimate, which
#    shows all the clusters, weighted by duration
subjects_dir = os.path.join(mainDir) 

# blue blobs are for condition A < condition B, red for A > B
brain = stc_all_cluster_vis.plot(surface='inflated',
    hemi='both', views='lateral', subjects_dir=subjects_dir,
    time_label='Duration significant (ms)', size=(800, 800),
    smoothing_steps=10)

# brain.save_image('clusters.png')

#%% Statistics 
"""
1) Parametric Hypothesis Testing assumes Normal distribution or iid 
2) Non-parametric Hypothesis Testing does not rely on any assumption and is usually done by
   methods such as Bootstrap analysis & Permutation Test 
   
a) Paired data has dependent samples since the data is acquired from the same subject (multiple data)
Paired data: there are two measurements from each patient, one before treatment and one after treatment.

These two measurements relate to one another, we are interested in the difference between the two measurements (the
log ratio) to determine whether a gene has been up-regulated or down-regulated in breast cancer following that treatment.   

                                                                                                                
b) Unpaired data has independent samples as the data is acqured from two different/distinct subjects 

c) Complex data has more than two Groups (ANOVA)

###########################
The significance level (alpha) is related to the degree of certainty you require in
order to reject the null hypothesis in favor of the alternative e.g. alpha = 0.05

The p-value is the probability of observing the given sample result under the
assumption that the null hypothesis is true. 

If the p-value is less than alpha, then you reject the null hypothesis.
For example, if alpha = 0.05 and the p-value is 0.03, then you reject the null hypothesis

##########################
Confidence intervals: a range of values that have a chosen probability of
containing the true hypothesized quantity.
###########################

Steps of Hypthesis testing:
    
1. Determine the null and alternative hypothesis, using
mathematical expressions if applicable.

2. Select a significance level (alpha).

3. Take a random sample from the population of interest.

4. Calculate a test statistic from the sample that provides
information about the null hypothesis.

5. Decision

>>> If the value of the statistic is consistent with the null hypothesis
then do not reject H0.

>>> If the value of the statistic is not consistent with the null
hypothesis, then reject H0 and accept the alternative hypothesis.

##########################
"""

#%%
#fname = 'S9017_ses1_cond1_block0001-rh.stc'
#stc = mne.read_source_estimate(fname)
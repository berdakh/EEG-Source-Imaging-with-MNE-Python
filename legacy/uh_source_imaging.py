# -*- coding: utf-8 -*-
"""
Created on Tue Jun 12 14:56:30 2018

@author: Berdakh
"""
import numpy as np
import os.path as op
import matplotlib.pyplot as plt
import os
import mne
from mne.minimum_norm import make_inverse_operator, apply_inverse, apply_inverse_epochs

#%%
filename = 'S9014_ses6_closeloop_block0009-epo.fif'
subjectName = 'S9014'
subjectDir = 'C:\\uhdata\\freesurfer'

os.chdir(os.path.join(subjectDir,subjectName))


raw_fname = op.join(os.getcwd(), filename)
epochs = mne.read_epochs(raw_fname, proj=True, preload=True, verbose=None)
epochs.set_eeg_reference(ref_channels = "average", projection=True) 
epochs.apply_proj()
tmin, tmax = epochs.time_as_index([-1, 0.5])


#%%
#fname_label = [subjectDir + '/S9014/labels/rh.BA1.label',
#               subjectDir + '/S9014/labels/rh.BA2.label',
#               subjectDir + '/S9014/labels/lh.BA1.label',
#               subjectDir + '/S9014/labels/lh.BA2.label']

#rh.BA1.label
#rh.BA2.label
#rh.BA3a.label
#lh.BA1.label
#lh.BA2.label
#lh.BA3a.label

# read label(s)
#labels = [mne.read_label(ss) for ss in fname_label]


#%% Computing a covariance matrix
"""
Many methods in MNE, including source estimation and some classification
algorithms, require covariance estimations from the recordings.
In this tutorial we cover the basics of sensor covariance computations and
construct a noise covariance matrix that can be used when computing the
minimum-norm inverse solution. For more information, see :ref:`BABDEEEB`.
"""
# compute noise covariance matrix 
noise_cov  = mne.compute_covariance(epochs, keep_sample_mean=True, 
                             tmin=tmin, 
                             tmax=tmax)

fig_cov, fig_spectra = mne.viz.plot_cov(noise_cov, epochs.info)

#%% Average the epochs 
evoked = epochs.average().pick_types(eeg=True)
evoked.plot(time_unit='s')
evoked.plot_topomap(times=np.linspace(0.05, 0.15, 5),time_unit='s')

# Show whitening
evoked.plot_white(noise_cov, time_unit='s')

#del epochs  # to save memory
#%% Inverse modeling: MNE/dSPM on evoked and raw data
# Read the forward solution and compute the inverse operator
fname_fwd = os.getcwd() + '\S9014-fwd.fif'
fwd = mne.read_forward_solution(fname_fwd)

# make an MEG inverse operator
info = evoked.info
inverse_operator = make_inverse_operator(info, fwd, noise_cov,
                                         loose=0.2, depth=0.8)
 
#%% Compute inverse solution

method = "dSPM"
snr = 4.
lambda2 = 1. / snr ** 2
stc = apply_inverse(evoked, inverse_operator, lambda2,
                    method=method)

#%% Visualization
plt.figure()
plt.plot(1e3 * stc.times, stc.data[::100, :].T)
plt.xlabel('time (ms)')
plt.ylabel('%s value' % method)
plt.show()

#%%
vertno_max, time_max = stc.get_peak(hemi='rh')
subjects_dir = subjectDir

surfer_kwargs = dict(
    hemi='both', subjects_dir=subjects_dir,
    clim= 'auto', views='lateral',
    initial_time=[0.2], time_unit='s', smoothing_steps=None, 
    cortex='bone', time_viewer = True)


brain = stc.plot(**surfer_kwargs)
brain.add_foci(vertno_max, coords_as_verts=True, hemi='rh', color='red')
brain.add_text(0.1, 0.9, 'dSPM (plus location of maximal activation)', 'title',font_size=14)

#%%

stc_vec = apply_inverse(evoked, inverse_operator, lambda2,
                        method=method, pick_ori='vector')
brain = stc_vec.plot(**surfer_kwargs)
brain.add_text(0.1, 0.9, 'Vector solution', 'title', font_size=20)
#del stc_vec
# You can write it to disk with::
#     >>> from mne.minimum_norm import write_inverse_operator
#     >>> write_inverse_operator('sample_audvis-meg-oct-6-inv.fif',
#                                inverse_operator)
#%%
for mi, (method, lims) in enumerate((('dSPM', [8, 12, 15]),
                                     ('sLORETA', [3, 5, 7]),
                                     ('eLORETA', [0.75, 1.25, 1.75]),)):
    
#    surfer_kwargs['clim']['lims'] = lims
    stc1 = apply_inverse(evoked, inverse_operator, lambda2,
                        method=method, pick_ori=None)
    brain = stc1.plot(figure=mi, **surfer_kwargs)
    brain.add_text(0.1, 0.9, method, 'title', font_size=20)

#%%
from mne.time_frequency import csd_morlet
from mne.beamformer import make_dics, apply_dics_csd


csd = csd_morlet(epochs, tmin=0, tmax=0.25, 
                 frequencies=np.linspace(2, 4))

# Compute DICS spatial filter and estimate source power.
filters = make_dics(epochs.info, fwd, csd, real_filter = True)

stc, freqs = apply_dics_csd(csd, filters)

message = 'DICS source power in the 8-12 Hz frequency band'
brain = stc.plot(surface='inflated', hemi='rh', subjects_dir=subjects_dir,
                 time_label=message, time_viewer = True)

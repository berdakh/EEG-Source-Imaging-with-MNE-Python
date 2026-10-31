"""
EEG processing and Movement related cortical potentials! 
==================================================
"""
import mne
import numpy as np
import os.path as op
import matplotlib.pyplot as plt 
import os, glob
   
#%% path to the EEG data 
subjects_dir = 'C:\\uhdata\\ES9014\\s14'
#filename = 'S9014_ses6_closeloop_block0009.vhdr'
os.chdir(subjects_dir)
#a = os.listdir(u'.')

#os.chdir(subjects_dir)
# find all files with .vmrk extension

#fname = op.join(subjects_dir, filename)
#d = mne.io.read_raw_brainvision(fname, preload=True)
#%% Read all file in the directory
allfiles = glob.glob("*.vhdr") 
  
afile = dict()
dat = dict()
for ii,jj in enumerate(allfiles):
#    afile.append(allfiles[ii])   
    afile.update({ii:jj})
    d = mne.io.read_raw_brainvision(jj, preload=True)
    dat.update({ii:d})
#    dt = mne.io.concatenate_raws([d],dat.update({ii:d}))

# dd = mne.io.concatenate_raws([dat[0], dat[1], dat[2]])
#    dt = mne.io.read_raw_brainvision(jj, preload=True)
d = dat[0]

for kk, tt in enumerate(dat):
    if kk < len(dat):
        mne.io.concatenate_raws([d, dat[kk]])
    print(kk)

filename = allfiles[0]   
#%% Removing power-line noise with notch filtering
# plot events 
##plt.plot(d._data[-1])
d.notch_filter(np.arange(60, 241, 60), filter_length='auto',phase='zero')

#%% Get standard electrode positions 
montage = mne.channels.read_montage(kind = 'standard_1020')
d.set_montage(montage)
# One could also load subject specific 3D electrodes using the mne.channels.montage.read_dig_montage
# elec = mne.channels.montage.read_dig_montage(hsp=None, hpi=None, elp='S9011_ses14_electrodefile.elp',point_names=ch)

#%% 
# High-pass filtering to remove slow drifts
# To remove slow drifts, you can high pass.
d.filter(0.1, None, l_trans_bandwidth='auto', filter_length='auto',phase='zero')
#d.plot_psd(area_mode='range', tmax=10.0, picks=picks)
#%%##############################################################################
# Removing power-line noise with low-pass filtering
# low pass filtering below 50 Hz
d.filter(None, 40., h_trans_bandwidth='auto', filter_length='auto',phase='zero')
#d.plot_psd(area_mode='range', tmax=10.0, picks=picks)
#%%##############################################################################
# Downsampling and decimation
#d.resample(60, npad="auto")  # set sampling frequency to 100Hz
#%% BAD CHANNEL IDENTIFICATION 
#picks = mne.pick_types(d.info, meg=True, eeg=True, eog=True, stim=False, exclude='bads')

pick_chans = ['Fp1','Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'FC5', 'FC1', 'FC2', 'FC6', 'T7', 'C3', 'Cz',
 'C4', 'T8', 'CP5', 'CP1', 'CP2', 'CP6', 'P7', 'P3', 'Pz', 'P4', 'P8', 'AFz', 'O1', 'Oz',
 'O2', 'FCz', 'AF7', 'AF3', 'AF4', 'AF8', 'F5', 'F1', 'F2', 'F6', 'FC3', 'FC4', 'C5',
 'C1', 'C2', 'C6', 'CP3', 'CPz', 'CP4', 'P5', 'P1', 'P2', 'P6', 'PO7', 'PO3', 'POz', 'PO4',
 'PO8']

picks = mne.pick_channels(d.info["ch_names"], pick_chans)
baseline = (None, -0.5)  # means from the first instant to t = 0
# Define Epochs and compute an ERP for the movement onset condition.
events = mne.find_events(d, stim_channel=None, output='onset', 
                         consecutive='increasing', min_duration=0, 
                         shortest_event=1, mask=None,
                         uint_cast=False, mask_type='and', 
                         initial_event=False, verbose=None)

event_id, tmin, tmax = {'MRCP': 16}, -1.5, 1
reject = dict(eeg=25e-4)
epochs_params = dict(events=events, event_id=16, 
                     tmin=tmin, tmax=tmax,reject=reject)

# Segment continuous EEG data around the event onset 
ep = mne.Epochs(d, **epochs_params,picks=picks, 
                baseline=baseline,preload=True) 
#ep.plot_sensors()

#%% Drop channels used for EMG 
#ch_names = ['TP7','TP9', 'FT9', 'FT10', 'TP10']
#drop_ch_names =  ['L_B1', 'L_B2', 'L_T1', 'L_T2', 'R_B1', 'R_B2', 'R_T1', 'R_T2','STI 014']
#ep.drop_channels(ch_names)
#ep.info['bads'] = ch_names
#d.info['bads'] = ['Fp1', 'Fp2', 'AF7']
#%% Bad channel removal
#import outlierdetection
from outlierdetection import idOutliers

data = np.rollaxis(ep.get_data(), 0, 2)

A = idOutliers(data.reshape(56,-1), dim=1, 
               threshold=(None,1.6), maxIter=1, feat="var")

bad = list(A[1])
ch_names = ep.info['ch_names']

b=[]

for bads in bad:
    b.append(ch_names[bads])    
print(b)

#%%
ep.info['bads'] = b
ep.interpolate_bads(reset_bads='True', mode = 'accurate')
ep.set_eeg_reference(ref_channels = "average", projection=True) 

#%% Artifact Correction with ICA
"""
ICA finds directions in the feature space
corresponding to projections with high non-Gaussianity. We thus obtain
a decomposition into independent components, and the artifact's contribution
is localized in only a small number of components.
These components have to be correctly identified and removed.

If EOG or ECG recordings are available, they can be used in ICA to
automatically select the corresponding artifact components from the
decomposition. To do so, you have to first build an Epoch object around
blink or heartbeat event.
"""
from mne.preprocessing import ICA
# ICA parameters:
n_components = 10  # if float, select n_components by explained variance of PCA
method = 'fastica'  # for comparison with EEGLAB try "extended-infomax" here
decim = 3  # we need sufficient statistics, not all time points -> saves time

# we will also set state of the random number generator - ICA is a
# non-deterministic algorithm, but we want to have the same decomposition
# and the same order of components each time this tutorial is run
random_state = 23
picks_eeg = mne.pick_types(ep.info, meg=False, eeg=True, 
                           eog=False, stim=False, exclude='bads')
###############################################################################
# Define the ICA object instance
ica = ICA(n_components=n_components, method=method, random_state=random_state)
print(ica)
#%%##############################################################################
# we avoid fitting ICA on crazy environmental artifacts that would
# dominate the variance and decomposition
reject = dict(eeg=40e-6)

ica.fit(ep, picks=picks_eeg, reject = reject, decim=decim)
print(ica)
#ica.plot_components()  # can you spot some potential bad guys?
#%%##############################################################################
# Component properties
# Let's take a closer look at properties of first three independent components.
# first, component 0:
#ica.plot_properties(d, picks=0)
# we can see that the data were filtered so the spectrum plot is not
# very informative, let's change that:
#ica.plot_properties(d, picks=0, psd_args={'fmax': 35.})
#%%##############################################################################
# we can also take a look at multiple different components at once:
#ica.plot_properties(d, picks=[1, 2], psd_args={'fmax': 35.})
#%% INTERACTIVE PLOT 
#ica.plot_components(picks=range(10), inst=d)

#%% Advanced artifact detection
#%% We simplify things by setting the maximum number of components to reject
eog_inds, scores = ica.find_bads_eog(ep, ch_name = 'Fp1', threshold=1)  # find via correlation
#ica.plot_scores(scores, exclude=eog_inds)  # look at r scores of components
# we can see that only one component is highly correlated and that this
# component got detected by our correlation analysis (red).
#ica.plot_sources(d, exclude=eog_inds)  # look at source time course
# We can take a look at the properties of that component, now using the
# data epoched with respect to EOG events. We will also use a little bit of smoothing along the trials axis in the
# epochs image:
#ica.plot_properties(d, picks=eog_inds, psd_args={'fmax': 35.},image_args={'sigma': 1.})
ica.exclude.extend(eog_inds)
# from now on the ICA will reject this component even if no exclude
# parameter is passed, and this information will be stored to disk
# on saving uncomment this for reading and writing
# ica.save('my-ica.fif')
# ica = read_ica('my-ica.fif')
ep = ica.apply(ep, exclude=eog_inds)
# Perform CAR operation
#%% 
# To do the low-pass and high-pass filtering in one step you can do
# a *band-pass* filter by running the following: band-pass filtering in the range .1 Hz - 4 Hz
ep.filter(0.1, 4., l_trans_bandwidth='auto', h_trans_bandwidth='auto',
          filter_length='auto', phase='zero')

#%% save the data as -epochs 
# mne.io.write_info('tmp-ave.fif', d)
filename2save = "".join([filename.split(".")[0],'-epo.fif'])
ep.save(filename2save)

ep.plot_image(combine='gfp', group_by='type',
              sigma=2., cmap="YlGnBu_r") 

ep.average().plot().savefig(filename.split("_b")[0])
ep.average().plot_image().savefig('plot_image.png')

#%% some visualization examples 
"""
d.average().plot_image().savefig('plot_image.png')
ep.average().plot()
ep.plot_topo_image(vmin=-10, vmax=10, title='ERF images', sigma=2.,
                       fig_facecolor='w', font_color='k')
ep.plot_image(combine='gfp', group_by='type', sigma=2., cmap="YlGnBu_r")
"""

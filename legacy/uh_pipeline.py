"""
EEG processing and Movement related cortical potentials! 
==================================================
"""
import mne
import numpy as np
import os.path as op
#import glob, os
   
#%% path to the EEG data 
subjects_dir = 'C:\\Users\\Berdakh\\Google Drive\\uhdata\\data\\Subject_S9009'
#os.chdir(subjects_dir)
#all_files = glob.glob("*.vmrk") # find all files with .vmrk extension

filename = 'S9011_ses1_cond1_block0001.vhdr'

fname = op.join(subjects_dir, filename)
d = mne.io.read_raw_brainvision(fname, preload=True)

#% Extracting events and plotting events 
events = mne.find_events(d, stim_channel=None, output='onset', 
                         consecutive='increasing', min_duration=0, 
                         shortest_event=1, mask=None,
                         uint_cast=False, mask_type='and', 
                         initial_event=False, verbose=None)

#%% Removing power-line noise with notch filtering
picks = mne.pick_types(d.info, eeg=True, eog=False,stim=False, exclude='bads')
d.notch_filter(np.arange(60, 241, 60), picks=picks, filter_length='auto',phase='zero')
#d.plot_psd(area_mode='range', tmax=10.0, picks=picks)
# plot events 
#import matplotlib.pyplot as plt
##plt.plot(d._data[-1])
#print(d.info)
#%% load the datafile, and extract the variables
from scipy.io import loadmat
import scipy
import numpy as np

data=loadmat('ERPdata.mat')
X     =data['X']
Y     =data['Y'].reshape(-1) # ensure is 1d
fs    =data['fs'][0] # ensure is scalar
Cnames=data['Cnames']
Cpos  =data['Cpos']
ylabel='time (s)'
yvals =np.arange(0,X.shape[1])/fs  # element labels for 2nd dim of X


#%% Define Epochs and compute an ERP for the movement onset condition.
event_id, tmin, tmax = {'MRCP': 16}, -2.5, 1
epochs_params = dict(events=events, event_id=16, tmin=tmin, tmax=tmax,
                     reject=None)

# segment continuous EEG data around the event onset 
d = mne.Epochs(d, **epochs_params, detrend = 0, preload=True, baseline=(-0.2, 0.0)) 
#%%evoked_no_ref.plot_topomap(times=[0.1], size=3., title=title)
X2 = scipy.signal.detrend(X1, axis=2, type='linear')


#%% Drop channels used for EMG 
#ch_names = ['TP9', 'FT9', 'FT10', 'TP10']
drop_ch_names =  ['L_B1', 'L_B2', 'L_T1', 'L_T2', 'R_B1', 'R_B2', 'R_T1', 'R_T2','STI 014']
d.drop_channels(drop_ch_names)

#d.info['bads'] = ['Fp1', 'Fp2', 'AF7']

#%% Get standard electrode positions 
montage = mne.channels.read_montage(kind = 'standard_1020', ch_names = d.info['ch_names'])
d.set_montage(montage)
# One could also load subject specific 3D electrodes using the mne.channels.montage.read_dig_montage
# elec = mne.channels.montage.read_dig_montage(hsp=None, hpi=None, elp='S9011_ses14_electrodefile.elp',point_names=ch)

#%% 
# High-pass filtering to remove slow drifts
# To remove slow drifts, you can high pass.
# .. warning:: There can be issues using high-passes greater than 0.1 Hz
#              (see examples in :ref:`tut_filtering_hp_problems`),
#              so apply high-pass filters with caution.
d.filter(1, None, l_trans_bandwidth='auto', filter_length='auto',
           phase='zero')
#d.plot_psd(area_mode='range', tmax=10.0, picks=picks)
#%%##############################################################################
# Removing power-line noise with low-pass filtering
# If you're only interested in low frequencies, below the peaks of power-line
# noise you can simply low pass filter the data.
# low pass filtering below 50 Hz
d.filter(None, 40., h_trans_bandwidth='auto', filter_length='auto',
           phase='zero')
#d.plot_psd(area_mode='range', tmax=10.0, picks=picks)
#%%##############################################################################
# Downsampling and decimation
# When performing experiments where timing is critical, a signal with a high
# sampling rate is desired. However, having a signal with a much higher
# sampling rate than necessary needlessly consumes memory and slows down
# computations operating on the data. To avoid that, you can downsample
# your time series. Since downsampling raw data reduces the timing precision
# of events, it is recommended only for use in procedures that do not require
# optimal precision, e.g. computing EOG or ECG projectors on long recordings.
#
# .. note:: A *downsampling* operation performs a low-pass (to prevent
#           aliasing) followed by *decimation*, which selects every
#           :math:`N^{th}` sample from the signal. See
#           :func:`scipy.signal.resample` and
#           :func:`scipy.signal.resample_poly` for examples.
# Data resampling can be done with *resample* methods.
d.resample(60, npad="auto")  # set sampling frequency to 100Hz
#d.plot_psd(area_mode='range', tmax=10.0, picks=picks)

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
n_components = 20  # if float, select n_components by explained variance of PCA
method = 'fastica'  # for comparison with EEGLAB try "extended-infomax" here
decim = 3  # we need sufficient statistics, not all time points -> saves time

# we will also set state of the random number generator - ICA is a
# non-deterministic algorithm, but we want to have the same decomposition
# and the same order of components each time this tutorial is run
random_state = 23
picks_eeg = mne.pick_types(d.info, meg=False, eeg=True, eog=False,
                           stim=False, exclude='bads')
###############################################################################
# Define the ICA object instance
ica = ICA(n_components=n_components, method=method, random_state=random_state)
print(ica)
#%%##############################################################################
# we avoid fitting ICA on crazy environmental artifacts that would
# dominate the variance and decomposition
ica.fit(d, picks=picks_eeg, decim=decim)
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
eog_inds, scores = ica.find_bads_eog(d, ch_name = 'Fp2', threshold=2.5)  # find via correlation
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
d = ica.apply(d, exclude=eog_inds)
# Perform CAR operation
d.set_eeg_reference(ref_channels = "average", projection=False) 
#%% 
# To do the low-pass and high-pass filtering in one step you can do
# a *band-pass* filter by running the following: band-pass filtering in the range .1 Hz - 4 Hz
d.filter(.1, 4., l_trans_bandwidth='auto', h_trans_bandwidth='auto',
          filter_length='auto', phase='zero')

# interpolate bad channels if exist 
d.interpolate_bads(reset_bads='True', mode = 'accurate')
#%% save the data as -epochs 
# mne.io.write_info('tmp-ave.fif', d)
filename2save = "".join([filename.split(".")[0],'-epo.fif'])
d.save(filename2save)

#%% some visualization examples 
"""
d.average().plot_image()
d.average().plot()
d.plot_topo_image(vmin=-10, vmax=10, title='ERF images', sigma=2.,
                       fig_facecolor='w', font_color='k')
d.plot_image(combine='gfp', group_by='type', sigma=2., cmap="YlGnBu_r")
"""
 
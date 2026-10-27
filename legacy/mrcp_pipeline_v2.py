# -*- coding: utf-8 -*-
"""
Created on Sun Jul  8 17:14:21 2018
@author: Berdakh
"""
import mne
import os, glob
from scipy.io import loadmat
import h5py
import numpy as np
#%%
def matlabprep(filename, classfilename):   
    
    fieldname = filename.split('.')[0]    
    cvfieldname = classfilename.split('.')[0]
    
    #%% Load matlab file 
    arrays = {}
    f = h5py.File(filename, 'r')
    
    for k, v in f.items():
        arrays[k] = np.array(v)
    
    dat = arrays[fieldname]
    print(dat.shape)
    
    data = np.rollaxis(dat, 2, 1)
    print(data.shape)
    
    trials = len(data[:,1,1])
    n_channels = len(data[1,:,1])
    
    #%% Load class file 
    labels = {}
    ff = h5py.File(classfilename, 'r')
    
    for kk, vv in ff.items():
        print(kk, vv)
        labels[kk] = np.array(vv)
        
    y = labels[cvfieldname]
    
    #%% Load channel labels 
    cc = loadmat('chanlabels.mat')
    cc = cc['chanlabels']
    cc = cc.tolist()
    cc = cc[0][:]
    
    chlabels = [j.tolist() for j in cc]
    ch_names = [item for list in chlabels for item in list]
    
    #%% channel labels 
    sampling_rate = 250
    info = mne.create_info(n_channels, sampling_rate)
    print(info)
    
    #%% Creating :class:`Epochs <mne.Epochs>` objects
    # ---------------------------------------------
    ## To create an :class:`mne.Epochs` object from scratch, you can use the
    # :class:`mne.EpochsArray` class, which uses a numpy array directly without
    # wrapping a raw object. The array must be of `shape(n_epochs, n_chans,
    # n_times)`. The proper units of measure are listed above.
    sfreq = 250
    
    # Initialize an info structure
    info = mne.create_info(
            ch_names = ch_names[:n_channels],
            ch_types = ['eeg']*n_channels,
            sfreq=sfreq )    
    #%% ##############################################################################
    # It is necessary to supply an "events" array in order to create an Epochs
    # object. This is of `shape(n_events, 3)` where the first column is the sample
    # number (time) of the event, the second column indicates the value from which
    # the transition is made from (only used when the new value is bigger than the
    # old one), and the third column is the new event value.
    
    # Create an event matrix: 10 events with alternating event codes
    y = [int(i) for i in y]
    #ch_names = [item for list in chlabels for item in list]
    
    ev = [int(i) for i in range(trials)]
    
    events = np.zeros([trials, 3])
    events[:,2] = y[:]
    
    events[:,0] = ev 
    events = events.astype('int')
    
    labeltypes = np.unique(events[:,2])
    print(labeltypes)
    
    #%%##############################################################################
    # More information about the event codes: subject had 9 classes     
    event_id =  {0 :'nothing',                
                 1 : 'baseline',
                 2 : 'baseEyeCls',
                 3 : 'drawing',  
                 4 : 'coloring',  
                 5 : 'writing',  
                 6 : 'cut', 
                 7 : 'paste', 
                 8 : 'plan', 
                 9 : 'correction'}    
    
    # find event ids 
    event_idd = [(event_id[i],int(i)) for i in labeltypes]
    event_id = dict(event_idd)    
    #%%##############################################################################
    # Finally, we specify the beginning of an epoch (the end will be inferred
    # from the sampling frequency and n_samples)
    # Trials were cut from -0.5 to 1.0 seconds
    tmin = -0.5
    
    ###############################################################################
    # Now we can create the :class:`mne.EpochsArray` object
    ep = mne.EpochsArray(data, info, events, tmin, event_id)
    
    # load a standard montage with 3D electrode positions 
    montage = mne.channels.read_montage(kind = 'standard_1020')
    ep.set_montage(montage)
    
    filename2save = filename.split('.')[0] + '_epo.fif'
    ep.save(filename2save)
    print(ep)
    
    # We can treat the epochs object as we would any other
#    _ = ep['drawing'].average().plot(time_unit='s')
    return ep 

#%% the following list of command will be run if you press F5 
    
if __name__ == '__main__':      
    # location of the files 
    mainDir = 'C:\\uhdata\\Jesus_data'
    os.chdir(mainDir)
    
    file = []
    file = glob.glob("*.mat")
    ffile = [file[1:10], file[10:]]
    
    d = []
    for jj, fname in enumerate(ffile[0]):
        classfilename = fname 
        filename = ffile[1][jj]     
        try:
            d.append(matlabprep(filename, classfilename))
        except Exception as err:
            print(err)    

#%%
# alternatively run this code on a subject by subject basis 
            
#filename = 'data_s4.mat'
#classfilename = 'cv_s6'
             
#d1 = matlabprep(filename, classfilename)
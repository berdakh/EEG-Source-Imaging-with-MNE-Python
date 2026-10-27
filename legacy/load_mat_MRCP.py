# -*- coding: utf-8 -*-
"""
Created on Fri May  3 10:46:54 2019

@author: berdakh.abibullaev
"""
import numpy as np
from scipy.io import loadmat 
import mne, glob 

#%%
def data2mne(data, y, sfreq, event_id, chan_names):
    """This function converts 3D array of EEG data into 
    MNE data structure for further analysis.    
    Input: 
        data = should have the following structure: [trials x chans x samples]
        sfreq = sampling frequency 
        y = [1, col] shape >>> class label array  [+/- 1]
        
    Output: mne data structure         
    ######### Example Usage #########    
    # [trials x chans x samples] 
    data = np.rollaxis(data, 2, 1)
    
    from nuClass import Preproc
    a = Preproc()    
    event_id = {1 :'pos', -1 : 'neg'}
    sfreq = 256    
    ep = a.data2mne(data, y, sfreq, event_id)
    """                 
    # More information about the event codes      
    event_id = event_id #{1 :'pos', -1 : 'neg'}        
    # trials = len(data[:,1,1])
    n_channels = len(data[1,:,1])    
    # ch_names = [str(i) for i in range(n_channels)]        
    # Initialize an info structure
    info = mne.create_info(
            ch_names = chan_names,
            ch_types = ['eeg']*n_channels,
            sfreq = sfreq )           
    print('Event created :', info)        
    # Create an event matrix: events with alternating event codes
    evLen = y.shape[0]
    ev = [i for i in range(evLen)]    
    # prepare events 
    events = np.zeros([evLen, 3])
    events[:,0] = ev 
    events[:,2] = y[:]
    events = events.astype('int')        
    #print('Events :', events)        
    # find event ids 
   # event_idd = [(event_id[i], int(i)) for i in events[:,2]]
    event_id = {'Onset': 1}
   # event_id = dict(event_idd)
    # Specify the beginning of an epoch 
    # (the end will be inferred from the sampling frequency and n_samples)
    tmin = -1                
    # Create the :class:`mne.EpochsArray` object
    epochs = mne.EpochsArray(data, info, events, tmin)    
    # read and applye the standard montage file
    montage = mne.channels.read_montage(kind = 'standard_1020')
    epochs.set_montage(montage)                
    # quick visualization 
    # epochs.average().plot()    
    return epochs

#%%
chans = ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'FC5', 'FC1', 'FC2', 'FC6',
          'T7', 'C3', 'Cz', 'C4', 'T8', 'CP5', 'CP1', 'CP2', 'CP6', 'P7', 'P3', 'Pz',
           'P4', 'P8', 'AFz', 'O1', 'Oz', 'O2', 'FCz', 'AF7', 'AF3', 'AF4', 'AF8',
           'F5', 'F1', 'F2', 'F6', 'FC3', 'FC4', 'C5', 'C1', 'C2', 'C6', 'CP3', 'CPz',
            'CP4', 'P5', 'P1', 'P2', 'P6', 'PO7', 'PO3', 'POz', 'PO4', 'PO8']

event_id = {'16': 16}
sfreq = 500



#%% Find matlab files 
import pickle

def batchMne(name):        
    matfiles = glob.glob('*.mat')           
    epochsall = [];    
    for filename in matfiles:        
        a = loadmat(filename)
        
        # channels x time x trials
        X = a['X']
        Y = np.ones(a['X'].shape[-1])
        Y = Y.astype(float)
            
        data = np.rollaxis(X, 2, 0)        
        epochs = data2mne(data, Y.T, sfreq, event_id, chans)     
        
        epochs.average().plot()
        # here we have all data in MNE format 
        epochsall.append(epochs)
        
        fname = name + '-epoall.pickle'
        
        with open(fname, 'wb') as file:
            pickle.dump(epochsall, file, protocol = pickle.HIGHEST_PROTOCOL)

#%%            
import os
            
foldertype = 'S'
maindir = 'C:\\uhdata\\freesurfer\\'

os.chdir(maindir)
mainfolders = os.listdir(u'.')

for fname in mainfolders:
    try:
        if fname[:1] == foldertype: # find all folders starting with ES
            os.chdir(maindir + fname)              
            print('Processing:', fname)                        
            batchMne(fname) 
            
    except Exception as err:
        print("Something went wrong while copying the data >>>", fname)       
        print(err)
        
os.chdir(maindir)
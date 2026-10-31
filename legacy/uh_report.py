# -*- coding: utf-8 -*-
"""
Created on Sun Jul  1 14:03:29 2018

@author: Berdakh
"""
import mne
import os
import glob
from mne.minimum_norm import make_inverse_operator, apply_inverse

import matplotlib
import numpy as np  # noqa
from mayavi import mlab  # noqa
from surfer import Brain  # noqa

#%% GLOBAL VARIABLES 
mainDir = 'C:\\uhdata\\Jesus_data'
defFilename = 'default-source-name' + '-src.fif'  
 
method = 'MNE'
tminmax = {'tmin':-1, 'tmax':-0.5} 
initial_time = [0.0] # which source to visualize 
# go to the main directory

os.chdir(mainDir)
# find folders 
folder = os.listdir(u'.')             
   
# get folders starts with 'S' - freesurfer directory
subfolders = [f for f in folder if f[0] == 'S']
subject = subfolders[0]

curdir = os.path.join(mainDir, subject)
os.chdir(curdir) 

# 1) locate EEG files 
eegfiles = glob.glob('*-epo.fif')

# 2) locate the Forward solution and read it
fname_fwd = os.path.join(curdir, subject) +'-fwd.fif'
fwd = mne.read_forward_solution(fname_fwd)

stcs = []
# 3) for each eegfiles perform inverse mapping 
for file in eegfiles:
    # 4) read epochs 
    eegfilePath = os.path.join(curdir,file)        
    epochs = mne.read_epochs(eegfilePath, proj=True,preload=True, verbose=None)         
    
    #        tmin, tmax = epochs.time_as_index([-1, -0.5])  
    # 5) calculate noise covariance matrix 
    noise_cov  = mne.compute_covariance(epochs, keep_sample_mean=True, 
                                        tmin=tminmax['tmin'], tmax=tminmax['tmax'])      
    
    evoked = epochs.average().pick_types(eeg=True)
    
    # 6) make inverse operator 
    info = evoked.info
    inverse_operator = make_inverse_operator(info, fwd, noise_cov, loose=0.2, depth=None) 
    
    # 7) apply inverse solution 
    snr = 3.
    lambda2 = 1. / snr ** 2
    stc = apply_inverse(evoked, inverse_operator, lambda2, method= method)        
    
    # 8) save the source result 
    fname2save = file.split('-')[0] 
    stcs.append(stc)
    stcs.save(fname2save, ftype='stc', verbose=None) 

#%%
mainDir = 'C:\\uhdata\\freesurfer'
#        mainDir = Inverse.mainDir
initial_time = [0.0] # Inverse.initial_time

os.chdir(mainDir)
folder = os.listdir(u'.')

subfolders = [f for f in folder if f[0] == 'S']    

#%%
for subject in subfolders:        
    
    # go to the directory             
    curdir = os.path.join(mainDir, subject)
    os.chdir(curdir)
    
    # locate source files
    stcfiles = glob.glob('*-rh.stc')           
    
    for ii, fname in enumerate(stcfiles):
        
        # read the source files 
        print(fname)
        stc = mne.read_source_estimate(os.path.join(curdir, fname))
        plottitle = fname.split('_c')[0]     

        try:# plot the surface  
            # Get labels for FreeSurfer 'aparc' cortical parcellation with 34 labels/hemi
            labels_parc = mne.read_labels_from_annot(subject, parc='aparc', subjects_dir = mainDir)
            
            # plot the sources 
            brain = stc.plot(subject=subject, surface='inflated', hemi='both', colormap='auto', 
                         time_label='auto', smoothing_steps=10, transparent=True, alpha=0.8, 
                         time_viewer=False, subjects_dir=None, figure=None, views='dor', 
                         colorbar=True, clim='auto', cortex='classic', size=800, background='black',
                         foreground='white', 
                         initial_time = initial_time, time_unit='s')
        
            vertno_max, time_max = stc.get_peak(hemi='lh')
            brain.add_foci(vertno_max, coords_as_verts=True, hemi='lh', color='red')
            
            vertno_max_rh, time_max_rh = stc.get_peak(hemi='rh')
            brain.add_foci(vertno_max_rh, coords_as_verts=True, hemi='rh', color='blue')
            
            brain.add_text(0.1, 0.9, plottitle, 'title', font_size=18) 
        
        except Exception as errMessage:
            print('Something went wrong with subject >>>', fname)
            print(errMessage)
            pass
        """
        views : str | list
            View to use. See surfer.Brain(). Supported views: ['lat', 'med', 'fos',
            'cau', 'dor' 'ven', 'fro', 'par']. Using multiple views is not
            supported for mpl backend.
        """            
        labelo = []
        try:    
            #brain.add_annotation
            labelo.append([label for label in labels_parc if label.name == 'postcentral-lh'][0]) 
            labelo.append([label for label in labels_parc if label.name == 'postcentral-rh'][0]) 
            labelo.append([label for label in labels_parc if label.name == 'precentral-lh'][0] )
            labelo.append([label for label in labels_parc if label.name == 'precentral-rh'][0] )               
            labelo.append([label for label in labels_parc if label.name == 'inferiorparietal-lh'][0]) 
            labelo.append([label for label in labels_parc if label.name == 'inferiorparietal-rh'][0]) 
            labelo.append([label for label in labels_parc if label.name == 'superiorfrontal-rh'][0]) 
            labelo.append([label for label in labels_parc if label.name == 'superiorfrontal-lh'][0])                 
         
            for label in labelo:
                brain.add_label(label, borders=True)
                
        except Exception as errMessage:
            print(errMessage)
            pass         
        
        try: # make a directory called 'results'
            os.mkdir(os.path.join(curdir,'results'))
        except Exception as errMessage:
            print(errMessage)                                      
          
        finally:
            os.chdir(os.path.join(curdir,'results'))                
            brain.save_image(fname.split('_c')[0])                        
            brain.close()
            
os.chdir(mainDir)   
#%%
pngfiles = glob.glob('*.png')
pngfiles.sort(key = lambda x:x[0])
pngfiles.sort(key = lambda x:x[1])

print(pngfiles)
#%% subplot 
mainDir = 'C:\\uhdata\\freesurfer'

os.chdir(mainDir)        
folder = os.listdir(u'.')        
folders = [f for f in folder if f[0] == 'S']   

for subject in folders:    
    try: # go to the 'results' directory
#        resultsDir = os.path.join(os.path.join(mainDir, subject),'results')
        os.chdir(os.path.join(mainDir, subject))
        
        #%% find all files with .png extension             
        pngfiles = glob.glob('*.png')
        pngfiles.sort(key = lambda x:x[0])
        pngfiles.sort(key = lambda x:x[1])
        print(pngfiles)
       
        fig = matplotlib.pyplot.figure()
        for ii, filename in enumerate(pngfiles):        
            f = matplotlib.pyplot.subplot(4,4,ii+1)
            f.set_axis_off()
            f.set_xlabel('ses:'+str(ii+1))#    f.set_figheight(15)
            fig.set_figwidth(30)
            fig.set_figheight(30)
            fig.tight_layout()
            img = matplotlib.image.imread(filename)    
            matplotlib.pyplot.imshow(img)   

        figname = subject + '_subplot'+ '.png'
        matplotlib.pyplot.savefig(figname) 
  #%%      
    except Exception as errMessage:
        print(errMessage)       

#%%         
from uhClass import Inverse

time = [0, 0.1, 0.25, 0.3]

for time1 in time:    
    a = Inverse()     
    a.initial_time = time1
    a.folder_name = 'results_white' + str(a.initial_time)
    a.plotsources()

#%% load electrode positions 
import os, mne 
from uhClass import MRCP

ch = MRCP.pick_chans_default

epochs_name = 'S9007_ses3_closeloop_block0000-epo.fif'
epochs_path = os.path.join('C:\\uhdata\\freesurfer\\S9007', epochs_name)                       
epochs = mne.read_epochs(epochs_path)
epochs.plot_sensors(show_names=True)

#%%
mon = mne.channels.read_montage(kind = 'standard_1020')

b = mon.ch_names[:69]
b[2] = 'Nasion'
#%%
elec = mne.channels.montage.read_dig_montage(hsp=None, hpi=None, transform = False,
                                             elp='S9007_session11_electrodefile.elp', point_names=b) 


#%% 
epochs.set_montage(elec) 
  
epochs.plot_sensors(show_names=True)

    
    
    
    
    
    
    
    
    
    
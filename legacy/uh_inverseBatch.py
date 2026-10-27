"""
Created on Thu Jun 21 10:08:14 2018
@author: Berdakh
"""
import os
import mne
import glob
from mne.minimum_norm import make_inverse_operator, apply_inverse
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.pyplot import savefig

#%% batch load all the source spaces and calculate inverse + save it in the directory 
mainDir = 'C:\\uhdata\\freesurfer'
os.chdir(mainDir)
folder = os.listdir(u'.')
mne.utils.set_config("SUBJECTS_DIR", 'C:\\uhdata\\freesurfer', set_env=True)  

#[ expression for item in list if conditional ]
subfolders = [f for f in folder if f[0] == 'S']

method = "MNE"
True 
for subject in subfolders:        
    curdir = os.path.join(mainDir,subject)
    os.chdir(curdir)    
    eegfiles = glob.glob('*-epo.fif')
    
    fname_fwd = os.path.join(curdir, subject)+'-fwd.fif'
    fwd = mne.read_forward_solution(fname_fwd)
    
    for file in eegfiles:                
        eegfile = os.path.join(curdir,file)        
        epochs = mne.read_epochs(eegfile, proj=True,preload=True, verbose=None)         
        epochs.set_eeg_reference(ref_channels = "average", projection=True)
#        epochs.apply_proj()        
        tmin, tmax = [-1, -0.5]
#        tmin, tmax = epochs.time_as_index([-1, -0.5])
        noise_cov  = mne.compute_covariance(epochs, keep_sample_mean=True,tmin=tmin,tmax=tmax)        
        evoked = epochs.average().pick_types(eeg=True)
#        evoked.plot(time_unit='s')
        info = evoked.info
        inverse_operator = make_inverse_operator(info, fwd, noise_cov, loose=0.2, depth=None)        
        snr = 3.
        lambda2 = 1. / snr ** 2
        stc = apply_inverse(evoked, inverse_operator, lambda2, method=method)        
        fname = file.split('-')[0]    
        stc.save(fname, ftype='stc', verbose=None)
                              
#%% LOAD THE SOURCES to VISUALIZE and save it into the directory 
def plotsources():   
    
    mainDir = 'C:\\uhdata\\freesurfer'
    os.chdir(mainDir)
    folder = os.listdir(u'.')
    
    #[ expression for item in list if conditional ]
    subfolders = [f for f in folder if f[0] == 'S']    
    
    for subject in subfolders:     
        
        curdir = os.path.join(mainDir,subject)
        os.chdir(curdir)
        stcfiles = glob.glob('*-rh.stc')
        
#        plottitle = [jj.split('_c')[0] for jj in stcfiles]
        #%
        for ii, fname in enumerate(stcfiles): 
            print('>>>',ii+1)
            stc = mne.read_source_estimate(os.path.join(curdir,fname))
            plottitle = fname.split('_c')[0]     

            try:# plot the surface  
                # Get labels for FreeSurfer 'aparc' cortical parcellation with 34 labels/hemi
                labels_parc = mne.read_labels_from_annot(subject, parc='aparc',subjects_dir=mainDir)
                # plot the sources 
                brain = stc.plot(subject=subject, surface='inflated', hemi='both', colormap='auto', 
                             time_label='auto', smoothing_steps=10, transparent=True, alpha=0.8, 
                             time_viewer=False, subjects_dir=None, figure=None, views='dor', 
                             colorbar=True, clim='auto', cortex='classic', size=800, background='black',
                             foreground='white', 
                             initial_time=[0.25], time_unit='s', backend='auto', spacing='oct5')
            
                vertno_max, time_max = stc.get_peak(hemi='lh')
                brain.add_foci(vertno_max, coords_as_verts=True, hemi='lh', color='red')
                
                vertno_max_rh, time_max_rh = stc.get_peak(hemi='rh')
                brain.add_foci(vertno_max_rh, coords_as_verts=True, hemi='rh', color='blue')
                
                brain.add_text(0.1, 0.9, plottitle, 
                               'title', font_size=18) 
            
            except Exception:
                print('Something went wrong with subject >>>', fname)
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
                    
            except Exception as e:
                print(e)
                pass         
            
            try:
                os.mkdir(os.path.join(curdir,'results'))
            except Exception as e:
                print(e)
                pass                  
              
                os.chdir(os.path.join(curdir,'results'))                
                brain.save_image(fname.split('_c')[0])    
#                brain.save_movie(fname.split('_b')[0])
                brain.close()
                os.chdir(mainDir)      
                
#%% load multiple .png files in the directory and save it as subplot 
def subplott():
    pngfiles = glob.glob('*.png')
    #fig, ax = plt.subplots(4, 4, sharex='all', sharey='all')
    fig = plt.figure()
    
    for ii, filename in enumerate(pngfiles):        
        f = plt.subplot(4,4,ii+1)
        f.set_axis_off()
        f.set_xlabel('ses:'+str(ii+1))#    f.set_figheight(15)
        fig.set_figwidth(30)
        fig.set_figheight(30)
        fig.tight_layout()
        img=mpimg.imread(filename)    
        plt.imshow(img)        
   
    savefig( 'stitched.png')
   
#%%
#a = ['ses1','ses2','ses3','ses4','ses5','ses6','ses7','ses8','ses9','ses10',
#     'ses11','ses12','ses13','ses14']

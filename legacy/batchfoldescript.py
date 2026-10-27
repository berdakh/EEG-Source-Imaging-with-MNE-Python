# -*- coding: utf-8 -*-
"""
Created on Mon Jun 18 19:54:38 2018
THIS IS THE MAIN SCRIPT FOR BATCH PROCESSING 
This script calls batchfolder.py ----> uh_prepfn.py
@author: Berdakh
"""

#import batchfolder
import timeit
import mne
import os, glob
from uh import MRCP

mainDir = 'C:\\Users\\Berdakh\\uhdata'
os.chdir(mainDir)

# select only folders with "ES" prefix 

folder = [j for j in os.listdir(u'.')[1:-1] if 'ES' == j[:2]]

folder = folder[-1]
#%% 
start_time = timeit.default_timer()

def batchEntireDirectory():    
    
    for subjectname in folder:
        try:        
            print(subjectname) # jj is for example 'ES9009'
            batchPrepSubfolder(subjectname, mainDir)
            
        except Exception:
            print("Something went wrong while loading the data!")       
            pass    
    
    print('Time taken to finish', timeit.default_timer() - start_time)#%%


#%% 
def batchPrepSubfolder(subjectname, mainDir):       
    # path to the EEG data in a given subject directory 
    subjects_dir = os.path.join(mainDir,subjectname)    
    os.chdir(subjects_dir)    
    
    # list all the folders in the subjects directory (session specific)    
    session_folders = os.listdir(subjects_dir)    
    
    dat=[]    
    for folders in session_folders:    
        
        if folders[0] == 's': 
            # work with all the folders that start with 's'                        
            session_dir = os.path.join(subjects_dir,folders)           
            # enter the directory and find all files associated with .vhdr file extentions 
            os.chdir(session_dir)       
            allvhdr = glob.glob("*.vhdr") 
            dat = []
            for jj, filename in enumerate(allvhdr):    
                try:
                    # get the list of the files in the cwd 
                    d = mne.io.read_raw_brainvision(filename, preload=True)
                    dat.append(d)           
                except Exception as erMessage:
                    print(erMessage)       
                    pass  
                
            #  start concatinating if dat list is not empty 
            if bool(dat):            
                print('>>>>>>> Concatenating Files <<<<<<<')
                d0 = mne.io.concatenate_raws(dat)
                dat = []
                
            print('***Starting BATCH preprocessing at:***', session_dir)   
# =============================================================================
#   SIGNAL PROCESSING PIPE STARTS HERE 
# =============================================================================
            try:
                epoch_param = {'tmin':-1.5,
                               'tmax': 1,
                               'event_id': 16}                
                filename = allvhdr[0].split(".")[0]
                
                eeg = MRCP(filename2save = filename,
                          data_dir = session_dir,
                          epoch_param = epoch_param)
                
                ep = eeg.preproc(d0)   
                
                ep = eeg.ICA(ep)
                # save the file 
                eeg.save2epochs(ep)
                  
            except Exception as erMessage:
                print(erMessage)
                pass             

"""
# list comprehension solution:
dat = [mne.io.read_raw_brainvision(jj, preload=True) for jj in allvhdr]    
d0 = mne.io.concatenate_raws(dat)
"""







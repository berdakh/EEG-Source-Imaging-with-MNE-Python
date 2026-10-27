# -*- coding: utf-8 -*-
"""
Created on Jun 27 19:54:38 2018
THIS IS THE MAIN SCRIPT FOR BATCH PROCESSING EEG data 
This script calls uhClass.py 
@author: Berdakh
"""
#%% ##############################################################################
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
            epoch_param = {'tmin':-1,
                           'tmax': 1,
                           'event_id': 16}   
            try:
                
               filename = allvhdr[0].split(".")[0]

            except Exception:
                filename = 'subject'
                
            # we create an object MRCP which is bundles with methods
            # and attributes to process the data related to NRI project 
                          
            eeg = MRCP(filename2save = filename,
                      data_dir = session_dir,
                      epoch_param = epoch_param)

            try: 
                ep = eeg.preprocEEG(d0)                   
                ep.average().plot()
                
            except Exception as err:
                print(err)
                pass 
            
            filename = "".join([filename,'-epo.mat'])
            print('Saving file to :', filename)
            ep.save(filename, verbose = True)  
            data = np.moveaxis(ep.get_data(), 0, -1)
            
            matname = filename 
            io.savemat(matname, {'X':data})
            
            # pdb.set_trace()
            # pdb.set_trace = lambda: None

#%%#####
import timeit
import mne
import os, glob
from uhClass import MRCP
import pdb
import scipy.io as io
import numpy as np

get_ipython().run_line_magic('matplotlib', 'inline')
start_time = timeit.default_timer()

mainDir = 'C:\\uhdata'
#mainDir = '/home/hero/uhdata'

os.chdir(mainDir)
# select only folders with "ES" prefix 
folder = [j for j in os.listdir(u'.') if 'E' == j[:1]]

for subjectname in folder:
    print(subjectname) # jj is for example 'ES9009'
    batchPrepSubfolder(subjectname, mainDir)        
    
print('Time taken to finish', timeit.default_timer() - start_time)     


#%% move matlab files 
import shutil

source = 'C:\\uhdata' 
destination = 'C:\\uhdata\\freesurfer\\'       
foldername = 'ES' 
os.chdir(source)

#%% move all matlab files 

mainfolders = os.listdir(u'.')

for fname in mainfolders:

    if fname[:2] == foldername:
        subjectdir = os.path.join(source, fname)
        os.chdir(subjectdir)
        subfolders = os.listdir(u'.')
        
        # for each subject in the provided subfolders 
        for s in subfolders:
            if s[0] == 's':
                sessiondir = os.path.join(subjectdir, s)
                os.chdir(sessiondir)
                file = glob.glob("*.mat") # find files to move

                for files in file:                                
                    shutil.copy(os.path.join(sessiondir,files),
                            destination + fname[1:])         
        pass
    
#%% find and remove files
import os

file = glob.glob("*epo.mat")
for f in file:
    os.unlink(os.getcwd() + '\\' + f)
        
    
    
# -*- coding: utf-8 -*-
"""
This main function for batch processing of EEG data in the given subject directory
1) ES9009 >>>> S1....S15 

Created on Mon Jun 18 10:00:30 2018
@author: Berdakh
"""
import mne
import os, glob
import uh_prepfn 

def prep(subjectname, mainDir):    
    #%% path to the EEG data 
    subjects_dir = os.path.join(mainDir,subjectname)    
    os.chdir(subjects_dir)    
    # list all the folders in the subjects directory     
    dir_folders = os.listdir(subjects_dir)    
    
    dat=[]    
    for folder in dir_folders:           
        if folder[0] == 's': # work with all the folders that start with 's' and ignore others                         
            session_dir = os.path.join(subjects_dir,folder)            
            # enter the directory and find all files associated with .vhdr file extentions 
            os.chdir(session_dir)       
            allvhdr = glob.glob("*.vhdr") 
            
            for ii, jj in enumerate(allvhdr):    
                try:
                    # get the list of the files in the cwd 
                    d = mne.io.read_raw_brainvision(jj, preload=True)
                    dat.append(d)           
                except Exception as erMessage:
                    print(erMessage)       
                    pass  
            #  start concatinating if dat is not empty 
            if bool(dat):            
                print('>>>>>>> Concatenating Files <<<<<<<')
                d0 = mne.io.concatenate_raws(dat)
                dat = []
                
            print('***Starting BATCH preprocessing at:***' session_dir)            
            try:
                ep = uh.preproc(d0,allvhdr[0])
                ep = uh.ICA(ep) 
                 _ = uh.save(ep,allvhdr[0])
                  
            except Exception as erMessage:
                print(erMessage)
                pass             
"""
# list comprehension solution:
dat = [mne.io.read_raw_brainvision(jj, preload=True) for jj in allvhdr]    
d0 = mne.io.concatenate_raws(dat)
"""
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 18 10:00:30 2018

The script finds a list of brainvisition files 
in the given directory for a subject and 
applies a standard preprocessing method 
defined in the uh_prepfn.py 

Parameters:
    subjectname: is the folder name associated with a subject
    default directory is os.getcwd() or 'C:\\uhdata\\'
@author: Berdakh
"""
import mne
import os, glob
import uh_prepfn 
import timeit

mainDir = '/home/hero/uhdata/'
subjectname = 'S9023'

def batchfolder(subjectname, mainDir):    
    #%% path to the EEG data 
    start_time = timeit.default_timer()
    
    subjects_dir = mainDir + subjectname
    #filename = 'S9014_ses6_closeloop_block0009.vhdr'
    os.chdir(subjects_dir)
    folders = os.listdir(u'.')
    #os.chdir(subjects_dir)
    # find all files with .vmrk extension
    
    #fname = op.join(subjects_dir, filename)
    #d = mne.io.read_raw_brainvision(fname, preload=True)
    #%% Read all file in the directory
    #folder = folders[5:]
    dat=[]    
    for pp in folders:       
        print('$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$\n'
              'Processing folder:::::', pp,
              '\n$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$')    
        session_dir = subjects_dir +'/'+ pp    
        os.chdir(session_dir)       
        allfiles = glob.glob("*.vhdr") 
        
        for ii, jj in enumerate(allfiles):    
            try:            
                d = mne.io.read_raw_brainvision(jj, preload=True)
                dat.append(d)           
            except Exception:
                print("Something went wrong while loading the data!")       
                pass  
            
        if bool(dat):            
            print('>>>>>>> Concatenating Files <<<<<<<')
            d0 = mne.io.concatenate_raws(dat)
            dat = []
            print('Starting BATCH preprocessing')
            uh_prepfn.preproc(d0,allfiles[0])  
    """
    # list comprehension     
        raw_files = [mne.io.read_raw_brainvision(f, preload=True)
                    for f in allfiles]    
        raw = mne.io.concatenate_raws(raw_files)
    """         
    print('Time taken to finish', timeit.default_timer() - start_time)

if __name__ == '__main__':
    
    mainDir = '/home/hero/uhdata/'
    subjectname = 'ES9023'

    batchfolder(subjectname, mainDir)
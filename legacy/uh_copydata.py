# -*- coding: utf-8 -*-
"""
Created on Mon Jun 18 22:58:10 2018
@author: Berdakh
"""
import shutil 
import os 
import glob

maindir = 'C:\\uhdata\\'
os.chdir(maindir)
mainfolders = os.listdir(u'.')
destination = 'C:\\uhdata\\freesurfer\\' ## destination path

fname = 'ES9009'
for fname in mainfolders:
    try:
        if fname[:2] == 'ES': 
            subjectdir = os.path.join(maindir, fname)
            os.chdir(subjectdir)            
            subfolders = os.listdir(u'.')            

            for s in subfolders: 
                if s[0] == 's':
                    sessiondir = os.path.join(subjectdir, s)
                    os.chdir(sessiondir)                
                    file = glob.glob("*.fif") # find files to move 
                    
                    for files in file: 
                        shutil.copy(os.path.join(sessiondir,files), 
                                destination + fname[1:])         
    except Exception:
        print("Something went wrong while copying the data >>>", fname)       
        pass    
    
os.chdir(maindir)
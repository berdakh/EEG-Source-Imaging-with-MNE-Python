# -*- coding: utf-8 -*-
"""
Created on Mon Jun 18 22:58:10 2018
@author: Berdakh
"""
import shutil 
import os 
import glob
#%%
filetype = "*.mat"
foldertype = 'XS'

maindir = 'C:\\uhdata\\'
os.chdir(maindir)
mainfolders = os.listdir(u'.')
destination = 'C:\\uhdata\\freesurfer\\' ## destination path

for fname in mainfolders:
    try:
        if fname[:2] == foldertype: # find all folders starting with ES
            os.chdir(maindir + fname)            
            subfolders = os.listdir(u'.')
            
            for s in subfolders: # inside the session folder 
                os.chdir(maindir + fname + '\\' + s)
                file = glob.glob(filetype) # find files to move 
                
                for files in file: 
                    shutil.copy(maindir + fname + '\\' + s + '\\' + files, 
                                destination + fname[1:])           
        
    except Exception:
        print("Something went wrong while copying the data >>>", fname)       
        pass    

os.chdir(maindir)

#%%
# move files from one folder to another 

filetype = "*-epo.mat"
foldertype = 'S'

maindir = 'C:\\uhdata\\freesurfer\\'

os.chdir(maindir)
mainfolders = os.listdir(u'.')
destination = 'C:\\uhdata\\freesurfer\\chanlabel\\' ## destination path

for fname in mainfolders:
    try:
        if fname[:1] == foldertype: # find all folders starting with ES
            os.chdir(maindir + fname)   
            
            subfolders = os.listdir(u'.')
            file = glob.glob(filetype) # find files to move 
          
            for files in file: 
                   shutil.move(maindir + fname + '\\' + files, 
                               destination)                       
          
    except Exception as err:
        print("Something went wrong while copying the data >>>", fname)       
        print(err)

os.chdir(maindir)



# -*- coding: utf-8 -*-
"""
Created on Tue Jun 12 09:24:23 2018

@author: Berdakh
"""
mydir = 'C:\\Users\\Berdakh\\Google Drive\\uhdata\\data\\Subject_S9009';

import glob, os
os.chdir(mydir)
for file in glob.glob("*.vmrk"):
    
    print(file)
"""
Created on Sun Jun 10 20:15:59 2018
uh_forward solution preparation 
@author: Berdakh
"""
import os
import os.path as op
import mne
import numpy as np  # noqa
from mayavi import mlab  # noqa
from surfer import Brain  # noqa
import glob

#%% GLOBAL VARIABLES 
mainDir = 'C:\\Users\\Berdakh\\uhdata\\freesurfer'
os.chdir(mainDir)
folder = os.listdir(u'.')
#%%
#%% STEP 3: CO-REGISTRATION STEP (A MANUAL STEP)
#mne.gui.coregistration(subject=folder[4], subjects_dir=mainDir)
# at this stage you need to use GUI.

#%%
for subject in folder:     
    if subject[0] == 'S': # if folder starts with 'S' then proceed 
        os.chdir(os.path.join(mainDir,subject))
        #mne.gui.coregistration(subject=subjectName, subjects_dir=subjectDir)
    #% STEP 1: COMPUTER THE SOURCE SPACE (SOURCE GRID ON MRI)
        """
        Compute Source Space
        The source space defines the position of the candidate source locations. 
        The following code compute such a cortical source space with an OCT-5 resolution.
        """
        filename2save = "".join([subject,'-src.fif'])
        src = mne.setup_source_space(subject, spacing='oct5', subjects_dir = mainDir) 
        info = mne.write_source_spaces(filename2save,src, overwrite=True)            
        #src.plot(head=True,brain=False,skull=False,subjects_dir = subjectDir)
   
    #% STEP 2: COMPUTER THE FORWARD SOLUTION
        """
        The BEM solution requires a BEM model which describes the geometry 
        of the head the conductivities of the different tissues.
        
        NOTE that the BEM does not involve any use of the trans file. 
        The BEM only depends on the head geometry and conductivities. 
        It is therefore independent from the EEG data and the head position.
        
        The forward operator, commonly referred to as the gain or leadfield matrix
        requires the co-registration later on. 
        """
        conductivity = (0.3, 0.006, 0.3)  # for three layers
        model = mne.make_bem_model(subject, 
                                   subjects_dir=mainDir, 
                                   conductivity=conductivity,
                                   verbose=None)
        #bem2save = "".join([subjectName,'-bem.fif'])
        #mne.write_bem_surfaces(bem2save, model)     
        bem_sol = mne.make_bem_solution(model)
        # save bem solution 
        bem2solution = "".join([subject,'-bemsol.fif'])
        mne.write_bem_solution(bem2solution, bem_sol)
    
#        #%% STEP 3: CO-REGISTRATION STEP (MANUAL STEP)
#        mne.gui.coregistration(subject=subjectName, subjects_dir=subjectDir)
#        # at this stage you need to use GUI.
    
        #% MAKE FORWARD SOLUTION  
        trans = op.join(os.getcwd(), "".join([subject,'-trans.fif'])) 
        # 
        eegfile = os.path.join(os.path.join(mainDir,subject), glob.glob("*-epo.fif")[0])
    #    info = mne.io.read_info(raw_fname)
    #    
    #    # Here we look at the dense head, which isn't used for BEM computations but
    #    # is useful for coregistration.
    #    mne.viz.plot_alignment(info, trans=trans, subject=subjectName, subjects_dir=subjectDir, 
    #                           surfaces=['head'], coord_frame='head', meg=None, 
    #                           eeg='original', 
    #                           dig=True, ecog=False, src=src, mri_fiducials=False, bem=None, 
    #                           seeg=False,show_axes=False, fig=None, 
    #                           interaction ='trackball', verbose=None)
        #% will take a few mintues
    #    folder = os.listdir(u'.')    
        fwd = mne.make_forward_solution(eegfile, trans=trans, src=src, bem=bem_sol,
                                        meg=False, eeg=True, mindist=5.0)
        print(fwd)
    
        #% 
        fwdname = "".join([subject,'-fwd.fif'])
        mne.write_forward_solution(fwdname, fwd, overwrite=True, verbose=None)
        print("Processing finished for >>>", subject)    

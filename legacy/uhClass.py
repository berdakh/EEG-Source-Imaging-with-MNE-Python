import mne
import os
import numpy
import glob
from mne.minimum_norm import make_inverse_operator, apply_inverse

import matplotlib
import matplotlib.pyplot as plt
import numpy as np  # noqa
#from mayavi import mlab  # noqa
#from surfer import Brain  # noqa

#%%
class Forward(object):
    #%% GLOBAL CLASS VARIABLES
    mainDir = 'C:\\uhdata\\freesurfer'
    defFilename = 'default-source-name' + '-src.fif'
    foldername = 'S' # look for a foldername that stats with 'S' 
    
    def __init__(self, mainDir = mainDir,
                 filename2save = defFilename):

        self.mainDir = mainDir
        self.filename2save = filename2save
        

#%% CO-REGISTRATION STEP (MANUAL STEP)
    @staticmethod 
    def coregister(subjectName):
        """
        Usage example:
            
        from uhClass import Forward 
        subjectName = 'S9017'
        
        Forward.coregister(subjectName)
        Forward.plotAlignment(subjectName)
        """        
        mne.gui.coregistration(subject = subjectName, subjects_dir = Forward.mainDir)               

#%% Forward solution
    def forwardSolBatch(self):        
        """
        Usage example:
        --------------
        from uhClass import Forward
        a = Forward()        
        a.forwardSolBatch()        
        """
        #mainDir = 'C:\\uhdata\\freesurfer'
        
        os.chdir(self.mainDir)
        folder = os.listdir(u'.')

        for subject in folder:            
    
            if subject[0] == self.foldername: # if folder starts with 'S' then proceed
                os.chdir(os.path.join(self.mainDir,subject))
    
                #% STEP 1: COMPUTER THE SOURCE SPACE (SOURCE GRID ON MRI)
    
                """
                The source space defines the position of the candidate source locations.
                The following code compute such a cortical source space with an
                OCT-5 resolution.
                """
    
                filename2save = "".join([subject,'-src.fif'])
                
                src = mne.setup_source_space(subject, spacing='oct5', subjects_dir = self.mainDir)                
                mne.write_source_spaces(filename2save, src, overwrite=True)
    
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
                                           subjects_dir=self.mainDir,
                                           conductivity=conductivity,
                                           verbose=None)
    
                #bem2save = "".join([subjectName,'-bem.fif'])
                #mne.write_bem_surfaces(bem2save, model)
                
                bem_sol = mne.make_bem_solution(model)
                
                bem2solution = "".join([subject,'-bemsol.fif'])
                mne.write_bem_solution(bem2solution, bem_sol)
    
                #%% STEP 3: CO-REGISTRATION STEP (MANUAL STEP)
                # Forward.coregister(subject)
    
                #% MAKE FORWARD SOLUTION
                trans = os.path.join(os.getcwd(), "".join([subject,'-trans.fif']))
    
                eegfile = os.path.join(os.path.join(self.mainDir,subject), glob.glob("*-epo.fif")[0])    
                fwd = mne.make_forward_solution(eegfile, trans=trans, src=src, bem=bem_sol, meg=False, eeg=True, mindist=5.0)
                    
                fwdname = "".join([subject,'-fwd.fif'])
                mne.write_forward_solution(fwdname, fwd, overwrite=True, verbose=None)               
                print("Processing finished for >>>", subject)
                

#%% Plot alignment
    @staticmethod 
    def plotAlignment(subjectName):
        
        """
        Usage example:
        -----------------------------
        
        from uhClass import Forward     
        subjectName = 'xS9017'
        Forward.plotAlignment(subjectName)            
        """

        import glob, os
        # mainDir = 'C:\\uhdata\\freesurfer'
        
        mainDir = Forward.mainDir
        path = os.path.join(mainDir, subjectName)
        os.chdir(path)

        trans = glob.glob('*-trans.fif')
        # bemsol = glob.glob('*-bemsol.fif')
        
        src = glob.glob('*-src.fif')
        epo = glob.glob('*-epo.fif')

        info = mne.io.read_info(epo[0])

        # read source space
        stc = mne.source_space.read_source_spaces(os.path.join(path, src[0]))

        # read bem solution
        # bem_path = os.path.join(path, bemsol)
        # bem = mne.read_bem_solution(bem_path)

        # read transformation matrix
        trans_path = os.path.join(path, trans[0])
        trans = mne.read_trans(trans_path)
        
       # Here we look at the dense head, which isn't used for BEM computations but
       # is useful for coregistration.
       
        mne.viz.plot_alignment(info, trans=trans, 
                               subject=subjectName, 
                               subjects_dir=mainDir,
                               surfaces=['head'], coord_frame='head', meg=None,
                               eeg='original', dig=True, ecog=False, 
                               src=stc, mri_fiducials=False, bem=None,
                               seeg=False, show_axes=False, fig=None,
                               interaction ='trackball', verbose=None)                
# =============================================================================
# Inverse solution
# =============================================================================
class Inverse(object):    
    """ 
        EEG inverse mapping wrapper class to MNE-Python
        It binds all the necessary functions to perform
        Source Localication and Mapping or Vizualisation
    """    
    mainDir = 'C:\\uhdata\\freesurfer'
    defMethod = 'MNE'
    tminmax = {'tmin':-0.7, 'tmax':-0.5} # covariance tmin/tmax 
    foldername = 'S'
    
    initial_time = 0.0 # which source to visualize

    folder_name = 'results_' + str(initial_time)

    def __init__(self,
                 mainDir = mainDir,
                 method = defMethod,
                 tmin = tminmax['tmin'],
                 tmax = tminmax['tmax'],
                 initial_time = initial_time):

        self.mainDir = mainDir
        self.method = method
        self.tmin = tmin
        self.tmax = tmax
        self.initial_time = initial_time

    def __str__(self):
        return print("Inverse mapping object wrapper to MNE")
    
# =============================================================================
#%%  BATCH INVERSE FOR NRI PROJECT
# =============================================================================
    def batchInverse(self):
        """
        Batch load all the source spaces and calculate inverse + save it in the directory
            This funciton requires the following input paramters:
                1) A directory where all subject folders are located
                2) Epoched, preprocessed EEG files with -epo.fif extension
                3) A subject specific "FORWARD" solution with -fwd.fif extention
        
           
        Example usage:
        ---------------------------------------------------
        from uhClass import Inverse
        
        a = Inverse()
        a.dataDir = 'C:\\uhdata\\freesurfer'
        a.defMethod = 'MNE'
        a.tminmax = {'tmin':-0.2, 'tmax':0}
        
        a.initial_time = 0.0 # a time point for source visualization
        a.batchInverse()               
         
        """
        
        os.chdir(self.mainDir)            
        # go to the main directory
         # find folders
        folder = os.listdir(u'.')

        # get folders starts with 'S' - freesurfer directory
        subfolders = [f for f in folder if f[:1] == self.foldername]

        for subject in subfolders:

            curdir = os.path.join(self.mainDir, subject)
            os.chdir(curdir)

            # 1) locate EEG files
            eegfiles = glob.glob('*-epo.fif')

            # 2) locate the Forward solution and read it
            fname_fwd = os.path.join(curdir, subject) +'-fwd.fif'
            fwd = mne.read_forward_solution(fname_fwd)

            # 3) for each eegfiles perform inverse mapping
            for file in eegfiles:

                # 4) read epochs
                eegfilePath = os.path.join(curdir,file)
                epochs = mne.read_epochs(eegfilePath, proj=True, preload=True, verbose=None)
                epochs.set_eeg_reference(ref_channels = "average", projection=True)


        #        tmin, tmax = epochs.time_as_index([-1, -0.5])
                # 5) calculate noise covariance matrix
                noise_cov  = mne.compute_covariance(epochs, keep_sample_mean=True, tmin=self.tmin, tmax=self.tmax)
                evoked = epochs.average().pick_types(eeg=True)
                
                # 6) make inverse operator
                info = evoked.info
                inverse_operator = make_inverse_operator(info, fwd, noise_cov, loose=0.2, depth=None)

                # 7) apply inverse solution
                snr = 3.
                lambda2 = 1. / snr ** 2
                stc = apply_inverse(evoked, inverse_operator, lambda2, method= self.method)

                # 8) save the source result
                fname2save = file.split('-')[0]
                stc.save(fname2save, ftype='stc', verbose=None)


        print(">>> Inverse mapping is complete <<<")
        
        # go back to the main directory
        os.chdir(self.mainDir)

# =============================================================================
#%%  BATCH load and visualize the sources
# =============================================================================
#    @staticmethod    
    def plotsources(self):

        """
        BATCH LOAD THE SOURCES to VISUALIZE and save it into the directory
        INPUT: source files *-rh.stc or *-lh.stc

        """        
        mainDir = Inverse.mainDir
        folder_name = self.folder_name
        os.chdir(mainDir)
        
        folder = os.listdir(u'.')
        subfolders = [f for f in folder if f[0] == 'S']

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
                    brain = stc.plot(subject=subject, surface='pial', hemi='both', colormap='auto',
                                 time_label='auto', smoothing_steps=10, transparent=True, alpha=0.8,
                                 time_viewer=False, subjects_dir=None, figure=None, views='dor',
                                 colorbar=True, clim='auto', cortex='classic', size=800, background='black',
                                 foreground='white',
                                 initial_time = self.initial_time, time_unit='s', backend='auto', spacing='oct5')

                    vertno_max, time_max = stc.get_peak(hemi='lh')
                    brain.add_foci(vertno_max, coords_as_verts=True, hemi='lh', color='red')

                    vertno_max_rh, time_max_rh = stc.get_peak(hemi='rh')
                    brain.add_foci(vertno_max_rh, coords_as_verts=True, hemi='rh', color='blue')

                    brain.add_text(0.1, 0.9, plottitle, 'title', font_size=36)

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
#                   labelo.append([label for label in labels_parc if label.name == 'inferiorparietal-lh'][0])
#                   labelo.append([label for label in labels_parc if label.name == 'inferiorparietal-rh'][0])
#                   labelo.append([label for label in labels_parc if label.name == 'superiorfrontal-rh'][0])
#                   labelo.append([label for label in labels_parc if label.name == 'superiorfrontal-lh'][0])

                    for label in labelo:
                        brain.add_label(label, borders=1)

                except Exception as errMessage:
                    print(errMessage)
                    pass

                try: # make a directory called 'results'
                    os.mkdir(os.path.join(curdir, folder_name))
                except Exception as errMessage:
                    print(errMessage)

                finally:
                    os.chdir(os.path.join(curdir, folder_name))
                    brain.save_image(fname.split('_c')[0])
                    brain.close()

        os.chdir(mainDir)
# =============================================================================
#%% PLOT previously saved PNG files
# =============================================================================
    def subplottPNG(self):
        """
        This file locates all .png file saved from batch inverse
        localizaiton and saves them as subplot.
        See >>> [def plotsources(self,mainDir):]
        for how batch saving is done

        """
        os.chdir(self.mainDir)
        folder = os.listdir(u'.')
        folders = [f for f in folder if f[0] == 'S']

        for subject in folders:

            try: # go to the 'results' directory
                resultsDir = os.path.join(os.path.join(self.mainDir, subject),'results')
                os.chdir(resultsDir)

                # find all files with .png extension
                pngfiles = glob.glob('*.png')
                pngfiles.sort(key = lambda x:x[0])
                pngfiles.sort(key = lambda x:x[1])

                fig = plt.figure()

                for ii, filename in enumerate(pngfiles):
                    f = plt.subplot(4,4,ii+1)
                    f.set_axis_off()
                    f.set_xlabel('ses:'+str(ii+1))#    f.set_figheight(15)
                    fig.set_figwidth(30)
                    fig.set_figheight(30)
                    fig.tight_layout()
                    img = matplotlib.image.imread(filename)
                    plt.imshow(img)

                figname = subject + '_subplot'+ '.png'
                matplotlib.pyplot.savefig(figname)

            except Exception as errMessage:
                print(errMessage)

# =============================================================================
# MRCP processing class
class MRCP(object):
# =============================================================================
    default_dir =  '/home/hero/uhdata'

    # segmentation parameters
    epoch_param_default = {'tmin':-1,
                           'tmax': 1,
                           'event_id': 16}

    pick_chans_default = ['Fp1','Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'FC5', 'FC1', 'FC2', 'FC6', 'T7', 'C3', 'Cz',
                          'C4', 'T8', 'CP5', 'CP1', 'CP2', 'CP6', 'P7', 'P3', 'Pz', 'P4', 'P8', 'AFz', 'O1', 'Oz',
                          'O2', 'FCz', 'AF7', 'AF3', 'AF4', 'AF8', 'F5', 'F1', 'F2', 'F6', 'FC3', 'FC4', 'C5',
                          'C1', 'C2', 'C6', 'CP3', 'CPz', 'CP4', 'P5', 'P1', 'P2', 'P6', 'PO7', 'PO3', 'POz', 'PO4',
                          'PO8']

    # 3D sensor positions
    montage_default = mne.channels.read_montage(kind = 'standard_1020')

    filename_default = 'filename_default'
    MRCP_bandpass = False # [0.1, 4] Hz
    plot2save = True

# =============================================================================
#  CLASS CONSTRUCTOR - INITIATE ATTRIBUTES/VARIABLES OF A CLASS INSTANCE
# =============================================================================
    def __init__(self,
                 filename2save = filename_default,
                 montage = montage_default,
                 pick_chans = pick_chans_default,
                 epoch_param = epoch_param_default,
                 data_dir = default_dir,
                 MRCP = MRCP_bandpass,
                 plot = True):

        # CLASS instance variables
        self.filename2save = filename2save
        self.montage = montage
        self.pick_chans = pick_chans
        self.epoch_param = epoch_param
        self.data_dir = data_dir
        self.MRCP_bandpass = MRCP
        self.plot = plot
        

    def __str__(self):
        return print(f"Object for data >>>: {self.filename2save}")
# =============================================================================
#%% read EEG 
# =============================================================================
    @staticmethod 
    def readEEGraw(filename, subject):  
        """
        filename, subject should be a (str)
        ------------------------------------------
        Example: 
        ------------------------------------------    
        filename = 'S9009_ses1_cond1_block0000.vhdr'
        subject = 'ES9009' 
        
        from uhClass import MRCP
        d = MRCP.readEEGraw(filename, subject)         
        
        """
        # subject = 'ES9007'  
        datapath = os.path.join(MRCP.default_dir, subject)
        os.chdir(datapath)
        
        folders = os.listdir(datapath)
        folder = [f for f in folders if f[0] == 's' ]
        
        for dir in folder:
            os.chdir(os.path.join(datapath, dir))
            file = glob.glob(filename)
            
            if file:                
               print('>>>>>>>>>>>>> file loaded from >>>>>>>>>>>>>>>>>:', os.getcwd())
               filepath = os.path.join(os.getcwd(), filename)             
               try:
                   dat = mne.io.read_raw_brainvision(filepath, preload=True)
                   plt.plot(dat._data[-1])
                   plt.xlabel('Event Markers (samples)')
                   plt.ylabel('Event Values')
                   plt.title('Events')
               except Exception as err:
                   print(err)                   
               break
        # plot events         
        print(dat.info)            
        return dat    
    
    #%% read EEG epoch 
    @staticmethod
    def readEEGepoch(eegfilename, mainDir):  
        """
        filename, subject == (str)
        ------------------------------------------
        Example: 
        ------------------------------------------    
        filename = 'S9007_ses10_closeloop_block0000-epo.fif'
               
        from uhClass import MRCP
        filedir = 'C:\\uhdata\\freesurfer'
        ep = MRCP.readEEGepoch(filename, filedir)         
        
        """
        # subject = 'ES9007'  
        datapath = os.path.join(mainDir)
        os.chdir(datapath)
        
        folders = os.listdir(datapath)
        
        for dir in folders:
            
            os.chdir(os.path.join(datapath, dir))
            file = glob.glob(eegfilename)
            
            if file:
               print('>>>>>>>>>>>>> file loaded from >>>>>>>>>>>>>>>>>:', os.getcwd())
               filepath = os.path.join(os.getcwd(), eegfilename)             
               dat = mne.read_epochs(filepath, preload=True)                    
               break            
        return dat 
    
#% =============================================================================
# APPLY STANDARD PREPROCESSING & RETURN THE SEGMENTED DATA
# =============================================================================
#%% preprocess EEG 
    def preprocEEG(self, d):
        """
        see batchEntireDirectory()             
        Example: 
        ------------------------------------------    
        filename = 'S9009_ses1_cond1_block0000.vhdr'
        subject = 'ES9009' 
        
        from uhClass import MRCP
        d = MRCP.readEEGraw(filename, subject)         
        
        obj = MRCP()        
        dclean = obj.preprocEEG(d)
        
        """

        print('Working with files at :', self.data_dir)

        # Removing power-line noise with notch filtering
        d.notch_filter(numpy.arange(60, 241, 60), filter_length='auto',phase='zero')
        #% Set 3D electrode positions
        try:
            if self.montage:
                d.set_montage(self.montage)
                
        except Exception as error:
            print(error)
            
        else:
            print('Setting standard Electrode positions')
            # import mne            
            montage = mne.channels.read_montage(kind = 'standard_1020')
            d.set_montage(montage)
            
        finally:
            print('d.info')
            
        # Load subject specific 3D electrodes using the mne.channels.montage.read_dig_montage
        # elec = mne.channels.montage.read_dig_montage(hsp=None, hpi=None, elp='S9011_ses14_electrodefile.elp',point_names=ch)

        #% Pre-process start
        # High-pass filtering to remove slow drifts
        d.filter(0.1, None, l_trans_bandwidth='auto', filter_length='auto',phase='zero')
        #d.plot_psd(area_mode='range', tmax=10.0, picks=picks)

        # Removing power-line noise with low-pass filtering low pass filtering below 60 Hz
        d.filter(None, 40., h_trans_bandwidth='auto', filter_length='auto',phase='zero')        
        
        #d.plot_psd(area_mode='range', tmax=10.0, picks=picks)

        # Downsampling and decimation
        # d.resample(60, npad="auto")

        #% Select subset of channels by defining a [pick_chans] list
        
        if self.pick_chans:
            picks = mne.pick_channels(d.info["ch_names"], self.pick_chans)
        else:
            picks = None

        #% Define Epochs and compute an ERP for the movement onset condition.
        baseline = (None, -0.5)
        
        events = mne.find_events(d, stim_channel=None, output='onset',
                                 consecutive='increasing', min_duration=0,
                                 shortest_event=1, mask=None,
                                 uint_cast=False, mask_type='and',
                                 initial_event=False, verbose=None)

        # define the type of an event to extract. Note epoch_param is user defined!
        epochs_params = dict(events=events, event_id = self.epoch_param['event_id'],
                             tmin= self.epoch_param['tmin'], tmax = self.epoch_param['tmax'],
                             reject = None)

        # Segment continuous EEG data around the event onset
        ep = mne.Epochs(d, **epochs_params, picks=picks, baseline=baseline, preload=True)
        
        
#       #% Bad channel identificaiton and removal
#        data = numpy.rollaxis(ep.get_data(), 0, 2)
#
#        # Outlier detection
#        A = MRCP.idOutliers(data.reshape(len(self.pick_chans),-1), dim=1,
#                            threshold=(None, 2), maxIter=1, feat="var")
#
#        bad = list(A[1])
#        ch_names = ep.info['ch_names']
#        b=[]
#
#        for bads in bad:
#            b.append(ch_names[bads])
#        
#        print('BAD CHANNELS found >>>', b)
#        ep.info['bads'] = b
#        ep.interpolate_bads(reset_bads='True', mode = 'accurate')
#        ep.set_eeg_reference(ref_channels = "average", projection=True)
#        ep.apply_proj()
#        ep.info['bads'] = []
        
        return ep
# =============================================================================
#%% PERFORM ICA DECOMPOSITION TO REMOVE EYE BLINKS
# =============================================================================
    def ICA(self, ep):
    #%% Artifact Correction with ICA
        """
        ICA finds directions in the feature space corresponding to projections with high non-Gaussianity.
        We obtain a decomposition into independent components, and the artifact's contribution
        is localized in only a small number of components.These components have to be correctly identified and removed.

        If EOG or ECG recordings are available, they can be used in ICA to
        automatically select the corresponding artifact components from the
        decomposition. To do so, you have to first build an Epoch object around
        blink or heartbeat event.
        """
        
        from mne.preprocessing import ICA
        # ICA parameters:
        n_components = 20  # if float, select n_components by explained variance of PCA
        method = 'fastica'
        decim = 3  # need sufficient statistics, not all time points -> saves time

        # Set state of the random number generator - ICA is a
        # non-deterministic algorithm, but we want to have the same decomposition
        # and the same order of components each time

        picks_eeg = mne.pick_types(ep.info, meg=False, eeg=True, 
                                   eog=False, stim=False, exclude='bads')

        # Define the ICA object instance
        ica = ICA(n_components=n_components, method=method, random_state = 23)
        print(ica)
         

        # avoid fitting ICA on crazy environmental artifacts that would
        # dominate the variance and decomposition
        reject = dict(eeg=40e-6)

        ica.fit(ep, picks=picks_eeg, reject = reject, decim=decim)

        if self.ICAplot:
            ica.plot_components()  # can you spot some potential bad guys?

        #% Artifact detection
        eog_inds, scores = ica.find_bads_eog(ep, ch_name = 'Fp1', threshold=1)  # find via correlation
        #ica.plot_scores(scores, exclude=eog_inds)  # look at r scores of components
        ica.exclude.extend(eog_inds)

        # apply ICA
        ep = ica.apply(ep, exclude=eog_inds)

        if self.MRCP_bandpass: # this is basically for NIKUNJ data (by default it is bandpassed)
            # Extract MRCP and return a *band-pass* filtered signal in the range .1 Hz - 4 Hz
            ep.filter(None, 4., l_trans_bandwidth='auto', h_trans_bandwidth='auto',
                      filter_length='auto', phase='zero')

        return ep
# =============================================================================
#%% PLOT standard EEG figures for report     
# =============================================================================        
    def plotERP(self, ep):
        """
        import mne 
        
        filename = 'S9007_ses10_closeloop_block0000-epo.fif'
               
        from uhClass import MRCP
        filedir = 'C:\\uhdata\\freesurfer'
        ep = MRCP.readEEGepoch(filename, filedir) 
                
        erp = MRCP()        
        erp.plotERP(ep)
        
        """
        import os 
        import matplotlib.pyplot as plt
        
        try:
            filename = ep.filename.split('\\')[-1].split('.fif')[0]
            filename = 'plotsEEG_'+filename.split('_')[0]            
        except Exception as err:            
            filename = 'plots_eeg_file' 
            print(err)                    
        finally:
            print('Saving ERP plots at >>>>', os.getcwd())
        
        try:
            os.mkdir(os.path.join(os.getcwd(), filename))  
            os.chdir(os.path.join(os.getcwd(), filename)) 
        except Exception as err:
            print(err)                      
         
        
        ep = ep.interpolate_bads(reset_bads='True', mode = 'accurate')
        ep.info['bads'] = []
        
        ep.plot_psd(area_mode='range',fmin=0, fmax=40, tmax=10.0).savefig(filename + '_psd')

#       picks = ['FC2', 'C4', 'Cz', 'C5', 'FC1'] 
        
        ep.plot_image(picks = None, cmap='interactive', sigma=1)     
        
        plt.savefig(filename + '_image')   
        
        bands = [(0, 4, 'Delta'), (4, 8, 'Theta'), (8, 12, 'Alpha'),
                 (12, 30, 'Beta'), (30, 45, 'Gamma')]       
      
        ep.plot_psd_topomap(bands=bands, vmin=None, vmax=None, 
                            tmin=0, tmax=0.5).savefig(filename + '_psd_topo')
        
        ep.plot_sensors().savefig(filename + '_sensors_')           
        
        ep.plot_topo_image(vmin=-25, vmax=25, title='ERF images', sigma=3.,
                           fig_facecolor='w', font_color='k').savefig(filename + '_image_topo')  
         
        ep.average().plot().savefig(filename + 'erp_average_')
        ep.average().plot_image().savefig(filename + '_erp_average_image')
        print('Saving ERP plots at >>>>', os.getcwd())
#        plt.close("all")
        
# =============================================================================
#%%  SAVE FILE TO A DISK
# =============================================================================
    def save2epochs(self, ep):        
        os.chdir(self.data_dir)        
        # save the data as -epochs
        filename = "".join([self.filename2save,'-epoch.fif'])
        print('Saving file to :', filename)
        ep.save(filename, verbose = True)

        if self.plot is True:
            ep.plot_image(combine='gfp', group_by='type', sigma=2., cmap="YlGnBu_r")
            ep.average().plot().savefig(self.filename2save)
            ep.average().plot_image().savefig('plot_image.png')

            import matplotlib.pyplot as plt
            plt.close("all")
   
# =============================================================================
#%% FIND fif and move from source to destination directory
# =============================================================================
    def findfif2move(self, source, destination, foldername):
        """
        This function tries to find files with .fif extention
        in the provided 'source' location and to move it to the
        'destination' folder

        foldername = is the name of the folder that starts with specific name
        for examle >>> fname = 'ES'
        
        Usage:
        ---------------------    
        from uhClass import MRCP
        a = MRCP()
        
        source = '/home/hero/uhdata'
 
        destination = '/home/hero/uhdata/freesurfer/'       

        foldername = 'ES'    
        a.findfif2move(source,destination,foldername)

        """
        import glob
        import shutil

        os.chdir(source)
        mainfolders = os.listdir(u'.')

        for fname in mainfolders:
            try:
                if fname[:2] == foldername:
                    subjectdir = os.path.join(source, fname)
                    os.chdir(subjectdir)
                    subfolders = os.listdir(u'.')
                    
                    # for each subject in the provided subfolders 
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
        os.chdir(source)
# =============================================================================
#%%  REMOVE OUTLIERS
# =============================================================================
    @staticmethod
    def idOutliers(X, dim = 1, threshold=(None,2), maxIter=2, feat="var"):
        '''Removes outliers from X
        
        Removes outliers from X based on robust coviarance variance/mean
        computation.

        Parameters
        ----------
        data : list of datapoints (numpy arrays) or a single numpy array.
        dim : the dimension along with outliers need to be detected.
        threshold : the upper and lower bound of the thershold in a tuple expressed
        in standard deviations.
        maxIter : number of iterations that need to be performed.
        feat : feature on which outliers need to be based. "var" for variance, "mu"
        , for mean or a numpy array of the same shape as X.

        Returns
        -------
        out : a tuple of a list of inlier indices and a list of outlier indices.
        '''
    #%
        if feat=="var":
            feat = numpy.sqrt(numpy.abs(numpy.var(X,dim)))
        elif feat =="mu":
            feat = numpy.mean(X,dim)
        elif isinstance(feat,numpy.array):
            if not (all([isinstance(x,(int , float)) for x in feat]) and feat.shape == X.shape):
                raise Exception("Unrecognised feature type.")
        else:
            raise Exception("Unrecognised feature type.")
    #%
        outliers = []
        inliers = numpy.array(list(range(0,max(feat.shape))))

        mufeat = numpy.zeros(maxIter)
        stdfeat = numpy.zeros(maxIter)

        for i in range(0,maxIter):
            mufeat = numpy.mean(feat[inliers]) # median
            stdfeat = numpy.std(feat[inliers]) # std across channels

            if threshold[0] is None:
                high = mufeat + threshold[1]*stdfeat # median + 3* standard deviation
                bad = (feat > high)
                print(">>> Threshold: ", threshold[1])

            elif threshold[1] is None:
                low = mufeat+threshold[0]*stdfeat
                bad = (feat < low)
                print(">>> Threshold: ", threshold[0])

            else:
                high = mufeat+threshold[1]*stdfeat
                low = mufeat+threshold[0]*stdfeat
                bad = (feat > high) * (feat < low)

            if not any(bad):
                break
            else:
                outliers = outliers + list(inliers[bad])
                inliers = inliers[[ not x for x in bad]]

        print("== OUTLIERS FOUND ===", outliers)
        return (list(inliers), outliers)    
# =============================================================================
#%% Read Polhemus Sensor Positions      
# =============================================================================
    @staticmethod
    def read3Dsensor(filename):
#        filename = 'sensor.elp'  
        with open(filename) as f:
            c = f.readlines()
        
        ch_names = []
        fiducials = []
        loc = []
        
        for ii, f in enumerate(c):
            
            if f[:2] == '%N':
                ch_names.append(f[3:6])
            
            if f[:2] == '%F':
                fiducials.append(f[3:].rstrip().split('\t'))
            
            if f[:2] == '0.':
                loc.append(f.rstrip().split('\t'))        
                print(f)         
        
        sensor = dict()         
        sensor['ch_names'] = ch_names[1:]
        sensor['loc'] = loc       
        sensor['fiducials'] = fiducials
        
        return sensor 
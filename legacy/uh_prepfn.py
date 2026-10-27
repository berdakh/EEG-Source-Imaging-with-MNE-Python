import mne
import os
import numpy

directory =  os.getcwd()
# segmentation parameters    
epoch_param = {'tmin':-1.5,
              'tmax': 1,
              'event_id': 16}
    
pick_chans = ['Fp1','Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'FC5', 'FC1', 'FC2', 'FC6', 'T7', 'C3', 'Cz',
     'C4', 'T8', 'CP5', 'CP1', 'CP2', 'CP6', 'P7', 'P3', 'Pz', 'P4', 'P8', 'AFz', 'O1', 'Oz',
     'O2', 'FCz', 'AF7', 'AF3', 'AF4', 'AF8', 'F5', 'F1', 'F2', 'F6', 'FC3', 'FC4', 'C5',
     'C1', 'C2', 'C6', 'CP3', 'CPz', 'CP4', 'P5', 'P1', 'P2', 'P6', 'PO7', 'PO3', 'POz', 'PO4',
     'PO8']
    
def preproc(d, filename, montage, pick_chans, epoch_param):    
    print('*****************************************')
    print('Working with files at :',directory)      
    print('*****************************************')    
    # Removing power-line noise with notch filtering 
    d.notch_filter(numpy.arange(60, 241, 60), filter_length='auto',phase='zero')
    
#%% Set 3D electrode positions     
    try: 
        if montage:
            d.set_montage(montage)
    except Exception as error:
        print(error)
    else:
        print('Setting standard Electrode positions')
        montage = mne.channels.read_montage(kind = 'standard_1020')
        d.set_montage(montage)
    finally:
        print('d.info')   
        
    # Load subject specific 3D electrodes using the mne.channels.montage.read_dig_montage
    # elec = mne.channels.montage.read_dig_montage(hsp=None, hpi=None, elp='S9011_ses14_electrodefile.elp',point_names=ch)

#%% Pre-process start    
    # High-pass filtering to remove slow drifts
    d.filter(0.1, None, l_trans_bandwidth='auto', filter_length='auto',phase='zero')
    #d.plot_psd(area_mode='range', tmax=10.0, picks=picks)
    
    # Removing power-line noise with low-pass filtering low pass filtering below 60 Hz
    d.filter(None, 40., h_trans_bandwidth='auto', filter_length='auto',phase='zero')
    #d.plot_psd(area_mode='range', tmax=10.0, picks=picks)
    
    # Downsampling and decimation
    # d.resample(60, npad="auto")  # set sampling frequency to 60Hz 
    
#%% Select subset of channels by defining a [pick_chans] list 
    if pick_chans: 
        picks = mne.pick_channels(d.info["ch_names"], pick_chans)
    else:
        picks = None
               
#%% Define Epochs and compute an ERP for the movement onset condition.
    baseline = (None, -0.5)     
    events = mne.find_events(d, stim_channel=None, output='onset', 
                             consecutive='increasing', min_duration=0, 
                             shortest_event=1, mask=None,
                             uint_cast=False, mask_type='and', 
                             initial_event=False, verbose=None)
    
    # define the type of an event to extract. Note epoch_param is user defined!      
    epochs_params = dict(events=events, event_id = epoch_param['event_id'], 
                         tmin=epoch_param['tmin'], tmax=epoch_param['tmax'],
                         reject=dict(eeg=25e-4))
    
    # Segment continuous EEG data around the event onset 
    ep = mne.Epochs(d, **epochs_params, picks=picks, baseline=baseline,preload=True) 
    #ep.plot_sensors()    
    
#%% Bad channel identificaiton and removal     
    data = numpy.rollaxis(ep.get_data(), 0, 2)    
    A = idOutliers(data.reshape(56,-1), dim=1, threshold=(None, 2), maxIter=1, feat="var")
    
    bad = list(A[1])
    ch_names = ep.info['ch_names']    
    b=[]
    
    for bads in bad:
        b.append(ch_names[bads])    
    print('Bad channels :', b)
    
    ep.info['bads'] = b
    ep.interpolate_bads(reset_bads='True', mode = 'accurate')    
    ep.set_eeg_reference(ref_channels = "average", projection=False) 
    
    # return the clean data 
    return ep 

def ICA(ep, MRCP):    
#%% Artifact Correction with ICA
    """
    ICA finds directions in the feature space corresponding to projections with high non-Gaussianity.
    We obtain a decomposition into independent components, and the artifact's contribution
    is localized in only a small number of components. 
    These components have to be correctly identified and removed.
    
    If EOG or ECG recordings are available, they can be used in ICA to
    automatically select the corresponding artifact components from the
    decomposition. To do so, you have to first build an Epoch object around
    blink or heartbeat event. 
    """    
    from mne.preprocessing import ICA
    # ICA parameters:
    n_components = 10  # if float, select n_components by explained variance of PCA
    method = 'fastica' 
    decim = 3  # need sufficient statistics, not all time points -> saves time
    
    # Set state of the random number generator - ICA is a
    # non-deterministic algorithm, but we want to have the same decomposition
    # and the same order of components each time  
    
    random_state = 23
    picks_eeg = mne.pick_types(ep.info, meg=False, eeg=True, 
                               eog=False, stim=False, exclude='bads')

    # Define the ICA object instance
    ica = ICA(n_components=n_components, method=method, random_state=random_state)
    
    print(ica)
    
    # avoid fitting ICA on crazy environmental artifacts that would
    # dominate the variance and decomposition
    reject = dict(eeg=40e-6)
    
    ica.fit(ep, picks=picks_eeg, reject = reject, decim=decim)
    
    print(ica)
    #ica.plot_components()  # can you spot some potential bad guys?
    
    #% Advanced artifact detection
    
    #  We simplify things by setting the maximum number of components to reject
    eog_inds, scores = ica.find_bads_eog(ep, ch_name = 'Fp1', threshold=1)  # find via correlation
    #ica.plot_scores(scores, exclude=eog_inds)  # look at r scores of components
    ica.exclude.extend(eog_inds)
    
    # apply ICA 
    ep = ica.apply(ep, exclude=eog_inds)
    
    if MRCP:
        # Extract MRCP and return a *band-pass* filtered signal in the range .1 Hz - 4 Hz
        ep.filter(None, 4., l_trans_bandwidth='auto', h_trans_bandwidth='auto',
                  filter_length='auto', phase='zero')        
    return ep 


def savefile(ep, filename, directory, plot):   

    # save the data as -epochs 
    filename2save = "".join([filename.split(".")[0],'-epo.fif'])
    print('Saving file to :', filename2save)
    
    ep.save(filename2save, verbose = True)
    
    if plot is True:
        ep.plot_image(combine='gfp', group_by='type', sigma=2., cmap="YlGnBu_r")
        ep.average().plot().savefig(filename.split("_b")[0])
        ep.average().plot_image().savefig('plot_image.png')
        import matplotlib.pyplot as plt
        plt.close("all")
        
    
def idOutliers(X, dim=1, threshold=(None,2), maxIter=2, feat="var"):
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

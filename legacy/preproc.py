import numpy as np
import scipy, scipy.signal
import sklearn
from pylab import *
from h5py import File
#%%
def butterFilter(data, band, N = 2, fs = 100, dim = 1, filter_type = 'lowpass'):
    '''
    Applies temporal butterworth filter
    Inputs:
        data        : epochs x time x channels
		tmp = data.reshape(-1, data.shape[-1]) # concatenate all the trials by channels

        band        : min max of to be filtered frequency; if only 1 values is given it is assumed low pass filter
        N           : filter order
        fs          : sampling rate
        dim         : 1 ; assumes structure of data (time = first dimension)
        filter_type : default is low pass (if single value is given else band pass)
    returns filtered data
    '''
    band = np.array(band) / float(fs)
    if len(band) == 2:
        filter_type = 'bandpass'

    Notch = np.array([49,51]) / fs
    bb,aa = scipy.signal.butter(N = N, Wn = Notch, btype = 'bandstop')
    b, a = scipy.signal.butter(N = N, Wn = band, btype = filter_type)
    fdata = scipy.signal.filtfilt(bb,aa, data, method ='gust', axis = dim)
    fdata = scipy.signal.filtfilt(b,a, fdata, method = 'gust', axis = dim)
    return fdata
#%%
def car(data):
  '''
  Return a common average reference (CAR) spatial filter for n channels.
  The common average reference is a re-referencing scheme that is
  commonly used when no dedicated reference is given. Since the
  average signal is subtracted from each sensor'stdPower signal, it reduces
  signals that are common to all sensors, such as far-away noise.
  Parameters
  ----------
  n : int
    The number of sensors to filer.
  Returns
  -------
  W : array
    Spatial filter matrix of shape (n, n) where n is the number of
    sensors. Each row of W is single spatial filter.
  Examples
  --------
  >>> car(4)
  array([[ 0.75, -0.25, -0.25, -0.25],
         [-0.25,  0.75, -0.25, -0.25],
         [-0.25, -0.25,  0.75, -0.25],
         [-0.25, -0.25, -0.25,  0.75]])
  '''
  n = data.shape[-1]
  W = np.eye(n) - 1 / float(n)
  return  data.dot(W)
#%%
def stdPreproc(data, band,  fSample, cap = None, numStd = 3, calibration = 1):
    '''
    The standard preprocessing pipeline includes:
        Linear detrending
        Bad channel removal based on power outlier detection
        Feature standardization
        Temporal filtering using Butterworth filter
        Spatial filtering using CAR filter

    Inputs:
        data        : trial x time x channel
        band        : frequencies to filter at [low (high)]
        fSample     : the sampling frequency
        cap         : containing the channel information
        numStd      : how many standard deviations is should for boundary in outlier detection
        calibration : check for performing outlier detection; set to false when playing the game

    returns preprocessed data
    '''
    # linear detrending
    data       = scipy.signal.detrend(data, axis = 1)


    meanData = np.mean(data)    # global mean
    stdData  = np.std(data)     # global std

    # only remove channels in calibration; otherwise SVM will error
    if calibration :
        # bad channel removal: look for power outliers
        tmp = data.reshape(-1, data.shape[-1]) # concatenate all the trials by channels
        power = []
        for i in tmp.T:
            i,_ = scipy.signal.welch(i, fSample, axis = 0)
            power.append(i)

        power         = np.array(power)
        meanPower     = np.mean(power) # mean over all channels all epochs
        stdPower      = np.std(power)   # std over all channels and epochs

        # locate the badchannels
        badChannels,_ = np.where(np.logical_or(power > meanPower + numStd * stdPower,
                                    power < meanPower - numStd *stdPower))

        # set the defaults to use all
        channels                = np.ones(data.shape[-1], dtype = bool)
        # bad channels can be none; only remove if there is a channel to remove

        badChannels             = np.unique(badChannels)
        channels[badChannels]   = 0
        data                    = data[..., channels]

        if cap != None:
            if badChannels.size == 0:
                output = 'None'
            else:
                output = cap[channels == False, 0]
            print(('Removing channels :\n {0}'.format(output)))
    else:
        channels = None

    # feature normalization
    data = (data - meanData) / stdData


    #temporal filter
    data = butterFilter(data, band = band, fs = fSample)

    # spatial filter
    data = car(data)
    return data, channels


if __name__ == '__main__':
    '''
    Edit the file to preprocess data for a single subject
    '''
    file = '../Data/calibration_subject_15.hdf5'

    from scipy import signal
    from h5py import File
    import sklearn.preprocessing
    with File(file, 'r') as f:
        rawData = f['rawData/IM'].value
        procData = f['procData/IM'].value
        cap      = f['cap'].value
        procData, _    = stdPreproc(rawData, [0, 50], 250, cap = cap)
        # print(procData.shape)

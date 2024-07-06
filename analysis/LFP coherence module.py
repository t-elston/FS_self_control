'''LFP coherence module'''

import numpy as np
import scipy.stats as stats
from scipy import signal

def coherence(lfp1, lfp2, window='hann', fs=1e3, axis=-1):
    # computes the coherence between 2 equal-length LFP samples
    
    nts = lfp1.shape[axis]

    # create window array
    if type(window) in [str, tuple]:
        window = signal.get_window(window, nts)

    win_array = np.expand_dims(window, axis=list(range(lfp1.ndim-1))).swapaxes(-1, axis) # match lfp dims
    
    # Fourier transform LFP1 and compute spectrum
    xf = np.fft.rfft(signal.detrend(lfp1, axis=axis, type='constant') * win_array, axis=axis)
    sxx = np.real(xf * np.conj(xf))
    
    # Fourier transform LFP2 and compute spectrum
    yf = np.fft.rfft(signal.detrend(lfp2, axis=axis, type='constant') * win_array, axis=axis)
    syy = np.real(yf * np.conj(yf))
    
    # Compute cross-spectrum between LFP1 and LFP2
    sxy = np.real(xf * np.conj(yf))
    
    coh = np.abs(sxy)**2 / (sxx * syy) # get the coherence
    freqs = np.fft.rfftfreq(nts, 1/fs) # get the sample frequencies

    return freqs, coh

def multitaper_coherence(lfp1, lfp2, nw=4, n_tapers=3, fs=1e3, axis=-1):
    # compute multi-taper coherence, using Slepian windows
    #
    # nw: time bandwidth product (default = 4)
    # ntapers: first n slepian tapers to use (default = 3; should be <= 2*nw - 1, low bias for ntapers << 2*nw - 1)
    # ci: confidence interval probability
    
    if n_tapers is None:
        n_tapers = 2*nw - 1

    nts = lfp1.shape[axis] # number of time samples in lfp
    slepians = signal.windows.dpss(nts, nw, ntapers) # get slepian windows

    # compute a coherence estimate for each slepian taper
    coh = []
    for ii in range(slepians.shape[0]):
        freqs, coh_i = coherence(lfp1, lfp2, window=slepians[ii], fs=fs, axis=axis)
        coh.append(coh_i)
    coh = np.stack(coh, axis=-1)

    coh_mean = coh.mean(axis=axis)

    return freqs, coh_mean

#----------------------------------------------------------------
def mt_coherogram(lfp1, lfp2, n_perseg, n_overlap, fq_min, fq_max, fs):
    """Computes a multi-tamper coherogram for an LFP stream of arbitrary length.
       By default, 3 tapers are used.

    Inputs: 
        lfp1 (ndarray): Datastream for one LFP channel
        lfp2 (ndarray): Datastream for a different LFP channel
        n_perseg(int):  Window over which each piece of the coherogram is evaluated
        fq_min (int):   Minimum frequency coherence is computed at
        fq_max (int):   Maximum frequency coherence is evaluated at
        fs (int):       Sampling frequency (in Hz) of the LFP data (typically 1000 Hz)
            
    Returns:
        ts (ndarray):   Timestamps of each element of the coherogram
        coh (ndarray):  A n_frequencies x n_times array where each element describes the coherence between
                        lfp1 and lfp2 at each timestep and frequency
        freqs(ndarray): Array detailing which frquencies define the rows of coh     
    """
    
    if n_overlap < 1:
        n_overlap = int(n_overlap * n_perseg) # convert to number of segments if a fraction provided (noverlap <= 1)

    # find out the times to compute coherence at   
    ts = np.arange(0, lfp1.shape[0] + (n_perseg - n_overlap), n_perseg - n_overlap)

    n_freqs = ((fq_max - fq_min) * n_perseg // 1000) + 1

    # compute slepian tapers
    nw = 4 # time-bandwidth product
    n_tapers = 3 # number of tapers to use

    # define the starts and stops of each window to assess coherence over
    # first columns is starts, second is stops
    win_details = np.zeros(shape=(len(ts), 2))

    # loop over each timestep and find appropriate windows. Truncate windows at the 
    # very beginning and end of of the data stream
    for i_t in range(len(ts)):

        # make centered windows
        win_details[i_t, 0] = ts[i_t] - np.floor((n_perseg/2)) # window starts
        win_details[i_t, 1] = ts[i_t] + np.floor((n_perseg/2)) # window ends

        # is the left border of the window before the start of the session?
        if win_details[i_t, 0] < 0: 
            # then set the window start to zero
            win_details[i_t, 0] = 0

        # is the right border of the window longer than the end of the session?
        if win_details[i_t, 1] > lfp1.shape[0]:
            # then set the window end to the end of the session
            win_details[i_t, 1] = lfp1.shape[0]

    # run the coherence on the first window to see how many frequencies are obtained
    freqs, test_coh = multitaper_coherence(lfp1[win_details[0, 0]: win_details[0, 1]],
                                                lfp2[win_details[0, 0]: win_details[0, 1]],
                                                nw, n_tapers, fs)
    
    n_freqs = len(freqs)
    # intialize an array to accumulate the coherence data into
    coh = np.zeros(shape=(n_freqs, len(ts)))

    print('there are ' + str(str(n_freqs) + 'freqs'))

    # # loop over each timestep
    # for t in range(len(ts)):
    #     _, coh[:, t] = multitaper_coherence(lfp1[win_details[t, 0]: win_details[t, 1]],
    #                                             lfp2[win_details[t, 0]: win_details[t, 1]],
    #                                             nw, n_tapers, fs)

    




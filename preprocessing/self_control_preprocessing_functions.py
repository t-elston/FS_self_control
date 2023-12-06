"""
Module for preprocessing data from the self-control experiment. 
"""

import os
import glob
import pandas as pd
import numpy as np
import h5py
import matplotlib.pyplot as plt
import scipy.signal as sig
import scipy.stats as stats
from pathlib import Path
from scipy.ndimage import gaussian_filter
from tqdm import tqdm

def extract_behavior_from_h5(top_parent_dir):

    print('Extracting trial-by-trial response and eye data.')

    # get names+paths of .h5 files
    fnames = glob.glob(os.path.join(top_parent_dir, '*.h5'))

    for i in range(len(fnames)):
        
        # load the file and get its name       
        f = h5py.File(fnames[i],'r')
        fname = os.path.basename(fnames[i])[0:-3]
        
        ftrials  = list(f['ML'].keys())[1:-1]        
        sessiondf = pd.DataFrame()
        
        for t in range(len(ftrials)):
                    
            sessiondf.at[t,'fname'] = fname
            sessiondf.at[t,'tnum'] = t
            sessiondf.at[t,'use'] = f['ML'][ftrials[t]]['UserVars']['UseTrial'][0]
            sessiondf.at[t,'state'] = f['ML'][ftrials[t]]['UserVars']['rule'][0]
            sessiondf.at[t,'state_cue'] = f['ML'][ftrials[t]]['UserVars']['rtype'][0]
            sessiondf.at[t,'forced'] = f['ML'][ftrials[t]]['UserVars']['forced'][0]
            
            # figure out the option values depending on whether it was a free or forced choice
            if sessiondf.at[t,'forced'] == 0:
            
                sessiondf.at[t,'l_val'] = f['ML'][ftrials[t]]['UserVars']['OptionVals'][0]
                sessiondf.at[t,'r_val'] = f['ML'][ftrials[t]]['UserVars']['OptionVals'][1]

            else:
                if f["ML"][ftrials[t]]['UserVars']['SideChosen'][0] == -1:
                    sessiondf.at[t,'l_val'] = f['ML'][ftrials[t]]['UserVars']['OptionVals'][0]
                    sessiondf.at[t,'r_val'] = np.NaN
                else:
                    sessiondf.at[t,'l_val'] = np.NaN
                    sessiondf.at[t,'r_val'] = f['ML'][ftrials[t]]['UserVars']['OptionVals'][0]
        
            sessiondf.at[t,'ch_val'] = f['ML'][ftrials[t]]['UserVars']['ChosenVal'][0]
            sessiondf.at[t,'picked_best'] = f['ML'][ftrials[t]]['UserVars']['PickedBestOpt'][0]
            sessiondf.at[t,'rt'] = f['ML'][ftrials[t]]['UserVars']['RT'][0]
            sessiondf.at[t,'side'] = f['ML'][ftrials[t]]['UserVars']['SideChosen'][0]   
            
            # get the event code for stim on for saccade detection
            event_codes = np.array([f['ML'][ftrials[t]]['BehavioralCodes']['CodeNumbers']])
            event_times = np.array([f['ML'][ftrials[t]]['BehavioralCodes']['CodeTimes']])
            stim_on_time = np.round(event_times[event_codes==40] / 2).astype(int)
            stim_off_time = np.round(event_times[event_codes==41] / 2).astype(int)

            # get eye data for the trial
            eye = np.squeeze(np.array([f['ML'][ftrials[t]]['AnalogData']['Eye']]))
            
            # set some dummy vars for the number, time, side, and values of saccades
            sessiondf.at[t, 'n_sacc'] = np.NaN    # nsaccs
            sessiondf.at[t, 'sacc1_t'] = np.NaN   # sacc 1
            sessiondf.at[t, 'sacc1_val'] = np.NaN   # sacc 1
            sessiondf.at[t, 'sacc1_side'] = np.NaN   # sacc 1
            sessiondf.at[t, 'sacc2_t'] = np.NaN   # sacc 2
            sessiondf.at[t, 'sacc2_val'] = np.NaN   # sacc 2
            sessiondf.at[t, 'sacc2_side'] = np.NaN   # sacc 2
            sessiondf.at[t, 'sacc3_t'] = np.NaN   # sacc 3
            sessiondf.at[t, 'sacc3_val'] = np.NaN   # sacc 3
            sessiondf.at[t, 'sacc3_side'] = np.NaN   # sacc 3
            sessiondf.at[t, 'sacc4_t'] = np.NaN   # sacc 4
            sessiondf.at[t, 'sacc4_val'] = np.NaN   # sacc 4
            sessiondf.at[t, 'sacc4_side'] = np.NaN   # sacc 4
            sessiondf.at[t, 'sacc5_t'] = np.NaN   # sacc 5
            sessiondf.at[t, 'sacc5_val'] = np.NaN   # sacc 5
            sessiondf.at[t, 'sacc5_side'] = np.NaN   # sacc 5

            # detect saccades
            if (stim_on_time.size > 0) & (stim_off_time.size > 0):
            
                # pull a window of eye_data around when the choice options appear
                eye = eye[:,stim_on_time[0]:stim_off_time[0]]
                
                # get speed of eye movements in each direction
                dx = np.diff(eye[0,:])
                dy = np.diff(eye[1,:])

                # get 2d speed
                eye_speed = stats.zscore(np.hypot(dx,dy))

                # find the saccades
                sacc_ix = sig.find_peaks(eye_speed, 4, distance = 35)[0]
                
                # store some info about the saccades
                # how many saccades?
                sessiondf.at[t, 'n_sacc'] = len(sacc_ix)
                
                # loop over number of saccades
                for sacc_num, sacc_sample in enumerate(sacc_ix):

                    # what side of the screen was this?
                    x_pos = np.mean(eye[0,sacc_sample+1:sacc_sample+10])
                    
                    # was the saccade to the left?
                    if x_pos < -5:
                        sacc_side = -1
                        sacc_val = sessiondf.at[t,'l_val']
                        
                    # otherwise, was the saccade was to the right?    
                    elif x_pos > 5:
                        sacc_side = 1
                        sacc_val = sessiondf.at[t,'r_val']

                    # no saccade was detectable
                    elif (x_pos < 5) & (x_pos > -5):
                        sacc_side = 0
                        sacc_val = sessiondf.at[t,'ch_val']

                    # get time, relative to onset of pics, the sacc occurred
                    sacc_time = sacc_sample*2 # data was sampled at 500Hz

                    'sacc' + str(sacc_num+1)+'_t'
                    sessiondf.at[t, 'sacc' + str(sacc_num+1)+'_t'] = sacc_time
                    sessiondf.at[t, 'sacc' + str(sacc_num+1)+'_side'] = sacc_side
                    sessiondf.at[t, 'sacc' + str(sacc_num+1)+'_val'] = sacc_val

                    #plot eye movement for the trial
                    # plt.figure()
                    # colors = cm.magma(np.linspace(0,1,len(eye_speed)))
                    
                    # plt.subplot(1,2,1)
                    # plt.scatter(eye[0,0:len(eye_speed)], eye[1,0:len(eye_speed)], color=colors)
                    # plt.xlim([-15,15])
                    # plt.ylim([-15,15])
                    
                    # plt.subplot(1,2,2)
                    # xvals = np.arange(len(eye_speed))*2
                    # plt.plot(xvals,eye[0,0:len(eye_speed)], label = 'eye x')
                    # plt.plot(xvals,eye_speed, label = 'speed')
                    # plt.scatter(xvals, np.ones(len(eye_speed))*-12, color=colors)

                    # plt.ylim([-15,15])
                    # plt.plot([sacc_ix*2,sacc_ix*2],plt.ylim(),color='black',linewidth=1)
                    # plt.legend()
                    # plt.show()
            
                    xx=[]

        # save the data as a .csv in top_parent_dir
        save_name = top_parent_dir + '/' + fname + '_bhv.csv'
        print(save_name)
        sessiondf.to_csv(save_name)
        print('Saved data as .csv in original directory.')


def make_spike_tables_and_combine_data(top_parent_dir, save_dir, spike_params):

    # get the name of this recording
    rec_name = Path(top_parent_dir).stem[0:-3]

    # create the file to save into
    save_name = save_dir + '/' + rec_name

    # Create HDF5 file
    with h5py.File(save_name, 'w') as hf:
        pass  # No need to write anything initially, just creating the file

    # specify the offset and step_size (in milliseconds) for making spike tables
    t_offset = spike_params['t_offset']
    step_size = spike_params['step_size']
    win_size = spike_params['win_size']
    align_event = spike_params['align_event']

    # load the event_codes + event_times for this directory
    task_events = np.load(top_parent_dir + '/sync_event_codes.npy')

    # load the info about which brain area was on which probe
    brain_areas = pd.read_excel(top_parent_dir + '/brain_areas.xlsx')

    # load the behavior
    csv_files = glob.glob(f"{top_parent_dir}/*.csv")[0]
    bhv = pd.read_csv(csv_files)

    # find the start and stop of each trial
    trial_starts = np.argwhere(task_events[:,0] == 9).flatten()
    trial_ends = np.argwhere(task_events[:,0] == 18).flatten() + 1
    last_end_time = round(task_events[-1, 1])

    # Get a list of subdirectories - one per brain area
    sub_dirs = [d for d in os.listdir(top_parent_dir) if os.path.isdir(os.path.join(top_parent_dir, d))]

    # now loop over each directory
    for dir_ix, this_dir in enumerate(sub_dirs):

        index_contained = brain_areas[brain_areas['stream'].apply(lambda x: this_dir.find(x) != -1)].index
        this_brain_area = str(brain_areas['region'].iloc[index_contained].values[0])

        # load the spike times associated with this directory
        spike_times = np.load(top_parent_dir + '/' + this_dir +'/sync_spike_times.npy')
        spike_clusters = np.load(top_parent_dir + '/' + this_dir +'/spike_clusters.npy')

        # load the cluster labels associated with this directory
        cluster_labels=pd.read_csv(top_parent_dir + '/' + this_dir + '/cluster_KSLabel.tsv',sep='\t')

        # find the 'good' clusters
        cluster_ix = (cluster_labels['KSLabel'] == 'good') | (cluster_labels['KSLabel'] == 'mua')
        good_cluster_nums = cluster_labels['cluster_id'].loc[cluster_ix]

        print(str(len(good_cluster_nums)) + ' putative units found in ' + this_brain_area)

        # now get ready to extract the firing rates
        # specify the bin_centers
        bin_centers = np.arange(-1*t_offset, t_offset+step_size, step_size)
        n_bins = len(bin_centers)

        # initialize a n_trials x n_times x n_units array to store results in
        firing_rates = np.empty(shape=(len(trial_ends), n_bins, len(good_cluster_nums)))
        firing_rates[:] = np.nan

        dense_FR = np.empty(shape=(len(trial_ends), 2*t_offset, len(good_cluster_nums)))
        dense_FR[:] = np.nan

        # now loop over these putatively good units and make spike tables
        unit_names = []
        for u_ix, cluster_num in tqdm(enumerate(good_cluster_nums)):

            unit_names.append(rec_name + '_' + this_brain_area +'_u' + str(cluster_num))

            # get the spikes associated with this unit
            #u_spike_times = spike_times[spike_clusters.flatten() == cluster_num]
            unit_names.append(rec_name + '_' + this_dir +'_u' + str(cluster_num))

            # Round spike times and filter based on last_end_time
            u_spike_times = np.round(spike_times[spike_clusters.flatten() == cluster_num])
            u_spike_times = np.delete(u_spike_times, np.where(u_spike_times > last_end_time-1)[0])

            # Initialize an empty array for the spike train
            u_spike_train = np.zeros(shape=last_end_time)

            # Set 1s at spike times
            u_spike_train[u_spike_times.astype(int)] = 1

            # now loop over the trials
            for t in range(len(trial_ends)):

                # get the events and times of this trial
                trial_events = task_events[trial_starts[t] : trial_ends[t], :]

                # make a spike table if a choice was presented (40 is the event code for pics on)
                if np.any(trial_events[:,0] == align_event):

                    # pics_on_time = trial_events[np.argwhere(trial_events[:,0] == align_event).flatten(), 1]
                    # trial_spikes = u_spike_times - pics_on_time
                    # bins = np.arange(-1*t_offset, t_offset + 2*step_size, step_size)
                    # firing_rates[t,:, u_ix] = gaussian_filter(np.histogram(trial_spikes, bins)[0]*(1000/step_size), 
                    #                             sigma=2)
                    
                    pics_on_time = int(trial_events[np.argwhere(trial_events[:,0] == align_event).flatten(), 1])
                    dense_FR[t,:,u_ix] = u_spike_train[pics_on_time - t_offset : pics_on_time + t_offset]

        firing_rates, ts = window_smooth(win_size, step_size, dense_FR, np.arange(-1*t_offset, t_offset), 1000)

        firing_rates = firing_rates*1000

        # find units to reject
        u_mean_FR = np.nanmean(firing_rates, axis=(0, 1))
        units2del = u_mean_FR < 2
        firing_rates2 = np.delete(firing_rates, units2del, axis=2)
        u_mean_FR2 = np.delete(u_mean_FR, units2del, axis=0)
        unit_names2 = [name for name, mask in zip(unit_names, units2del) if not mask]
        cluster_labels2 = cluster_labels['KSLabel'].loc[~units2del].tolist()
        z_fr = stats.zscore(firing_rates2, axis=2, nan_policy='omit')

        print('saving ' +str(len(cluster_labels2)) +' units')

        # now actually save the data
        # dataframes have their own method
        bhv.to_hdf(save_name, key='bhv', mode='a')

        with h5py.File(save_name, 'a') as hf:
            # Write data into the file with the specified variable name
            hf.create_dataset(this_brain_area + '_FR', data=firing_rates2)
            hf.create_dataset(this_brain_area + '_zFR', data=z_fr)
            hf.create_dataset(this_brain_area + '_u_names', data=unit_names2)
            hf.create_dataset(this_brain_area + '_KS_labels', data=cluster_labels2)
            if dir_ix == 0:
                hf.create_dataset('ts', data=bin_centers)


def window_smooth(win_size, step_size, indata, in_times, fs):
    """
    INPUTS
    win_size = size of window for boxcar mean (in milliseconds)
    step_size = how much to move the window between downsamples
    indata = data with shape n_trials x n_timesteps x n_units
    in_times = original timestamps for each sample
    fs       = sampling frequency of indata

    OUTPUTS
    outdata = downsampled indata
    outtimes = downsampled timestamps for each new sample
    """

    # scale the step size to the sampling frequency
    scaled_step = round((fs * step_size) / 1000)
    scaled_winsize = round((fs * win_size) / 1000)

    n_new_times = int(len(in_times) / scaled_step)
    n_trials = len(indata[:, 0, 0])
    n_units = len(indata[0, 0, :])

    # initialize output
    outdata = np.empty(shape=(n_trials, n_new_times + 1, n_units))
    outtimes = np.empty(shape=(n_new_times + 1))

    for win_ix, window_center in enumerate(range(0, len(in_times), scaled_step)):

        # get bounds of the window
        win_start = window_center - round(scaled_winsize / 2)
        win_end = window_center + round(scaled_winsize / 2)

        # be sure we don't go over the edge with the windows
        if win_start < 0: win_start = 0
        if win_end > len(in_times): win_end = len(in_times)

        outtimes[win_ix] = int(in_times[window_center])
        outdata[:, win_ix, :] = np.nanmean(indata[:, win_start:win_end, :], axis=1)

    return outdata, outtimes

"""
Module for preprocessing data from the self-control experiment. 
"""

import os
import glob
import pandas as pd
import numpy as np
import h5py
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import scipy.signal as sig
import scipy.stats as stats
from pathlib import Path
from scipy.ndimage import gaussian_filter
from tqdm import tqdm
import random
import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)


def extract_behavior_from_h5(base_folder):

    print('Extracting trial-by-trial response and eye data.')

    # get names+paths of .h5 files
    fnames = glob.glob(os.path.join(base_folder, '*.h5'))

    for i in range(len(fnames)):
        
        # load the file and get its name       
        f = h5py.File(fnames[i],'r')
        fname = os.path.basename(fnames[i])[0:-3]
        
        ftrials  = list(f['ML'].keys())[1:-1]        
        sessiondf = pd.DataFrame()

        # find some characteristics of the eye tracking data
        #eye_mean, eye_std = get_mean_and_std_eyespeed(f, ftrials)
        # historical and stable values so everything is always on the same scale
        eye_mean = .1
        eye_std = .2
        
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
            
                # get speed of eye movements in each direction
                dx = np.diff(eye[0,:])
                dy = np.diff(eye[1,:])

                # get this trial's 2d speed
                eye_speed = np.hypot(dx,dy)
                raw_eye_speed = eye_speed

                # z score the eye speed
                eye_speed = (raw_eye_speed - eye_mean) / eye_std

                # pull a window of eye_data around when the choice options appear
                eye = eye[:,stim_on_time[0]:stim_off_time[0]]
                eye_speed = eye_speed[stim_on_time[0]:stim_off_time[0]]

                # find the saccades
                sacc_ix = sig.find_peaks(eye_speed, 2, distance = 35)[0]
                
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

                    # #plot eye movement for the trial
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


                    # # check if stimulation was applied and which current
                    # sessiondf.at[t, 'stim'] = 0
                    # stim_code = 10

                    # if np.sum(event_codes == 201) > 0:
                    #     sessiondf.at[t, 'stim'] = 25
                    #     stim_code = 201

                    # if np.sum(event_codes == 202) > 0:
                    #     sessiondf.at[t, 'stim'] = 50
                    #     stim_code = 202

                    # if np.sum(event_codes == 203) > 0:
                    #     sessiondf.at[t, 'stim'] = 75
                    #     stim_code = 203

                    # # find when stimulation was applied
                    # if stim_code != 10:
                    #     stim_onset_time = np.round(event_times[event_codes==stim_code] / 2).astype(int)[0]
                    # else:
                    #     stim_onset_time = np.round(event_times[event_codes==10] / 2).astype(int)[0]
                    #     stim_onset_time = stim_onset_time - 200

                    # # calculate mean eye speed during the 200 ms after stim onset (data were sampled at 500 Hz)
                    # sessiondf.at[t, 'stim_eyespeed'] = np.mean(raw_eye_speed[stim_onset_time: stim_onset_time+100]) 

                    xx=[]

        # save the data as a .csv in top_parent_dir
        save_name = base_folder + '/' + fname + '_bhv.csv'
        print(save_name)
        sessiondf.to_csv(save_name)
        print('Saved data as .csv in original directory.')


def make_spike_tables_and_combine_data(base_folder, save_dir, spike_params):

    # get the name of this recording
    rec_name = Path(base_folder).stem[0:-3]

    # create the file to save into
    save_name = save_dir + '/' + rec_name + '.h5'

    # Create HDF5 file
    with h5py.File(save_name, 'w') as hf:
        pass  # No need to write anything initially, just creating the file

    # specify the offset and step_size (in milliseconds) for making spike tables
    t_offset = spike_params['t_offset']
    step_size = spike_params['step_size']
    win_size = spike_params['win_size']
    align_event = spike_params['align_event']

    # load the event_codes + event_times for this directory
    task_events = np.load(base_folder + '/sync_event_codes.npy')

    # load the info about which brain area was on which probe
    brain_areas = pd.read_csv(base_folder + '/brain_areas.csv')['brain_area']

    # load the behavior
    bhv = pd.read_csv(get_path_from_dir(base_folder, '_bhv'))

    # find the start and stop of each trial
    trial_starts = np.argwhere(task_events[:,0] == 9).flatten()
    trial_ends = np.argwhere(task_events[:,0] == 18).flatten() + 1
    last_end_time = round(task_events[-1, 1])

    # Get a list of subdirectories - one per brain area
    sub_dirs = [d for d in os.listdir(base_folder) if os.path.isdir(os.path.join(base_folder, d))]

    # now loop over each directory
    for dir_ix, this_dir in enumerate(sub_dirs):

        this_brain_area = str(brain_areas[dir_ix])

        # load the spike times associated with this directory
        spike_times = np.load(base_folder + this_dir + '/' + 'ks4_out/sorter_output/sync_spike_times.npy')
        spike_clusters = np.load(base_folder + this_dir + '/' + 'ks4_out/sorter_output/spike_clusters.npy')

        # load the cluster metrics
        cluster_labels=pd.read_csv(base_folder + this_dir + '/' + 'ks4_out/sorter_output/cluster_KSLabel.tsv',sep='\t')
        quality_metrics = pd.read_csv(base_folder + this_dir + '/' + 'ks4_out/sorter_output/quality_metrics.csv')

        templates_average = np.load(base_folder + this_dir + '/' + 'ks4_out/sorter_output/templates.npy')

        # find the clusters
        try:
            cluster_ix = (quality_metrics['presence_ratio'] >= .9) & (quality_metrics['pct_isi_violations'] <= .1)
            good_cluster_nums = quality_metrics['Unnamed: 0'].loc[cluster_ix].values
        except:
            cluster_ix = (cluster_labels['KSLabel'] == 'good') | (cluster_labels['KSLabel'] == 'mua')
            good_cluster_nums = cluster_labels['cluster_id'].loc[cluster_ix]

        n_candidate_clusters = len(good_cluster_nums)

        print(str(n_candidate_clusters) + ' putative units found in ' + this_brain_area)

        # calculate the spike locations and mean waveforms for the selected units
        u_mean_waves, u_positions, u_ch = get_unit_waves_and_positions(templates_average[cluster_ix,:,:])

        # now get ready to extract the firing rates
        # specify the bin_centers
        bin_centers = np.arange(-1*t_offset, t_offset+step_size, step_size)
        n_bins = len(bin_centers)

        # initialize a n_trials x n_times x n_units array to store results in
        firing_rates = np.empty(shape=(len(trial_ends), n_bins, n_candidate_clusters))
        firing_rates[:] = np.nan

        dense_FR = np.empty(shape=(len(trial_ends), 2*t_offset, n_candidate_clusters))
        dense_FR[:] = np.nan

        # now loop over these putatively good units and make spike tables
        unit_names = []
        for u_ix, cluster_num in tqdm(enumerate(good_cluster_nums)):

            unit_names.append(rec_name + '_' + this_brain_area +'_u' + str(cluster_num))

            # Round spike times and filter based on last_end_time
            u_spike_times = np.round(spike_times[spike_clusters.flatten() == cluster_num])
            u_spike_times = np.delete(u_spike_times, np.where(u_spike_times > last_end_time-1)[0])

            # Initialize an empty array for the spike train
            u_spike_train = np.zeros(shape=last_end_time)

            # Set 1s at spike times
            u_spike_train[u_spike_times.astype(int)] = 1

            # now loop over the trials
            for t in range(len(trial_ends)-1):

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

        start_ix = np.argmin(np.abs(bin_centers- -1000))
        end_ix = np.argmin(np.abs(bin_centers- 1000))

        firing_rates = firing_rates[:,start_ix:end_ix+1,:]
        ts = ts[start_ix:end_ix+1]

        # find units to reject
        u_mean_FR = np.nanmean(firing_rates, axis=(0, 1))
        units2del = u_mean_FR < 1
        firing_rates2 = np.delete(firing_rates, units2del, axis=2)
        u_mean_FR2 = np.delete(u_mean_FR, units2del, axis=0)
        unit_names2 = [name for name, mask in zip(unit_names, units2del) if not mask]
        z_fr = np.zeros_like(firing_rates2)
        z_fr[:] = np.nan
        # zscore the units
        for i_u in range(z_fr.shape[2]):

            i_u_mean = np.nanmean(firing_rates2[:,:, i_u])
            i_u_std = np.nanstd(firing_rates2[:,:, i_u])
            z_fr[:,:,i_u] = (firing_rates2[:,:, i_u] - i_u_mean) / i_u_std

        u_ch2 = np.delete(u_ch, units2del)
        u_positions2 = np.delete(u_positions, units2del, axis=0)
        u_mean_waves2 = np.delete(u_mean_waves, units2del, axis=0)

        print('saving ' +str(len(u_mean_FR2)) +' units')

        # now actually save the data
        # dataframes have their own method
        bhv.to_hdf(save_name, key='bhv', mode='a')

        with h5py.File(save_name, 'a') as hf:
            # Write data into the file with the specified variable name
            hf.create_dataset(this_brain_area + '_FR', data=firing_rates2)
            hf.create_dataset(this_brain_area + '_zFR', data=z_fr)
            hf.create_dataset(this_brain_area + '_u_names', data=unit_names2)
            hf.create_dataset(this_brain_area + '_channels', data=u_ch2)
            hf.create_dataset(this_brain_area + '_locations', data=u_positions2)
            hf.create_dataset(this_brain_area + '_mean_wf', data=u_mean_waves2)

            if dir_ix == 0:
                hf.create_dataset('ts', data=ts)

            # Get the keys (dataset names) present in the HDF5 file
            keys = list(hf.keys())  # List of keys in the HDF5 file

            # Store the keys as an attribute named 'dataset_names'
            hf.attrs['dataset_names'] = keys


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

def get_path_from_dir(base_folder, file_name):
    """
    Search for a file within the specified directory and its subdirectories
    by matching the given file name.

    Args:
    - base_folder (str): The directory path to start the search from.
    - file_name (str): The specific file name or part of the file name to be matched.

    Returns:
    - str or None: The path to the first file found with the given name,
      or None if no file is found.
    """
    base_folder = Path(base_folder)
    
    # Iterate through all files and directories in the base_folder
    for file in base_folder.glob('**/*'):
        if file.is_file() and file_name in file.name:
            return file.resolve()  # Return the path of the first file found with the given name
    
    # Print an error message if no file with the given name is found
    print(f"No '{file_name}' file found in the directory.")
    return None

def get_mean_and_std_eyespeed(f, ftrials):
    """
    Calculate the mean and standard deviation of eye movement speed
    during specific trial periods defined by stimulus onset and offset times.

    Args:
    - f (dict): Data dictionary containing eye movement and event information.
    - ftrials (list): List of indices or identifiers for specific trials.

    Returns:
    - tuple: A tuple containing the mean and standard deviation of eye movement speed.
    """
    all_eye_speed = np.array([])

    for t in range(len(ftrials)):
        # get the event code for stim on for saccade detection
        event_codes = np.array([f['ML'][ftrials[t]]['BehavioralCodes']['CodeNumbers']])
        event_times = np.array([f['ML'][ftrials[t]]['BehavioralCodes']['CodeTimes']])
        stim_on_time = np.round(event_times[event_codes==40] / 2).astype(int)
        stim_off_time = np.round(event_times[event_codes==41] / 2).astype(int)

        # get eye data for the trial
        eye = np.squeeze(np.array([f['ML'][ftrials[t]]['AnalogData']['Eye']]))

        # detect saccades
        if (stim_on_time.size > 0) & (stim_off_time.size > 0):
            # get speed of eye movements in each direction
            dx = np.diff(eye[0, :])
            dy = np.diff(eye[1, :])

            # get 2d speed
            eye_speed = np.hypot(dx, dy)

            all_eye_speed = np.concatenate([all_eye_speed, eye_speed])

    eye_mean = np.nanmean(all_eye_speed)
    eye_std = np.nanstd(all_eye_speed)
    return eye_mean, eye_std

def generate_channel_map(y_pitch, total_channels, x_pitch):

    """
    Generate a channel map consisting of vertical and horizontal positions.

    Args:
    - y_pitch (numeric): The pitch or step size for the vertical positions.
    - total_channels (int): The total number of channels in the map.
    - x_pitch (numeric): The pitch or step size for the horizontal positions.

    Returns:
    - vertical_position (numpy.ndarray): An array representing vertical positions
      of the channels, calculated based on the specified y_pitch.
    - horizontal_position (numpy.ndarray): An array representing horizontal positions
      of the channels, calculated based on the specified x_pitch. These positions alternate
      between positive and negative values.

    This function generates a channel map by calculating vertical positions based on the y_pitch
    and creating horizontal positions that alternate between positive and negative values
    using the x_pitch. The resulting arrays represent the vertical and horizontal positions
    of the channels in the map.
    """
    # Calculate the number of pairs (since each pair consists of two elements)
    num_pairs = total_channels // 2

    # Create an array of indices for the pairs
    indices = np.arange(num_pairs)

    # Calculate values for the pairs using vectorized operations
    values = indices * y_pitch

    # Duplicate values for each pair
    vertical_position = np.repeat(values, 2)

    # Create an array of alternating signs for each pair and scale by x_offset
    signs = np.tile([1, -1], num_pairs)[:total_channels]  # Create alternating signs for each pair
    horizontal_position = signs * x_pitch  # Scale by x_offset

    return vertical_position, horizontal_position


def get_unit_waves_and_positions(templates_average):
    """
    Obtain unit wave characteristics and positions based on input templates_average.

    Args:
    - templates_average (numpy.ndarray): A 3D array containing mean waveform templates for units.
      The dimensions are: (n_units, n_samples, n_channels).

    Returns:
    - u_wave_means (numpy.ndarray): A 2D array representing mean waveforms for each unit.
    - u_position (numpy.ndarray): A 2D array containing the positions of units in a channel map.
      Each row represents the horizontal and vertical positions of a unit.
    - u_max_ch (int): The channel index where the signal was maximal for the last unit.

    This function processes the `templates_average` array to extract characteristics and positions
    for units. It calculates mean waveforms for each unit, determines the channel with the maximal
    signal for each unit, and assigns horizontal and vertical positions to the units based on a
    local channel map. The final output includes arrays for mean waveforms (`u_wave_means`),
    positions of units (`u_position`), and the channel index with maximal signal for the last unit
    (`u_max_ch`).
    """
    # get characteristics of this recording
    n_units, n_samples, n_channels = templates_average.shape

    u_position = np.zeros(shape=(n_units, 2))
    u_wave_means = np.zeros(shape=(n_units, n_samples))
    u_max_ch = np.zeros(shape=(n_units, )).astype(int)

    # create a local channel map
    vertical_position, horizontal_position = generate_channel_map(15, 384, 40)

    for u in range(n_units):
        u_waves = templates_average[u, :, :]
        no_zero_u_waves = u_waves.copy()
        no_zero_u_waves[no_zero_u_waves == 0] = np.nan
        u_wave_means[u, :] = np.nanmean(no_zero_u_waves, axis=1)
        u_waves = templates_average[u, :, :]

        # find the location where the signal was biggest
        _, max_ch = np.unravel_index(np.argmin(u_waves), u_waves.shape)
        
        u_max_ch[u] = max_ch

        if horizontal_position[max_ch] < 0:
            u_position[u,0] = horizontal_position[max_ch] + random.uniform(-10, 20)
        else:
            u_position[u,0] = horizontal_position[max_ch] + random.uniform(-20, 10)

        u_position[u,1] = vertical_position[max_ch] + random.uniform(-20, 20)

    return u_wave_means, u_position, u_max_ch


def generate_channel_map(y_pitch, total_channels, x_pitch):

    """
    Generate a channel map consisting of vertical and horizontal positions.

    Args:
    - y_pitch (numeric): The pitch or step size for the vertical positions.
    - total_channels (int): The total number of channels in the map.
    - x_pitch (numeric): The pitch or step size for the horizontal positions.

    Returns:
    - vertical_position (numpy.ndarray): An array representing vertical positions
      of the channels, calculated based on the specified y_pitch.
    - horizontal_position (numpy.ndarray): An array representing horizontal positions
      of the channels, calculated based on the specified x_pitch. These positions alternate
      between positive and negative values.

    This function generates a channel map by calculating vertical positions based on the y_pitch
    and creating horizontal positions that alternate between positive and negative values
    using the x_pitch. The resulting arrays represent the vertical and horizontal positions
    of the channels in the map.
    """
    # Calculate the number of pairs (since each pair consists of two elements)
    num_pairs = total_channels // 2

    # Create an array of indices for the pairs
    indices = np.arange(num_pairs)

    # Calculate values for the pairs using vectorized operations
    values = indices * y_pitch

    # Duplicate values for each pair
    vertical_position = np.repeat(values, 2)

    # Create an array of alternating signs for each pair and scale by x_offset
    signs = np.tile([1, -1], num_pairs)[:total_channels]  # Create alternating signs for each pair
    horizontal_position = signs * x_pitch  # Scale by x_offset

    return vertical_position, horizontal_position
import glob
import numpy as np
import pickle
import sys
import os
from pathlib import Path
sys.path.append("C:/Users/ebtbl/OneDrive/Desktop/NeuroTime/")

from scipy.io import loadmat
from neurotime.NeuronData import NeuronData
from neurotime.data_utils import SignalTimes, SignalValues

root_path = Path(__file__).resolve().parent.parent.joinpath('data/matlab/')
both_paths = glob.glob(str(root_path.joinpath('both')) + "/*.mat")
sta_paths = glob.glob(str(root_path.joinpath('sta')) + "/*.mat")
vol_paths = glob.glob(str(root_path.joinpath('vol')) + "/*.mat")

paths_list = [both_paths, sta_paths, vol_paths]
for paths in paths_list:
    for path in paths:
        mat_contents = loadmat(path)
        neuron_data_mat = mat_contents['NeuronData'][0]

        tst = neuron_data_mat['trial_start_times'][0]
        tet = neuron_data_mat['trial_end_times'][0]
        spike_times = neuron_data_mat['spike_times'][0]

        st = neuron_data_mat['signal_times'][0]
        sv = neuron_data_mat['signal_values'][0]

        signal_times = SignalTimes()
        signal_values = SignalValues()

        signal_times.reward = st['reward'][0][0]
        signal_values.reward = sv['reward'][0][0]
        signal_times.choice_side = st['choice_side'][0][0]
        signal_values.choice_side = sv['choice_side'][0][0]
        signal_times.choice_stimulus = st['choice_stimulus'][0][0]
        signal_values.choice_stimulus = sv['choice_stimulus'][0][0]

        # construct the neuron data object
        neuron_data = NeuronData(trial_start_times = tst, trial_end_times = tet, 
                                    spike_times=spike_times, signal_times=signal_times, 
                                    signal_values=signal_values)
        
        output_path = path.replace('matlab', 'neuron_data')
        output_path = output_path[:-4] + ".npy"
        
        with open(output_path, 'wb') as output_file:
            pickle.dump(neuron_data, output_file)


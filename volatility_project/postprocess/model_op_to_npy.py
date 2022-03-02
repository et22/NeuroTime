import glob
import numpy as np
import pickle
import sys
sys.path.append("C:/Users/ebtbl/OneDrive/Desktop/NeuroTime/")

import os
from pathlib import Path
import multiprocessing as mp

from neurotime.ARMAXModel import ARMAXModel, ModelDefinition

if __name__ == '__main__':
    ses_area = np.hstack((np.zeros(shape=(108)), np.ones(shape=(117)), 2*np.ones(shape=(91))))
    area_label = ["dlPFC", "OFC", "ACC"]
    model_combos = ["TR", "Int", "Sea", "Exo_rew", "Exo_csi", "Exo_cti"]

    root_path = Path(__file__).resolve().parent.parent.joinpath('data/model_op/')
    both_paths = glob.glob(str(root_path.joinpath('both')) + "/*.npy")
    sta_paths = glob.glob(str(root_path.joinpath('sta')) + "/*.npy")
    vol_paths = glob.glob(str(root_path.joinpath('vol')) + "/*.npy")

    paths_list = [both_paths, sta_paths, vol_paths]
    paths_names = ['both', 'stable', 'volatile']
    for p_idx, paths in enumerate(paths_list):
        timescales = dict()
        amplitudes = dict()
        significant = dict()
        keys = ['ar_intrinsic', 'ar_seasonal', 'ex_reward', 'ex_choice_side', 'ex_choice_stimulus']
        for key in keys:
            timescales[key] = [[], [], []]
            amplitudes[key] = [[], [], []]
            significant[key] = [[], [], []]

        
        for n_idx, path in enumerate(paths):
            with open(path, 'rb') as input_file:
                neuron_results = pickle.load(input_file)
            ses_num = int(path.split('ses_', 1)[1][:-4])-1
            params = neuron_results['params'] 
            sigs = neuron_results['sig'] 
            for k_idx, key in enumerate(keys):
                timescales[key][int(ses_area[ses_num])].append(params[key + '_tau'])
                if not key[0:2] == 'ar':
                    amplitudes[key][int(ses_area[ses_num])].append(params[key + '_amp'])
                significant[key][int(ses_area[ses_num])].append(sigs[k_idx+1])

        pname = paths_names[p_idx]
        output_path = str(Path(__file__).resolve().parent.parent.joinpath(f'data/postprocessed/{pname}.npy'))
        with open(output_path, 'wb') as output_file:
            pickle.dump((timescales, amplitudes, significant), output_file)
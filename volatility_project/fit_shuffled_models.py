import glob
import numpy as np
import pickle
import sys
sys.path.append("C:/Users/ebtbl/OneDrive/Desktop/NeuroTime/")

import os
from pathlib import Path
import multiprocessing as mp

from neurotime.ARMAXModel import ARMAXModel, ModelDefinition

def fit_neurons(p_idx, paths, n_idx, path, paths_list):
    n_lists = len(paths_list)
    n_paths = len(paths)
    n_shuffles = 3
    print(f'fitting set {p_idx}/{n_lists}, neuron {n_idx}/{n_paths}')
    output_path = path.replace('neuron_data', 'model_op_shuffle')

    if not os.path.exists(output_path):
        with open(path, 'rb') as input_file:
                neuron_data = pickle.load(input_file)

        neuron_params = [dict() for _ in range(n_shuffles)]
        neuron_rsqs = []
        model_combos = ["TR", "Int", "Sea", "Exo_rew", "Exo_csi", "Exo_cti"]

        for mod in model_combos:
            model_definition = ModelDefinition()
            if mod == "TR":
                model_definition.addTRComponent(name = 'reward')
                model_definition.addTRComponent(name = 'choice_side')
                model_definition.addTRComponent(name = 'choice_stimulus')
            elif mod == "Int":
                model_definition.addARComponent(name = 'intrinsic')
            elif mod == "Sea":
                model_definition.addARComponent(name = 'seasonal')
            elif mod == "Exo_rew":
                model_definition.addExoComponent(name = 'reward')
            elif mod == "Exo_csi":
                model_definition.addExoComponent(name = 'choice_side')
            elif mod == "Exo_cti":
                model_definition.addExoComponent(name = 'choice_stimulus')
            else:
                raise ValueError(f"invalid model name {mod}")

            # now, we construct the ARMAX model
            armax_model = ARMAXModel(model_definition=model_definition)

            X, y, i_size, s_size = neuron_data.to_armax_input(model = armax_model, model_definition = model_definition, bin_size = 50)

            # next, call fit function
            r_sqs, shuffle_params = armax_model.fit_shuffled(X, y, n_shuffles=n_shuffles)

            for idx, params in enumerate(shuffle_params):
                armax_model.params = params
                params = armax_model.get_params(intrinsic_size=i_size, seasonal_size=s_size)
                neuron_params[idx].update(params) 
            neuron_rsqs.append(r_sqs)
        
        neuron_results = dict()
        neuron_results['params'] = neuron_params
        neuron_results['mods'] = model_combos
        neuron_results['rsq'] = neuron_rsqs

        with open(output_path, 'wb') as output_file:
            pickle.dump(neuron_results, output_file)

if __name__ == '__main__':
    root_path = Path(__file__).resolve().parent.joinpath('data/neuron_data/')
    both_paths = glob.glob(str(root_path.joinpath('both')) + "/*.npy")
    sta_paths = glob.glob(str(root_path.joinpath('sta')) + "/*.npy")
    vol_paths = glob.glob(str(root_path.joinpath('vol')) + "/*.npy")

    #pool = mp.Pool(mp.cpu_count())

    paths_list = [both_paths, sta_paths, vol_paths] #both_paths, 
    for p_idx, paths in enumerate(paths_list):
        #pool.starmap(fit_neurons, [(p_idx, paths, n_idx, path, paths_list) for n_idx, path in enumerate(paths)])
        for n_idx, path in enumerate(paths):
            try:
                fit_neurons(p_idx, paths, n_idx, path, paths_list)
            except Exception as e:
                print(e)
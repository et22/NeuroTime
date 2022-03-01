import pickle
import multiprocessing as mp
print("Number of processors: ", mp.cpu_count())

input_path = "volatility_project/data/model_op/both/model_op_neu_1_ses_1.npy"
with open(input_path, 'rb') as input_file:
    neuron_results = pickle.load(input_file)


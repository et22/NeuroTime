from scipy.io import loadmat
from neurotime.NeuronData import NeuronData
from neurotime.data_utils import SignalTimes, SignalValues

from neurotime.ARMAXModel import ARMAXModel, ModelDefinition

data_path = './example_data/example_neuron_data.mat'
mat_contents = loadmat(data_path)
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

# here, we add 3 task relevant, 2 autoregressive, and 2 exogenous terms to the model
model_definition = ModelDefinition()
# model_definition.addTRComponent(name = 'reward')
# model_definition.addTRComponent(name = 'choice_side')
# model_definition.addTRComponent(name = 'choice_stimulus')
# model_definition.addARComponent(name = 'intrinsic')
model_definition.addARComponent(name = 'seasonal')
# model_definition.addExoComponent(name = 'reward')
# model_definition.addExoComponent(name = 'choice_side')
# model_definition.addExoComponent(name = 'choice_stimulus')

# now, we construct the ARMAX model
armax_model = ARMAXModel(model_definition=model_definition)


X, y, i_size, s_size = neuron_data.to_armax_input(model = armax_model, model_definition = model_definition, bin_size = 50)

# next, call fit function
armax_model.fit(X, y)

# finally, we report fitting results
r_sq = armax_model.score(X, y)
print(f'Model Performance: R^2={r_sq}')

params = armax_model.get_params(intrinsic_size=i_size, seasonal_size=s_size)
print(f'Model parameters: {params}')
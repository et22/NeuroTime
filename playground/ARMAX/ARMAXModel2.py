from lmfit.models import Model, LinearModel, ExponentialModel, ConstantModel, ExpressionModel

import numpy as np

"""
mean_sc = 5
mean_fr_mod = ConstantModel(prefix='mean_sc')
model1 = ExpressionModel('mean_sc*x', independent_vars=['x'])
model = model1 + mean_fr_mod + LinearModel(prefix='exogenous') + LinearModel(prefix='intrinsic') + LinearModel(prefix='seasonal') + ExponentialModel(prefix='choice_memory')
#model = ConstantModel(param_names = 'mean_sc')
print(model.param_names)


# independent vars for autoreg
independent_vars = ["sc_bin_1", "sc_bin_2", ...]
params = ["alpha_1", "alpha_2", ...]

# independent vars for memory
independent_vars = [""]
params = ["mean_sc", ]

# independent vars for exo
independent_vars = [""]
params = [""]
"""
"""

mean_sc = 5
Z = np.arange(0,10)
U = np.arange(0, 10)
alpha_intrinsic = np.arange(0,5)
intrinsic_bins = np.arange(0,5)
alpha_seasonal = np.arange(0,5)
seasonal_bins = np.arange(0,5)
amp_reward = .5
amp_choice = .7
amp_stimulus = .8
time_diff_reward = np.arange(0,5)
tau_reward = 2
signal_reward = np.arange(0,5)
time_diff_choice = np.arange(0,5)
tau_choice = 2
signal_choice = np.arange(0,5)
time_diff_stimulus = np.arange(0,5)
tau_stimulus = 2
signal_stimulus = np.arange(0,5)
"""

def fcn2min(params, x, data):
    return model(params, x) - data

    
def model(params, x):
    mem_exp = lambda t, tau, signal: np.exp(-t/tau)*signal

    v = params.valuesdict()
    alpha_intrinsic = np.array([v['alpha_intrinsic_1'], v['alpha_intrinsic_2'], v['alpha_intrinsic_3'], v['alpha_intrinsic_4'], v['alpha_intrinsic_5']])
    alpha_seasonal = np.array([v['alpha_seasonal_1'], v['alpha_seasonal_2'], v['alpha_seasonal_3'], v['alpha_seasonal_4'], v['alpha_seasonal_5']])

    model_output = mean_sc + np.dot(Z, U) + \
        np.dot(alpha_intrinsic, intrinsic_bins) + np.dot(alpha_seasonal, seasonal_bins) + \
        mean_sc*v['amp_reward']*np.sum([mem_exp(t, v['tau_reward'], signal) for t, signal in zip(time_diff_reward, signal_reward)]) + \
        mean_sc*v['amp_choice']*np.sum([mem_exp(t, v['tau_choice'], signal) for t, signal in zip(time_diff_choice, signal_choice)]) + \
        mean_sc*v['amp_stimulus']*np.sum([mem_exp(t, v['tau_stimulus'], signal) for t, signal in zip(time_diff_stimulus, signal_stimulus)])

print(model_output)
"""
class ARMAXModel():
    def __init__(self):
        self.model = ConstantModel(prefix='mean_sc')

    def __call__(self, inputs):
        output = self.model(inputs)
        return output

    def addAutoRegTerm(self, name: str, depth: int):
        for i in range(depth):
            # define each autoregressive term in sum
            alpha_param = name + "_alpha_" + str(i)
            ind_var = name + "_bin_" + str(i)
            model = alpha_param + " * " + ind_var
            term = ExpressionModel(model, independent_vars=[ind_var])

            # add each term to model
            self.model = self.model + term
    
    def addMemTerm(self, name: str, depth: int):
        mean_sc = name + "_meanfr"
        for i in range(depth):
            # define independent vars in sum

            # define each autoregressive term in sum
            alpha_param = name + "_alpha_" + str(i)
            model = alpha_param + " * " + ind_var
            term = ExpressionModel(model, independent_vars=[ind_var])

            # add each term to model
            self.model = self.model + term
           

    def addExoTerm(self, name: str, mask_duration: int):
        exoTerm = ExogenousComponent(name = name, mask_duration = mask_duration)
        self.addTerm(exoTerm.getModel())

"""


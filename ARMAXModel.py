import numpy as np
from scipy.optimize import minimize

from .ModelDefinition import ModelDefinition

# TODO priorities
# - transform nerondata to model input based on model definition/construction
# - actually test the model, and write demo notebook
# - write code to extract intrinsic and seasonal timescales using eignvalue stuff 

# - write cv/more complex fitting function
# - write autocorrelation timescale estimation method 
class ARMAXModel():
    def __init__(self, model_definition: ModelDefinition):
        """
        Initializes an ARMAXModel from a ModelDefinition. 
        Attributes:
        model_definition : ModelDefinition
            describes model components
        param_labels : list[str]
            names of model parameters
        param_bounds : list[list[int]]
            upper and lower bounds for each parameter
        signal_labels : list[str]
            names of model signals
        model : func
            function to predict spike counts   
        params : list[float]
            parameter values of the model   

        Methods:
        fit(X, y)
            fits parameters to minimize mse cost function
        predict(X)
            returns predicted spiek counts given input
        score(X, y)
            returns r^2 value given input and actual spike counts
        set_params(param_names, param_values)
            sets model parameters
        get_params()
            returns model parameters

        """
        self.model_definition = model_definition
        self.__initialize_model(model_definition)
        self.__initialize_parameters()

    def fit(self, X, y):
        vars, times = self.__separate_predictors(X)
        min_func = lambda params: np.square(y-self.model(params, vars, times))

        res = minimize(min_func, self.params)

        if not res.success:
            raise ValueError(f'optimization failed: {res.message}')
        else:
            self.params = res.x

    def predict(self, X):
        vars, times = self.__separate_predictors(X)
        y_pred = self.model(self.params, vars, times)
        return y_pred

    def score(self, X, y):
        y_pred = self.predict(X)
        corr_mtx = np.corrcoef(y, y_pred)
        r_sq = corr_mtx[0, 1] ** 2
        return r_sq


    def __separate_predictors(self, X):
        n = len(self.signal_labels)
        vars = X[:,:n]
        times = X[:,n:]
        return vars, times

    def set_params(self, param_names, param_values):
        for param_name, param_value in zip(param_names, param_values):
            if param_name in self.param_labels:
                idx = self.param_labels.index(param_name)
                self.params[idx] = param_value
            else:
                raise ValueError(f"{param_name} is not a valid parameter name.")

    def get_params(self):
        param_dict = dict()
        for name, value in zip(self.param_labels, self.params):
            param_dict[name] = value

        return param_dict

    def __initialize_model(self, model_definition: ModelDefinition):
        """
        Generates model function from a ModelDefinition. 
        """
        param_bounds = []
        param_labels = []
        signal_labels = []

        for tr in model_definition.tr_comps:
            param_labels.append(tr.name)
            param_bounds.append(tuple(tr.param_bounds))
            
            signal_labels.append(tr.name)

        for ar in model_definition.ar_comps:
            param_labels += [f'{ar.name}_{idx}' for idx in range(ar.depth)]
            param_bounds += [tuple(ar.param_bounds) for _ in range(ar.depth)]
            
            signal_labels += [f'{ar.name}_{idx}' for idx in range(ar.depth)]

        dp_len = len(signal_labels)

        for exo in model_definition.exo_comps:
            param_labels.append(f'{exo.name}_amp')
            param_bounds.append(tuple(exo.amp_bounds))

            param_labels.append(f'{exo.name}_tau')
            param_bounds.append(tuple(exo.tau_bounds))

            signal_labels += [f'{exo.name}_{idx}' for idx in range(exo.depth)]

        # add mean firing rate to model
        signal_labels.append(model_definition.fr_comp.name)

        def exo_fun(vars, delta_time, tau):
            op = np.exp(-delta_time/tau)*vars
            return np.sum(op, axis=1) # sum along rows and return

        def model_fun(params, vars, times):
            output = np.sum(vars[:,:dp_len]*params[:dp_len]*times[:,:dp_len], axis = 1) # first take product
            output += vars[:, -1] # add mean fr to each pred
            vt_idx = dp_len
            for idx, exo in enumerate(model_definition.exo_comps):
                p_idx = dp_len+2*idx
                next_vt_idx = vt_idx + exo.depth
                output += vars[:, -1]*params[p_idx]*exo_fun(vars[:,vt_idx:next_vt_idx], times[:,vt_idx:next_vt_idx], params[p_idx+1])
                vt_idx = next_vt_idx

        self.model = model_fun
        self.param_bounds = param_bounds
        self.param_labels = param_labels 
        self.signal_labels = signal_labels
    
    def __initialize_parameters(self):
        params = []
        for bounds in self.param_bounds:
            if len(bounds) == 0:
                bounds = (0, 1)
            params.append(np.random.uniform(low=bounds[0], high=bounds[1]))

        self.params = params
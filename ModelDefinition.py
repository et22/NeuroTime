from .DataClasses import FRComponent, TRComponent, ARComponent, ExoComponent

class ModelDefinition():
    def __init__(self):
        self.fr_comp = FRComponent('mean_fr')
        self.tr_comps = []
        self.ar_comps = []
        self.exo_comps = []

    def addTRComponent(self, name: str, param_bounds: list[int] = [-5, 5]):
        """
        Adds task relevant component to model. 
        Optional keyword arguments:
        name:   Name of task relevant component. Must be the name of a signal field in 
                the input NeuronData.
        param_bounds: [lower bound, upper bound] for the beta parameter for the task relevant component
        """
        self.tr_comps.append(TRComponent(name, param_bounds))

    def addARComponent(self, name: str = 'intrinsic', depth: int = 5, param_bounds: list[int] = [-5, 5]):
        """
        Adds autoregressive component to model. 
        Optional keyword arguments:
        name:   'intrinsic' or 'seasonal'
        depth:  Number of previous bins to regress over
        param_bounds:  [lower bound, upper bound] for all the beta parameters for the AR component
        """
        self.ar_comps.append(ARComponent(name, depth, param_bounds))

    def addExoComponent(self, name: str, depth: int = 5, amp_bounds: list[int] = [-5, 5], tau_bounds: list[int] = [0, 30]):
        """
        Adds exogenous component to model. 
        Optional keyword arguments:
        name:   Name of exogneous component. Must be the name of a signal field in 
                the input NeuronData.
        depth:  Number of previous trials to regress over
        amp_bounds: [lower bound, upper bound] for the amplitude parameter for the exogenous component
        tau_bounds: [lower bound, upper bound] for the tau parameter for the exogenous component
        """
        self.exo_comps.append(ExoComponent(name, depth, amp_bounds, tau_bounds))


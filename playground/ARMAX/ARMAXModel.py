import numpy as np
import time
from lmfit import Model, Parameter, report_fit

from AutoRegComponent import AutoRegComponent
from MemoryComponent import MemoryComponent
from ExogenousComponent import ExogenousComponent
from ConstantComponent import ConstantComponent

class ARMAXModel():
    def __init__(self):
        self.component_names = []
        self.model = Model()

    def __call__(self, inputs, params):
        for input in inputs:
            output = 0
            for name in self.component_names:
                comp_func = getattr(self, name)
                output += comp_func(input[name], params[name])
            return output
            
    def addAutoRegTerm(self, name: str):
        self.component_names.append(name)
        setattr(self, name, AutoRegComponent(name = name))
    
    def addMemTerm(self, name: str):
        self.component_names.append(name)
        setattr(self, name, MemoryComponent(name = name))    

    def addExoTerm(self, name: str, mask_duration: int):
        self.component_names.append(name)
        setattr(self, name, ExogenousComponent(name = name, mask_duration = mask_duration))

    def addConstantTerm(self, name: str):
        self.component_names.append(name)
        setattr(self, name, ConstantComponent(name = name))

    def fit(self, neuron_data):
        pass
    def train(self, name: str):
        pass
    def validate(self, name: str):
        pass

if __name__ == '__main__':
    model = ARMAXModel()
    model.addConstantTerm('mean_fr')
    model.addAutoRegTerm('intrinsic')
    model.addExoTerm('choice', mask_duration=500)
    model.addMemTerm('choice_mem')

    choice_input = {'signal_time': 5,
                    'signal_value': 10,
                    'time': 10}

    choice_mem_input = {'time_diffs': 50,
                        'signal_values': 1,
                        'mean_fr': 5}

    input = {'mean_fr': 5, 
             'intrinsic': np.array([5, 10, 15, 5, 10]),
             'choice': choice_input,
             'choice_mem': choice_mem_input}
    
    choice_mem_params = {'amp': 2, 
                         'tau': 50}
    
    params = {'mean_fr': None,
              'intrinsic': np.array([5, 10, 15, 5, 10]),
              'choice': 3, 
              'choice_mem': choice_mem_params}
              
    print(model(input, params))


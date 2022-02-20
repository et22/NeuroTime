from ModelComponent import ModelComponent
import numpy as np

class MemoryComponent(ModelComponent):
    def __init__(self, name: str):
        super(MemoryComponent, self).__init__(name)

    def __call__(self, input, params):
        exp = np.sum(np.exp(-input["time_diffs"]/params["tau"])*input["signal_values"])
        return input["mean_fr"]*params["amp"]*exp
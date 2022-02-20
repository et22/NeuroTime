import numpy as np

from ModelComponent import ModelComponent

class AutoRegComponent(ModelComponent):
    def __init__(self, name):
        super(AutoRegComponent, self).__init__(name)

    def __call__(self, input, params):
        return np.dot(input, params)

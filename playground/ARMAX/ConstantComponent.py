from ModelComponent import ModelComponent

class ConstantComponent(ModelComponent):
    def __init__(self, name):
        super(ConstantComponent, self).__init__(name)

    def __call__(self, input, _):
        return input

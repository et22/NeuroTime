from ModelComponent import ModelComponent

class ExogenousComponent(ModelComponent):
    def __init__(self, name: str, mask_duration: int):
        super(ExogenousComponent, self).__init__(name)
        self.mask_duration = mask_duration

    def __call__(self, input, params):
        if input["signal_time"] <= input["time"] < input["signal_time"] + self.mask_duration:
            return input["signal_value"]*params
        else: 
            return 0


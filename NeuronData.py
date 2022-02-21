import numpy as np
from dataclasses import dataclass

@dataclass
class NeuronData:
    trial_start_times: np.ndarray
    trial_end_times: np.ndarray
    spike_times: np.ndarray
    signal_times: SignalTimes
    signal_values: SignalValues
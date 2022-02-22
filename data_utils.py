import numpy as np
from dataclasses import dataclass

@dataclass
class SignalTimes:
    pass

@dataclass
class SignalValues:
    pass

@dataclass
class Signal:
    signal_time: np.ndarray
    signal_value: np.ndarray

@dataclass
class FRComponent:
    name: str

@dataclass
class TRComponent:
    name: str
    param_bounds: list[int]

@dataclass
class ARComponent:
    name: str
    depth: int
    param_bounds: list[int]

@dataclass
class ExoComponent:
    name: str
    depth: int
    amp_bounds: list[int]
    tau_bounds: list[int]

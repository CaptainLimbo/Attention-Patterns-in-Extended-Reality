import numpy as np
from typing import Optional, Dict

from offline.data import GazeData
from offline.modules import Module


class IVTFixationDetector(Module):
    def __init__(self, velocity_threshold: float = 30) -> None:
        self.velocity_threshold = velocity_threshold

    def update(self, data: GazeData) -> GazeData:
        data.fixation = data.velocity < self.velocity_threshold

        return data


class IVTSaccadeDetector(Module):
    def __init__(self, velocity_threshold: float = 30) -> None:
        self.velocity_threshold = velocity_threshold

    def update(self, data: GazeData) -> GazeData:
        data.saccade = ~data.fixation
        return data


class SmoothPursuitDetector(Module):
    def __init__(
        self, low_velocity_threshold: float = 10, high_velocity_threshold: float = 40
    ) -> None:
        self.low_velocity_threshold = low_velocity_threshold
        self.high_velocity_threshold = high_velocity_threshold

    def update(self, data: GazeData) -> GazeData:
        data.smooth_pursuit = (data.velocity > self.low_velocity_threshold) & (
            data.velocity < self.high_velocity_threshold
        )

        return data


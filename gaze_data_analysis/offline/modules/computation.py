import numpy as np
from scipy.signal import savgol_filter
from scipy.ndimage import median_filter
from scipy.stats import mode

from offline.data import GazeData
from offline.modules import Module
import offline.utils as utils


def discard_short_periods(signal, all_extracted_periods, short_fixation_threshold=0.06):
    cleaned_periods = [
        period
        for period in all_extracted_periods
        if signal.start_timestamp[period[1]] - signal.start_timestamp[period[0]] >= short_fixation_threshold
    ]
    return cleaned_periods

def find_all_periods(signal, indices=[], data=None):
    start = None
    all_periods = []
    for i in range(len(signal)-1):
        if start is None:
            if signal[i] and indices[i] + 1 == indices[i + 1]:
                start = i
        else: # there is a start
            if not signal[i] or indices[i] + 1 != indices[i + 1]:
                all_periods.append((start, i))
                start = None
    if start is not None and indices[len(signal) - 1] - 1 == indices[len(signal)-2]:
        all_periods.append((start, len(signal) - 1))
    return all_periods

class DurationDistanceVelocity(Module):
    def __init__(self, window_size=1) -> None:
        self.window_size = window_size

    def update(self, data: GazeData) -> GazeData:
        if self.window_size > 1:
            data.duration = np.roll(data.start_timestamp, -self.window_size // 2, axis=0) - np.roll(data.start_timestamp, self.window_size // 2, axis=0)
            data.duration = data.duration[self.window_size // 2: -self.window_size // 2]
            data.distance = utils.angular_distance(
                np.roll(data.gaze_direction, -self.window_size // 2, axis=0),
                np.roll(data.gaze_direction, self.window_size // 2, axis=0)
            )[self.window_size // 2: -self.window_size // 2]
            for i in range(1, len(data.indices) - 2):
                if data.indices[i+1] != data.indices[i] + 1:
                    data.duration[i - self.window_size // 2] = data.start_timestamp[i] - data.start_timestamp[i - self.window_size // 2]
                    data.distance[i - self.window_size // 2] = utils.single_angular_distance(
                        data.gaze_direction[i],
                        data.gaze_direction[i - self.window_size // 2]
                    )

            assert len(data.duration) == len(data.distance)
        else:
            data.duration = (np.roll(data.start_timestamp, -1, axis=0) - data.start_timestamp)[:-1]
            data.distance = utils.angular_distance(
                data.gaze_direction[:-1],
                data.gaze_direction[1:]
            )
            for i in range(len(data.duration)):
                if data.indices[i+1] != data.indices[i] + 1:
                    data.duration[i] = data.start_timestamp[i+1] - data.start_timestamp[i]
                    data.distance[i] = utils.single_angular_distance(data.gaze_direction[i+1], data.gaze_direction[i])
        data.velocity = data.distance / data.duration
        assert np.all(data.duration > 0)
        assert np.all(data.velocity >= 0)
        data.start_timestamp = data.start_timestamp[:-1] if self.window_size == 1 else data.start_timestamp[self.window_size // 2: -self.window_size // 2]
        data.gaze_direction = data.gaze_direction[:-1] if self.window_size == 1 else data.gaze_direction[self.window_size // 2: -self.window_size // 2]
        data.gaze_target = data.gaze_target[:-1] if self.window_size == 1 else data.gaze_target[self.window_size // 2: -self.window_size // 2]
        if hasattr(data, "label"):
            data.label = data.label[:-1] if self.window_size == 1 else data.label[self.window_size // 2: -self.window_size // 2]
        if len(data.indices) >0:
            data.indices = data.indices[:-1] if self.window_size == 1 else data.indices[self.window_size // 2: -self.window_size // 2]
        return data


class MedianFilter(Module):
    def __init__(self, attr: str, window_size: int = 5) -> None:
        self.attr = attr
        self.window_size = window_size

    def update(self, data: GazeData) -> GazeData:
        signal = getattr(data, self.attr)
        previous_signal = signal.copy()
        for i in range(len(signal) - self.window_size //2 - 1):
            if self.window_size == 5 and data.indices[i+2] - data.indices[i-2] == 4:
                signal[i] = median_filter(previous_signal[i-2:i+3], size=(5, 1))[2]
            elif data.indices[i+1] - data.indices[i-1] == 2:
                signal[i] = median_filter(previous_signal[i-1:i+2], size=(5, 1))[1]
        
        setattr(data, self.attr, signal)

        return data


class ModeFilter(Module):
    def __init__(self, attr: str, window_size: int = 3) -> None:
        self.attr = attr
        self.window_size = window_size

    def update(self, data: GazeData) -> GazeData:
        signal = getattr(data, self.attr)
        signal = [1 if s else 0 for s in signal]
        for i in range(len(signal) - self.window_size):
            signal[i + self.window_size // 2] = mode(
                signal[i : i + self.window_size], keepdims=True
            )[0][0]
        setattr(data, self.attr, signal)

        return data


class ROIFilter(Module):
    def __init__(self, attr: str, window_size: int = 5) -> None:
        self.attr = attr
        self.window_size = window_size

    def update(self, data: GazeData) -> GazeData:
        signal = getattr(data, self.attr)
        for i in range(len(signal) - self.window_size):
            windowed_signal = signal[i : i + self.window_size]
            val = max(set(windowed_signal), key=list(windowed_signal).count)

            if val != 'other':
                signal[i + self.window_size // 2] = val
        setattr(data, self.attr, signal)

        return data


class MovingAverageFilter(Module):
    def __init__(self, attr: str, window_size: int = 5) -> None:
        self.attr = attr
        self.window_size = window_size

    def update(self, data: GazeData) -> GazeData:
        signal = getattr(data, self.attr)
        for i in range(len(signal) - self.window_size):
            signal[i + self.window_size // 2] = np.mean(
                signal[i : i + self.window_size]
            )
        setattr(data, self.attr, signal)

        return data


class SavgolFilter(Module):
    def __init__(self, attr: str, window_size: int = 3, order: int = 1) -> None:
        self.attr = attr
        self.window_size = window_size
        self.order = order

    def update(self, data: GazeData) -> GazeData:
        signal = getattr(data, self.attr)
        signal = savgol_filter(
            signal, window_length=self.window_size, polyorder=self.order
        )
        setattr(data, self.attr, signal)
        return data


class AggregateFixations(Module):
    def __init__(
        self, short_fixation_threshold=0.06, merge_interval_threshold=0.075, merge_direction_threshold=0.5
    ) -> None:
        super().__init__()
        self.short_fixation_threshold = short_fixation_threshold
        self.merge_interval_threshold = merge_interval_threshold
        self.merge_direction_threshold = merge_direction_threshold

    def merge_fixation_periods(self, all_extracted_periods, merge_interval_threshold=0.075, merge_direction_threshold=0.5, data=None):
        i = 0
        while i < len(all_extracted_periods) - 1:
            if data.start_timestamp[all_extracted_periods[i+1][0]] - data.start_timestamp[all_extracted_periods[i][1]] < merge_interval_threshold:
                last_direction = data.gaze_direction[all_extracted_periods[i][1]]
                next_direction = data.gaze_direction[all_extracted_periods[i+1][0]]
                if not utils.single_angular_distance(last_direction, next_direction) > merge_direction_threshold:
                    all_extracted_periods[i] = (all_extracted_periods[i][0], all_extracted_periods[i+1][1])
                    all_extracted_periods.pop(i+1)
                    continue
    
            i += 1
        return all_extracted_periods

        

    def update(self, data: GazeData) -> GazeData:
        fixations = []
        start = None
        all_periods = find_all_periods(data.fixation, indices=data.indices)
        data.fixation = np.zeros(len(data.fixation), dtype=bool)

        all_periods = self.merge_fixation_periods(all_periods, self.merge_interval_threshold, self.merge_direction_threshold, data)
        all_periods = discard_short_periods(data, all_periods, self.short_fixation_threshold)

        for start, i in all_periods:
            gaze_targets = list(data.gaze_target[start : i+1])
            data.fixation[start: i+1] = True

            
            target = max(set(gaze_targets), key=gaze_targets.count)

            fixations.append(
                {
                    "start_timestamp": data.start_timestamp[start],
                    "end_timestamp": data.start_timestamp[i],
                    "duration": data.start_timestamp[i]
                    - data.start_timestamp[start],
                    "centroid": np.average(
                        data.gaze_direction[start : i+1],
                        axis=0,
                        weights=data.duration[start : i+1],
                    ),
                    "target": target,
                }
            )
            start = None

        data.fixations = fixations
        return data


class AggregateSaccades(Module):
    def update(self, data: GazeData) -> GazeData:
        saccades = []
        all_saccades = find_all_periods(data.saccade, indices=data.indices)
        # print("Number of saccades: ", len(all_saccades))
        for inner_start, inner_end in all_saccades:
            saccades.append(
                {
                    "start_timestamp": data.start_timestamp[inner_start],
                    "end_timestamp": data.start_timestamp[inner_end],
                    "duration": data.start_timestamp[inner_end]
                    - data.start_timestamp[inner_start],
                    "amplitude": utils.angular_distance(
                        data.gaze_direction[inner_start, np.newaxis],
                        data.gaze_direction[inner_end, np.newaxis],
                    ),
                    "velocity": np.mean(
                        data.velocity[inner_start : inner_end + 1]
                    ),
                    "peak_velocity": np.max(
                        data.velocity[inner_start : inner_end + 1]
                    ),
                }
            )

        data.saccades = saccades
        del data.saccade
        return data


class AggregateSmoothPursuits(Module):
    def __init__(
        self,
        aggregate_to_fixations=True, short_sp_threshold=0.06, merge_interval_threshold=0.075,
    ) -> None:
        super().__init__()
        self.aggregate_to_fixations = aggregate_to_fixations
        self.short_sp_threshold = short_sp_threshold
        self.merge_interval_threshold = merge_interval_threshold
        self.away_from_puzzle_targets = ["mascot", "progressbar", "timer"]

        
    def merge_sp_with_target_constraints(self, all_extracted_periods, merge_interval_threshold=0.075, merge_direction_threshold=0.5, data=None):
        i = 0
        while i < len(all_extracted_periods) - 1:
            if data.start_timestamp[all_extracted_periods[i+1][0]] - data.start_timestamp[all_extracted_periods[i][1]] < merge_interval_threshold:
                gaze_target_1_list = list(data.gaze_target[all_extracted_periods[i][0] : all_extracted_periods[i][1] + 1])
                gaze_target_2_list = list(data.gaze_target[all_extracted_periods[i+1][0] : all_extracted_periods[i+1][1] + 1])
                gaze_target_1 = max(set(gaze_target_1_list), key=gaze_target_1_list.count)
                gaze_target_2 = max(set(gaze_target_2_list), key=gaze_target_2_list.count)
                if gaze_target_1 == gaze_target_2:
                    last_direction = data.gaze_direction[all_extracted_periods[i][1]]
                    next_direction = data.gaze_direction[all_extracted_periods[i+1][0]]
                    if not utils.angular_distance([last_direction], [next_direction])[0] > merge_direction_threshold:
                        all_extracted_periods[i] = (all_extracted_periods[i][0], all_extracted_periods[i+1][1])
                        all_extracted_periods.pop(i+1)
                        continue
            i += 1
        # print("Merged", original_length - len(all_extracted_periods), "periods")
        return all_extracted_periods

    
    def update(self, data: GazeData) -> GazeData:
        smooth_pursuits = []
        all_smooth_pursuits = find_all_periods(data.smooth_pursuit, indices=data.indices)
        all_smooth_pursuits = self.merge_sp_with_target_constraints(all_smooth_pursuits, data=data)
        all_smooth_pursuits = discard_short_periods(data, all_smooth_pursuits, self.short_sp_threshold)

        for start, i in all_smooth_pursuits:
                    gaze_targets = list(data.gaze_target[start : i + 1])
                    if (
                        not (
                            "hints" in gaze_targets and (any([target in gaze_targets for target in self.away_from_puzzle_targets])) or "puzzle" in gaze_targets and (any([target in gaze_targets for target in self.away_from_puzzle_targets])))
                    ):
                        # a valid smooth pursuit
                        # catheter, ventricle or majority
                        target = (
                            "hints"
                            if "hints" in gaze_targets
                            else (
                                "puzzle"
                                if "puzzle" in gaze_targets
                                else max(set(gaze_targets), key=gaze_targets.count)
                            )
                        )
                        data.fixation[start: i+1] = True
                        # print("Smooth pursuit target: ", gaze_targets)
                        smooth_pursuits.append(
                            {
                                "start_timestamp": data.start_timestamp[start],
                                "end_timestamp": data.start_timestamp[i],
                                "start_index": start,
                                "end_index": i,
                                "duration": data.start_timestamp[i]
                                - data.start_timestamp[start],
                                "amplitude": utils.angular_distance(
                                    data.gaze_direction[start, np.newaxis],
                                    data.gaze_direction[i, np.newaxis],
                                ),
                                "velocity": np.mean(
                                    data.velocity[start : i + 1]
                                ),
                                "target": target,
                            }
                        )
                        if self.aggregate_to_fixations:
                            data.fixations.append(
                                {
                                    "start_timestamp": data.start_timestamp[
                                        start
                                    ],
                                    "end_timestamp": data.start_timestamp[i],
                                    "duration": data.start_timestamp[i]
                                    - data.start_timestamp[start],
                                    "target": target,
                                }
                            )
                    start = None
        data.smooth_pursuits = smooth_pursuits
        return data



if __name__ == "__main__":
    a = np.array([[2,3,4],[-1, 4, 6], [9, 5, 5]])
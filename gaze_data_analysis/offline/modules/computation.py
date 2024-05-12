import numpy as np
from scipy.signal import savgol_filter
from scipy.ndimage import median_filter
from scipy.stats import mode

from offline.data import GazeData
from offline.modules import Module
import offline.utils as utils


def discard_short_periods(all_extracted_periods, minimum_fixation_steps):
    original_length = len(all_extracted_periods)
    # print("Original length", original_length)
    cleaned_periods = [
        period
        for period in all_extracted_periods
        if period[1] - period[0] >= minimum_fixation_steps
    ]
    # print("Discarded", original_length - len(cleaned_periods), "periods")
    return cleaned_periods

def find_all_periods(signal, indices=[]):
    start = None
    all_periods = []
    for i in range(len(signal)-1):
        if start is None:
            if signal[i] and (len(indices)==0 or indices[i] + 1 == indices[i + 1]):
                start = i
        else: # there is a start
            if not signal[i] or (len(indices) > 0 and indices[i] + 1 != indices[i + 1]):
                all_periods.append((start, i))
                # if len(indices) > 0 and indices[i] - indices[start] > 400:
                #     print("start", start, "end", i, "indices", indices[start], indices[i])
                #     print(indices[start:i+1])
                start = None
    if start is not None and (len(indices) == 0 or indices[len(signal) - 1] - 1 == indices[len(signal)-2]):
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
            assert len(data.duration) == len(data.distance)
        else:
            data.duration = (np.roll(data.start_timestamp, -1, axis=0) - data.start_timestamp)[:-1]
            data.distance = utils.angular_distance(
                data.gaze_direction[:-1],
                data.gaze_direction[1:]
            )
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
        # signal = medfilt(signal, kernel_size=self.window_size)
        signal = median_filter(signal, size=self.window_size, axes=0)
        setattr(data, self.attr, signal)

        return data


class ModeFilter(Module):
    def __init__(self, attr: str, window_size: int = 3) -> None:
        self.attr = attr
        self.window_size = window_size

    def update(self, data: GazeData) -> GazeData:
        signal = getattr(data, self.attr)
        # if true false signal, convert to 0, 1 signal
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
        self, minimum_fixation_steps=3, interval_steps_tolerance=5, merge_direction_threshold=0.5
    ) -> None:
        super().__init__()
        self.minimum_fixation_steps = minimum_fixation_steps
        self.interval_steps_tolerance = interval_steps_tolerance
        self.merge_direction_threshold = merge_direction_threshold

    def merge_fixation_periods(self, all_extracted_periods, interval_steps_tolerance, merge_direction_threshold=0.5, data=None):
        original_length = len(all_extracted_periods)
        # print("Original length", original_length)
        i = 0
        while i < len(all_extracted_periods) - 1:
            if all_extracted_periods[i+1][0] - all_extracted_periods[i][1] < interval_steps_tolerance and (len(data.indices)==0 or data.indices[all_extracted_periods[i+1][0]] - data.indices[all_extracted_periods[i][1]] < interval_steps_tolerance):
                last_direction = data.gaze_direction[all_extracted_periods[i][1]]
                next_direction = data.gaze_direction[all_extracted_periods[i+1][0]]
                if not utils.angular_distance([last_direction], [next_direction])[0] > merge_direction_threshold:
                    all_extracted_periods[i] = (all_extracted_periods[i][0], all_extracted_periods[i+1][1])
                    all_extracted_periods.pop(i+1)
                    continue
            i += 1
        # print("Merged", original_length - len(all_extracted_periods), "periods")
        return all_extracted_periods

    def discard_short_periods(self, all_extracted_periods):
        return [
            period
            for period in all_extracted_periods
            if period[1] - period[0] >= self.minimum_fixation_steps
        ]
        

    def update(self, data: GazeData) -> GazeData:
        fixations = []
        start = None
        all_periods = find_all_periods(data.fixation, indices=data.indices if data.sliced else [])

        all_periods = self.merge_fixation_periods(all_periods, self.interval_steps_tolerance, self.merge_direction_threshold, data)
        all_periods = discard_short_periods(all_periods, self.minimum_fixation_steps)

        for start, i in all_periods:
            gaze_targets = list(data.gaze_target[start : i+1])
            # print(f"fixation gaze targets: {gaze_targets}")
            data.fixation[start: i] = True
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
        # print(f"Number of fixations: {len(fixations)}")
        # del data.fixation
        return data


class AggregateSaccades(Module):
    def update(self, data: GazeData) -> GazeData:
        saccades = []
        all_saccades = find_all_periods(data.saccade, indices=data.indices if data.sliced else [])
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
        aggregate_to_fixations=True, minimum_sp_steps=5, interval_steps_tolerance=5
    ) -> None:
        super().__init__()
        self.aggregate_to_fixations = aggregate_to_fixations
        self.minimum_sp_steps = minimum_sp_steps
        self.interval_steps_tolerance = interval_steps_tolerance
        self.away_from_puzzle_targets = ["mascot", "progressbar", "timer"]
    
    def update(self, data: GazeData) -> GazeData:
        smooth_pursuits = []
        all_smooth_pursuits = find_all_periods(data.smooth_pursuit, indices=data.indices if data.sliced else [])
        # all_smooth_pursuits = self.merge_periods_with_target_constraints(all_smooth_pursuits, data)
        all_smooth_pursuits = discard_short_periods(all_smooth_pursuits, self.minimum_sp_steps)

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

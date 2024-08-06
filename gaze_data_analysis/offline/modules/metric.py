import numpy as np
from typing import List

from offline.data import GazeData
from offline.modules import Module


class FixationMetrics(Module):
    def update(self, data: GazeData) -> GazeData:
        fixation_metrics = {
            "count": 0,
            "fixation_time_prop": 0,
            "duration_mean": 0,
            "fixation_rate": 0,
            "time_to_first": 0,
        }

        if len(data.fixations) > 0:
            fixation_metrics["count"] = len(data.fixations)

            durations = np.array([fixation["duration"] for fixation in data.fixations])
            fixation_metrics["fixation_time_prop"] = np.sum(durations) / data.get_total_duration()
            fixation_metrics["duration_mean"] = np.mean(durations)
            fixation_metrics["fixation_rate"] = len(data.fixations) / data.get_total_duration()
            fixation_metrics["time_to_first"] = (
                data.fixations[0]["start_timestamp"] - data.start_timestamp[0]
            )

        data.fixation_metrics = fixation_metrics

        return data
    

class SaccadeMetrics(Module):
    def update(self, data: GazeData) -> GazeData:
        saccade_metrics = {
            "count": 0,
            "duration_total": 0,
            "duration_mean": 0,
            "duration_std": 0,
            "amplitude_total": 0,
            "amplitude_mean": 0,
            "amplitude_std": 0,
            "velocity_mean": 0,
            "velocity_std": 0,
            "peak_velocity_mean": 0,
            "peak_velocity_std": 0,
            "peak_velocity": 0,
        }

        if len(data.saccades) > 0:
            saccade_metrics["count"] = len(data.saccades)

            durations = np.array([saccade["duration"] for saccade in data.saccades])
            saccade_metrics["duration_total"] = np.sum(durations)
            saccade_metrics["duration_mean"] = np.mean(durations)
            saccade_metrics["duration_std"] = np.std(durations)

            amplitudes = np.array([saccade["amplitude"] for saccade in data.saccades])
            saccade_metrics["amplitude_total"] = np.sum(amplitudes)
            saccade_metrics["amplitude_mean"] = np.mean(amplitudes)
            saccade_metrics["amplitude_std"] = np.std(amplitudes)

            velocities = np.array([saccade["velocity"] for saccade in data.saccades])
            saccade_metrics["velocity_mean"] = np.mean(velocities)
            saccade_metrics["velocity_std"] = np.std(velocities)

            peak_velocities = np.array(
                [saccade["peak_velocity"] for saccade in data.saccades]
            )
            saccade_metrics["peak_velocity_mean"] = np.mean(peak_velocities)
            saccade_metrics["peak_velocity_std"] = np.std(peak_velocities)
            saccade_metrics["peak_velocity"] = np.max(peak_velocities)

        data.saccade_metrics = saccade_metrics

        return data


class SmoothPursuitMetrics(Module):
    def update(self, data: GazeData) -> GazeData:
        smooth_pursuit_metrics = {
            "count": 0,
            "duration_total": 0,
            "duration_mean": 0,
            "duration_std": 0,
            "amplitude_total": 0,
            "amplitude_mean": 0,
            "amplitude_std": 0,
            "velocity_mean": 0,
            "velocity_std": 0,
        }

        if hasattr(data, "smooth_pursuits") and len(data.smooth_pursuits) > 0:
            smooth_pursuit_metrics["count"] = len(data.smooth_pursuits)

            durations = np.array(
                [pursuit["duration"] for pursuit in data.smooth_pursuits]
            )
            smooth_pursuit_metrics["duration_total"] = np.sum(durations)
            smooth_pursuit_metrics["duration_mean"] = np.mean(durations)
            smooth_pursuit_metrics["duration_std"] = np.std(durations)

            amplitudes = np.array(
                [pursuit["amplitude"] for pursuit in data.smooth_pursuits]
            )
            smooth_pursuit_metrics["amplitude_total"] = np.sum(amplitudes)
            smooth_pursuit_metrics["amplitude_mean"] = np.mean(amplitudes)
            smooth_pursuit_metrics["amplitude_std"] = np.std(amplitudes)

            velocities = np.array(
                [pursuit["velocity"] for pursuit in data.smooth_pursuits]
            )
            smooth_pursuit_metrics["velocity_mean"] = np.mean(velocities)
            smooth_pursuit_metrics["velocity_std"] = np.std(velocities)

        data.smooth_pursuit_metrics = smooth_pursuit_metrics

        return data

class ROIMetrics(Module):
    def __init__(self, rois: List[str]) -> None:
        self.rois = rois

    def update(self, data: GazeData) -> GazeData:
        roi_metrics = {}

        for roi in self.rois: 
            roi_metrics[f"{roi}_count"] = 0
            roi_metrics[f"{roi}_duration_total"] = 0
            roi_metrics[f"{roi}_duration_mean"] = 0
            roi_metrics[f"{roi}_fixation_rate"] = 0
            roi_metrics[f"{roi}_fixation_prop"] = 0


            fixations = [
                fixation for fixation in data.fixations if fixation["target"] == roi
            ]

            if len(fixations) > 0:
                roi_metrics[f"{roi}_count"] = len(fixations)

                durations = np.array([fixation["duration"] for fixation in fixations])
                roi_metrics[f"{roi}_duration_total"] = np.sum(durations)
                roi_metrics[f"{roi}_duration_mean"] = np.mean(durations)
                roi_metrics[f"{roi}_fixation_rate"] =  np.sum(durations)
                roi_metrics[f"{roi}_fixation_prop"] = np.mean(durations)

        data.roi_metrics = roi_metrics

        return data

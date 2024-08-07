import numpy as np
import pandas as pd
import os

import offline.modules as m
from offline.data import SudokuARGazeData, SudokuVRGazeData, aggregate_data

DATA_DIR = "dataset/AR_samples"
ROIS = ["puzzle", "hints", "timer", "progressbar", "mascot"]
LABELS = [0, 1, 3]
PARTICIPANTS = list(range(1, 20))

def write_data_to_results(data_list, duration_list, results, identifier="base", label_index=-1):
    aggr_data = aggregate_data(data_list, all_rois=ROIS)
    total_duration = sum(duration_list)
    for metrics_name in [
        "fixation_metrics",
        "saccade_metrics",
        "roi_metrics",
    ]:
        metrics = getattr(aggr_data, f"{metrics_name}")
        if metrics_name not in results:
            results[metrics_name] = np.full(
                ((len(metrics) + 1) * (2+len(LABELS)), len(PARTICIPANTS)),
                fill_value=-1,
                dtype=np.float64,
            ) 
        
        col = PARTICIPANTS.index(participant)
        row = 0
        if identifier == "final":
            row = len(metrics) + 1
        if label_index != -1:
            row = (label_index+2) * (len(metrics) + 1)

        for i, val in enumerate(metrics.values()):
            results[metrics_name][row+i, col] = val
        results[metrics_name][row+len(metrics), col] = total_duration


if __name__ == "__main__":

    modules = [
        m.DurationDistanceVelocity(window_size=3), 
        m.SavgolFilter(attr="velocity", window_size=3, order=1),
        m.IVTFixationDetector(velocity_threshold=30),
        m.AggregateFixations(),
        m.SmoothPursuitDetector(low_velocity_threshold=30, high_velocity_threshold=100),
        m.AggregateSmoothPursuits(
            aggregate_to_fixations=True, 
        ),
        m.IVTSaccadeDetector(velocity_threshold=30),
        m.AggregateSaccades(),
        m.FixationMetrics(),
        m.SaccadeMetrics(),
        m.ROIMetrics(rois=ROIS),
    ]

    results = {}
    full_features = {}
    final_data = []
    final_duration = []
    final_data_label = [[] for _ in LABELS]
    final_duration_label = [[] for _ in LABELS]

    for root, dirs, files in os.walk(DATA_DIR):
        count_final = len([file for file in files if "csv" in file]) - 1
        for i, file in enumerate(files):
            participant = int(file.split("_")[0][1:])
            # sanity check
            if participant not in PARTICIPANTS:
                continue

            label_data = SudokuARGazeData(os.path.join(root, file)) if "AR" in DATA_DIR else SudokuVRGazeData(os.path.join(root, file))
            if not label_data.is_data_useable():
                continue
            total_duration = label_data.get_total_duration()
            if "base" in file:
                for module in modules:
                    label_data = module.update(label_data)
                write_data_to_results([label_data], [total_duration], results, identifier="base")
            else:
                for i, label in enumerate(LABELS):
                    period_data =  SudokuARGazeData(os.path.join(root, file), label=label) if "AR" in DATA_DIR else SudokuVRGazeData(os.path.join(root, file), label=label)
                    
                    slice_result = period_data.is_data_useable()

                    if not slice_result:
                        final_data_label[i].append(None)
                        final_duration_label[i].append(0)
                    else:
                        duration = period_data.get_total_duration()
                        for module in modules:
                            period_data = module.update(period_data)
                        final_data_label[i].append(period_data)
                        final_duration_label[i].append(duration)
                    if len(final_data_label[i]) == count_final:
                        write_data_to_results([data for data in final_data_label[i] if data], final_duration_label[i], results, identifier="final", label_index=i)
                        final_data_label[i], final_duration_label[i] = [], []
                # full trial
                label_data =  SudokuARGazeData(os.path.join(root, file)) if "AR" in DATA_DIR else SudokuVRGazeData(os.path.join(root, file))
                if not label_data.is_data_useable():
                    continue
                for module in modules:
                    label_data = module.update(label_data)
                final_data.append(label_data)
                final_duration.append(total_duration)
                if len(final_data) == count_final:
                    write_data_to_results(final_data, final_duration, results, identifier="final")
                    final_data, final_duration = [], []

    for name, result in results.items():
        np.savetxt(f"{'AR' if 'AR' in DATA_DIR else 'VR'}_{name}.csv", result, delimiter=",")

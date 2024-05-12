from datetime import datetime, timezone
import numpy as np
import pandas as pd
from typing import List
from itertools import chain


class GazeData:
    def __init__(self, file_path: str) -> None:
        self.file_path: str = file_path
        self.start_timestamp: float
        self.gaze_direction: np.ndarray
        self.indices = []
        self.duration = 0
        self.sliced = False

        # self.load_data(file_path)

    def load_data(self, file_path: str) -> None:
        raise NotImplementedError("Subclasses of GazeData should implement load_data.")

    def __len__(self):
        return len(self.indices)

    def slice_label(self, label):
        if label == -1:
            return True
        indices = self.label == label
        self.start_timestamp = self.start_timestamp[indices]
        self.gaze_direction = self.gaze_direction[indices]
        self.label = self.label[indices]
        self.gaze_target = self.gaze_target[indices]
        self.indices = np.where(indices)[0]
        self.sliced = True
        if len(self.indices) <= 5:
            return False
        return True
    
    def get_total_duration(self) -> float:
        if len(self.indices) == 0:
            return self.start_timestamp[-1] - self.start_timestamp[0]
        total_duration = 0
        for i, step in enumerate(self.indices):
            if i == 0:
                continue
            # print(step, self.indices[i-1])
            if step - self.indices[i-1] == 1:
                total_duration += self.start_timestamp[i] - self.start_timestamp[i-1]
        return total_duration

class SudokuVRGazeData(GazeData):
    def __init__(self, file_path: str, label=None) -> None:
        super().__init__(file_path)
        self.load_data(file_path, label)
    
    def load_data(self, file_path: str, label=None) -> None: 
        data = pd.read_csv(
            file_path,
            usecols=[
                "RealTime",
                "CombinedGaze",
                "CombinedGazeConfidence",
                "IntersectWithUseful",
                "IntersectWithNormal",
                "IntersectWithPuzzle",
                "IntersectWithMascot",
                "IntersectWithProgressBar",
                "IntersectWithTimer",
                "HintReceived",
                "Mistake",
                "MascotDistraction",
                "AudioDistraction"
            ],
        )
        data = data.dropna(axis=0, how="any")
        data = self.preprocess_data(data)
        self.gaze_direction = data[["CombinedGaze_x", "CombinedGaze_y", "CombinedGaze_z"]].to_numpy()
        self.start_timestamp = data["RealTime"].to_numpy(); 
        self.label = data["Label"].to_numpy()
        self.gaze_target = data["target"].to_numpy()

        
        if len(self.indices) == 0:
            self.indices = np.arange(len(data)) 

    def preprocess_data(self, data):
        # keep only rows with CombinedGazeConfidence == 1
        data = data.loc[data["CombinedGazeConfidence"] == 1, :].copy(deep=True)
        data = self.convert_label_columns(data)
        data = self.compute_combined_gaze(data)
        data = self.convert_target_columns(data)

        data["RealTime"] =  self.preprocess_time_data(data["RealTime"])
        return data
    
    def convert_label_columns(self, df, unique_labels=True):
        # Initialize the Label column with 0
        df["Label"] = 0

        df.loc[df["MascotDistraction"] == 1, "Label"] = 1

        # Rule 2: If Mistake has value 1, and in the following 300 rows, none of FalseMistake has value 1, then Label should have value 2 for the previous rows starting from the row that contains the most recent "1" in the column "HintReceived"
        # Use a rolling window of size 120 to check the condition
        mistake_indices = df[df["Mistake"] == 1].index
        for mistake_index in mistake_indices:
            hint_received_indices = (
                df.loc[:mistake_index, "HintReceived"]
                .loc[df["HintReceived"] == 1]
                .index
            )

            if len(hint_received_indices) > 0:
                label_index = (
                    hint_received_indices[-2]
                    if len(hint_received_indices) > 1
                    else hint_received_indices[-1] - 120  # 2s prior to a mistake
                )
                df.loc[label_index:mistake_index, "Label"] = 2 if unique_labels else 1
        
        audio_distraction_indices = df[df["AudioDistraction"] == 1].index
        for audio_index in audio_distraction_indices:
            df.loc[audio_index : audio_index + 120, "Label"] = 3 if unique_labels else 1

        df.loc[df["AudioDistraction"] == 1, "Label"] = 3 if unique_labels else 1

        # Return the modified dataframe
        df.drop(
            [
                "MascotDistraction",
                "Mistake",
                "HintReceived",
                "AudioDistraction"
            ],
            axis=1,
            inplace=True,
        )
        return df
    
    def split_columns_and_save(self, feature_df, col, split_num=3):
        if split_num == 3:
            feature_df[[f"{col}_x", f"{col}_y", f"{col}_z"]] = (
                feature_df[col].str.strip("()").str.split("|", expand=True)
            )
        elif split_num == 4:
            feature_df[[f"{col}_x", f"{col}_y", f"{col}_z", f"{col}_o"]] = (
                feature_df[col].str.strip("()").str.split("|", expand=True)
            )
        feature_df[f"{col}_x"] = pd.to_numeric(feature_df[f"{col}_x"])
        feature_df[f"{col}_y"] = pd.to_numeric(feature_df[f"{col}_y"])
        feature_df[f"{col}_z"] = pd.to_numeric(feature_df[f"{col}_z"])
        if split_num == 4:
            feature_df[f"{col}_o"] = pd.to_numeric(feature_df[f"{col}_o"])
        feature_df = feature_df.drop(col, axis=1)
        return feature_df

    def compute_combined_gaze(self, df):
        df = self.split_columns_and_save(df, "CombinedGaze", split_num=3)
        
        return df
    
    def convert_target_columns(self, df):
        df["target"] = "other"
        df.loc[df["IntersectWithPuzzle"] == 1, "target"] = "puzzle"
        df.loc[df["IntersectWithUseful"] == 1, "target"] = "hints"
        df.loc[df["IntersectWithMascotDistraction"] == 1, "target"] = "mascot"
        df.loc[df["IntersectWithProgressBar"] == 1, "target"] = "progressbar"
        df.loc[df["IntersectWithTimer"] == 1, "target"] = "timer"
        
        df.drop(
            ["IntersectWithUseful", "IntersectWithNormal", "IntersectWithPuzzle", "IntersectWithMascotDistraction", "IntersectWithProgressBar", "IntersectWithTimer"],
            axis=1,
            inplace=True,
        )
        return df

    def preprocess_time_data(self, time_data):
        parse_time_data = np.vectorize(
            lambda x: datetime.strptime(x, "%H:%M:%S:%f")
            .replace(tzinfo=timezone.utc)
            .timestamp()
        )

        time_data = parse_time_data(time_data)
        time_data = time_data - np.min(time_data)

        return time_data


class SudokuARGazeData(GazeData):
    def __init__(self, file_path: str) -> None:
        super().__init__(file_path)
        self.load_data(file_path)

    def load_data(self, file_path: str) -> None:
        data = pd.read_csv(
            file_path,
            usecols=[
                "RealTime",
                "FixationPoint",
                "LeftEyeCenter",
                "RightEyeCenter",
                "IntersectWithUseful",
                "IntersectWithNormal",
                "IntersectWithPuzzle",
                "IntersectWithMascot",
                "IntersectWithProgress",
                "IntersectWithTimer",
                "MascotDistraction",
                "Mistake",
                "FalseMistake",
                "HintReceived",
                "AudioDistraction",
                "BlockingDistraction",
                "ArtificialDistraction",
            ],
        )
        data = data.dropna(axis=0, how="any")
        
        data = self.preprocess_data(data)
        self.start_timestamp = data["RealTime"].to_numpy()

        self.gaze_direction = data[["gaze_x", "gaze_y", "gaze_z"]].to_numpy()

        self.label = data["Label"].to_numpy()
        self.gaze_target = data["target"].to_numpy()

    def preprocess_data(self, data):
        data = self.convert_label_columns(data)
        data = self.compute_combined_gaze(data)
        data = self.convert_target_columns(data)
        data["RealTime"] = self.preprocess_time_data(data["RealTime"])
        return data

    def split_columns_and_save(self, feature_df, col, split_num=3):
        if split_num == 3:
            feature_df[[f"{col}_x", f"{col}_y", f"{col}_z"]] = (
                feature_df[col].str.strip("()").str.split("|", expand=True)
            )
        elif split_num == 4:
            feature_df[[f"{col}_x", f"{col}_y", f"{col}_z", f"{col}_o"]] = (
                feature_df[col].str.strip("()").str.split("|", expand=True)
            )
        feature_df[f"{col}_x"] = pd.to_numeric(feature_df[f"{col}_x"])
        feature_df[f"{col}_y"] = pd.to_numeric(feature_df[f"{col}_y"])
        feature_df[f"{col}_z"] = pd.to_numeric(feature_df[f"{col}_z"])
        if split_num == 4:
            feature_df[f"{col}_o"] = pd.to_numeric(feature_df[f"{col}_o"])
        feature_df = feature_df.drop(col, axis=1)
        return feature_df

    def compute_combined_gaze(self, df):
        df = self.split_columns_and_save(df, "LeftEyeCenter", split_num=3)
        df = self.split_columns_and_save(df, "RightEyeCenter", split_num=3)
        df = self.split_columns_and_save(df, "FixationPoint", split_num=3)
        df["gaze_x"] = (
            df["FixationPoint_x"] - (df["LeftEyeCenter_x"] + df["RightEyeCenter_x"]) / 2
        )
        df["gaze_y"] = (
            df["FixationPoint_y"] - (df["LeftEyeCenter_y"] + df["RightEyeCenter_y"]) / 2
        )
        df["gaze_z"] = (
            df["FixationPoint_z"] - (df["LeftEyeCenter_z"] + df["RightEyeCenter_z"]) / 2
        )
        for col in ["FixationPoint", "LeftEyeCenter", "RightEyeCenter"]:
            for direction in ["x", "y", "z"]:
                df = df.drop(f"{col}_{direction}", axis=1)
        return df

    def convert_label_columns(self, df, unique_labels=True):
        # Initialize the Label column with 0
        df["Label"] = 0

        df.loc[df["MascotDistraction"] == 1, "Label"] = 1
        # if FalseMistake is 1, in all previous 180 rows FalseMistakeWindow should be 1 
        df["FalseMistakeWindow"] = 0
        false_mistake_indices = df[df["FalseMistake"] == 1].index
        for false_mistake_index in false_mistake_indices:
            df.loc[false_mistake_index - 180 : false_mistake_index, "FalseMistakeWindow"] = 1
        mistake_indices = df[df["Mistake"] == 1].index
        for mistake_index in mistake_indices:
            if df.loc[mistake_index, "FalseMistakeWindow"] == 0:
                hint_received_indices = (
                    df.loc[:mistake_index, "HintReceived"]
                    .loc[df["HintReceived"] == 1]
                    .index
                )

                if len(hint_received_indices) > 0:
                    label_index = (
                        hint_received_indices[-2]
                        if len(hint_received_indices) > 1
                        else hint_received_indices[-1] - 120  # 2s prior to a mistake
                    )
                    df.loc[label_index:mistake_index, "Label"] = 2 if unique_labels else 1
        df.drop("FalseMistakeWindow", axis=1, inplace=True)

        # Rule 3: If AudioDistraction has value 1, then Label should have value 3 in the following 90 rows
        audio_distraction_indices = df[df["AudioDistraction"] == 1].index
        for audio_index in audio_distraction_indices:
            df.loc[audio_index : audio_index + 120, "Label"] = 3 if unique_labels else 1

        df.loc[df["AudioDistraction"] == 1, "Label"] = 3 if unique_labels else 1
        df.drop(
            [
                "MascotDistraction",
                "Mistake",
                "FalseMistake",
                "HintReceived",
                "AudioDistraction",
                "BlockingDistraction",
                "ArtificialDistraction",
            ],
            axis=1,
            inplace=True,
        )
        # length of label 2
        # print(f"Length of label 2: {len(df[df['Label'] == 2])}")
        # print(f"Total length: {len(df)}")
        return df

    def convert_target_columns(self, df):
        df["target"] = "other"
        df.loc[df["IntersectWithPuzzle"] == 1, "target"] = "puzzle"
        df.loc[df["IntersectWithUseful"] == 1, "target"] = "hints"
        df.loc[df["IntersectWithMascot"] == 1, "target"] = "mascot"
        df.loc[df["IntersectWithProgress"] == 1, "target"] = "progressbar"
        df.loc[df["IntersectWithTimer"] == 1, "target"] = "timer"
        
        df.drop(
            ["IntersectWithUseful", "IntersectWithNormal", "IntersectWithPuzzle", "IntersectWithMascot", "IntersectWithProgress", "IntersectWithTimer"],
            axis=1,
            inplace=True,
        )
        return df

    def preprocess_time_data(self, time_data):
        parse_time_data = np.vectorize(
            lambda x: datetime.strptime(x, "%H:%M:%S:%f")
            .replace(tzinfo=timezone.utc)
            .timestamp()
        )

        time_data = parse_time_data(time_data)
        time_data = time_data - np.min(time_data)

        return time_data


def aggregate_data(all_data: List[GazeData], all_rois=[1, 2, 3]) -> GazeData:
    new_data = GazeData(file_path="")
    if hasattr(all_data[0], "fixation_metrics"):
        fixation_metrics = {
            "count": 0,
            "duration_total": 0,
            "duration_mean": 0,
            "duration_std": 0,
            "time_to_first": 0,
        }
        all_fixation_durations = np.concatenate([np.array([fixation["duration"] for fixation in data.fixations]) for data in all_data])
        fixation_metrics["count"] = len(all_fixation_durations)
        fixation_metrics["duration_total"] = np.sum(all_fixation_durations)
        fixation_metrics["duration_mean"] = np.mean(all_fixation_durations)
        fixation_metrics["duration_std"] = np.std(all_fixation_durations)
        fixation_metrics["time_to_first"] = all_data[0].fixation_metrics["time_to_first"]
        new_data.fixation_metrics = fixation_metrics
    if hasattr(all_data[0], "saccade_metrics"):
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
        all_saccades_durations = np.concatenate([np.array([saccade["duration"] for saccade in data.saccades]) for data in all_data])
        saccade_metrics["count"] = len(all_saccades_durations)
        saccade_metrics["duration_total"] = np.sum(all_saccades_durations)
        saccade_metrics["duration_mean"] = np.mean(all_saccades_durations)
        saccade_metrics["duration_std"] = np.std(all_saccades_durations)
        all_saccades_amplitudes = np.concatenate([np.array([saccade["amplitude"] for saccade in data.saccades]) for data in all_data])
        saccade_metrics["amplitude_total"] = np.sum(all_saccades_amplitudes)
        saccade_metrics["amplitude_mean"] = np.mean(all_saccades_amplitudes)
        saccade_metrics["amplitude_std"] = np.std(all_saccades_amplitudes)
        all_saccades_velocities = np.concatenate([np.array([saccade["velocity"] for saccade in data.saccades]) for data in all_data])
        saccade_metrics["velocity_mean"] = np.mean(all_saccades_velocities)
        saccade_metrics["velocity_std"] = np.std(all_saccades_velocities)
        all_saccades_peak_velocities = np.concatenate([np.array([saccade["peak_velocity"] for saccade in data.saccades]) for data in all_data])
        saccade_metrics["peak_velocity_mean"] = np.mean(all_saccades_peak_velocities)
        saccade_metrics["peak_velocity_std"] = np.std(all_saccades_peak_velocities)
        saccade_metrics["peak_velocity"] = np.max(all_saccades_peak_velocities)
        new_data.saccade_metrics = saccade_metrics
    if hasattr(all_data[0], "roi_metrics"):
        roi_metrics = {}
        for roi in all_rois:  #  + ["other"]:
            # roi_metrics[f"{roi}_count_prop"] = 0
            roi_metrics[f"{roi}_count"] = 0
            roi_metrics[f"{roi}_duration_total"] = 0
            roi_metrics[f"{roi}_duration_mean"] = 0
            roi_metrics[f"{roi}_duration_std"] = 0

            fixations = list(chain([[
                fixation for fixation in data.fixations if fixation["target"] == roi
            ] for data in all_data]))[0]

            if len(fixations) > 0:
                roi_metrics[f"{roi}_count"] = len(fixations)

                durations = np.array([fixation["duration"] for fixation in fixations])
                roi_metrics[f"{roi}_duration_total"] = np.sum(durations)
                roi_metrics[f"{roi}_duration_mean"] = np.mean(durations)
                roi_metrics[f"{roi}_duration_std"] = np.std(durations)
        
        for roi_1 in all_rois:
            for roi_2 in all_rois:
                roi_1_to_roi_2 = 0
                for data in all_data:
                    for i in range(len(data.fixations) - 1):
                        if (
                            data.fixations[i]["target"] == roi_1
                            and data.fixations[i + 1]["target"] == roi_2
                        ):
                            roi_1_to_roi_2 += 1

                    roi_metrics[f"{roi_1}_to_{roi_2}_count"] = roi_1_to_roi_2

        new_data.roi_metrics = roi_metrics
    return new_data


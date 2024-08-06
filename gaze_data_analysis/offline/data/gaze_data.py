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

    def load_data(self, file_path: str) -> None:
        raise NotImplementedError("Subclasses of GazeData should implement load_data.")

    def __len__(self):
        return len(self.indices)
    
    def is_data_useable(self):
        return len(self.raw_data) >= 30
    
    def slice_label(self, label):
        if label == -1:
            return
        self.raw_data = self.raw_data.loc[self.raw_data["Label"] == label]

    def split_columns_and_save(self, feature_df, col, split_num=3):
        if split_num == 3:
            try:
                feature_df[[f"{col}_x", f"{col}_y", f"{col}_z"]] = (
                    feature_df[col].str.strip("()").str.split("|", expand=True)
                )
            except:
                print(feature_df)
                exit()
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
    
    def convert_label_columns(self, df, unique_labels=True):
        # Initialize the Label column with 0
        df["Label"] = 0

        df.loc[df["MascotDistraction"] == 1, "Label"] = 1

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

        # Rule 3: If AudioDistraction has value 1, then Label should have value 3 in the following 120 rows
        audio_distraction_indices = df[df["AudioDistraction"] == 1].index
        for audio_index in audio_distraction_indices:
            # find the next indiex with HintReceived being True
            hint_received_indices = (
                df.loc[audio_index:, "HintReceived"]
                .loc[df["HintReceived"] == 1]
                .index
            )
            if len(hint_received_indices) > 0:
                label_index = hint_received_indices[0]
            else:
                label_index = audio_index + 120
            df.loc[audio_index : label_index, "Label"] = 3 if unique_labels else 1

        df.loc[df["AudioDistraction"] == 1, "Label"] = 3 if unique_labels else 1
        df.drop(
            [
                "MascotDistraction",
                "Mistake",
                "HintReceived",
                "AudioDistraction",
            ],
            axis=1,
            inplace=True,
        )
        return df
       
    def get_total_duration(self) -> float:
        total_duration = 0
        for i, step in enumerate(self.indices):
            if i == 0:
                continue
            # might be blinks or breaks due to data split. If split, must be more than 60 frames
            if step - self.indices[i-1] <= 60:
                total_duration += self.start_timestamp[i] - self.start_timestamp[i-1]
        return total_duration
    
    def preprocess_time_data(self, time_data):
        parse_time_data = np.vectorize(
            lambda x: datetime.strptime(x, "%H:%M:%S:%f")
            .replace(tzinfo=timezone.utc)
            .timestamp()
        )

        time_data = parse_time_data(time_data)
        time_data = time_data - np.min(time_data)

        return time_data


class SudokuVRGazeData(GazeData):
    def __init__(self, file_path: str, label=-1) -> None:
        super().__init__(file_path)
        self.load_data(file_path, label)
    
    def load_data(self, file_path: str, label=-1) -> None: 
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
        data['RealTime'] = self.preprocess_time_data(data['RealTime'])
        start_index = data.loc[data["HintReceived"] == 1].index[1]
        end_index = data.loc[data["HintReceived"] == 1].index[-1]
        data = data.loc[start_index:end_index]
        data.reset_index(drop=True, inplace=True)
        data = self.convert_label_columns(data)
        self.raw_data = data
        self.slice_label(label)
        if not self.is_data_useable():
            return
        self.preprocess_data(self.raw_data)
        
        if len(self.indices) == 0:
            self.indices = data.index.to_numpy()

    def preprocess_data(self, data):
        # drop blinks
        data = data.loc[data["CombinedGazeConfidence"] == 1, :].copy(deep=True)
        self.start_timestamp = data["RealTime"].to_numpy(); 
        data = self.compute_combined_gaze(data)
        data = self.convert_target_columns(data)
        self.gaze_direction = data[["CombinedGaze_x", "CombinedGaze_y", "CombinedGaze_z"]].to_numpy()
        self.label = data["Label"].to_numpy()
        self.gaze_target = data["target"].to_numpy()

        return data
    

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

class SudokuARGazeData(GazeData):
    def __init__(self, file_path: str, label=-1) -> None:
        super().__init__(file_path)
        
        self.label = label
        self.load_data(file_path, label=label)

    
    def drop_blinks(self, data):
        self.blink_indices = data.loc[(data["LeftEyeBlinking"] == 1) | (data["RightEyeBlinking"] == 1)].index
        data = data.loc[(data["LeftEyeBlinking"] == 0) & (data["RightEyeBlinking"] == 0)]
        data = data.drop(["LeftEyeBlinking", "RightEyeBlinking"], axis=1)
        return data

    def load_data(self, file_path: str, label=-1) -> None:
        data = pd.read_csv(
            file_path,
            usecols=[
                "RealTime",
                "FixationPoint",
                "LeftEyeCenter",
                "RightEyeCenter",
                "LeftEyeBlinking",
                "RightEyeBlinking",
                "LeftEyeOpenAmount",
                "RightEyeOpenAmount",
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
            ],
        )
        # clean the data, start from the second HintReceived being 1, end at the last HintReceived being 1
        data = data.dropna(axis=0, how="any")
        data['RealTime'] = self.preprocess_time_data(data['RealTime'])

        start_index = data.loc[data["HintReceived"] == 1].index[1]
        end_index = data.loc[data["HintReceived"] == 1].index[-1]
        data = data.loc[start_index:end_index]
        data.reset_index(drop=True, inplace=True)
        data = self.convert_label_columns(data)
        self.raw_data = data
        self.slice_label(label)
        if not self.is_data_useable():
            return
        self.preprocess_data(self.raw_data)

        
    def preprocess_data(self, data):
        data = self.drop_blinks(data)
        data = self.compute_combined_gaze(data)
        data = self.convert_target_columns(data)
        self.start_timestamp = data["RealTime"].to_numpy()

        self.gaze_direction = data[["gaze_x", "gaze_y", "gaze_z"]].to_numpy()

        self.label = data["Label"].to_numpy()
        self.gaze_target = data["target"].to_numpy()

        self.indices = data.index.to_numpy()
        return data

    def convert_label_columns(self, df, unique_labels=True):
        # Initialize the Label column with 0
        df["Label"] = 0

        # Apply the rules in the provided order
        # Rule 1: If DevilDistraction has value 1, then Label should also have value 1
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

        # Rule 3: If AudioDistraction has value 1, then Label should have value 3 in the following 120 rows
        audio_distraction_indices = df[df["AudioDistraction"] == 1].index
        for audio_index in audio_distraction_indices:
            # find the next index with HintReceived being True
            hint_received_indices = (
                df.loc[audio_index:, "HintReceived"]
                .loc[df["HintReceived"] == 1]
                .index
            )
            if len(hint_received_indices) > 0:
                label_index = hint_received_indices[0]
            else:
                label_index = audio_index + 120
            df.loc[audio_index : label_index, "Label"] = 3 if unique_labels else 1

        df.drop(
            [
                "MascotDistraction",
                "Mistake",
                "FalseMistake",
                "HintReceived",
                "AudioDistraction",
            ],
            axis=1,
            inplace=True,
        )
        return df
    
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
    


    def convert_target_columns(self, df):
        df["target"] = "other"
        df.loc[df["IntersectWithPuzzle"] == 1, "target"] = "puzzle"
        df.loc[df["IntersectWithUseful"] == 1, "target"] = "hints"
        df.loc[df["IntersectWithTrajectory"] == 1, "target"] = "trajectory"
        df.loc[df["IntersectWithMascot"] == 1, "target"] = "Mascot"
        df.loc[df["IntersectWithProgress"] == 1, "target"] = "progressbar"
        df.loc[df["IntersectWithTimer"] == 1, "target"] = "timer"
        
        df.drop(
            ["IntersectWithUseful", "IntersectWithNormal", "IntersectWithPuzzle", "IntersectWithMascot", "IntersectWithProgress", "IntersectWithTimer", "IntersectWithTrajectory"],
            axis=1,
            inplace=True,
        )
        return df
    


def aggregate_data(all_data: List[GazeData], all_rois=[1, 2, 3]) -> GazeData:
    if len(all_data) == 0:
        new_data = GazeData(file_path="")
        new_data.fixation_metrics = {
            "count": -1,
            "fixation_time_prop": -1,
            "duration_mean": -1,
            "fixation_rate": -1,
            "time_to_first": -1,
        }
        new_data.saccade_metrics = {
            "count": -1,
            "duration_total": -1,
            "duration_mean": -1,
            "duration_std": -1,
            "amplitude_total": -1,
            "amplitude_mean": -1,
            "amplitude_std": -1,
            "velocity_mean": -1,
            "velocity_std": -1,
            "peak_velocity_mean": -1,
            "peak_velocity_std": -1,
            "peak_velocity": -1,
        }
        new_data.roi_metrics = {}
        for roi in all_rois:
            new_data.roi_metrics[f"{roi}_count"] = -1
            new_data.roi_metrics[f"{roi}_duration_total"] = -1
            new_data.roi_metrics[f"{roi}_duration_mean"] = -1
            new_data.roi_metrics[f"{roi}_fixation_rate"] = -1
            new_data.roi_metrics[f"{roi}_fixation_prop"] = -1
            
        return new_data
    
    if len(all_data) == 1:
        return all_data[0]
    new_data = GazeData(file_path="")
    if hasattr(all_data[0], "fixation_metrics"):
        fixation_metrics = {
            "count": 0,
            "fixation_time_prop": 0,
            "duration_mean": 0,
            "fixation_rate": 0,
            "time_to_first": 0,
        }
        all_fixation_durations = np.concatenate([np.array([fixation["duration"] for fixation in data.fixations]) for data in all_data])

        all_total_durations = np.array([data.get_total_duration() for data in all_data])

        fixation_metrics["count"] = len(all_fixation_durations)
        fixation_metrics["fixation_time_prop"] = np.sum(all_fixation_durations) / np.sum(all_total_durations)
        
        fixation_metrics["duration_mean"] = np.mean(all_fixation_durations)
        fixation_metrics["fixation_rate"] = len(all_fixation_durations) / np.sum(all_total_durations)
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
        all_total_durations = np.array([data.get_total_duration() for data in all_data])
        roi_metrics = {}
        for roi in all_rois:
            roi_metrics[f"{roi}_count"] = 0
            roi_metrics[f"{roi}_duration_total"] = 0
            roi_metrics[f"{roi}_duration_mean"] = 0
            roi_metrics[f"{roi}_fixation_rate"] = 0
            roi_metrics[f"{roi}_fixation_prop"] = 0

            fixations = list(chain([[
                fixation for fixation in data.fixations if fixation["target"] == roi
            ] for data in all_data]))[0]

            if len(fixations) > 0:
                roi_metrics[f"{roi}_count"] = len(fixations)

                durations = np.array([fixation["duration"] for fixation in fixations])
                roi_metrics[f"{roi}_duration_total"] = np.sum(durations)
                roi_metrics[f"{roi}_duration_mean"] = np.mean(durations)
                roi_metrics[f"{roi}_fixation_rate"] = len(durations) / np.sum(durations)
                roi_metrics[f"{roi}_fixation_prop"] = np.sum(durations) / np.sum(durations)
        new_data.roi_metrics = roi_metrics
    return new_data
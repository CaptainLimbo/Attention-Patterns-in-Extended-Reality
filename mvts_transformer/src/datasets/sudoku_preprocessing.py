import shutil
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
import os
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler

TIME_FORMAT = "%H:%M:%S:%f"

warnings.filterwarnings("ignore")

AR_ACS = [10] * 19
VR_ACS = [10] * 19

AR_PARTICIPANTS = list(range(1, 21))
VR_PARTICIPANTS = list(range(1, 21))    

def grouping_ACS(score):
    # <10: 0; 10-50: 1; >50: 2
    if score <= 27:
        return 0
    elif score <= 41:
        return 1
    else:
        return 2

AR_USED_COLUMNS = [
    "LeftEyeOpenAmount",
    "RightEyeOpenAmount",
    "LeftEyeCenter",
    "RightEyeCenter",
    "LeftEyeGazeDirectionPuzzle",
    "RightEyeGazeDirectionPuzzle",
    "IntersectWithUseful",
    "IntersectWithNormal",
    "IntersectWithPuzzle",
    "MascotDistraction",
]

VR_USED_COLUMNS = [
    "LeftDialation",
    "RightDialation",
    "LeftGaze",
    "RightGaze",
    "IntersectWithUseful",
    "IntersectWithPuzzle",
    "IntersectWithMascotDistraction",
    "IntersectWithTimer",
    "IntersectWithProgressBar",
    "MascotDistraction",
]



def merge_VR_intersections(df):
    df["IntersectWithNormal"] = df["IntersectWithMascotDistraction"] | df["IntersectWithTimer"] | df["IntersectWithProgressBar"]
    df.drop(["IntersectWithMascotDistraction", "IntersectWithTimer", "IntersectWithProgressBar"], axis=1, inplace=True)
    
    df = df[["LeftGaze", "RightGaze", "IntersectWithUseful", "IntersectWithNormal","IntersectWithPuzzle", "MascotDistraction"]] if "LeftDialation" not in df.columns else df[["LeftDialation", "RightDialation", "LeftGaze", "RightGaze", "IntersectWithUseful", "IntersectWithNormal","IntersectWithPuzzle", "MascotDistraction"]]
    return df


def convert_label_column(df):
    df["Label"] = 0
    df.loc[df["MascotDistraction"] == 1, "Label"] = 1

    df.drop(
        [
            "MascotDistraction",
        ],
        axis=1,
        inplace=True,
    )
    return df


def split_columns_and_save(df, col, split_num=3):
    if split_num == 3:
        df[[f"{col}_x", f"{col}_y", f"{col}_z"]] = (
            df[col].str.strip("()").str.split("|", expand=True)
        )
    elif split_num == 4:
        df[[f"{col}_x", f"{col}_y", f"{col}_z", f"{col}_o"]] = (
            df[col].str.strip("()").str.split("|", expand=True)
        )
    df[f"{col}_x"] = pd.to_numeric(df[f"{col}_x"])
    df[f"{col}_y"] = pd.to_numeric(df[f"{col}_y"])
    df[f"{col}_z"] = pd.to_numeric(df[f"{col}_z"])
    if split_num == 4:
        df[f"{col}_o"] = pd.to_numeric(df[f"{col}_o"])
    df = df.drop(col, axis=1)
    return df



def clean_dataframe(df, use_center=True, use_amount=False, use_target=False):
    flag = 'ar'
    if "LeftDialation" in df.columns:
        flag = 'vr'
    if not use_center:
        if flag == 'ar':
            df.drop(["LeftEyeCenter", "RightEyeCenter"], axis=1, inplace=True)
    if not use_amount:
        if flag == 'ar':
            df.drop(["LeftEyeOpenAmount", "RightEyeOpenAmount"], axis=1, inplace=True)
        else:
            df.drop(["LeftDialation", "RightDialation"], axis=1, inplace=True)
    if not use_target:
        if flag == 'ar':
            df.drop(["IntersectWithUseful", "IntersectWithNormal", "IntersectWithPuzzle"], axis=1, inplace=True)
        else:
            df.drop(["IntersectWithUseful", "IntersectWithPuzzle", "IntersectWithMascotDistraction", "IntersectWithTimer", "IntersectWithProgressBar"], axis=1, inplace=True)
    return df

def raw_eye_tracking_to_time_series(
    input_dataframe,
    start_id=0,
    window_size=120,
    step_size=120,
    split="time",
    ts_label=None,
    use_center=True,
    use_amount=False,
    use_target=False
):
    """
    conversion of the raw eye tracking data to multivariate time series.
    Input csv is of the the following format:
        each row is a single timestamp with columns for all variates, including a label indicating whether a mistake was made
    We want to convert it to two pandas dataframe, one for features `feature_df` and another for labels `label_df`; and also obtain a mapping `id_mapping` from a index to the entries in the dataframe for extracting the features and labels. `feature_df[id_mapping[i]]` should be the 1-th time-series sample, which should be of the shape (seq_length, feat_dim), while `label_df[id_mapping[i]]` should be the corresponding label, which should be of the shape (num_labels, ) which should be (2, ) in this case where my label is binary.
    Each data sample should have a seq length of 60, meaning that it should span 60 timestamps (rows). We would like to use a sliding window with a step size of 10 rows to go over the input csv to extract all time-series. If in a window of 60 timestamps, a value "1" in the mistake column occurs, that sample would have a corresponding label of "1" in `label_df`, otherwise "0". Write me the python code using pandas and any other packages necessary to implement this method.
    """
    # Load the input CSV file
    df = input_dataframe.copy()
    df = clean_dataframe(df, use_center=use_center, use_amount=use_amount, use_target=use_target)
    columns_to_split = ["LeftEyeCenter", 
        "RightEyeCenter"] if "LeftEyeCenter" in df.columns else []
    columns_to_split += [
        "LeftEyeGazeDirectionPuzzle",
        "RightEyeGazeDirectionPuzzle",
    ] if "FalseMistake" in df.columns else ["LeftGaze", "RightGaze"]

    if "FalseMistake" not in df.columns and "IntersectWithPuzzle" in df.columns: # meaning it's VR
        df = merge_VR_intersections(df)
    
    for col in columns_to_split:
        df = split_columns_and_save(df, col, split_num=3)

    feature_df = convert_label_column(df)


    # fill the nan with 0
    feature_df.fillna(0, inplace=True)
    num_samples = start_id
    if split == "time":
        train_dfs, val_dfs, test_dfs = [], [], []
        val_start, val_end, test_start, test_end = get_train_val_test_by_time(
            len(feature_df),
            val_ratio=0.1,
            test_ratio=0.1,
            minimum_time=120,
            no_test=True,
            no_val=True,
        )
        if val_start == None:  # only train
            train_dfs.append(feature_df)
        elif test_start == None:
            val_dfs.append(feature_df.iloc[val_start:val_end])
            train_dfs.extend(
                [
                    feature_df.iloc[:val_start],
                    feature_df.iloc[val_end:],
                ]
            )
        else:
            val_dfs.append(feature_df.iloc[val_start:val_end])
            test_dfs.append(feature_df.iloc[test_start:test_end])
            train_dfs.extend(
                [
                    feature_df.iloc[: min(val_start, test_start)],
                    feature_df.iloc[
                        min(val_end, test_end) : max(val_start, test_start)
                    ],
                    feature_df.iloc[max(val_end, test_end) :],
                ]
            )
        train_final_df, train_label_df, num_samples, _ = extract_from_dfs(
            source_dfs=train_dfs,
            window_size=window_size,
            step_size=step_size,
            num_samples=num_samples,
            ts_label=ts_label,
        )
        if val_dfs == []:
            return (
                (train_final_df, train_label_df),
                (None, None),
                (None, None),
                num_samples,
            )
        val_final_df, val_label_df, num_samples, _ = extract_from_dfs(
            source_dfs=val_dfs,
            window_size=window_size,
            step_size=step_size,
            num_samples=num_samples,
            ts_label=ts_label,
        )
        if test_dfs == []:
            return (
                (train_final_df, train_label_df),
                (val_final_df, val_label_df),
                (None, None),
                num_samples,
            )
        test_final_df, test_label_df, num_samples, _ = extract_from_dfs(
            source_dfs=test_dfs,
            window_size=window_size,
            step_size=step_size,
            num_samples=num_samples,
            ts_label=ts_label,
        )

        return (
            (train_final_df, train_label_df),
            (val_final_df, val_label_df),
            (
                test_final_df,
                test_label_df,
            ),
            num_samples,
        )


def extract_from_dfs(
    source_dfs, window_size, step_size, num_samples, ts_label=None,
):
    new_df = pd.DataFrame()
    labels = []
    target_feature_indices = []
    feature_indices = []
    inner_start = num_samples
    start = num_samples
    if ts_label is not None:
        for source_df in source_dfs:
            index = 0
            feature_df = source_df.drop("Label", axis=1)
            while index < len(source_df) - window_size + 1:
                # skip not full window and window that span two phases
                if index + window_size > len(source_df):
                    break
                ind = np.arange(index, index + window_size)
                labels.append(ts_label)
                new_df = pd.concat([new_df, feature_df.iloc[ind]], ignore_index=True)
                target_feature_indices += [num_samples] * window_size
                num_samples += 1
                index += step_size

        label_num = 3
        # append to the labels so that the one hot encoding can be done properly, even if not all labels are present in one user's data
        label_df = pd.get_dummies(labels + list(range(label_num))).astype("float32")

        # drop the added ones
        label_df = label_df.iloc[: len(labels)]

        # convert to one hot
        # label_df = pd.get_dummies(label_df[0])
        new_df["ts_index"] = target_feature_indices
        new_df.set_index("ts_index", inplace=True)
        label_df["ts_index"] = list(range(start, num_samples))
        label_df.set_index("ts_index", inplace=True)
        return new_df, label_df, num_samples, labels

    
    for feature_df in source_dfs:
        # feature_df.reset_index(drop=True, inplace=True)
        if len(feature_df) < window_size:
            continue
        start_index = 0
        clean_df = feature_df.drop("Label", axis=1)
        # label can be 0, 1, or 2. if shorter than window size, end at the last one, start new 
        
        while start_index < len(feature_df):
            # df index might not be consecutive. Find the last index of the window if it's consecutive
            end_index = start_index + 60
            label = feature_df["Label"].iloc[start_index]
            while end_index < start_index + window_size and end_index + 1 < len(feature_df) and feature_df["Label"].iloc[end_index] == label:
                end_index += 1
                if feature_df.index[end_index] != feature_df.index[end_index - 1] + 1:
                    break
            if end_index <= start_index + 60: # discard windows that are <0.5s
                start_index += 1
                continue
            labels.append(label)
            new_df = pd.concat([new_df, clean_df.iloc[start_index:end_index]], ignore_index=True)
            feature_indices += [num_samples] * (end_index - start_index)
            num_samples += 1
            start_index = end_index
    label_df = pd.get_dummies(labels + list(range(2))).astype("float32")
    
    # drop the added ones
    label_df = label_df.iloc[: len(labels)]


    new_df["ts_index"] = feature_indices
    new_df.set_index("ts_index", inplace=True)
    label_df["ts_index"] = list(range(inner_start, num_samples))
    label_df.set_index("ts_index", inplace=True)
    
    
    return new_df, label_df, num_samples, labels


def convert_raw_data(
    read_root_path,
    save_root_path,
    normalize=True,
    split_strategy="time",
    window_size=120,
    step_size=120,
    label=None,
    use_center=True, use_amount=False, use_target=False
):
    if split_strategy == "time":
        dataframes_by_owner = {"train": {}, "val": {}, "test": {}}
    num_samples = 0
    open_amount_distribution = {}
    participants = AR_PARTICIPANTS if "AR" in read_root_path else VR_PARTICIPANTS
    flag = "AR" if "AR" in read_root_path else "VR"
    left_normalizer_name = "LeftEyeOpenAmount"
    right_normalizer_name = "RightEyeOpenAmount"
    for root, dirs, files in os.walk(read_root_path):
        for file in files:
            # first round, only to get the distribution of open amount 
            if normalize and file.endswith("base.csv"):
                owner = int(root.split(os.path.sep)[-1][1:])
                if owner not  in participants:
                    continue
                reference_df = pd.read_csv(os.path.join(root, file))
                if flag == "AR":
                    scale_left, scale_right = MinMaxScaler(), MinMaxScaler()
                    scale_left.fit(reference_df[left_normalizer_name].values.reshape(-1, 1))
                    scale_right.fit(
                        reference_df[right_normalizer_name].values.reshape(-1, 1)
                    )
                    open_amount_distribution[owner] = (
                        scale_left,
                        scale_right,
                    )
        for file in files:
            if file.endswith('.csv') and 'final' in file:
                owner = int(root.split(os.path.sep)[-1][1:])
                if owner not in participants:
                    continue
                use_columns = AR_USED_COLUMNS if "AR" in read_root_path else VR_USED_COLUMNS
                ts_label=None if label is None else label[owner-1]
                dataframe = pd.read_csv(os.path.join(root, file), usecols=use_columns)
                # normalize only those that are not -1
                
                if normalize and flag == "AR":
                                    
                    dataframe[left_normalizer_name] = open_amount_distribution[owner][0].transform(dataframe[left_normalizer_name].values.reshape(-1, 1)).round(3)
                    dataframe[right_normalizer_name] = open_amount_distribution[owner][1].transform(dataframe[right_normalizer_name].values.reshape(-1, 1)).round(3)
                
                if split_strategy == "time":
                    (
                        (train_x, train_y),
                        (val_x, val_y),
                        (test_x, test_y),
                        num_samples,
                    ) = raw_eye_tracking_to_time_series(
                        dataframe,
                        start_id=num_samples,
                        window_size=window_size,
                        step_size=step_size,
                        split=split_strategy,
                        ts_label=ts_label,
                        use_center=use_center,
                        use_amount=use_amount,
                        use_target=use_target
                    )
                    for split, x, y in zip(
                        ["train", "val", "test"],
                        [train_x, val_x, test_x],
                        [train_y, val_y, test_y],
                    ):
                    
                        if x is None:
                            continue
                        if owner not in dataframes_by_owner[split]:
                            dataframes_by_owner[split][owner] = [x, y]
                        else:
                            dataframes_by_owner[split][owner][0] = pd.concat(
                                [dataframes_by_owner[split][owner][0], x]
                            )
                            dataframes_by_owner[split][owner][1] = pd.concat(
                                [dataframes_by_owner[split][owner][1], y]
                            )

    if os.path.exists(save_root_path):
        shutil.rmtree(save_root_path)
    for ssplit in ["train", "val", "test"]:
        if not os.path.exists(os.path.join(save_root_path, f"{ssplit}")):
            os.makedirs(os.path.join(save_root_path, f"{ssplit}"))
    if split_strategy == "time":
        for split in ["train", "val", "test"]:
            for owner in dataframes_by_owner[split]:
                x, y = dataframes_by_owner[split][owner]
                if not os.path.exists(os.path.join(save_root_path, f"{split}/{owner}")):
                    os.makedirs(os.path.join(save_root_path, f"{split}/{owner}"))
                if x is not None:
                    x.to_csv(
                        os.path.join(save_root_path, f"{split}/{owner}/feature_df.csv")
                    )
                    y.to_csv(
                        os.path.join(save_root_path, f"{split}/{owner}/label_df.csv")
                    )

def normalize_column(array, normalizer, outliers=-1.0):
    # only normalize the non-outliers
    backup = array.copy()
    normalized = normalizer.transform(array).round(3)
    if outliers == [-1.0]:
        cleaned = np.where(np.isin(backup, outliers), 0, normalized)
    else:
        cleaned = np.where(np.isin(backup, outliers), backup, normalized)
    return cleaned
    

def get_train_val_test_by_time(
    total_time, val_ratio=0.1, test_ratio=0.1, minimum_time=60, no_test=False, no_val=False
):
    """
    find two non-overlapping intervals of length val_ratio and test_ratio in the total time. If < minimum_time, find length = minimum_time
    """
    if no_test:
        if total_time < minimum_time * 2 or no_val:
            return None, None, None, None
        val_length = max(int(total_time * val_ratio), minimum_time)
        val_start = np.random.randint(0, total_time - val_length)
        return val_start, val_start + val_length, None, None

    if total_time < minimum_time * 3:
        return None, None, None, None

    val_length = max(int(total_time * val_ratio), minimum_time)
    test_length = max(int(total_time * test_ratio), minimum_time)

    val_start = np.random.randint(0, total_time - val_length - test_length)
    test_start = np.random.randint(val_start + val_length, total_time - test_length + 1)

    if np.random.rand() > 0.5:
        val_start, test_start = test_start, val_start
    return val_start, val_start + val_length, test_start, test_start + test_length


if __name__ == "__main__":
    np.random.seed(42)
    convert_raw_data("datasets/AR_samples", "datasets/Sudoku_Split_Time/AR_direction_center_target/Mascot", label=None, use_center=True, use_amount=False, use_target=True)
    convert_raw_data("datasets/AR_samples", "datasets/Sudoku_Split_Time/AR_direction_amount_target/ACS", label=list(map(grouping_ACS, AR_ACS)), use_center=False, use_amount=True, use_target=True)

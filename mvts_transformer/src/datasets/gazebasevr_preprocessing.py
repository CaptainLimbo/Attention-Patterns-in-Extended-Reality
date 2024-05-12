from collections import defaultdict
import shutil
import json
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import pickle
import warnings
import os
from tqdm import tqdm

# to ignore warnings
warnings.filterwarnings("ignore")


def spherical_to_cartesian(horizontal_column, vertical_column):
    # first convert to radians
    horizontal = np.radians(horizontal_column)
    vertical = np.radians(vertical_column)
    x = np.sin(horizontal)
    z = np.cos(vertical) * (1 - x**2)
    y = np.tan(vertical) * z
    return round(x, 3), round(y, 3), round(z, 3)


def preprocess_df(
    df, subsampling=True, openamount_placeholder=1, intersect_placeholder=0, use_blink=True, use_center=False, use_openamount=False, use_targets=False
):
    # remove columns xT, yT, zT
    df = df.drop(["n", "x", "y", "xT", "yT", "zT"], axis=1)
    if subsampling:
        df = df.iloc[::4]
        # reset index
        df = df.reset_index(drop=True)

    # add a blink column for both left and right eyes.  set to 1 if the values are all nans
    if use_blink:
        df["blink_left"] = (
            df[["lx", "ly", "clx", "cly", "clz"]].isna().all(axis=1).astype(int)
        )
        df["blink_right"] = (
            df[["rx", "ry", "crx", "cry", "crz"]].isna().all(axis=1).astype(int)
        )

    # convert spherical to cartesian for both left and right eyes
    df["lx"], df["ly"], df["lz"] = spherical_to_cartesian(df["lx"], df["ly"])
    df["rx"], df["ry"], df["rz"] = spherical_to_cartesian(df["rx"], df["ry"])
    # add openamount for both left and right eyes and set all values to 1
    df["openamount_left"] = openamount_placeholder
    df["openamount_right"] = openamount_placeholder
    # add three columns for intersection with useful, relevant, and puzzle, and set all to 0
    df["intersectuseful"] = intersect_placeholder
    df["intersectrelevant"] = intersect_placeholder
    df["intersectpuzzle"] = intersect_placeholder

    used_columns = []
    if use_openamount:
    # reorder columns
        used_columns += ["openamount_left", "openamount_right"]
    if use_blink:
        used_columns += ["blink_left", "blink_right"]
    if use_targets:
        used_columns += ["intersectuseful", "intersectrelevant", "intersectpuzzle"]
    if use_center:
        used_columns += ["clx", "cly", "clz", "crx", "cry", "crz"]
    used_columns += ["lx", "ly", "lz", "rx", "ry", "rz"]



    df = df[used_columns]
    # fill nans with 0
    df = df.fillna(0)
    return df


def raw_eye_tracking_to_time_series(
    df,
    start_id=0,
    window_size=120,
    step_size=60,
    openamount_placeholder=1,
    use_blink=True,
    use_center=True,
    use_openamount=False,
    use_targets=False
):
    df = preprocess_df(df, openamount_placeholder=openamount_placeholder, use_blink=use_blink, use_center=use_center, use_openamount=use_openamount, use_targets=use_targets)
    num_samples = start_id
    train_dfs, val_dfs = [], []
    val_start, val_end, _, _ = get_train_val_test_by_time(
        len(df), no_test=True, val_ratio=0.1
    )
    if val_start == None:  # only train
        train_dfs.append(df)
    else:
        val_dfs.append(df.iloc[val_start:val_end])
        train_dfs.extend(
            [
                df.iloc[:val_start],
                df.iloc[val_end:],
            ]
        )

    train_final_df, train_label_df, num_samples = extract_from_dfs(
        source_dfs=train_dfs,
        window_size=window_size,
        step_size=step_size,
        num_samples=num_samples,
    )
    if val_dfs == []:
        return (
            (train_final_df, train_label_df),
            (None, None),
            (None, None),
            num_samples,
        )
    val_final_df, val_label_df, num_samples = extract_from_dfs(
        source_dfs=val_dfs,
        window_size=window_size,
        step_size=step_size,
        num_samples=num_samples,
    )
    return (
        (train_final_df, train_label_df),
        (val_final_df, val_label_df),
        num_samples,
    )


def extract_from_dfs(
    source_dfs,
    window_size,
    step_size,
    num_samples,
):
    target_df = pd.DataFrame()
    target_labels = []
    target_feature_indices = []
    start = num_samples
    # need to deal with label differently. Distance/ADHD/Trial comes from outer info, phase comes from raw data
    for source_df in source_dfs:
        index = 0
        feature_df = source_df
        while index < len(source_df) - window_size + 1:
            # skip not full window and window that span two phases
            if index + window_size > len(source_df):
                break
            ind = np.arange(index, index + window_size)
            target_labels.append(0)
            target_df = pd.concat([target_df, feature_df.iloc[ind]], ignore_index=True)
            target_feature_indices += [num_samples] * window_size
            num_samples += 1
            index += step_size

    label_df = pd.get_dummies(target_labels).astype("float32")

    # convert to one hot
    # label_df = pd.get_dummies(label_df[0])
    target_df["ts_index"] = target_feature_indices
    target_df.set_index("ts_index", inplace=True)
    label_df["ts_index"] = list(range(start, num_samples))
    label_df.set_index("ts_index", inplace=True)
    return target_df, label_df, num_samples


def convert_raw_data(
    read_root_path,
    save_root_path,
    use_task=["VRG"],
    window_size=120,
    step_size=60,
    openamount_placeholder=1,
    use_blink=True,
    use_center=False,
    use_openamount=False,
    use_targets=False
):
    np.random.seed(42)
    num_samples = 0
    dataframes_by_owner = {"train": {}, "val": {}}
    for root, dirs, files in os.walk(read_root_path):
        for file in files:
            if file.endswith(".csv"):
                if not any([task in file for task in use_task]):
                    continue
                owner = file[2:6]
                (train_x, train_y), (val_x, val_y), num_samples = (
                    raw_eye_tracking_to_time_series(
                        pd.read_csv(os.path.join(root, file)),
                        start_id=num_samples,
                        window_size=window_size,
                        step_size=step_size,
                        openamount_placeholder=openamount_placeholder,
                        use_blink=use_blink,
                        use_center=use_center,
                        use_openamount=use_openamount,
                        use_targets=use_targets
                    )
                )
                for split, x, y in zip(
                    ["train", "val"], [train_x, val_x], [train_y, val_y]
                ):
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
    for split in ["train", "val"]:
        for owner, (x, y) in dataframes_by_owner[split].items():
            if not os.path.exists(os.path.join(save_root_path, f"{split}/{owner}")):
                os.makedirs(os.path.join(save_root_path, f"{split}/{owner}"))
            x.to_csv(os.path.join(save_root_path, f"{split}/{owner}/feature_df.csv"))
            y.to_csv(os.path.join(save_root_path, f"{split}/{owner}/label_df.csv"))

def get_train_val_test_by_time(
    total_time, val_ratio=0.1, test_ratio=0.1, minimum_time=90, no_test=True
):
    """
    find two non-overlapping intervals of length val_ratio and test_ratio in the total time. If < minimum_time, find length = minimum_time
    """
    if no_test:
        if total_time < minimum_time * 2:
            return None, None, None, None
        val_length = max(int(total_time * val_ratio), minimum_time)
        val_start = np.random.randint(0, total_time - val_length)
        return val_start, val_start + val_length, None, None

    if total_time < minimum_time * 3:
        return None, None, None, None
    # find the length of val and test
    val_length = max(int(total_time * val_ratio), minimum_time)
    test_length = max(int(total_time * test_ratio), minimum_time)
    # find the start of val and test
    val_start = np.random.randint(0, total_time - val_length - test_length)
    test_start = np.random.randint(val_start + val_length, total_time - test_length + 1)
    # randomly swap val and test
    if np.random.rand() > 0.5:
        val_start, test_start = test_start, val_start
    return val_start, val_start + val_length, test_start, test_start + test_length


if __name__ == "__main__":
    convert_raw_data(
        "datasets/pretrain/gazebasevr/data",
        "datasets/pretrain/cleaned_direction_amount",
        use_task=["VID", "VRG", "PUR", "TEX", "RAN"],
        openamount_placeholder=1,
        use_blink=False,
        use_center=False,
        use_openamount=True,
        use_targets=False,
    )

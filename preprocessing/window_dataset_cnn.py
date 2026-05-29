"""
Precompute per-sensor windowed datasets for 1D-CNN training.

Reads the full CSV, extracts 4s windows (400 timesteps, 2s stride),
saves 7 .npz files (one per sensor node) with run-stratified splits.

Usage:
    python preprocessing/window_dataset_cnn.py
"""

import os, sys
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from digital_twin.utils import ensure_dir

INPUT_CSV = "dataset_20260527_193221/full_bridge_response.csv"
OUTPUT_DIR = "data/processed/cnn_windows"
WINDOW_SIZE = 400  # 4 seconds at 100 Hz
STRIDE = 200       # 2 seconds
SEED = 42

CHANNEL_NAMES = ["acceleration", "displacement", "strain_proxy"]


def extract_windows_for_sensor(sub, window_size, stride):
    """Return (windows, labels, run_ids, damage_cases) for one sensor."""
    sub = sub.sort_values(["run_id", "damage_case", "time"]).reset_index(drop=True)

    windows = []
    labels = []
    run_ids = []
    damage_cases = []

    # Group by individual run (run_id × damage_case)
    for (rid, dc), grp in sub.groupby(["run_id", "damage_case"]):
        grp = grp.sort_values("time")
        n = len(grp)
        if n < window_size:
            continue
        num_w = (n - window_size) // stride + 1

        for w in range(num_w):
            start = w * stride
            end = start + window_size
            chunk = np.column_stack([
                grp[ch].values[start:end] for ch in CHANNEL_NAMES
            ])
            windows.append(chunk)
            labels.append(grp["label"].iloc[0])
            run_ids.append(rid)
            damage_cases.append(dc)

    return np.array(windows), np.array(labels, dtype=int)


def main():
    print(f"Loading: {INPUT_CSV}")
    df = pd.read_csv(INPUT_CSV)
    print(f"  Rows: {len(df)}, Columns: {list(df.columns)}")
    print(f"  Sensor nodes: {sorted(df['sensor_node'].unique())}")

    ensure_dir(OUTPUT_DIR)

    sensor_nodes = sorted(df["sensor_node"].unique())

    # Build per-run keys for stratified split
    run_keys = df[["run_id", "damage_case"]].drop_duplicates().reset_index(drop=True)
    run_keys["label"] = run_keys["damage_case"].map({
        "healthy": 0, "reduced_girder_stiffness": 1,
        "bearing_failure": 2, "deck_cracking": 3,
    })

    # Stratified split: 70/15/15
    train_keys, temp_keys = train_test_split(
        run_keys, test_size=0.30, random_state=SEED,
        stratify=run_keys["label"],
    )
    val_keys, test_keys = train_test_split(
        temp_keys, test_size=0.50, random_state=SEED,
        stratify=temp_keys["label"],
    )

    split_map = {}
    for split_name, key_df in [("train", train_keys), ("val", val_keys), ("test", test_keys)]:
        for _, row in key_df.iterrows():
            split_map[(row["run_id"], row["damage_case"])] = split_name

    print(f"  Splits: train={len(train_keys)} runs, val={len(val_keys)}, test={len(test_keys)}")

    for sn in sensor_nodes:
        sub = df[df["sensor_node"] == sn].copy()
        windows, labels = extract_windows_for_sensor(sub, WINDOW_SIZE, STRIDE)

        # Build split mask
        run_key_pairs = list(zip(
            sub.groupby(["run_id", "damage_case"]).groups.keys()
        ))
        # Actually we need per-window split assignment
        w_split = []
        temp = sub.groupby(["run_id", "damage_case"])
        w_idx = 0
        for (rid, dc), grp in temp:
            nw = (len(grp) - WINDOW_SIZE) // STRIDE + 1
            if nw < 1:
                continue
            s = split_map.get((rid, dc), "train")
            w_split.extend([s] * nw)

        w_split = np.array(w_split)
        train_mask = w_split == "train"
        val_mask = w_split == "val"
        test_mask = w_split == "test"

        # Per-channel normalization on training set
        means = windows[train_mask].mean(axis=(0, 1))
        stds = windows[train_mask].std(axis=(0, 1))
        stds[stds < 1e-12] = 1.0

        windows_norm = (windows - means) / stds

        out_path = os.path.join(OUTPUT_DIR, f"sensor_{sn}.npz")
        np.savez_compressed(
            out_path,
            data=windows_norm,
            labels=labels,
            train_idx=np.where(train_mask)[0],
            val_idx=np.where(val_mask)[0],
            test_idx=np.where(test_mask)[0],
            means=means,
            stds=stds,
        )

        n_train = train_mask.sum()
        n_val = val_mask.sum()
        n_test = test_mask.sum()
        print(f"  Sensor {sn}: {len(windows)} windows "
              f"(train={n_train}, val={n_val}, test={n_test}), "
              f"shape={windows_norm.shape}")

    print(f"\nSaved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()

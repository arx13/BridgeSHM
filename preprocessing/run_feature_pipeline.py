"""
Extract windowed features from the full 3D dataset CSV.

Usage:
    python preprocessing/run_feature_pipeline.py \
        --input dataset_20260527_193221/full_bridge_response.csv \
        --output data/processed/3d_windowed_features.csv \
        --window 2.0 --stride 1.0

Output matches the schema expected by models/classifier_data_loader.py
"""

import argparse
import os
import sys
import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from preprocessing.features import (
    extract_features_from_window, add_cross_sensor_features
)
from digital_twin.utils import ensure_dir


def window_run_sensors(df_run, window_size, stride, dt=0.01):
    """
    Given a DataFrame for one run (7 sensors × N time steps),
    yield dicts for each window × sensor that match
    `extract_features_from_window` expectations.

    Columns: time, sensor_node, acceleration, displacement, strain_proxy, ...
    """
    window_steps = int(window_size / dt)
    stride_steps = int(stride / dt)

    if df_run.empty:
        return

    run_id = df_run["run_id"].iloc[0]
    damage_case = df_run["damage_case"].iloc[0]

    # Pivot: for each sensor_node, collect time series
    sensors = sorted(df_run["sensor_node"].unique())
    n_steps = len(df_run[df_run["sensor_node"] == sensors[0]])

    series = {}
    for sn in sensors:
        sub = df_run[df_run["sensor_node"] == sn].sort_values("time")
        series[sn] = {
            "time": sub["time"].values,
            "acceleration": sub["acceleration"].values,
            "displacement": sub["displacement"].values,
            "strain_proxy": sub["strain_proxy"].values,
        }

    n_windows = max(0, (n_steps - window_steps) // stride_steps + 1)

    for w in range(n_windows):
        start_idx = w * stride_steps
        end_idx = start_idx + window_steps
        t_start = series[sensors[0]]["time"][start_idx]
        t_end = series[sensors[0]]["time"][end_idx - 1]

        for sn in sensors:
            window = {
                "window_id": f"{run_id}_{w}",
                "damage_case": damage_case,
                "run_id": run_id,
                "sensor_node": sn,
                "window_pos": w,
                "time_start": t_start,
                "time_end": t_end,
                "acceleration": series[sn]["acceleration"][start_idx:end_idx].tolist(),
                "displacement": series[sn]["displacement"][start_idx:end_idx].tolist(),
                "strain_proxy": series[sn]["strain_proxy"][start_idx:end_idx].tolist(),
            }
            yield window


def extract_features(full_csv, output_csv, window_size, stride, dt=0.01):
    print(f"Loading dataset: {full_csv}")
    df = pd.read_csv(full_csv)
    print(f"  Rows: {len(df)}, Columns: {list(df.columns)}")
    print(f"  Damage cases: {list(df['damage_case'].unique())}")
    print(f"  Runs: {df['run_id'].nunique()}")

    # Group by (run_id, damage_case) — each group is one run
    groups = df.groupby(["run_id", "damage_case"])
    all_windows = []
    gbar = tqdm(groups, desc="Window+Feature", unit="run")

    for (run_id, damage_case), grp in gbar:
        for window in window_run_sensors(grp, window_size, stride, dt):
            row = extract_features_from_window(window, dt=dt)
            all_windows.append(row)

    if not all_windows:
        print("No windows extracted. Check window/stride params.")
        return

    feature_df = pd.DataFrame(all_windows)
    print(f"\nPer-sensor features: {len(feature_df)} rows")

    # Add cross-sensor features
    cross = add_cross_sensor_features(feature_df)
    print(f"Cross-sensor features: {len(cross)} rows")

    # Merge: join cross-sensor features back onto per-sensor rows
    merge_keys = ["run_id", "window_pos", "damage_case", "label_binary"]
    feature_df = feature_df.merge(
        cross, on=merge_keys, how="left", suffixes=("", "_drop")
    )
    # Drop duplicate columns from merge
    for col in list(feature_df.columns):
        if col.endswith("_drop"):
            feature_df.drop(columns=[col], inplace=True)

    # Replace inf/nan
    feature_df = feature_df.replace([np.inf, -np.inf], np.nan).fillna(0)

    # Add damage_case_id matching user's label scheme: healthy=0, reduced_girder=1, bearing=2, deck=3
    label_order = {"healthy": 0, "reduced_girder_stiffness": 1, "bearing_failure": 2, "deck_cracking": 3}
    feature_df["damage_case_id"] = feature_df["damage_case"].map(label_order)

    ensure_dir(os.path.dirname(output_csv))
    feature_df.to_csv(output_csv, index=False)
    print(f"\nSaved: {output_csv}")
    print(f"  {len(feature_df)} rows, {len(feature_df.columns)} columns")
    print(f"  Features: {len(feature_df.columns) - 9}")  # 9 meta cols

    # Quick stats
    print(f"\n=== Per damage case ===")
    for name, grp in feature_df.groupby("damage_case"):
        print(f"  {name}: {grp['run_id'].nunique()} runs, {len(grp)} windows")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="dataset_20260527_193221/full_bridge_response.csv")
    parser.add_argument("--output", default="data/processed/3d_windowed_features.csv")
    parser.add_argument("--window", type=float, default=2.0, help="Window size in seconds")
    parser.add_argument("--stride", type=float, default=1.0, help="Stride in seconds")
    args = parser.parse_args()

    extract_features(args.input, args.output, args.window, args.stride)


if __name__ == "__main__":
    main()

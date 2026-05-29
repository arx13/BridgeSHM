"""
Run full dataset generation — 4 damage cases × N runs each.

Usage:
    python run_full_dataset.py [--runs 50] [--output dataset_YYYYMMDD_HHMMSS]

Resume: re-run the same command; already-completed run_ids are skipped.
"""

import argparse
import json
import os
import sys
import time
import traceback
from datetime import datetime
from copy import deepcopy

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from digital_twin.config import (
    BRIDGE_CONFIG, SIM_CONFIG, VEHICLE_LIBRARY,
    VBI_CONFIG, ENV_CONFIG, TRAFFIC_CONFIG, DAMAGE_CASES,
)
from digital_twin.bridge_simulation import run_simulation
from digital_twin.utils import ensure_dir


def load_existing_run_ids(csv_path):
    """Return set of run_ids already saved in a per-damage-case CSV."""
    if not os.path.exists(csv_path):
        return set()
    try:
        existing = pd.read_csv(csv_path, usecols=["run_id"])
        return set(existing["run_id"].unique())
    except (pd.errors.EmptyDataError, ValueError, KeyError):
        return set()


def main():
    parser = argparse.ArgumentParser(description="Generate full SHM dataset")
    parser.add_argument("--runs", type=int, default=50,
                        help="Runs per damage case (default: 50)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output directory (default: dataset_<timestamp>)")
    args = parser.parse_args()

    runs_per_case = args.runs

    if args.output:
        out_dir = args.output
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = f"dataset_{timestamp}"

    ensure_dir(out_dir)

    log_path = os.path.join(out_dir, "dataset.log")
    meta_path = os.path.join(out_dir, "metadata.json")

    # Log setup
    def log(msg):
        line = f"[{datetime.now():%H:%M:%S}] {msg}"
        print(line)
        with open(log_path, "a") as f:
            f.write(line + "\n")

    log(f"Output directory: {out_dir}")
    log(f"Runs per damage case: {runs_per_case}")
    log(f"Total runs: {len(DAMAGE_CASES) * runs_per_case}")

    # Save metadata
    meta = {
        "timestamp": datetime.now().isoformat(),
        "runs_per_case": runs_per_case,
        "total_runs": len(DAMAGE_CASES) * runs_per_case,
        "damage_cases": {k: {kk: vv for kk, vv in v.items() if kk != "enabled"}
                        for k, v in DAMAGE_CASES.items()},
        "bridge": {k: v for k, v in BRIDGE_CONFIG.items()
                   if isinstance(v, (int, float, str, list))},
        "vbi_mode": VBI_CONFIG.get("vbi_mode", "loose"),
        "dt": SIM_CONFIG["dt"],
        "total_time": SIM_CONFIG["total_time"],
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    log(f"Metadata saved to {meta_path}")

    # Create a VBI config with loose coupling for speed
    local_vbi = dict(VBI_CONFIG)
    local_vbi["vbi_mode"] = "loose"
    total_runs_all = len(DAMAGE_CASES) * runs_per_case

    global_run_counter = 0
    failures = []

    for damage_name, damage_case in DAMAGE_CASES.items():
        csv_path = os.path.join(out_dir, f"dataset_{damage_name}.csv")
        existing_ids = load_existing_run_ids(csv_path)
        if existing_ids:
            log(f"{damage_name}: {len(existing_ids)} runs already completed, "
                f"will skip them")

        for run_id in range(1, runs_per_case + 1):
            if run_id in existing_ids:
                global_run_counter += 1
                continue

            global_run_counter += 1
            t_start = time.time()

            try:
                sig, traf = run_simulation(
                    bridge_config=BRIDGE_CONFIG,
                    sim_config=SIM_CONFIG,
                    vehicle_library=VEHICLE_LIBRARY,
                    vbi_config=local_vbi,
                    env_config=ENV_CONFIG,
                    traffic_config=TRAFFIC_CONFIG,
                    damage_case_name=damage_name,
                    damage_case=damage_case,
                    run_id=run_id,
                    total_runs=runs_per_case,
                )

                sig["run_id"] = run_id
                sig["damage_case"] = damage_name
                traf["run_id"] = run_id
                traf["damage_case"] = damage_name

                elapsed = time.time() - t_start
                completion = sig["completion_ratio"].iloc[0]
                status = "OK" if completion >= 0.9 else f"PARTIAL({completion:.2f})"

                # Append signal to per-damage-case CSV
                sig.to_csv(csv_path, mode="a",
                           header=not os.path.exists(csv_path),
                           index=False)

                # Append traffic log (once per case)
                traf_csv = os.path.join(out_dir, f"traffic_{damage_name}.csv")
                traf.to_csv(traf_csv, mode="a",
                            header=not os.path.exists(traf_csv),
                            index=False)

                log(f"[{global_run_counter}/{total_runs_all}] "
                    f"{damage_name} run{run_id} — {status} ({elapsed:.1f}s)")

            except Exception as e:
                elapsed = time.time() - t_start
                log(f"[FAIL] {damage_name} run{run_id} — {e} "
                    f"({elapsed:.1f}s)")
                traceback.print_exc()
                failures.append({
                    "damage_case": damage_name,
                    "run_id": run_id,
                    "error": str(e),
                })
                # Continue to next run; don't abort the batch

    # --- Final summary ---
    total_success = sum(
        1 for name in DAMAGE_CASES
        for _ in range(runs_per_case)
        if os.path.exists(os.path.join(out_dir, f"dataset_{name}.csv"))
    )
    log("=" * 50)
    log(f"Dataset generation complete.")
    log(f"Directory: {out_dir}")
    log(f"Failures: {len(failures)}")

    for name in DAMAGE_CASES:
        csv_path = os.path.join(out_dir, f"dataset_{name}.csv")
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            log(f"  {name}: {len(df)} rows, "
                f"runs={df['run_id'].nunique()}, "
                f"accel=[{df['acceleration'].min():.3f}, {df['acceleration'].max():.3f}] m/s²")

    # Combine all into a single full dataset
    log("Combining all damage cases into full_bridge_response.csv ...")
    all_parts = []
    for name in DAMAGE_CASES:
        csv_path = os.path.join(out_dir, f"dataset_{name}.csv")
        if os.path.exists(csv_path):
            all_parts.append(pd.read_csv(csv_path))
    if all_parts:
        full = pd.concat(all_parts, ignore_index=True)
        full_path = os.path.join(out_dir, "full_bridge_response.csv")
        full.to_csv(full_path, index=False)
        log(f"Full dataset: {full_path} ({len(full)} rows)")
        meta["total_rows"] = len(full)
        meta["n_failures"] = len(failures)
        if failures:
            meta["failures"] = failures
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

    log("Done.")


if __name__ == "__main__":
    main()

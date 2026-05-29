import os
import pandas as pd

from digital_twin.config import BRIDGE_CONFIG, SIM_CONFIG, DAMAGE_CASES
from modal_analysis.free_vibration import run_free_vibration, ensure_dir

OUTPUT_DIR = "outputs/free_vibration"

def main():
    ensure_dir(OUTPUT_DIR)

    all_dfs = []

    for damage_case_name, damage_case_cfg in DAMAGE_CASES.items():
        print(f"Running free vibration for: {damage_case_name}")

        df = run_free_vibration(
            bridge_config=BRIDGE_CONFIG,
            sim_config=SIM_CONFIG,
            damage_case_name=damage_case_name,
            damage_case_cfg=damage_case_cfg,
            total_time=SIM_CONFIG["total_time"],
            dt=SIM_CONFIG["dt"]
        )

        out_path = os.path.join(OUTPUT_DIR, f"{damage_case_name}_free_vibration.csv")
        df.to_csv(out_path, index=False)
        all_dfs.append(df)

    full_df = pd.concat(all_dfs, ignore_index=True)
    full_df.to_csv(os.path.join(OUTPUT_DIR, "all_free_vibration_data.csv"), index=False)

    print("\n========== FREE VIBRATION COMPLETE ==========")
    print(f"Saved per-case files in: {OUTPUT_DIR}")
    print("Saved combined file: outputs/free_vibration/all_free_vibration_data.csv")
    print("============================================")

if __name__ == "__main__":
    main()
import os
import pandas as pd

from modal_analysis.modal_features import (
    ensure_dir,
    extract_modal_features,
    build_health_index
)

INPUT_PATH = "outputs/free_vibration/all_free_vibration_data.csv"
OUTPUT_DIR = "outputs/modal_features"

def main():
    ensure_dir(OUTPUT_DIR)

    print("Loading free vibration dataset...")
    df = pd.read_csv(INPUT_PATH)

    print("Extracting modal features...")
    modal_df = extract_modal_features(df, dt=0.01)
    modal_df.to_csv(os.path.join(OUTPUT_DIR, "modal_features.csv"), index=False)

    print("Building Bridge Health Index...")
    health_df = build_health_index(modal_df)
    health_df.to_csv(os.path.join(OUTPUT_DIR, "bridge_health_index.csv"), index=False)

    print("\n========== MODAL FEATURE SUMMARY ==========")
    print(modal_df)
    print("===========================================")

    print("\n========== BRIDGE HEALTH INDEX ==========")
    print(health_df)
    print("=========================================")

    print("\nSaved files:")
    print("outputs/modal_features/modal_features.csv")
    print("outputs/modal_features/bridge_health_index.csv")

if __name__ == "__main__":
    main()
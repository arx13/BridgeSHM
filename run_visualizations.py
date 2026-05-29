import os
import sys
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from validation.plots import (
    plot_time_series, plot_per_sensor_boxplot, plot_response_distribution,
    plot_feature_correlation_heatmap, plot_all_sensor_comparison
)
from validation.fft_analysis import (
    plot_fft_comparison, plot_multi_sensor_fft, plot_fft_all_cases
)
from validation.damage_analysis import (
    plot_damage_separation, plot_spatial_rms_pattern, plot_cross_sensor_ratios
)
from modal_analysis.plots import plot_frequency_comparison

RAW_PATH = "data/synthetic/module1_bridge_response_dataset.csv"
FEATURES_PATH = "data/processed/module2_windowed_features.csv"
MODAL_FREQ_PATH = "outputs/modal/modal_frequencies.csv"


def main():
    os.makedirs("outputs/validation", exist_ok=True)
    os.makedirs("outputs/modal", exist_ok=True)

    print("Loading raw response data...")
    raw_df = pd.read_csv(RAW_PATH)
    raw_df["damage_case"] = raw_df["damage_case"].str.strip()
    print(f"  {len(raw_df)} rows, {list(raw_df.columns)}")

    print("Loading feature data...")
    feat_df = pd.read_csv(FEATURES_PATH)
    print(f"  {len(feat_df)} rows, {len(feat_df.columns)} columns")

    print("Loading modal frequencies...")
    freq_df = pd.read_csv(MODAL_FREQ_PATH)
    print(f"  {len(freq_df)} rows")

    print("\n--- Validation Plots ---")
    for dc in ["healthy", "minor_midspan_damage", "moderate_midspan_damage", "support_damage"]:
        plot_time_series(raw_df, sensor_node=87, damage_case=dc, run_id=1, signal_col="acceleration")
    plot_per_sensor_boxplot(raw_df, signal_col="acceleration")
    plot_response_distribution(raw_df, signal_col="acceleration")
    plot_feature_correlation_heatmap(feat_df)
    for dc in ["healthy", "moderate_midspan_damage"]:
        plot_all_sensor_comparison(raw_df, damage_case=dc, run_id=1)

    print("\n--- FFT Analysis ---")
    plot_fft_comparison(raw_df, sensor_node=87, healthy_case="healthy",
                        damaged_case="moderate_midspan_damage")
    for dc in ["healthy", "moderate_midspan_damage"]:
        plot_multi_sensor_fft(raw_df, damage_case=dc, run_id=1, signal_col="acceleration")
    plot_fft_all_cases(raw_df, sensor_node=87, run_id=1)

    print("\n--- Damage Analysis ---")
    plot_damage_separation(feat_df)
    plot_spatial_rms_pattern(feat_df)
    plot_cross_sensor_ratios(feat_df)

    print("\n--- Modal Analysis ---")
    plot_frequency_comparison(freq_df)

    print("\nAll visualizations complete!")


if __name__ == "__main__":
    main()

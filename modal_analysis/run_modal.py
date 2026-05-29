from digital_twin.config import BRIDGE_CONFIG, DAMAGE_CASES
from modal_analysis.modal import run_modal_analysis
from modal_analysis.plots import (
    plot_frequency_comparison,
    plot_mode_shape,
    plot_mode_shape_comparison
)

def main():
    freq_df, mode_shapes = run_modal_analysis(
        bridge_config=BRIDGE_CONFIG,
        damage_cases=DAMAGE_CASES,
        num_modes=6
    )

    freq_df.to_csv("outputs/modal/modal_frequencies.csv", index=False)

    print("\n========== MODAL FREQUENCIES ==========")
    print(freq_df)
    print("=======================================\n")

    plot_frequency_comparison(freq_df)

    for damage_case in DAMAGE_CASES.keys():
        for mode_num in range(1, 4):  # plot first 3 modes
            plot_mode_shape(mode_shapes, damage_case, mode_num)

    for mode_num in range(1, 4):
        plot_mode_shape_comparison(mode_shapes, mode_num)

    print("Modal analysis complete.")
    print("Saved results to outputs/modal/")

if __name__ == "__main__":
    main()
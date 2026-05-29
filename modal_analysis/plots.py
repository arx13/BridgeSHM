import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 300,
    "font.size": 10, "axes.titlesize": 12, "axes.labelsize": 11,
    "legend.fontsize": 9, "figure.facecolor": "white", "axes.facecolor": "white"
})

DAMAGE_COLORS = {"healthy": "#2ecc71", "minor_midspan_damage": "#f39c12",
                 "moderate_midspan_damage": "#e74c3c", "support_damage": "#9b59b6"}
DAMAGE_ORDER = ["healthy", "minor_midspan_damage", "moderate_midspan_damage", "support_damage"]
SENSOR_POSITIONS = {67: 0, 72: 5, 77: 10, 87: 20, 97: 30, 107: 35, 117: 40}
BRIDGE_SPAN_LENGTHS = [0, 20, 40, 60]  # 3 spans of 20m each

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def plot_frequency_comparison(freq_df, save_dir="outputs/modal"):
    ensure_dir(save_dir)
    freq_wide = freq_df.pivot(index="mode_number", columns="damage_case", values="frequency_hz")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Absolute frequencies
    for dc in DAMAGE_ORDER:
        if dc in freq_wide.columns:
            marker = "o" if dc == "healthy" else "s"
            ls = "-" if dc == "healthy" else "--"
            ax1.plot(freq_wide.index, freq_wide[dc], marker=marker, linestyle=ls,
                     linewidth=1.5, markersize=8, color=DAMAGE_COLORS[dc],
                     label=dc.replace("_", " ").title())
    ax1.set_xlabel("Mode Number")
    ax1.set_ylabel("Frequency (Hz)")
    ax1.set_title("Natural Frequencies vs Damage Case")
    ax1.set_xticks(range(1, len(freq_wide.index) + 1))
    ax1.legend(framealpha=0.8)
    ax1.grid(True, alpha=0.3)

    # % change from healthy
    if "healthy" in freq_wide.columns:
        healthy_freq = freq_wide["healthy"].values
        for dc in DAMAGE_ORDER[1:]:
            if dc in freq_wide.columns:
                pct_change = (freq_wide[dc].values - healthy_freq) / healthy_freq * 100
                ax2.plot(freq_wide.index, pct_change, marker="s", linewidth=1.5,
                         markersize=8, color=DAMAGE_COLORS[dc],
                         label=dc.replace("_", " ").title())
                for i, p in enumerate(pct_change):
                    ax2.annotate(f"{p:.3f}%", (freq_wide.index[i], p),
                                textcoords="offset points", xytext=(0, 10),
                                fontsize=8, ha="center", color=DAMAGE_COLORS[dc])
        ax2.axhline(0, color="black", linewidth=0.5)
    ax2.set_xlabel("Mode Number")
    ax2.set_ylabel("Frequency Change from Healthy (%)")
    ax2.set_title("Relative Frequency Shift Due to Damage")
    ax2.set_xticks(range(1, len(freq_wide.index) + 1))
    ax2.legend(framealpha=0.8)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{save_dir}/frequency_comparison.png", bbox_inches="tight")
    plt.close()

    # Summary table
    print("\n========== FREQUENCY SHIFT SUMMARY (%) ==========")
    print(f"{'Mode':>6}", end="")
    for dc in DAMAGE_ORDER[1:]:
        print(f"{dc:>28}", end="")
    print()
    for mode in freq_wide.index:
        f_healthy = freq_wide.loc[mode, "healthy"]
        print(f"{mode:>6}", end="")
        for dc in DAMAGE_ORDER[1:]:
            if dc in freq_wide.columns:
                pct = (freq_wide.loc[mode, dc] - f_healthy) / f_healthy * 100
                print(f"{pct:>+26.4f}", end="")
        print()
    print("================================================\n")

def plot_mode_shape(mode_shapes, damage_case, mode_number, save_dir="outputs/modal"):
    ensure_dir(save_dir)
    data = mode_shapes[damage_case][mode_number]
    x = data["x"]
    y = data["mode_shape"]

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(x, y, linewidth=2, color=DAMAGE_COLORS.get(damage_case, "#3498db"))

    # Sensor node markers
    for sn, pos in SENSOR_POSITIONS.items():
        idx = np.argmin(np.abs(x - pos))
        ax.scatter(x[idx], y[idx], color="red", s=40, zorder=5, edgecolors="black", linewidths=0.5)
        ax.annotate(f"SN{sn}", (x[idx], y[idx]), textcoords="offset points",
                    xytext=(0, 12), fontsize=8, ha="center", color="red")

    # Span boundaries
    for span_end in BRIDGE_SPAN_LENGTHS[1:-1]:
        ax.axvline(span_end, color="grey", linestyle="--", linewidth=0.8, alpha=0.5)
        ax.text(span_end, ax.get_ylim()[1] * 1.05, f"{span_end}m", fontsize=7,
                ha="center", color="grey", alpha=0.6)

    # Damage location marker
    if "midspan" in damage_case:
        damage_x = 30
        ax.axvspan(damage_x - 1, damage_x + 1, color="red", alpha=0.12, label="Damage zone")
    elif "support" in damage_case:
        ax.axvspan(0, 0.5, color="red", alpha=0.12, label="Damage zone")

    ax.set_xlabel("Bridge Length (m)")
    ax.set_ylabel("Normalized Vertical Mode Shape")
    ax.set_title(f"Mode Shape | {damage_case.replace('_', ' ').title()} | Mode {mode_number}")
    if "midspan" in damage_case or "support" in damage_case:
        ax.legend(loc="upper right", framealpha=0.8)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-1.2, 1.2)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/mode_shape_{damage_case}_mode{mode_number}.png", bbox_inches="tight")
    plt.close()

def plot_mode_shape_comparison(mode_shapes, mode_number, save_dir="outputs/modal"):
    ensure_dir(save_dir)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))

    # Mode shapes
    for dc in DAMAGE_ORDER:
        if dc in mode_shapes and mode_number in mode_shapes[dc]:
            x = mode_shapes[dc][mode_number]["x"]
            y = mode_shapes[dc][mode_number]["mode_shape"]
            ax1.plot(x, y, linewidth=1.5, color=DAMAGE_COLORS[dc],
                     label=dc.replace("_", " ").title())

    for span_end in BRIDGE_SPAN_LENGTHS[1:-1]:
        ax1.axvline(span_end, color="grey", linestyle="--", linewidth=0.5, alpha=0.4)
    ax1.set_xlabel("Bridge Length (m)")
    ax1.set_ylabel("Normalized Vertical Mode Shape")
    ax1.set_title(f"Mode Shape Comparison | Mode {mode_number}")
    ax1.legend(framealpha=0.8)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(-1.2, 1.2)

    # Difference from healthy
    if "healthy" in mode_shapes:
        x_h = mode_shapes["healthy"][mode_number]["x"]
        y_h = mode_shapes["healthy"][mode_number]["mode_shape"]
        for dc in DAMAGE_ORDER[1:]:
            if dc in mode_shapes and mode_number in mode_shapes[dc]:
                x_d = mode_shapes[dc][mode_number]["x"]
                y_d = mode_shapes[dc][mode_number]["mode_shape"]
                y_diff = y_d - y_h
                ax2.plot(x_d, y_diff, linewidth=1.5, color=DAMAGE_COLORS[dc],
                         label=dc.replace("_", " ").title())

    ax2.axhline(0, color="black", linewidth=0.5)
    for span_end in BRIDGE_SPAN_LENGTHS[1:-1]:
        ax2.axvline(span_end, color="grey", linestyle="--", linewidth=0.5, alpha=0.4)
    ax2.set_xlabel("Bridge Length (m)")
    ax2.set_ylabel("Mode Shape Difference")
    ax2.set_title(f"Mode Shape Difference (Damaged − Healthy) | Mode {mode_number}")
    ax2.legend(framealpha=0.8)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{save_dir}/mode_shape_comparison_mode{mode_number}.png", bbox_inches="tight")
    plt.close()

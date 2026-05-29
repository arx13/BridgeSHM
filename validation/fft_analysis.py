import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from scipy.fft import fft, fftfreq
from scipy.signal import find_peaks

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 300,
    "font.size": 10, "axes.titlesize": 12, "axes.labelsize": 11,
    "legend.fontsize": 9, "figure.facecolor": "white", "axes.facecolor": "white"
})

DAMAGE_COLORS = {"healthy": "#2ecc71", "minor_midspan_damage": "#f39c12",
                 "moderate_midspan_damage": "#e74c3c", "support_damage": "#9b59b6"}

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def compute_fft(signal, dt):
    n = len(signal)
    yf = fft(signal)
    xf = fftfreq(n, dt)[:n // 2]
    amp = 2.0 / n * np.abs(yf[:n // 2])
    return xf, amp

def plot_fft_comparison(df, sensor_node, healthy_case="healthy", damaged_case="moderate_midspan_damage",
                        healthy_run=1, damaged_run=1, signal_col="acceleration",
                        dt=0.01, save_dir="outputs/validation"):
    ensure_dir(save_dir)

    healthy = df[(df["sensor_node"] == sensor_node) & (df["damage_case"] == healthy_case) & (df["run_id"] == healthy_run)][signal_col].values
    damaged = df[(df["sensor_node"] == sensor_node) & (df["damage_case"] == damaged_case) & (df["run_id"] == damaged_run)][signal_col].values
    if len(healthy) == 0 or len(damaged) == 0:
        return

    xf_h, amp_h = compute_fft(healthy, dt)
    xf_d, amp_d = compute_fft(damaged, dt)

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 10))

    # Full spectrum
    ax1.plot(xf_h, amp_h, color=DAMAGE_COLORS[healthy_case], linewidth=0.8, label="Healthy")
    ax1.plot(xf_d, amp_d, color=DAMAGE_COLORS[damaged_case], linewidth=0.8, alpha=0.85, label="Damaged")
    ax1.set_xlim(0, 50)
    ax1.set_ylabel("Amplitude")
    ax1.set_title(f"FFT Comparison | {signal_col.capitalize()} | Sensor {sensor_node}")
    ax1.legend(framealpha=0.8)
    ax1.grid(True, alpha=0.3)

    # Log scale
    ax2.plot(xf_h, amp_h, color=DAMAGE_COLORS[healthy_case], linewidth=0.8, label="Healthy")
    ax2.plot(xf_d, amp_d, color=DAMAGE_COLORS[damaged_case], linewidth=0.8, alpha=0.85, label="Damaged")
    ax2.set_xlim(0, 50)
    ax2.set_yscale("log")
    ax2.set_ylabel("Amplitude (log)")
    ax2.set_title("Log Scale View")
    ax2.legend(framealpha=0.8)

    # Difference spectrum
    peaks_h, _ = find_peaks(amp_h, height=np.max(amp_h) * 0.05, distance=20)
    peaks_d, _ = find_peaks(amp_d, height=np.max(amp_d) * 0.05, distance=20)
    freq_diff = xf_d[:min(len(amp_d), len(amp_h))]
    amp_diff = amp_d[:len(freq_diff)] - amp_h[:len(freq_diff)]
    ax3.bar(freq_diff, amp_diff, width=0.15, color="#e74c3c", alpha=0.6, label="Damaged - Healthy")
    ax3.axhline(0, color="black", linewidth=0.5)
    ax3.set_xlim(0, 50)
    ax3.set_xlabel("Frequency (Hz)")
    ax3.set_ylabel("Amplitude Difference")
    ax3.set_title("Spectral Difference (Damaged − Healthy)")
    ax3.legend(framealpha=0.8)

    plt.tight_layout()
    plt.savefig(f"{save_dir}/fft_compare_{signal_col}_sensor{sensor_node}.png", bbox_inches="tight")
    plt.close()

def plot_multi_sensor_fft(df, damage_case, run_id=1, signal_col="acceleration", dt=0.01, save_dir="outputs/validation"):
    ensure_dir(save_dir)
    sensors = sorted(df["sensor_node"].unique())
    fig, axes = plt.subplots(len(sensors), 1, figsize=(14, 12), sharex=True)
    color = DAMAGE_COLORS.get(damage_case, "#3498db")

    for idx, sn in enumerate(sensors):
        sig = df[(df["sensor_node"] == sn) & (df["damage_case"] == damage_case) & (df["run_id"] == run_id)][signal_col].values
        if len(sig) == 0:
            continue
        xf, amp = compute_fft(sig, dt)
        axes[idx].plot(xf, amp, color=color, linewidth=0.8)
        axes[idx].set_xlim(0, 50)
        axes[idx].set_yscale("log")
        axes[idx].set_ylabel(f"SN{sn}")
        axes[idx].grid(True, alpha=0.3)
        axes[idx].tick_params(labelsize=7)

        peaks, props = find_peaks(amp, height=np.max(amp) * 0.1, distance=20)
        for p in peaks[:3]:
            axes[idx].axvline(xf[p], color="red", linestyle="--", linewidth=0.5, alpha=0.5)
            axes[idx].text(xf[p] + 0.5, amp[p], f"{xf[p]:.1f}Hz", fontsize=6, color="red")

    axes[-1].set_xlabel("Frequency (Hz)")
    fig.suptitle(f"Per-Sensor FFT | {damage_case.replace('_', ' ').title()} Run {run_id} | {signal_col}", fontsize=13)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/multi_sensor_fft_{damage_case}_{signal_col}.png", bbox_inches="tight")
    plt.close()

def plot_fft_all_cases(df, sensor_node, run_id=1, signal_col="acceleration", dt=0.01, save_dir="outputs/validation"):
    ensure_dir(save_dir)
    fig, ax = plt.subplots(figsize=(14, 6))

    for dc in ["healthy", "minor_midspan_damage", "moderate_midspan_damage", "support_damage"]:
        sig = df[(df["sensor_node"] == sensor_node) & (df["damage_case"] == dc) & (df["run_id"] == run_id)][signal_col].values
        if len(sig) == 0:
            continue
        xf, amp = compute_fft(sig, dt)
        ax.plot(xf, amp, color=DAMAGE_COLORS.get(dc, "#3498db"), linewidth=0.8, alpha=0.85,
                label=dc.replace("_", " ").title())
        peaks, _ = find_peaks(amp, height=np.max(amp) * 0.1, distance=20)
        if len(peaks) > 0:
            dom_freq = xf[peaks[0]]
            ax.annotate(f"{dc[:6]} {dom_freq:.2f}Hz", xy=(dom_freq, amp[peaks[0]]),
                        fontsize=7, color=DAMAGE_COLORS.get(dc), alpha=0.8)

    ax.set_xlim(0, 50)
    ax.set_yscale("log")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Amplitude (log)")
    ax.set_title(f"FFT: All Damage Cases | Sensor {sensor_node} | {signal_col.capitalize()}")
    ax.legend(framealpha=0.8)
    ax.grid(True, alpha=0.3)

    # Annotate natural frequency bands
    for f_band, label in [(20, "Mode 1-2"), (38, "Mode 3-4"), (50, "Mode 5-6")]:
        ax.axvline(f_band, color="grey", linestyle="--", linewidth=0.5, alpha=0.3)
        ax.text(f_band, ax.get_ylim()[1] * 0.9, label, fontsize=7, color="grey", alpha=0.6)

    plt.tight_layout()
    plt.savefig(f"{save_dir}/fft_all_cases_sensor{sensor_node}_{signal_col}.png", bbox_inches="tight")
    plt.close()

def get_dominant_frequency(signal, dt):
    xf, amp = compute_fft(signal, dt)
    idx = np.argmax(amp[1:]) + 1
    return xf[idx], amp[idx]

import os
import numpy as np
import pandas as pd
from scipy.fft import fft, fftfreq

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def compute_modal_frequency(signal, dt=0.01, f_low=35, f_high=42):
    #Extract modal frequency using centroid (not peak).
    n = len(signal)
    yf = fft(signal)
    xf = fftfreq(n, dt)

    # Keep only positive frequencies
    mask = xf > 0
    xf = xf[mask]
    yf = np.abs(yf[mask])

    # Restrict to modal band
    band_mask = (xf >= f_low) & (xf <= f_high)

    if not np.any(band_mask):
        return 0.0, 0.0

    xf_band = xf[band_mask]
    yf_band = yf[band_mask]

    #NEW: frequency centroid (smooth, continuous)
    weighted_freq = np.sum(xf_band * yf_band) / (np.sum(yf_band) + 1e-12)

    peak_amp = np.max(yf_band)

    return weighted_freq, peak_amp

def compute_band_energy(signal, dt=0.01, f_low=20, f_high=40):
    n = len(signal)
    yf = fft(signal)
    xf = fftfreq(n, dt)

    mask = (xf >= f_low) & (xf <= f_high)
    return np.sum(np.abs(yf[mask]) ** 2)

def extract_modal_features(df, dt=0.01):
    # Extract modal-sensitive features per sensor and damage case.
    rows = []

    grouped = df.groupby(["damage_case", "sensor_node"])

    for (damage_case, sensor_node), group in grouped:
        group = group.sort_values("time").reset_index(drop=True)

        acc = group["acceleration"].values
        disp = group["displacement"].values
        strain = group["strain_proxy"].values

        freq_acc, amp_acc = compute_modal_frequency(acc, dt)
        freq_disp, amp_disp = compute_modal_frequency(disp, dt)
        freq_strain, amp_strain = compute_modal_frequency(strain, dt)

        row = {
            "damage_case": damage_case,
            "sensor_node": sensor_node,

            "acc_dominant_frequency": freq_acc,
            "disp_dominant_frequency": freq_disp,
            "strain_dominant_frequency": freq_strain,

            "acc_peak_amplitude": np.max(np.abs(acc)),
            "disp_peak_amplitude": np.max(np.abs(disp)),
            "strain_peak_amplitude": np.max(np.abs(strain)),

            "acc_rms": np.sqrt(np.mean(acc**2)),
            "disp_rms": np.sqrt(np.mean(disp**2)),
            "strain_rms": np.sqrt(np.mean(strain**2)),

            "acc_band_energy_20_40Hz": compute_band_energy(acc, dt),
            "disp_band_energy_20_40Hz": compute_band_energy(disp, dt),
            "strain_band_energy_20_40Hz": compute_band_energy(strain, dt),
        }

        rows.append(row)

    # ---------------------------------------------------------
    # NEW: normalize RMS pattern across sensors
    # This acts as a mode-shape proxy
    # ---------------------------------------------------------
    modal_df = pd.DataFrame(rows)

    normalized_rows = []

    for damage_case, group in modal_df.groupby("damage_case"):
        total_acc_rms = group["acc_rms"].sum() + 1e-12
        total_disp_rms = group["disp_rms"].sum() + 1e-12
        total_strain_rms = group["strain_rms"].sum() + 1e-12

        for _, row in group.iterrows():
            row_dict = row.to_dict()

            row_dict["acc_rms_normalized"] = row["acc_rms"] / total_acc_rms
            row_dict["disp_rms_normalized"] = row["disp_rms"] / total_disp_rms
            row_dict["strain_rms_normalized"] = row["strain_rms"] / total_strain_rms

            normalized_rows.append(row_dict)

    return pd.DataFrame(normalized_rows)

def build_health_index(modal_df):
    """
    Compare each damage case to healthy baseline and compute modal drift score.
    """
    healthy_df = modal_df[modal_df["damage_case"] == "healthy"].copy()

    rows = []

    for _, row in modal_df.iterrows():
        if row["damage_case"] == "healthy":
            continue

        sensor_node = row["sensor_node"]
        ref = healthy_df[healthy_df["sensor_node"] == sensor_node]

        if ref.empty:
            continue

        ref = ref.iloc[0]

        freq_shift = abs(row["acc_dominant_frequency"] - ref["acc_dominant_frequency"])
        rms_shift = abs(row["acc_rms"] - ref["acc_rms"])
        band_shift = abs(row["acc_band_energy_20_40Hz"] - ref["acc_band_energy_20_40Hz"])

        # Simple normalized modal drift score
        health_index = (
            0.2 * freq_shift / (ref["acc_dominant_frequency"] + 1e-6) +
            0.3 * rms_shift / (ref["acc_rms"] + 1e-6) +
            0.5 * band_shift / (ref["acc_band_energy_20_40Hz"] + 1e-6)
        )

        rows.append({
            "damage_case": row["damage_case"],
            "sensor_node": sensor_node,
            "frequency_shift": freq_shift,
            "rms_shift": rms_shift,
            "band_energy_shift": band_shift,
            "bridge_health_index": health_index
        })

    return pd.DataFrame(rows)
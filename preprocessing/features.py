import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis
from scipy.fft import fft, fftfreq

def compute_energy(signal):
    return np.sum(signal ** 2)

def band_energy(signal, fs=100, f_low=20, f_high=40):
    n = len(signal)
    yf = fft(signal)
    xf = fftfreq(n, 1/fs)
    mask = (xf >= f_low) & (xf <= f_high)
    return np.sum(np.abs(yf[mask])**2)

def band_energies(signal, fs=100):
    bands = [(0, 10), (10, 20), (20, 40), (40, 50)]
    result = {}
    for lo, hi in bands:
        result[f"band_energy_{lo}_{hi}Hz"] = band_energy(signal, fs, lo, hi)
    return result

def rms(x):
    return np.sqrt(np.mean(np.square(x)))

def crest_factor(x):
    r = rms(x)
    if abs(r) < 1e-12:
        return 0.0
    return np.max(np.abs(x)) / r

def compute_fft_features(signal, dt=0.01):
    try:
        n = len(signal)
        yf = fft(signal)
        xf = fftfreq(n, dt)[:n // 2]
        amp = 2.0 / n * np.abs(yf[:n // 2])
        if len(amp) <= 1:
            return {"dominant_frequency": 0.0, "dominant_amplitude": 0.0, "spectral_energy": 0.0}
        idx = np.argmax(amp[1:]) + 1
        d = {"dominant_frequency": xf[idx], "dominant_amplitude": amp[idx], "spectral_energy": np.sum(amp**2)}
        for k in d:
            if np.isnan(d[k]): d[k] = 0.0
        return d
    except:
        return {"dominant_frequency": 0.0, "dominant_amplitude": 0.0, "spectral_energy": 0.0}

def safe_stat(x, func, default=0.0):
    try:
        val = func(x)
        return default if (np.isnan(val) or np.isinf(val)) else val
    except:
        return default

def extract_statistical_features(signal, prefix):
    return {
        f"{prefix}_mean": safe_stat(signal, np.mean), f"{prefix}_std": safe_stat(signal, np.std),
        f"{prefix}_rms": safe_stat(signal, rms), f"{prefix}_max": safe_stat(signal, np.max),
        f"{prefix}_min": safe_stat(signal, np.min), f"{prefix}_ptp": safe_stat(signal, np.ptp),
        f"{prefix}_skew": safe_stat(signal, skew), f"{prefix}_kurtosis": safe_stat(signal, kurtosis),
        f"{prefix}_crest_factor": safe_stat(signal, crest_factor),
    }

def extract_features_from_window(window, dt=0.01):
    feature_row = {
        "window_id": window["window_id"], "damage_case": window["damage_case"],
        "run_id": window["run_id"], "sensor_node": window["sensor_node"],
        "window_pos": window.get("window_pos", 0),
        "time_start": window["time_start"], "time_end": window["time_end"],
        "label_binary": 0 if window["damage_case"] == "healthy" else 1
    }
    for signal_name in ["acceleration", "displacement", "strain_proxy"]:
        signal = np.array(window[signal_name])
        feature_row.update(extract_statistical_features(signal, signal_name))
        fft_feats = compute_fft_features(signal, dt=dt)
        for k, v in fft_feats.items():
            feature_row[f"{signal_name}_{k}"] = v
        feature_row[f"{signal_name}_energy"] = compute_energy(signal)
        for k, v in band_energies(signal, fs=1/dt).items():
            feature_row[f"{signal_name}_{k}"] = v
    return feature_row

def add_cross_sensor_features(df):
    """Compute spatial features across all 7 sensors per window position."""
    if "window_pos" not in df.columns:
        return pd.DataFrame(columns=["run_id", "window_id", "damage_case", "label_binary"])

    grouped = df.groupby(["run_id", "window_pos", "damage_case", "label_binary"])
    rows = []
    for keys, group in grouped:
        run_id, window_pos, damage_case, label_binary = keys
        row = {"run_id": run_id, "window_pos": window_pos, "damage_case": damage_case, "label_binary": label_binary}
        group = group.sort_values("sensor_node").reset_index(drop=True)
        n_sensors = len(group)

        for prefix in ["acceleration", "displacement", "strain_proxy"]:
            rms_vals = group[f"{prefix}_rms"].values.copy()
            rms_vals = np.nan_to_num(rms_vals, 0)
            total = rms_vals.sum() + 1e-12

            # Normalized RMS pattern (mode-shape)
            for i in range(n_sensors):
                row[f"{prefix}_norm_rms_{i}"] = rms_vals[i] / total

            # Energy ratios between adjacent sensors
            for i in range(n_sensors - 1):
                row[f"{prefix}_ratio_{i}_{i+1}"] = rms_vals[i] / (rms_vals[i+1] + 1e-12)

            # Spatial statistics
            row[f"{prefix}_spatial_max"] = np.max(rms_vals)
            row[f"{prefix}_spatial_min"] = np.min(rms_vals)
            row[f"{prefix}_spatial_range"] = np.ptp(rms_vals)
            row[f"{prefix}_spatial_mean"] = np.mean(rms_vals)
            row[f"{prefix}_spatial_std"] = np.std(rms_vals)
            row[f"{prefix}_spatial_skew"] = safe_stat(rms_vals, skew)
            row[f"{prefix}_spatial_kurtosis"] = safe_stat(rms_vals, kurtosis)

        rows.append(row)
    return pd.DataFrame(rows)

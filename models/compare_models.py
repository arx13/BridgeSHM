"""
Side-by-side comparison: Model A vs Model B (+ Gateway).

Loads results from both pipelines and prints comparison tables.

Usage:
    python models/compare_models.py
"""

import os, sys
import json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

MODEL_A_DIR = "outputs/cnn_multiclass"
MODEL_B_DIR = "outputs/cnn_binary"
GATEWAY_DIR = "outputs/cnn_gateway"
SENSOR_NODES = [34, 37, 39, 44, 47, 49, 54]


def load_results(path, expect_sensors=True):
    with open(path) as f:
        return json.load(f)


def print_comparison():
    print("=" * 80)
    print("Model Comparison: Model A (Multiclass CNN) vs Model B (Binary+Intensity+Gateway)")
    print("=" * 80)

    # Model A
    a_results = load_results(os.path.join(MODEL_A_DIR, "results.json"))

    # Model B
    b_results = load_results(os.path.join(MODEL_B_DIR, "results.json"))

    # Gateway
    if os.path.exists(os.path.join(GATEWAY_DIR, "results.json")):
        gw = load_results(os.path.join(GATEWAY_DIR, "results.json"))
    else:
        gw = None

    # Per-sensor comparison
    print(f"\n{'Sensor':>8}  {'A-Acc':>8}  {'A-F1':>8}  {'B-BinAcc':>10}  {'B-BinF1':>9}  {'B-IntAcc':>10}")
    print("-" * 65)

    a_sensors = {int(k): v for k, v in a_results.get("per_sensor", {}).items()}
    b_sensors = {int(k): v for k, v in b_results.items()}

    for sn in SENSOR_NODES:
        a_res = a_sensors.get(sn, {})
        b_res = b_sensors.get(sn, {})

        a_acc = a_res.get("test_accuracy", -1)
        a_f1 = a_res.get("test_f1", -1)
        b_bin_acc = b_res.get("binary_accuracy", -1)
        b_bin_f1 = b_res.get("binary_f1", -1)
        b_int_acc = b_res.get("intensity_accuracy", -1)

        print(f"{sn:>8}  {a_acc:>8.4f}  {a_f1:>8.4f}  {b_bin_acc:>10.4f}  {b_bin_f1:>9.4f}  {b_int_acc:>10.4f}")

    # Ensemble / Gateway comparison
    print(f"\n{'System':<30}  {'Accuracy':>10}  {'Macro-F1':>10}")
    print("-" * 55)

    a_ensemble = a_results.get("ensemble", {})
    a_ens_acc = a_ensemble.get("ensemble_accuracy", -1)
    a_ens_f1 = a_ensemble.get("per_class_f1", [-1])
    a_ens_f1_mean = np.mean(a_ens_f1) if isinstance(a_ens_f1, list) else -1
    print(f"{'A: Multiclass ensemble':<30}  {a_ens_acc:>10.4f}  {a_ens_f1_mean:>10.4f}")

    if gw:
        gw_acc = gw.get("test_accuracy", -1)
        gw_f1 = gw.get("test_macro_f1", -1)
        print(f"{'B: Gateway (RF fusion)':<30}  {gw_acc:>10.4f}  {gw_f1:>10.4f}")
        print(f"{'B: Gateway per-class F1':<30}  {str([f'{f:.3f}' for f in gw.get('per_class_f1', [])]):>30}")
    else:
        print(f"{'B: Gateway':<30}  {'not yet run':>10}")

    # Winner
    print("\n" + "=" * 80)
    if gw and gw.get("test_accuracy", 0) >= a_ens_acc:
        print("WINNER: Model B (Binary+Intensity CNN + RF Gateway)")
        print(f"  Accuracy: {gw.get('test_accuracy', 0):.4f} vs {a_ens_acc:.4f}")
    else:
        print("WINNER: Model A (Multiclass 1D-CNN ensemble)")
        print(f"  Accuracy: {a_ens_acc:.4f}")

    print("=" * 80)

    # Confusion matrix comparison
    print("\n\nModel A Ensemble Confusion Matrix:")
    cm_a = np.array(a_ensemble.get("confusion_matrix", []))
    if cm_a.size > 0:
        print(cm_a)

    if gw:
        cm_gw = np.array(gw.get("confusion_matrix", []))
        if cm_gw.size > 0:
            print("\nModel B Gateway Confusion Matrix:")
            print(cm_gw)


if __name__ == "__main__":
    print_comparison()

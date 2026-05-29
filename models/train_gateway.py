"""
Gateway fusion for Model B.

Loads per-sensor binary predictions + intensity softmax outputs
from the test set, trains a RandomForest(100, max_depth=8)
to map per-sensor predictions + sensor_x coords → 4-class damage type.

Features per window:
  7 sensors × (1 binary_pred + 3 intensity_softmax) + 7 × sensor_x = 35

Usage:
    python models/train_gateway.py
"""

import os, sys
import numpy as np
import json
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from digital_twin.utils import ensure_dir

BINARY_DIR = "outputs/cnn_binary"
OUTPUT_DIR = "outputs/cnn_gateway"
SEED = 42

# Sensor layout on girder 2
SENSOR_X = {
    "34": 5.0, "37": 10.0, "39": 15.0,
    "44": 25.0, "47": 30.0, "49": 35.0, "54": 45.0,
}
SENSOR_NODES = [str(s) for s in [34, 37, 39, 44, 47, 49, 54]]


def main():
    ensure_dir(OUTPUT_DIR)

    print("Loading per-sensor gateway data...")
    gw = np.load(os.path.join(BINARY_DIR, "gateway_data.npz"), allow_pickle=True)

    # Build feature matrix and labels
    all_feats = None
    all_labels = None

    for sn in SENSOR_NODES:
        arr = gw[sn].item() if isinstance(gw[sn], np.ndarray) else gw[sn]
        if isinstance(arr, np.ndarray) and arr.dtype == np.object_:
            arr = arr.item()

        # Handle different loading shapes
        binary_arr = np.array(arr["test_binary"]).ravel()
        intensity_arr = np.array(arr["test_intensity_raw"])
        labels = np.array(arr["test_labels"]).ravel()

        if intensity_arr.ndim == 1:
            # Reshape from list of lists
            intensity_arr = np.array(arr["test_intensity_raw"])

        if intensity_arr.ndim == 3:
            intensity_arr = intensity_arr.reshape(-1, 3)

        n = len(binary_arr)

        # Per-sensor features: binary + 3 intensity values + sensor_x
        sx = SENSOR_X[sn]
        sensor_feats = np.column_stack([
            binary_arr,
            intensity_arr[:, 0],
            intensity_arr[:, 1],
            intensity_arr[:, 2],
            np.full(n, sx),
        ])

        if all_feats is None:
            all_feats = sensor_feats
            all_labels = labels
        else:
            assert len(all_labels) == n, f"Label count mismatch for sensor {sn}"
            all_feats = np.column_stack([all_feats, sensor_feats])

    assert all_feats is not None, "No features loaded!"
    print(f"  Feature matrix: {all_feats.shape}")
    print(f"  Label distribution: {np.bincount(all_labels.astype(int), minlength=4)}")

    # Train/test split: use last 25% as test (same as original split)
    np.random.seed(SEED)
    n = len(all_labels)
    perm = np.random.RandomState(SEED).permutation(n)
    split = int(n * 0.75)
    train_idx = perm[:split]
    test_idx = perm[split:]

    x_train, y_train = all_feats[train_idx], all_labels[train_idx]
    x_test, y_test = all_feats[test_idx], all_labels[test_idx]

    # Train RF
    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=8,
        min_samples_leaf=5,
        random_state=SEED,
        class_weight="balanced",
        n_jobs=-1,
    )
    rf.fit(x_train, y_train)

    y_pred_train = rf.predict(x_train)
    y_pred_test = rf.predict(x_test)

    train_acc = accuracy_score(y_train, y_pred_train)
    test_acc = accuracy_score(y_test, y_pred_test)
    test_f1_macro = f1_score(y_test, y_pred_test, average="macro")
    cm = confusion_matrix(y_test, y_pred_test)

    print(f"  Train accuracy: {train_acc:.4f}")
    print(f"  Test accuracy:  {test_acc:.4f}")
    print(f"  Test macro-F1:  {test_f1_macro:.4f}")
    print(f"  Confusion matrix:\n{cm}")

    # Per-class F1
    per_class_f1 = f1_score(y_test, y_pred_test, average=None)
    print(f"  Per-class F1: {[f'{f:.4f}' for f in per_class_f1]}")

    # Feature importance
    imp = rf.feature_importances_
    print(f"  Total features: {len(imp)}")

    results = {
        "train_accuracy": float(train_acc),
        "test_accuracy": float(test_acc),
        "test_macro_f1": float(test_f1_macro),
        "per_class_f1": per_class_f1.tolist(),
        "confusion_matrix": cm.tolist(),
        "feature_importances": imp.tolist(),
        "n_estimators": 100,
        "max_depth": 8,
    }

    # Save model and results
    joblib.dump(rf, os.path.join(OUTPUT_DIR, "gateway_rf.pkl"))
    with open(os.path.join(OUTPUT_DIR, "results.json"), "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {OUTPUT_DIR}/")
    print("Done.")


if __name__ == "__main__":
    main()

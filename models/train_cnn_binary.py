"""
Model B: Per-sensor binary + intensity multi-task 1D-CNN.

USE_TF_BACKEND = True  # set KERAS_BACKEND=tensorflow before running

Each sensor trains a shared backbone with two heads:
1. Binary head (healthy vs damaged) — BinaryCrossentropy
2. Intensity head (3-level ordinal: low/medium/high) — CategoricalCrossentropy

Joint loss = binary_BCE + 0.5 * intensity_CE.
Intensity gradient only flows for damaged samples.

Per-sensor optimal binary threshold is learned on the validation set
by sweeping 0.01–0.99 and picking argmax F1.

Usage:
    python models/train_cnn_binary.py
"""

import os, sys

os.environ["KERAS_BACKEND"] = "tensorflow"

import numpy as np
import json
from tqdm import tqdm

import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers, regularizers

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from digital_twin.utils import ensure_dir

WINDOWS_DIR = "data/processed/cnn_windows"
OUTPUT_DIR = "outputs/cnn_binary"
SEED = 42
EPOCHS = 200
PATIENCE = 20
BATCH_SIZE = 64
LEARNING_RATE = 1e-3

SENSOR_NODES = [34, 37, 39, 44, 47, 49, 54]
NUM_CLASSES = 4
INTENSITY_CLASSES = 3  # low/medium/high
INPUT_SHAPE = (400, 3)

# Severity mapping: damage_case -> (binary, intensity_level)
#   0=healthy -> binary=0, intensity=-1 (ignored)
#   1=reduced_girder_stiffness (~10%) -> binary=1, intensity=0 (low)
#   2=bearing_failure (~15%) -> binary=1, intensity=1 (medium)
#   3=deck_cracking (~20%) -> binary=1, intensity=2 (high)
BINARY_MAP = {0: 0, 1: 1, 2: 1, 3: 1}
INTENSITY_MAP = {0: -1, 1: 0, 2: 1, 3: 2}

tf.random.set_seed(SEED)
np.random.seed(SEED)


def build_cnn_multitask():
    """Shared 1D-CNN backbone + binary + intensity heads."""
    inp = layers.Input(shape=INPUT_SHAPE)

    x = layers.BatchNormalization()(inp)

    x = layers.Conv1D(32, 7, padding="same", kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Dropout(0.2)(x)

    x = layers.Conv1D(64, 5, padding="same", kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Dropout(0.2)(x)

    x = layers.Conv1D(128, 3, padding="same", kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(0.3)(x)

    shared = layers.Dense(64, activation="relu", kernel_regularizer=regularizers.l2(1e-4))(x)

    # Binary head
    binary_out = layers.Dense(1, activation="sigmoid", name="binary")(shared)

    # Intensity head (3-level ordinal)
    intensity_out = layers.Dense(
        INTENSITY_CLASSES, activation="softmax", name="intensity"
    )(shared)

    model = keras.Model(inputs=inp, outputs=[binary_out, intensity_out])
    return model


def intensity_loss(y_true, y_pred):
    """Crossentropy for intensity, zero loss for healthy samples (label=-1)."""
    mask = tf.cast(tf.not_equal(y_true, -1), tf.float32)
    y_true_clamped = tf.clip_by_value(y_true, 0, 2)
    y_true_cat = tf.one_hot(tf.cast(y_true_clamped, tf.int32), depth=INTENSITY_CLASSES)
    loss = keras.losses.categorical_crossentropy(y_true_cat, y_pred)
    loss = loss * mask
    return tf.reduce_sum(loss) / tf.maximum(tf.reduce_sum(mask), 1.0)


def find_best_threshold(model, x_val, y_val_binary, y_val_intensity, n_thresh=99):
    """Sweep thresholds 0.01–0.99, return best threshold and F1."""
    pred_bin = model.predict(x_val, verbose=0)[0].ravel()
    best_f1 = 0.0
    best_th = 0.5

    thresholds = np.linspace(0.01, 0.99, n_thresh)
    for th in thresholds:
        pred_class = (pred_bin >= th).astype(int)
        tp = ((pred_class == 1) & (y_val_binary == 1)).sum()
        fp = ((pred_class == 1) & (y_val_binary == 0)).sum()
        fn = ((pred_class == 0) & (y_val_binary == 1)).sum()
        prec = tp / (tp + fp + 1e-12)
        rec = tp / (tp + fn + 1e-12)
        f1 = 2 * prec * rec / (prec + rec + 1e-12)
        if f1 > best_f1:
            best_f1 = f1
            best_th = th

    return float(best_th), float(best_f1)


def main():
    ensure_dir(OUTPUT_DIR)

    all_sensor_results = {}
    gateway_data = {}  # for fusion: sensor outputs on test set

    for sn in SENSOR_NODES:
        print(f"\n{'='*60}")
        print(f"Training Model B — Sensor {sn}")
        print(f"{'='*60}")

        npz = np.load(os.path.join(WINDOWS_DIR, f"sensor_{sn}.npz"))
        data = npz["data"]
        labels = npz["labels"]

        train_idx = npz["train_idx"]
        val_idx = npz["val_idx"]
        test_idx = npz["test_idx"]

        x_train, y_train = data[train_idx], labels[train_idx]
        x_val, y_val = data[val_idx], labels[val_idx]
        x_test, y_test = data[test_idx], labels[test_idx]

        # Build multi-task targets
        y_train_bin = np.array([BINARY_MAP[l] for l in y_train], dtype=np.float32)
        y_train_int = np.array([INTENSITY_MAP[l] for l in y_train], dtype=np.float32)
        y_val_bin = np.array([BINARY_MAP[l] for l in y_val], dtype=np.float32)
        y_val_int = np.array([INTENSITY_MAP[l] for l in y_val], dtype=np.float32)
        y_test_bin = np.array([BINARY_MAP[l] for l in y_test], dtype=np.float32)
        y_test_int = np.array([INTENSITY_MAP[l] for l in y_test], dtype=np.float32)

        print(f"  Train: {len(x_train)}, Val: {len(x_val)}, Test: {len(x_test)}")
        print(f"  Binary dist train: {np.bincount(y_train_bin.astype(int))}")
        print(f"  Intensity dist train: "
              f"{np.bincount(y_train_int[y_train_int >= 0].astype(int), minlength=3)}")

        model = build_cnn_multitask()
        model.summary()

        optimizer = keras.optimizers.AdamW(
            learning_rate=LEARNING_RATE, weight_decay=1e-4,
        )
        model.compile(
            optimizer=optimizer,
            loss={
                "binary": "binary_crossentropy",
                "intensity": intensity_loss,
            },
            loss_weights={"binary": 1.0, "intensity": 0.5},
            metrics={
                "binary": ["accuracy"],
                "intensity": ["accuracy"],
            },
        )

        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=PATIENCE,
                restore_best_weights=True, min_delta=1e-5,
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=0.5, patience=8,
                min_lr=1e-6, verbose=1,
            ),
            keras.callbacks.ModelCheckpoint(
                os.path.join(OUTPUT_DIR, f"sensor_{sn}_best.keras"),
                monitor="val_loss", save_best_only=True,
            ),
        ]

        model.fit(
            x_train, {"binary": y_train_bin, "intensity": y_train_int},
            validation_data=(x_val, {
                "binary": y_val_bin, "intensity": y_val_int,
            }),
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            callbacks=callbacks,
            verbose=2,
        )

        # Find best binary threshold on val set
        best_th, val_f1 = find_best_threshold(model, x_val, y_val_bin, y_val_int)
        print(f"  Best binary threshold: {best_th:.3f} (val F1: {val_f1:.4f})")

        # Evaluate on test set
        pred_bin, pred_int = model.predict(x_test, verbose=0)
        pred_bin_class = (pred_bin.ravel() >= best_th).astype(int)
        pred_int_class = np.argmax(pred_int, axis=1)

        # Binary metrics on test
        tp = ((pred_bin_class == 1) & (y_test_bin == 1)).sum()
        fp = ((pred_bin_class == 1) & (y_test_bin == 0)).sum()
        fn = ((pred_bin_class == 0) & (y_test_bin == 1)).sum()
        tn = ((pred_bin_class == 0) & (y_test_bin == 0)).sum()
        bin_prec = tp / (tp + fp + 1e-12)
        bin_rec = tp / (tp + fn + 1e-12)
        bin_f1 = 2 * bin_prec * bin_rec / (bin_prec + bin_rec + 1e-12)
        bin_acc = (tp + tn) / (tp + tn + fp + fn)

        # Intensity metrics (only on damaged samples)
        damaged_mask = y_test_bin == 1
        if damaged_mask.sum() > 0:
            int_acc = (pred_int_class[damaged_mask] == y_test_int[damaged_mask]).mean()
        else:
            int_acc = 0.0

        result = {
            "sensor_node": sn,
            "binary_threshold": best_th,
            "binary_accuracy": float(bin_acc),
            "binary_f1": float(bin_f1),
            "intensity_accuracy": float(int_acc),
            "confusion_matrix_binary": {
                "tp": int(tp), "fp": int(fp),
                "fn": int(fn), "tn": int(tn),
            },
        }
        all_sensor_results[str(sn)] = result

        # Store gateway data
        gateway_data[str(sn)] = {
            "test_binary": pred_bin_class.tolist(),
            "test_intensity_raw": pred_int.tolist(),  # softmax outputs
            "test_labels": y_test.tolist(),
            "test_intensity_labels": y_test_int.tolist(),
        }

        print(f"  Test binary acc: {bin_acc:.4f}, F1: {bin_f1:.4f}")
        print(f"  Test intensity acc (damaged only): {int_acc:.4f}")

        # Save model
        model.save(os.path.join(OUTPUT_DIR, f"sensor_{sn}_model.keras"))

        # Save threshold for TFLite deployment
        with open(os.path.join(OUTPUT_DIR, f"sensor_{sn}_threshold.json"), "w") as f:
            json.dump({"binary_threshold": best_th}, f)

    # Save gateway data
    np.savez_compressed(
        os.path.join(OUTPUT_DIR, "gateway_data.npz"),
        **gateway_data,
    )

    # Save all results
    with open(os.path.join(OUTPUT_DIR, "results.json"), "w") as f:
        json.dump(all_sensor_results, f, indent=2)

    print(f"\nResults saved to {OUTPUT_DIR}/")
    print("Done.")


if __name__ == "__main__":
    main()

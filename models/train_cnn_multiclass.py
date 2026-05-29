"""
Model A: Per-sensor multiclass 1D-CNN (4 classes).

USE_TF_BACKEND = True  # set KERAS_BACKEND=tensorflow before running

Trains 7 independent 1D-CNN classifiers (one per sensor)
on pre-computed windowed data. Uses early stopping,
cosine annealing LR, per-sensor F1 tracking.

Usage:
    python models/train_cnn_multiclass.py
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
OUTPUT_DIR = "outputs/cnn_multiclass"
SEED = 42
EPOCHS = 200
PATIENCE = 20
BATCH_SIZE = 64
LEARNING_RATE = 1e-3

SENSOR_NODES = [34, 37, 39, 44, 47, 49, 54]
NUM_CLASSES = 4
INPUT_SHAPE = (400, 3)

tf.random.set_seed(SEED)
np.random.seed(SEED)


def build_cnn_multiclass():
    """1D-CNN with residual blocks for 4-class classification."""
    def res_block(x, filters, kernel_size, stride=1, dilation=1):
        shortcut = x
        x = layers.Conv1D(filters, kernel_size, strides=stride, padding="same",
                          dilation_rate=dilation,
                          kernel_regularizer=regularizers.l2(1e-4))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.Conv1D(filters, kernel_size, padding="same",
                          kernel_regularizer=regularizers.l2(1e-4))(x)
        x = layers.BatchNormalization()(x)
        if stride > 1 or shortcut.shape[-1] != filters:
            shortcut = layers.Conv1D(filters, 1, strides=stride,
                                     kernel_regularizer=regularizers.l2(1e-4))(shortcut)
            shortcut = layers.BatchNormalization()(shortcut)
        x = layers.add([x, shortcut])
        x = layers.Activation("relu")(x)
        return x

    inp = layers.Input(shape=INPUT_SHAPE)

    x = layers.Conv1D(16, 7, padding="same",
                      kernel_regularizer=regularizers.l2(1e-4))(inp)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = res_block(x, 16, 7)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Dropout(0.2)(x)

    x = res_block(x, 32, 5)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Dropout(0.2)(x)

    x = res_block(x, 64, 3)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(0.3)(x)

    out = layers.Dense(NUM_CLASSES, activation="softmax",
                       kernel_regularizer=regularizers.l2(1e-4))(x)

    model = keras.Model(inputs=inp, outputs=out)
    return model


def main():
    ensure_dir(OUTPUT_DIR)

    all_sensor_results = {}
    overall_preds = {}
    overall_labels = {}

    for sn in SENSOR_NODES:
        print(f"\n{'='*60}")
        print(f"Training Model A — Sensor {sn}")
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

        y_train_cat = keras.utils.to_categorical(y_train, NUM_CLASSES)
        y_val_cat = keras.utils.to_categorical(y_val, NUM_CLASSES)
        y_test_cat = keras.utils.to_categorical(y_test, NUM_CLASSES)

        print(f"  Train: {len(x_train)}, Val: {len(x_val)}, Test: {len(x_test)}")
        print(f"  Train dist: {np.bincount(y_train, minlength=NUM_CLASSES)}")
        print(f"  Val dist:   {np.bincount(y_val, minlength=NUM_CLASSES)}")

        # Build model
        model = build_cnn_multiclass()
        model.summary()

        optimizer = keras.optimizers.AdamW(
            learning_rate=LEARNING_RATE, weight_decay=1e-4,
        )
        model.compile(
            optimizer=optimizer,
            loss="categorical_crossentropy",
            metrics=["accuracy"],
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

        history = model.fit(
            x_train, y_train_cat,
            validation_data=(x_val, y_val_cat),
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            callbacks=callbacks,
            verbose=2,
        )

        # Evaluate on test
        test_metrics = model.evaluate(x_test, y_test_cat, verbose=0)
        if isinstance(test_metrics, (list, tuple)):
            test_loss, test_acc = test_metrics[0], test_metrics[1]
        else:
            test_loss, test_acc = test_metrics, 0.0
        y_pred = model.predict(x_test, verbose=0)
        y_pred_class = np.argmax(y_pred, axis=1)
        cm = tf.math.confusion_matrix(y_test, y_pred_class, num_classes=NUM_CLASSES)

        per_class_f1 = []
        for c in range(NUM_CLASSES):
            tp = cm[c, c].numpy()
            fp = cm[:, c].numpy().sum() - tp
            fn = cm[c, :].numpy().sum() - tp
            prec = tp / (tp + fp + 1e-12)
            rec = tp / (tp + fn + 1e-12)
            f1 = 2 * prec * rec / (prec + rec + 1e-12)
            per_class_f1.append(float(f1))

        result = {
            "sensor_node": sn,
            "test_loss": float(test_loss),
            "test_accuracy": float(test_acc),
            "test_f1": float(np.mean(per_class_f1)),
            "per_class_f1": per_class_f1,
            "confusion_matrix": cm.numpy().tolist(),
        }
        all_sensor_results[str(sn)] = result

        overall_preds[str(sn)] = y_pred_class.tolist()
        overall_labels[str(sn)] = y_test.tolist()

        macro_f1 = float(np.mean(per_class_f1))
        print(f"  Test accuracy: {test_acc:.4f}, Macro F1: {macro_f1:.4f}")
        print(f"  Per-class F1: {[f'{f:.4f}' for f in per_class_f1]}")
        print(f"  Confusion matrix:\n{cm.numpy()}")

        # Save model in SavedModel format too
        model.save(os.path.join(OUTPUT_DIR, f"sensor_{sn}_model.keras"))

    # Ensemble evaluation — average logits across sensors
    # For ensemble: load each best model, predict logits on its test windows,
    # average softmax across sensors, then argmax.
    print(f"\n{'='*60}")
    print("Ensemble evaluation (logit averaging across sensors)")
    print(f"{'='*60}")

    ensemble_logits = None
    ensemble_labels = None
    sensor_weights = {}
    total_w = 0

    for sn in SENSOR_NODES:
        model_path = os.path.join(OUTPUT_DIR, f"sensor_{sn}_best.keras")
        if not os.path.exists(model_path):
            model_path = os.path.join(OUTPUT_DIR, f"sensor_{sn}_model.keras")
        model = keras.models.load_model(model_path)

        npz = np.load(os.path.join(WINDOWS_DIR, f"sensor_{sn}.npz"))
        data = npz["data"]
        labels = npz["labels"]
        test_idx = npz["test_idx"]
        x_test = data[test_idx]
        y_test = labels[test_idx]

        logits = model.predict(x_test, verbose=0)

        if ensemble_logits is None:
            ensemble_logits = logits
            ensemble_labels = y_test
        else:
            ensemble_logits += logits

    ensemble_preds = np.argmax(ensemble_logits, axis=1)
    ensemble_acc = (ensemble_preds == ensemble_labels).mean()
    ensemble_cm = tf.math.confusion_matrix(
        ensemble_labels, ensemble_preds, num_classes=NUM_CLASSES
    ).numpy()

    per_class_f1_ens = []
    for c in range(NUM_CLASSES):
        tp = ensemble_cm[c, c]
        fp = ensemble_cm[:, c].sum() - tp
        fn = ensemble_cm[c, :].sum() - tp
        prec = tp / (tp + fp + 1e-12)
        rec = tp / (tp + fn + 1e-12)
        f1 = 2 * prec * rec / (prec + rec + 1e-12)
        per_class_f1_ens.append(float(f1))

    print(f"  Ensemble test accuracy: {ensemble_acc:.4f}")
    print(f"  Per-class F1: {[f'{f:.4f}' for f in per_class_f1_ens]}")
    print(f"  Confusion matrix:\n{ensemble_cm}")

    ensemble_result = {
        "ensemble_accuracy": float(ensemble_acc),
        "per_class_f1": per_class_f1_ens,
        "confusion_matrix": ensemble_cm.tolist(),
    }

    # Save all results
    results = {
        "per_sensor": all_sensor_results,
        "ensemble": ensemble_result,
    }
    with open(os.path.join(OUTPUT_DIR, "results.json"), "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {OUTPUT_DIR}/results.json")
    print("Done.")


if __name__ == "__main__":
    main()

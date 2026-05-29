"""
Export the winning model as INT8 quantized TFLite.

Selects winner based on compare_models output, then:
1. Loads the keras model
2. Converts to TFLite with INT8 quantization (representative dataset from train set)
3. Embeds binary threshold (Model B) or softmax (Model A)
4. Saves per-sensor .tflite files + combined metadata json

Usage:
    python models/export_tflite.py [--model {a,b}]
"""

import os, sys, argparse, json

os.environ["KERAS_BACKEND"] = "tensorflow"

import numpy as np
import tensorflow as tf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from digital_twin.utils import ensure_dir

WINDOWS_DIR = "data/processed/cnn_windows"
MODEL_A_DIR = "outputs/cnn_multiclass"
MODEL_B_DIR = "outputs/cnn_binary"
OUTPUT_DIR = "outputs/tflite_export"
SENSOR_NODES = [34, 37, 39, 44, 47, 49, 54]
INPUT_SHAPE = (400, 3)


def representative_dataset_gen(data, n_samples=200):
    """Yield samples for INT8 calibration."""
    n = len(data)
    indices = np.random.choice(n, min(n_samples, n), replace=False)
    for i in indices:
        yield [data[i:i+1].astype(np.float32)]


def export_model_a():
    print("Exporting Model A (Multiclass 1D-CNN ensemble)...")
    ensure_dir(OUTPUT_DIR)

    metadata = {
        "model_type": "A_multiclass",
        "sensors": {},
        "num_classes": 4,
        "input_shape": list(INPUT_SHAPE),
    }

    for sn in SENSOR_NODES:
        model_path = os.path.join(MODEL_A_DIR, f"sensor_{sn}_best.keras")
        if not os.path.exists(model_path):
            model_path = os.path.join(MODEL_A_DIR, f"sensor_{sn}_model.keras")

        print(f"  Sensor {sn}...")
        model = tf.keras.models.load_model(model_path, compile=False)

        # Load training data for representative dataset
        npz = np.load(os.path.join(WINDOWS_DIR, f"sensor_{sn}.npz"))
        train_data = npz["data"][npz["train_idx"]]

        # Representative dataset
        def rep_gen():
            return representative_dataset_gen(train_data, n_samples=200)

        # Convert to TFLite with INT8
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = rep_gen
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8

        tflite_model = converter.convert()

        out_path = os.path.join(OUTPUT_DIR, f"sensor_{sn}.tflite")
        with open(out_path, "wb") as f:
            f.write(tflite_model)

        # Verify
        interpreter = tf.lite.Interpreter(model_content=tflite_model)
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        metadata["sensors"][str(sn)] = {
            "tflite_path": out_path,
            "input_dtype": str(input_details[0]["dtype"]),
            "input_shape": input_details[0]["shape"].tolist(),
            "output_dtype": str(output_details[0]["dtype"]),
            "output_shape": output_details[0]["shape"].tolist(),
            "input_quantization": str(input_details[0]["quantization"]),
            "output_quantization": str(output_details[0]["quantization"]),
        }
        print(f"    -> {out_path} ({len(tflite_model)} bytes)")

    json_path = os.path.join(OUTPUT_DIR, "metadata.json")
    with open(json_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"  Metadata: {json_path}")


def export_model_b():
    print("Exporting Model B (Binary + Intensity 1D-CNN)...")
    ensure_dir(OUTPUT_DIR)

    metadata = {
        "model_type": "B_binary_intensity",
        "sensors": {},
        "num_binary_classes": 2,
        "num_intensity_classes": 3,
        "input_shape": list(INPUT_SHAPE),
    }

    for sn in SENSOR_NODES:
        model_path = os.path.join(MODEL_B_DIR, f"sensor_{sn}_best.keras")
        if not os.path.exists(model_path):
            model_path = os.path.join(MODEL_B_DIR, f"sensor_{sn}_model.keras")

        # Load threshold
        th_path = os.path.join(MODEL_B_DIR, f"sensor_{sn}_threshold.json")
        with open(th_path) as f:
            threshold_data = json.load(f)
        threshold = threshold_data["binary_threshold"]

        print(f"  Sensor {sn} (threshold={threshold:.3f})...")
        model = tf.keras.models.load_model(model_path, compile=False)

        # Load training data for representative dataset
        npz = np.load(os.path.join(WINDOWS_DIR, f"sensor_{sn}.npz"))
        train_data = npz["data"][npz["train_idx"]]

        def rep_gen():
            return representative_dataset_gen(train_data, n_samples=200)

        # Convert to TFLite with INT8
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = rep_gen
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8

        tflite_model = converter.convert()

        out_path = os.path.join(OUTPUT_DIR, f"sensor_{sn}.tflite")
        with open(out_path, "wb") as f:
            f.write(tflite_model)

        # Verify
        interpreter = tf.lite.Interpreter(model_content=tflite_model)
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        metadata["sensors"][str(sn)] = {
            "tflite_path": out_path,
            "binary_threshold": threshold,
            "input_dtype": str(input_details[0]["dtype"]),
            "input_shape": input_details[0]["shape"].tolist(),
            "output_shapes": [
                od["shape"].tolist() for od in output_details
            ],
            "output_dtypes": [
                str(od["dtype"]) for od in output_details
            ],
        }
        print(f"    -> {out_path} ({len(tflite_model)} bytes)")

    json_path = os.path.join(OUTPUT_DIR, "metadata.json")
    with open(json_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"  Metadata: {json_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", choices=["a", "b"], default=None,
        help="Which model to export (default: check both, export whichever exists)",
    )
    args = parser.parse_args()

    if args.model == "a":
        export_model_a()
    elif args.model == "b":
        export_model_b()
    else:
        # Try to determine which model was trained
        a_exists = os.path.exists(os.path.join(MODEL_A_DIR, "results.json"))
        b_exists = os.path.exists(os.path.join(MODEL_B_DIR, "results.json"))

        if a_exists and b_exists:
            print("Both models trained — exporting Model A (simpler deployment)")
            export_model_a()
        elif a_exists:
            export_model_a()
        elif b_exists:
            export_model_b()
        else:
            print("No trained models found. Train models first.")
            sys.exit(1)

    print("\nDone. TFLite models ready for ESP32 deployment.")


if __name__ == "__main__":
    main()

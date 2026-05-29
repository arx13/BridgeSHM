import os
import sys
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GroupKFold

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

from models.classifier_data_loader import (
    load_features, split_by_run, prepare_classifier_data,
    save_scaler, save_label_encoder, add_sensor_onehot, get_feature_columns
)

OUTPUT_DIR = "outputs/classifiers"
RANDOM_STATE = 42


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def run_cv(df, n_splits=5):
    df = df.copy()
    add_sensor_onehot(df)
    df["group"] = df["run_id"].astype(str) + "_" + df["damage_case"]
    groups = pd.factorize(df["group"])[0]

    sensor_cols = [c for c in df.columns if c.startswith("sensor_")]
    feature_cols = get_feature_columns(df)
    feature_cols = [c for c in feature_cols if c not in sensor_cols and c != "group"] + sensor_cols

    X = df[feature_cols].values
    y_bin = df["label_binary"].values

    gkf = GroupKFold(n_splits=n_splits)

    print(f"Running {n_splits}-fold run-stratified CV ({len(np.unique(groups))} groups)...")
    cv_scores = []
    for fold, (train_idx, test_idx) in enumerate(gkf.split(X, y_bin, groups=groups)):
        rf = RandomForestClassifier(
            n_estimators=300, random_state=RANDOM_STATE,
            class_weight="balanced_subsample", n_jobs=2
        )
        rf.fit(X[train_idx], y_bin[train_idx])
        acc = accuracy_score(y_bin[test_idx], rf.predict(X[test_idx]))
        cv_scores.append(acc)
        print(f"  Fold {fold}: {acc:.4f}")

    mean_acc = np.mean(cv_scores)
    std_acc = np.std(cv_scores)
    print(f"\nRF CV Binary Accuracy: {mean_acc:.4f} ± {std_acc:.4f}")
    return mean_acc


def train_all_models(df, test_size=0.25, val_size=0.0, scale_for_mlp_svm=True):
    print("Splitting data (run-stratified)...")
    train_df, val_df, test_df = split_by_run(df, test_size=test_size, val_size=val_size)

    print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    data = prepare_classifier_data(train_df, val_df, test_df, scale=scale_for_mlp_svm)

    merged_X = np.vstack([data["X_train"], data["X_val"]]) if len(data["X_val"]) > 0 else data["X_train"]
    merged_y_bin = np.concatenate([data["y_train_bin"], data["y_val_bin"]]) if len(data["y_val_bin"]) > 0 else data["y_train_bin"]
    merged_y_multi = np.concatenate([data["y_train_multi"], data["y_val_multi"]]) if len(data["y_val_multi"]) > 0 else data["y_train_multi"]
    merged_X_scaled = np.vstack([data["X_train_scaled"], data["X_val_scaled"]]) if len(data["X_val_scaled"]) > 0 else data["X_train_scaled"]

    neg = (merged_y_bin == 0).sum()
    pos = (merged_y_bin == 1).sum()

    models = {}

    # RF (no scaling needed)
    print("\nTraining Random Forest (binary)...")
    rf_bin = RandomForestClassifier(
        n_estimators=300, random_state=RANDOM_STATE,
        class_weight="balanced_subsample", n_jobs=2
    )
    rf_bin.fit(merged_X, merged_y_bin)
    models["rf_binary"] = rf_bin
    joblib.dump(rf_bin, os.path.join(OUTPUT_DIR, "rf_binary.pkl"))

    print("Training Random Forest (multiclass)...")
    rf_multi = RandomForestClassifier(
        n_estimators=300, random_state=RANDOM_STATE,
        class_weight="balanced_subsample", n_jobs=2
    )
    rf_multi.fit(merged_X, merged_y_multi)
    models["rf_multiclass"] = rf_multi
    joblib.dump(rf_multi, os.path.join(OUTPUT_DIR, "rf_multiclass.pkl"))

    # XGBoost
    if HAS_XGB:
        print("\nTraining XGBoost (binary)...")
        xgb_bin = xgb.XGBClassifier(
            n_estimators=300, max_depth=8, learning_rate=0.05,
            subsample=0.7, colsample_bytree=0.7,
            reg_alpha=0.1, reg_lambda=1.0,
            scale_pos_weight=neg / pos,
            random_state=RANDOM_STATE, n_jobs=2, verbosity=0
        )
        xgb_bin.fit(merged_X, merged_y_bin)
        models["xgb_binary"] = xgb_bin
        joblib.dump(xgb_bin, os.path.join(OUTPUT_DIR, "xgb_binary.pkl"))

        print("Training XGBoost (multiclass)...")
        xgb_multi = xgb.XGBClassifier(
            n_estimators=300, max_depth=8, learning_rate=0.05,
            subsample=0.7, colsample_bytree=0.7,
            reg_alpha=0.1, reg_lambda=1.0,
            random_state=RANDOM_STATE, n_jobs=2, verbosity=0
        )
        xgb_multi.fit(merged_X, merged_y_multi)
        models["xgb_multiclass"] = xgb_multi
        joblib.dump(xgb_multi, os.path.join(OUTPUT_DIR, "xgb_multiclass.pkl"))

    # MLP (needs scaled data)
    print("\nTraining MLP (binary)...")
    mlp_bin = MLPClassifier(
        hidden_layer_sizes=(96, 48, 24), max_iter=300,
        random_state=RANDOM_STATE, early_stopping=True
    )
    mlp_bin.fit(merged_X_scaled, merged_y_bin)
    models["mlp_binary"] = mlp_bin
    joblib.dump(mlp_bin, os.path.join(OUTPUT_DIR, "mlp_binary.pkl"))

    print("Training MLP (multiclass)...")
    mlp_multi = MLPClassifier(
        hidden_layer_sizes=(96, 48, 24), max_iter=300,
        random_state=RANDOM_STATE, early_stopping=True
    )
    mlp_multi.fit(merged_X_scaled, merged_y_multi)
    models["mlp_multiclass"] = mlp_multi
    joblib.dump(mlp_multi, os.path.join(OUTPUT_DIR, "mlp_multiclass.pkl"))

    # SVM (needs scaled data)
    print("\nTraining SVM (binary)...")
    svm_bin = SVC(kernel="rbf", probability=True, random_state=RANDOM_STATE, class_weight="balanced")
    svm_bin.fit(merged_X_scaled, merged_y_bin)
    models["svm_binary"] = svm_bin
    joblib.dump(svm_bin, os.path.join(OUTPUT_DIR, "svm_binary.pkl"))

    save_scaler(data["scaler"], os.path.join(OUTPUT_DIR, "classifier_scaler.pkl"))
    save_label_encoder(data["label_encoder"], os.path.join(OUTPUT_DIR, "label_encoder.pkl"))

    print("\n========== TRAINING COMPLETE ==========")
    for name in sorted(models.keys()):
        print(f"  {name}: {os.path.join(OUTPUT_DIR, name)}.pkl")

    return models, data


def main():
    ensure_dir(OUTPUT_DIR)

    print("Loading features...")
    df = load_features()
    print(f"Loaded {len(df)} samples with {len(get_feature_columns(df))} features")

    # Run CV to estimate performance
    cv_acc = run_cv(df, n_splits=5)

    # Train final models
    models, data = train_all_models(df, test_size=0.25, val_size=0.0)

    # Quick test evaluation
    from sklearn.metrics import classification_report, confusion_matrix
    X_test = data["X_test"]
    y_test = data["y_test_bin"]
    rf_bin = models["rf_binary"]
    y_pred = rf_bin.predict(X_test)
    print(f"\nTest accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred, target_names=["Healthy", "Damaged"]))


if __name__ == "__main__":
    main()

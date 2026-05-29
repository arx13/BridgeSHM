import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import joblib

FEATURE_EXCLUDE_COLS = [
    "window_id", "damage_case", "run_id", "window_pos",
    "time_start", "time_end", "label_binary", "sensor_node"
]

META_COLS = FEATURE_EXCLUDE_COLS + ["damage_case_id"]


def load_features(data_path="data/processed/3d_windowed_features.csv"):
    df = pd.read_csv(data_path)
    df = df.replace([np.inf, -np.inf], np.nan).fillna(0)
    return df


def get_feature_columns(df):
    return [col for col in df.columns if col not in META_COLS]


def add_sensor_onehot(df):
    for sn in sorted(df["sensor_node"].unique()):
        existing = [c for c in df.columns if c.startswith("sensor_")]
        if f"sensor_{sn}" not in existing:
            df[f"sensor_{sn}"] = (df["sensor_node"] == sn).astype(int)
    return df


def split_by_run(df, test_size=0.2, val_size=0.2, random_state=42):
    runs = df[["run_id", "damage_case"]].drop_duplicates().reset_index(drop=True)
    runs["key"] = runs["run_id"].astype(str) + "_" + runs["damage_case"]

    # Split: train / temp (val+test)
    train_runs, temp_runs = train_test_split(
        runs, test_size=test_size + val_size,
        random_state=random_state, stratify=runs["damage_case"]
    )
    # Split temp into val / test
    if val_size > 0:
        val_ratio = val_size / (test_size + val_size)
        val_runs, test_runs = train_test_split(
            temp_runs, test_size=test_size / (test_size + val_size),
            random_state=random_state, stratify=temp_runs["damage_case"]
        )
    else:
        val_runs = temp_runs.iloc[:0]
        test_runs = temp_runs

    df["key"] = df["run_id"].astype(str) + "_" + df["damage_case"]

    train = df[df["key"].isin(train_runs["key"].values)].copy()
    test = df[df["key"].isin(test_runs["key"].values)].copy()
    val = df[df["key"].isin(val_runs["key"].values)].copy() if len(val_runs) > 0 else train.iloc[:0].copy()

    for d in [train, val, test]:
        d.drop(columns=["key"], inplace=True, errors="ignore")

    return train, val, test


def prepare_classifier_data(train_df, val_df, test_df, scale=True):
    train_df = add_sensor_onehot(train_df)
    val_df = add_sensor_onehot(val_df)
    test_df = add_sensor_onehot(test_df)

    feature_cols = get_feature_columns(train_df)
    sensor_cols = [c for c in train_df.columns if c.startswith("sensor_")]
    feature_cols = [c for c in feature_cols if c not in sensor_cols] + sensor_cols

    X_train = train_df[feature_cols].values
    X_val = val_df[feature_cols].values if len(val_df) > 0 else np.empty((0, len(feature_cols)))
    X_test = test_df[feature_cols].values

    y_train_bin = train_df["label_binary"].values
    y_val_bin = val_df["label_binary"].values if len(val_df) > 0 else np.array([])
    y_test_bin = test_df["label_binary"].values

    # Use damage_case_id if available (preserves label scheme), else fallback to LabelEncoder
    if "damage_case_id" in train_df.columns:
        y_train_multi = train_df["damage_case_id"].values
        y_val_multi = val_df["damage_case_id"].values if len(val_df) > 0 else np.array([])
        y_test_multi = test_df["damage_case_id"].values
        label_encoder = LabelEncoder()
        label_encoder.fit(train_df["damage_case_id"].values)
    else:
        label_encoder = LabelEncoder()
        y_train_multi = label_encoder.fit_transform(train_df["damage_case"])
        y_val_multi = label_encoder.transform(val_df["damage_case"]) if len(val_df) > 0 else np.array([])
        y_test_multi = label_encoder.transform(test_df["damage_case"])

    scaler = StandardScaler()
    if scale and len(X_train) > 0:
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val) if len(X_val) > 0 else X_val
        X_test_scaled = scaler.transform(X_test)
    else:
        X_train_scaled, X_val_scaled, X_test_scaled = X_train, X_val, X_test

    return {
        "X_train": X_train, "X_val": X_val, "X_test": X_test,
        "X_train_scaled": X_train_scaled,
        "X_val_scaled": X_val_scaled,
        "X_test_scaled": X_test_scaled,
        "y_train_bin": y_train_bin, "y_val_bin": y_val_bin, "y_test_bin": y_test_bin,
        "y_train_multi": y_train_multi,
        "y_val_multi": y_val_multi,
        "y_test_multi": y_test_multi,
        "feature_cols": feature_cols,
        "scaler": scaler,
        "label_encoder": label_encoder
    }


def save_scaler(scaler, path="outputs/classifiers/classifier_scaler.pkl"):
    joblib.dump(scaler, path)


def save_label_encoder(label_encoder, path="outputs/classifiers/label_encoder.pkl"):
    joblib.dump(label_encoder, path)

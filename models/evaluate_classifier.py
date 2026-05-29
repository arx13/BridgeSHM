import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import joblib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    accuracy_score, f1_score, roc_curve, auc, precision_recall_curve
)

from models.classifier_data_loader import load_features, split_by_run, prepare_classifier_data

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 300,
    "font.size": 10, "axes.titlesize": 12, "axes.labelsize": 11,
    "legend.fontsize": 9, "figure.facecolor": "white", "axes.facecolor": "white"
})

OUTPUT_DIR = "outputs/classifiers"
DAMAGE_COLORS = {"healthy": "#2ecc71", "minor_midspan_damage": "#f39c12",
                 "moderate_midspan_damage": "#e74c3c", "support_damage": "#9b59b6"}

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def plot_confusion_matrix(cm, labels, title, save_path):
    cm_norm = cm.astype("float") / cm.sum(axis=1, keepdims=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Raw counts
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax1,
                xticklabels=labels, yticklabels=labels, cbar=False,
                annot_kws={"size": 13, "weight": "bold"})
    ax1.set_xlabel("Predicted", fontsize=10)
    ax1.set_ylabel("True", fontsize=10)
    ax1.set_title("Confusion Matrix (Counts)")

    # Percentages
    sns.heatmap(cm_norm, annot=True, fmt=".1%", cmap="Blues", ax=ax2,
                xticklabels=labels, yticklabels=labels, cbar_kws={"label": "Proportion"},
                annot_kws={"size": 13, "weight": "bold"})
    ax2.set_xlabel("Predicted", fontsize=10)
    ax2.set_ylabel("True", fontsize=10)
    ax2.set_title("Confusion Matrix (Row %)")

    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()

def plot_roc_curves(models_dict, X, y_true, save_path):
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, label="Random (AUC=0.5)")

    for name, model in models_dict.items():
        if hasattr(model, "predict_proba"):
            y_score = model.predict_proba(X)[:, 1]
            fpr, tpr, _ = roc_curve(y_true, y_score)
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, linewidth=1.5, label=f"{name} (AUC={roc_auc:.3f})")

    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves — Binary Classification")
    ax.legend(loc="lower right", framealpha=0.8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()

def plot_precision_recall_curves(models_dict, X, y_true, save_path):
    fig, ax = plt.subplots(figsize=(8, 7))

    for name, model in models_dict.items():
        if hasattr(model, "predict_proba"):
            y_score = model.predict_proba(X)[:, 1]
            prec, rec, _ = precision_recall_curve(y_true, y_score)
            avg_prec = auc(rec, prec)
            ax.plot(rec, prec, linewidth=1.5, label=f"{name} (AP={avg_prec:.3f})")

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curves — Binary Classification")
    ax.legend(loc="lower left", framealpha=0.8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()

def plot_per_class_accuracy(models_dict, X_bin, y_bin, X_multi, y_multi,
                            class_names_bin, class_names_multi, save_path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Binary per-class accuracy
    bin_data = {}
    for name, model in models_dict.get("binary", {}).items():
        y_pred = model.predict(X_bin)
        cm = confusion_matrix(y_bin, y_pred)
        class_acc = cm.diagonal() / cm.sum(axis=1)
        bin_data[name] = class_acc

    x = np.arange(len(class_names_bin))
    width = 0.15
    for i, (name, accs) in enumerate(bin_data.items()):
        ax1.bar(x + i * width, accs, width, alpha=0.75, label=name[:10])
    ax1.set_xticks(x + width * (len(bin_data) - 1) / 2)
    ax1.set_xticklabels(class_names_bin)
    ax1.set_ylabel("Per-Class Accuracy")
    ax1.set_title("Binary: Per-Class Accuracy")
    ax1.legend(framealpha=0.8, fontsize=7)
    ax1.set_ylim(0, 1)
    ax1.grid(True, alpha=0.3, axis="y")

    # Multiclass per-class accuracy
    multi_data = {}
    for name, model in models_dict.get("multiclass", {}).items():
        y_pred = model.predict(X_multi)
        cm = confusion_matrix(y_multi, y_pred)
        class_acc = cm.diagonal() / cm.sum(axis=1)
        multi_data[name] = class_acc

    x2 = np.arange(len(class_names_multi))
    for i, (name, accs) in enumerate(multi_data.items()):
        ax2.bar(x2 + i * width, accs, width, alpha=0.75, label=name[:10])
    ax2.set_xticks(x2 + width * (len(multi_data) - 1) / 2)
    ax2.set_xticklabels([c.replace("_", " ").title() for c in class_names_multi], fontsize=8)
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=15, ha="right")
    ax2.set_ylabel("Per-Class Accuracy")
    ax2.set_title("Multiclass: Per-Class Accuracy")
    ax2.legend(framealpha=0.8, fontsize=7)
    ax2.set_ylim(0, 1)
    ax2.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()

def plot_feature_importance(feature_cols, importances, save_path):
    sorted_idx = np.argsort(importances)[::-1][:20]
    top_features = [feature_cols[i] for i in sorted_idx]
    top_importances = importances[sorted_idx]
    cumsum = np.cumsum(top_importances) / top_importances.sum()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    colors = plt.cm.YlOrRd(top_importances / top_importances.max())
    bars = ax1.barh(range(20), top_importances[::-1], color=colors[::-1], edgecolor="black", linewidth=0.5)
    ax1.set_yticks(range(20))
    ax1.set_yticklabels([n.replace("_", " ")[:30] for n in top_features[::-1]], fontsize=8)
    ax1.set_xlabel("Importance")
    ax1.set_title("Top 20 Feature Importances (RF Binary)")
    ax1.invert_yaxis()

    ax2.plot(range(1, 21), cumsum, marker="o", linewidth=1.5, color="#2c3e50")
    ax2.fill_between(range(1, 21), 0, cumsum, alpha=0.15, color="#2c3e50")
    ax2.axhline(0.5, color="red", linestyle="--", linewidth=0.8, alpha=0.5)
    ax2.text(15, 0.52, "50% threshold", fontsize=9, color="red", alpha=0.6)
    ax2.set_xlabel("Number of Features")
    ax2.set_ylabel("Cumulative Importance")
    ax2.set_title("Cumulative Feature Importance")
    ax2.set_xticks(range(1, 21))
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()

def evaluate_model(name, model, X, y_true, class_names, task="binary", proba=False):
    y_pred = model.predict(X)
    acc = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="weighted" if task == "multiclass" else "binary")

    print(f"\n========== {name.upper()} ({task.upper()}) ==========")
    print(f"Accuracy: {acc:.4f}  F1: {f1:.4f}")
    print("\nConfusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names, zero_division=0))

    auc_val = None
    if proba and hasattr(model, "predict_proba"):
        try:
            if task == "binary":
                y_score = model.predict_proba(X)[:, 1]
                auc_val = roc_auc_score(y_true, y_score)
            else:
                y_score = model.predict_proba(X)
                auc_val = roc_auc_score(y_true, y_score, multi_class="ovr")
            print(f"ROC AUC: {auc_val:.4f}")
        except Exception:
            pass
    print("========================================")
    return cm, auc_val, acc

def main():
    ensure_dir(OUTPUT_DIR)
    print("Loading features...")
    df = load_features()
    print("Splitting (80/20 run-stratified)...")
    train_df, val_df, test_df = split_by_run(df, test_size=0.2, val_size=0.0)
    data = prepare_classifier_data(train_df, val_df, test_df, scale=True)

    X_test = data["X_test"]
    X_test_scaled = data["X_test_scaled"]
    y_test_bin = data["y_test_bin"]
    y_test_multi = data["y_test_multi"]
    label_encoder = data["label_encoder"]
    class_names_multi = list(label_encoder.classes_)

    model_configs = [
        ("Random Forest", "rf_binary.pkl", "rf_multiclass.pkl", False),
        ("MLP", "mlp_binary.pkl", "mlp_multiclass.pkl", True),
        ("SVM", "svm_binary.pkl", "svm_multiclass.pkl", True),
        ("XGBoost", "xgb_binary.pkl", "xgb_multiclass.pkl", False),
    ]

    results = {}
    bin_models = {}
    multi_models = {}

    for name, bin_file, multi_file, use_scaled in model_configs:
        bin_path = os.path.join(OUTPUT_DIR, bin_file)
        if not os.path.exists(bin_path):
            print(f"\nSkipping {name} (model not found)")
            continue
        X = X_test_scaled if use_scaled else X_test

        model_bin = joblib.load(bin_path)
        cm_bin, auc_bin, acc_bin = evaluate_model(
            f"{name} Binary", model_bin, X, y_test_bin,
            ["Healthy", "Damaged"], task="binary", proba=True
        )
        plot_confusion_matrix(
            cm_bin, ["Healthy", "Damaged"], f"{name} — Binary",
            os.path.join(OUTPUT_DIR, f"{name.lower().replace(' ', '_')}_binary_cm.png")
        )
        results[f"{name}_binary"] = acc_bin
        bin_models[name] = model_bin

        multi_path = os.path.join(OUTPUT_DIR, multi_file)
        if os.path.exists(multi_path):
            model_multi = joblib.load(multi_path)
            cm_multi, auc_multi, acc_multi = evaluate_model(
                f"{name} Multiclass", model_multi, X, y_test_multi,
                class_names_multi, task="multiclass"
            )
            plot_confusion_matrix(
                cm_multi, class_names_multi, f"{name} — Multiclass",
                os.path.join(OUTPUT_DIR, f"{name.lower().replace(' ', '_')}_multiclass_cm.png")
            )
            results[f"{name}_multiclass"] = acc_multi
            multi_models[name] = model_multi

    # ROC + PR curves
    plot_roc_curves(bin_models, X_test, y_test_bin,
                    os.path.join(OUTPUT_DIR, "roc_curves.png"))
    plot_precision_recall_curves(bin_models, X_test, y_test_bin,
                                 os.path.join(OUTPUT_DIR, "precision_recall_curves.png"))

    # Per-class accuracy
    plot_per_class_accuracy(
        {"binary": bin_models, "multiclass": multi_models},
        X_test, y_test_bin, X_test, y_test_multi,
        ["Healthy", "Damaged"], class_names_multi,
        os.path.join(OUTPUT_DIR, "per_class_accuracy.png")
    )

    # Feature importance
    rf_path = os.path.join(OUTPUT_DIR, "rf_binary.pkl")
    if os.path.exists(rf_path):
        rf_bin = joblib.load(rf_path)
        plot_feature_importance(data["feature_cols"], rf_bin.feature_importances_,
                                os.path.join(OUTPUT_DIR, "rf_feature_importance.png"))

    print("\n========== SUMMARY ==========")
    for name, acc in sorted(results.items(), key=lambda x: -x[1]):
        print(f"  {name:30s}: {acc:.4f}")
    print("=============================")
    print("\nSaved plots:")
    for f in os.listdir(OUTPUT_DIR):
        if f.endswith(".png"):
            print(f"  {OUTPUT_DIR}/{f}")

if __name__ == "__main__":
    main()

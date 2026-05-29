import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 300,
    "font.size": 10, "axes.titlesize": 12, "axes.labelsize": 11,
    "legend.fontsize": 9, "figure.facecolor": "white", "axes.facecolor": "white"
})

DAMAGE_COLORS = {"healthy": "#2ecc71", "minor_midspan_damage": "#f39c12",
                 "moderate_midspan_damage": "#e74c3c", "support_damage": "#9b59b6"}
DAMAGE_ORDER = ["healthy", "minor_midspan_damage", "moderate_midspan_damage", "support_damage"]
SENSOR_POSITIONS = {67: 0, 72: 5, 77: 10, 87: 20, 97: 30, 107: 35, 117: 40}

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def aggregate_run_level(df, feat_cols=None):
    """Aggregate per (run_id, damage_case): median across all windows/sensors."""
    if feat_cols is None:
        meta = {"window_id", "damage_case", "run_id", "sensor_node",
                "window_pos", "time_start", "time_end", "label_binary"}
        feat_cols = [c for c in df.columns if c not in meta]
    run_agg = df.groupby(["run_id", "damage_case"])[feat_cols].median().reset_index()
    labels = df[["run_id", "damage_case", "label_binary"]].drop_duplicates()
    run_agg = run_agg.merge(labels, on=["run_id", "damage_case"])
    return run_agg


def plot_damage_separation(feature_df, save_dir="outputs/validation"):
    """t-SNE / PCA visualization showing damage case separation."""
    ensure_dir(save_dir)
    from sklearn.decomposition import PCA

    meta = {"window_id", "damage_case", "run_id", "sensor_node",
            "window_pos", "time_start", "time_end", "label_binary", "key", "group"}
    feat_cols = [c for c in feature_df.columns if c not in meta]

    run_agg = aggregate_run_level(feature_df, feat_cols)
    X = run_agg[[c for c in run_agg.columns if c not in {"run_id", "damage_case", "label_binary"}]].values
    y = run_agg["damage_case"].values

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    fig, ax = plt.subplots(figsize=(10, 7))
    for dc in DAMAGE_ORDER:
        mask = y == dc
        ax.scatter(X_pca[mask, 0], X_pca[mask, 1], c=DAMAGE_COLORS[dc],
                   label=dc.replace("_", " ").title(), s=60, edgecolors="black",
                   linewidths=0.5, alpha=0.8)
        # Label each point with run_id
        for i in np.where(mask)[0]:
            ax.annotate(str(int(run_agg.iloc[i]["run_id"])),
                        (X_pca[i, 0], X_pca[i, 1]), fontsize=7, alpha=0.7)

    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
    ax.set_title("PCA: Run-Level Feature Separation by Damage Case")
    ax.legend(framealpha=0.8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/pca_damage_separation.png", bbox_inches="tight")
    plt.close()

    return X_pca, y


def plot_spatial_rms_pattern(feature_df, save_dir="outputs/validation"):
    """Normalized RMS pattern across sensors for each damage case."""
    ensure_dir(save_dir)
    sensors = sorted(feature_df["sensor_node"].unique())

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for signal_idx, signal in enumerate(["acceleration", "displacement"]):
        ax = axes[signal_idx]
        norm_cols = [f"{signal}_norm_rms_{i}" for i in range(7)]

        run_agg = aggregate_run_level(feature_df, norm_cols)

        for dc in DAMAGE_ORDER:
            subset = run_agg[run_agg["damage_case"] == dc]
            pattern = subset[norm_cols].mean().values
            std = subset[norm_cols].std().values
            ax.errorbar(range(7), pattern, yerr=std, color=DAMAGE_COLORS[dc],
                        marker="o", linewidth=1.5, markersize=6,
                        label=dc.replace("_", " ").title(), capsize=3)

        ax.set_xticks(range(7))
        ax.set_xticklabels([f"SN{s}" for s in sensors], fontsize=8)
        ax.set_xlabel("Sensor Node")
        ax.set_ylabel("Normalized RMS")
        ax.set_title(f"{signal.capitalize()} RMS Pattern")
        ax.legend(framealpha=0.8, fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{save_dir}/spatial_rms_pattern.png", bbox_inches="tight")
    plt.close()


def plot_cross_sensor_ratios(feature_df, save_dir="outputs/validation"):
    """Cross-sensor energy ratios per damage case."""
    ensure_dir(save_dir)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.ravel()

    for idx, signal in enumerate(["acceleration", "displacement", "strain_proxy"]):
        ax = axes[idx]
        ratio_cols = [f"{signal}_ratio_{i}_{i+1}" for i in range(6)]

        run_agg = aggregate_run_level(feature_df, ratio_cols)

        x_pos = np.arange(6)
        width = 0.2
        for j, dc in enumerate(DAMAGE_ORDER):
            subset = run_agg[run_agg["damage_case"] == dc]
            means = subset[ratio_cols].mean().values
            ax.bar(x_pos + j * width, means, width, color=DAMAGE_COLORS[dc],
                   alpha=0.75, label=dc.replace("_", " ").title())

        ax.set_xticks(x_pos + width * 1.5)
        ax.set_xticklabels([f"S{i}-S{i+1}" for i in range(6)], fontsize=8)
        ax.set_ylabel("Energy Ratio")
        ax.set_title(f"{signal.capitalize()} Adjacent Sensor Ratios")
        ax.legend(framealpha=0.8, fontsize=7)
        ax.grid(True, alpha=0.3, axis="y")

    # Damage index heatmap
    ax = axes[3]
    norm_cols = [f"{s}_norm_rms_{i}" for s in ["acceleration", "displacement", "strain_proxy"] for i in range(7)]
    run_agg = aggregate_run_level(feature_df, norm_cols)
    healthy_mean = run_agg[run_agg["damage_case"] == "healthy"][norm_cols].mean().values + 1e-12
    damage_indices = {}
    for dc in DAMAGE_ORDER[1:]:
        dc_mean = run_agg[run_agg["damage_case"] == dc][norm_cols].mean().values
        damage_indices[dc] = np.abs((dc_mean - healthy_mean) / healthy_mean)

    di_df = pd.DataFrame(damage_indices, index=norm_cols)
    sns.heatmap(di_df, ax=ax, cmap="YlOrRd", annot=True, fmt=".2f",
                linewidths=0.5, cbar_kws={"label": "|Δ| / Healthy"})
    ax.set_title("Damage Index: Normalized RMS Change (%)")
    ax.set_ylabel("Sensor Feature")
    ax.set_xlabel("Damage Case")
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=15, ha="right")
    plt.setp(ax.yaxis.get_majorticklabels(), fontsize=6)

    plt.tight_layout()
    plt.savefig(f"{save_dir}/cross_sensor_ratios.png", bbox_inches="tight")
    plt.close()


def plot_feature_importance_map(feature_df, save_dir="outputs/validation"):
    """Heatmap of top feature values across damage cases (run-aggregated)."""
    ensure_dir(save_dir)
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import cross_val_score, GroupKFold

    meta = {"window_id", "damage_case", "run_id", "sensor_node",
            "window_pos", "time_start", "time_end", "label_binary", "key", "group"}
    feat_cols = [c for c in feature_df.columns if c not in meta]

    run_agg = aggregate_run_level(feature_df, feat_cols)
    X = run_agg[[c for c in run_agg.columns if c not in {"run_id", "damage_case", "label_binary"}]].values
    y = run_agg["label_binary"].values
    feat_names = [c for c in run_agg.columns if c not in {"run_id", "damage_case", "label_binary"}]

    rf = RandomForestClassifier(n_estimators=300, random_state=42, class_weight="balanced_subsample")
    rf.fit(X, y)
    top_k = 15
    idx = np.argsort(rf.feature_importances_)[::-1][:top_k]
    top_names = [feat_names[i] for i in idx]
    top_imps = rf.feature_importances_[idx]

    # Z-score normalize the top features for heatmap
    from scipy.stats import zscore
    X_top = X[:, idx]
    X_top_z = zscore(X_top, axis=0)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Importance bar chart
    colors = plt.cm.YlOrRd(top_imps / top_imps.max())
    ax1.barh(range(top_k), top_imps[::-1], color=colors[::-1], edgecolor="black", linewidth=0.5)
    ax1.set_yticks(range(top_k))
    ax1.set_yticklabels([n.replace("_", " ")[:25] for n in top_names[::-1]], fontsize=8)
    ax1.set_xlabel("Importance")
    ax1.set_title("Top 15 Features (RF Binary)")
    ax1.invert_yaxis()

    # Z-score heatmap per damage case
    dc_order = run_agg["damage_case"].values
    unique_dcs = sorted(set(dc_order))
    dc_to_idx = {dc: i for i, dc in enumerate(unique_dcs)}
    sort_order = np.argsort([dc_to_idx[dc] for dc in dc_order])

    sns.heatmap(X_top_z[sort_order].T, ax=ax2, cmap="RdBu_r", center=0,
                xticklabels=[dc.replace("_", " ")[:12] for dc in np.array(dc_order)[sort_order]],
                yticklabels=[n.replace("_", " ")[:20] for n in top_names], cbar_kws={"label": "Z-score"})
    ax2.set_title("Top Feature Values by Run (Z-score)")
    ax2.set_xlabel("Run (sorted by damage case)")
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha="right", fontsize=7)
    plt.setp(ax2.yaxis.get_majorticklabels(), fontsize=7)

    plt.tight_layout()
    plt.savefig(f"{save_dir}/feature_importance_map.png", bbox_inches="tight")
    plt.close()

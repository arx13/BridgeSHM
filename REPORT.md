# BridgeSHM — Structural Health Monitoring via 3D FE Simulation & Machine Learning

## Project Overview

BridgeSHM simulates **traffic-induced vibration data** from a 3-span prestressed concrete girder bridge using a **3D finite-element model in OpenSeesPy**, then trains machine learning classifiers to **detect and classify structural damage** from the sensor measurements. The goal is to develop a realistic, data-driven SHM pipeline that could be deployed on a real instrumented bridge.

The pipeline has four stages:

1. **3D FE Simulation** — A 3-span × 20 m continuous girder bridge is modelled in OpenSeesPy with shell elements (deck) and beam elements (girders). Random multi-vehicle traffic crosses the bridge with full vehicle-bridge interaction (VBI), road roughness, thermal modulation, and sensor noise. The transient dynamic response is solved via Newmark integration with Rayleigh damping.

2. **Windowing & Feature Extraction** — Raw time-series (7 sensors × 3 signals × 20 s × 100 Hz) is sliced into overlapping windows. Statistical, FFT, and spatial features are extracted per window.

3. **ML Classification** — Multiple models are trained and compared: Random Forest, XGBoost, MLP, SVM, per-sensor CNNs, multi-sensor CNN, autoencoder, and a **Gateway RF fusion** approach.

4. **TFLite Deployment** — The best binary CNN is exported to TFLite format for ESP32 microcontroller deployment.


## Repository Structure

```text
bridge_SHM/
│
├── run_batch.sh                        # Batch dataset generation script
├── train_classifiers.sh                # ML training wrapper
├── run_full_dataset.py                 # Main dataset generation entry point
├── run_visualizations.py               # Validation and analysis plots
├── requirements.txt                    # Simulation & ML dependencies
│
├── digital_twin/                       # Core bridge simulation engine
│   ├── __init__.py
│   ├── config.py                       # Bridge, traffic, and damage parameters
│   ├── bridge_simulation.py            # 3D FE model, VBI solver, sensors
│   ├── traffic.py                      # Vehicle models and traffic generation
│   └── utils.py                        # Data export and utility functions
│
├── preprocessing/                      # Data preparation pipeline
│   ├── __init__.py
│   ├── features.py                     # Statistical, FFT, and spatial features
│   ├── run_feature_pipeline.py         # Raw signals → feature dataset
│   └── window_dataset_cnn.py           # Raw signals → CNN windows
│
├── models/                             # Model training and evaluation
│   ├── __init__.py
│   ├── classifier_data_loader.py       # Run-stratified data loading
│   ├── train_classifier.py             # RF, XGBoost, MLP, and SVM training
│   ├── evaluate_classifier.py          # Metrics and confusion matrices
│   ├── train_cnn_binary.py             # Binary healthy/damaged CNN
│   ├── train_cnn_multiclass.py         # Multiclass CNN classifier
│   ├── train_gateway.py                # Gateway RF Fusion model
│   ├── compare_models.py               # Model benchmarking
│   └── export_tflite.py                # TensorFlow Lite export
│
├── validation/                         # Data exploration and SHM analysis
│   ├── __init__.py
│   ├── damage_analysis.py              # PCA, feature importance, RMS analysis
│   └── fft_analysis.py                 # Spectral and frequency analysis
│
├── modal_analysis/                     # Structural dynamics analysis
│   ├── __init__.py
│   ├── modal.py                        # Eigenvalue and mode-shape analysis
│   ├── free_vibration.py               # Free-decay simulations
│   ├── modal_features.py               # Modal feature extraction
│   ├── plots.py                        # Visualization utilities
│   ├── run_modal.py                    # Modal analysis runner
│   ├── run_free_vibration.py           # Free-vibration runner
│   └── run_modal_features.py           # Modal feature pipeline
│
└── outputs/                            # Generated results and trained models
    ├── cnn_binary/                     # Binary CNN models
    ├── cnn_multiclass/                 # Multiclass CNN models
    ├── cnn_gateway/                    # Gateway RF Fusion models
    └── tflite_export/                  # ESP32 deployment artifacts

```

### Directory Overview

| Directory         | Purpose                                                                          |
| ----------------- | -------------------------------------------------------------------------------- |
| `digital_twin/`   | High-fidelity 3D finite-element bridge simulation and vehicle–bridge interaction |
| `preprocessing/`  | Signal processing, feature extraction, and CNN dataset generation                |
| `models/`         | Machine-learning and deep-learning training pipelines                            |
| `validation/`     | Exploratory analysis and damage-sensitive feature investigation                  |
| `modal_analysis/` | Structural dynamics and modal parameter extraction                               |
| `datasets/`       | Generated SHM datasets from simulation runs                                      |
| `outputs/`        | Trained models, plots, evaluation metrics, and deployment artifacts              |


---

## The Simulation Engine

### Bridge Model (3D Grillage)

The bridge is modelled in OpenSeesPy with `ndm=3, ndf=6`:

| Parameter | Value |
|-----------|-------|
| Span layout | 3 × 20 m = 60 m total |
| Girders | 4 longitudinal, 2.5 m spacing, 8 m deck width |
| Deck | ShellMITC4, 0.25 m thickness |
| Material | E = 30–33.8 GPa (temperature-dependent), ν = 0.2 |
| Girder section | A = 0.35 m², Iz = 0.08 m⁴, Iy = 0.04 m⁴, J = 0.02 m⁴ |
| Mass | 875 kg/m (lumped at nodes) |
| Damping | Rayleigh damping (ζ ≈ 2–3%) |
| Discretization | 20 elements/span × 4 girders + shell deck + diaphragms every 5 m |

The model is defined in `digital_twin/bridge_simulation.py` as the `BridgeModel3D` class. It builds the mesh, applies boundary conditions (pinned at span ends, roller at piers), injects damage, and manages the recorder.

### Vehicle-Bridge Interaction (VBI)

Vehicles are modelled as **quarter-car axle systems** (2-DOF: sprung + unsprung mass) running across the bridge at realistic speeds:

| Vehicle Type | Axles | Axle Loads | Speed Range |
|---|---|---|---|
| Light car | 2 | 7–12 kN each | 10–80 km/h |
| SUV | 2 | 12–18 kN each | 10–80 km/h |
| Truck | 2 | 34–46 kN each | 10–60 km/h |
| Heavy truck | 3 | 54–66 kN each | 10–60 km/h |

The VBI coupling includes:
- **Road roughness** per ISO 8608 Class B (Gd(n₀) = 4 × 10⁻⁶ m³), generated as a 1-D random profile
- **Tyre stiffness** (Kt ≈ 1.5 × 10⁶ N/m) and **suspension stiffness/damping** (Ks ≈ 5 × 10⁵ N/m, Cs ≈ 1.5 × 10⁴ N·s/m)
- **Contact force** computed at each time step and applied to the bridge deck at the instantaneous axle position, interpolated between adjacent nodes

### Real-World Effects Simulated

| Effect | Implementation | Realism |
|--------|---------------|---------|
| Random traffic | Poisson arrival (λ = 3–5 veh/min), random lane/speed/weight/entry time | Captures natural traffic variability |
| Road roughness | ISO 8608 Class B random profile | Generates realistic high-frequency excitation |
| Thermal modulation | Seasonal + diurnal temperature cycle (5–45 °C), E(T) coefficient −0.3%/°C | Stiffness varies with ambient temperature |
| Thermal gradient | Linear through deck depth (+19 °C/−6 °C top-to-bottom) | Induces thermal curvature |
| Sensor noise | Gaussian noise, SNR 25–40 dB | Models realistic accelerometer noise |
| Material jitter | E ±2% per run | Captures manufacturing variability |
| Solver jitter | Tolerance varied ±10% | Captures numerical variability |

### Damage Cases

Damage is injected by **replacing affected beam-column elements** with reduced section properties over a localized region:

| Case | Description | Location | Severity | Physical Analog |
|---|---|---|---|---|
| **Healthy** | No damage | — | 0% | Baseline condition |
| **Reduced girder stiffness** | 10% E × I reduction | Span 2, Girder 2, 5-element region | 10% | Concrete cracking / loss of prestress |
| **Bearing failure** | 15% stiffness reduction | Span 1, Girder 1 near left support | 15% | Support settlement / bearing degradation |
| **Deck cracking** | 20% stiffness reduction | Span 2, midspan, full width | 20% | Transverse deck cracking |

Each case has **50 stochastic runs** with different traffic, temperature, and noise seeds, yielding **200 runs total** in the complete dataset.

---

## Dataset Organization

### Raw Simulation Output

The complete dataset is stored at `dataset_20260527_193221/` (866 MB):

```
dataset_20260527_193221/
├── metadata.json              ← Bridge config + simulation parameters
├── full_bridge_response.csv   ← Consolidated raw sensor data
├── dataset_healthy.csv
├── dataset_reduced_girder_stiffness.csv
├── dataset_bearing_failure.csv
├── dataset_deck_cracking.csv
├── traffic_healthy.csv
├── traffic_reduced_girder_stiffness.csv
├── traffic_bearing_failure.csv
├── traffic_deck_cracking.csv
└── dataset.log
```

Each CSV has **7 sensors × 3 signals × 2000 time steps × 50 runs = 700,000 rows** per damage case (2.8M total).

### Sensor Layout

Seven sensors along **Girder 2** (inner traffic lane):

| Sensor ID | Position | Location |
|-----------|----------|----------|
| SN67 | 0 m | Left abutment |
| SN72 | 5 m | Span 1 quarter point |
| SN77 | 10 m | Span 1 midspan |
| SN87 | 20 m | Pier 1 |
| SN97 | 30 m | Span 2 midspan |
| SN107 | 35 m | Span 2 quarter point |
| SN117 | 40 m | Pier 2 |

Each records:
- **Vertical acceleration** (m/s²) — primary signal
- **Vertical displacement** (m)
- **Strain proxy** — curvature-based, via finite difference of displacement triplets

### Train/Test Splitting

Critical design: **run-stratified splitting**. Data is grouped by `(run_id, damage_case)` — 40 groups total (10 runs × 4 cases). An 80/20 split yields 32 training groups + 8 test groups. This prevents **data leakage** where the same traffic event appears in both train and test sets.

The small number of groups (40) makes single-split evaluation noisy. **5-fold cross-validation** gives the most honest estimate.

---

## Model Selection: Why Gateway RF Fusion?

We evaluated five approaches on the same run-stratified data:

### 1. Traditional ML on Features

| Model | Binary Acc (CV) | Binary Acc (test) | Multiclass Acc | Notes |
|---|---|---|---|---|
| **Random Forest** | **69.4% ± 3.9%** | 71.25% | 70.0% | Strong but degraded by run-stratified split (honest) |
| **XGBoost** | — | **84.7%*** | **71.1%*** | *Likely from non-stratified split — overestimates |
| **MLP (96-48-24)** | — | 79.1%* | 60.1%* | Same caveat |
| **SVM (RBF)** | — | 63.5%* | — | Poor for imbalanced multiclass |

**The problem:** When evaluated honestly (run-stratified), the best RF achieves only **71.25% binary** with **8% healthy recall** — it almost always predicts "damaged" (trivial classifier in a 73%-damaged dataset).

### 2. Per-Sensor 1D-CNN (Model A — Multiclass)

Each sensor is trained independently as a 4-class classifier on 400-sample windows with 3 channels (accel, displacement, strain):

| Sensor | Multiclass Acc |
|---|---|
| Best (SN44) | 33.70% |
| Worst (SN39, SN47) | 19.26% |
| **Ensemble (logit average)** | **21.11%** |

25% random baseline for 4 classes. The CNN barely beats random — individual sensor signals are too dominated by traffic variability to discriminate damage classes.

### 3. Per-Sensor Binary CNN + RF Gateway (Model B)

Two-stage approach:
1. **Stage 1**: Train a binary CNN per sensor (healthy vs. damaged) with intensity regression
2. **Stage 2**: Stack all 7 CNN binary predictions + intensity estimates → train a **Random Forest gateway** on the 14-dimensional meta-feature vector for 4-class classification

Results:
- **Stage 1 binary CNN**: 73.33% per sensor (but all predict "damaged" — threshold 0.01)
- **Stage 2 Gateway RF**: **63.24% multiclass**, per-class F1: [0.625, 0.588, 0.778, 0.529]
- **Crucially**: the gateway achieves **balanced per-class performance** — healthy recall is no longer 8%

### 4. Multi-Sensor Conv2D

Stack all 7 sensors as a (7, 400, 3) tensor and apply 2D convolutions:
- **Test accuracy: 47.04%** — best deep learning result
- Per-class F1: [0.422, 0.459, 0.310, 0.724]
- Still far below feature-based ML

### 5. Autoencoder (Anomaly Detection)

Train on healthy-only windows, threshold reconstruction error:
- **Val accuracy: 47.73%** — no better than random
- Traffic variability produces larger reconstruction error than damage

### Why Gateway RF Fusion is Preferred

| Criterion | XGBoost (84.7%) | RF run-stratified | Gateway RF Fusion |
|---|---|---|---|
| **Honest evaluation** | (likely leaked) | (CV 69.4%) | (run-stratified) |
| **Healthy recall** | ~60% | **8%** (trivial) | **62.5%** (balanced) |
| **Multiclass capability** | 71.1% | 70.0% | 63.24% |
| **Per-class balance** | Moderate | Terrible (all "damaged") | **Best** (0.53–0.78 F1) |
| **Deployable as edge model** | Yes | Yes | Yes (RF is small) |

The **Gateway RF Fusion** is the **recommended model** for real-world deployment because:

1. **It doesn't just predict "damaged"** — its healthy recall is 62.5% vs. 8% for raw RF
2. **Class balance is the best** across all models (F1 range 0.53–0.78 vs. 0.31–0.72 for Conv2D)
3. **It exploits cross-sensor information** — the RF gateway sees all 7 sensors' binary decisions simultaneously, learning spatial patterns
4. **It's compact** — 7 binary CNN stages + 1 RF is deployable on embedded hardware

---

## Key Insights

1. **Cross-sensor spatial features outperform per-sensor temporal features** by a wide margin — the top 15 RF features are all spatial (RMS ratios, normalized RMS patterns).

2. **Traffic variability is the dominant noise source** — it masks damage-induced changes at individual sensors by 10–100×.

3. **Gateway RF Fusion is the most practical model** — it achieves 63.24% multiclass accuracy with **balanced per-class performance** (healthy F1 = 0.625, not 8%), making it the only model that doesn't just predict "damaged" always.

4. **Frequency shifts from damage are tiny** (0.0003–0.16%) — undetectable under traffic loading.

5. **Run-stratified evaluation is essential** — non-stratified splits show 84.7% accuracy but the honest estimate is 69.4% ± 3.9% (RF CV).

6. **The 3D FE model with VBI + thermal + noise creates realistic data** that captures the fundamental challenge of SHM: damage signals are buried in operational variability.

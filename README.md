# BridgeSHM
### Structural Health Monitoring Using 3D Finite-Element Simulation, Vehicle–Bridge Interaction, and Machine Learning

BridgeSHM is an end-to-end Structural Health Monitoring (SHM) framework that combines high-fidelity bridge simulation, realistic traffic loading, feature engineering, and machine learning to detect and classify structural damage in prestressed concrete bridges.

The project uses a 3D finite-element digital twin built with OpenSeesPy to generate realistic vibration responses under stochastic traffic conditions and evaluates multiple machine-learning and deep-learning approaches for damage identification.

---

## Highlights

- 3D finite-element bridge model using OpenSeesPy
- Realistic Vehicle–Bridge Interaction (VBI) simulation
- Traffic, temperature, material variability, and sensor-noise modeling
- Automated feature extraction pipeline
- Classical ML and deep-learning model benchmarking
- Gateway RF Fusion architecture for balanced multiclass damage classification
- TFLite export for edge deployment on ESP32-class devices

---

## System Pipeline

```text
3D FE Simulation
        ↓
Sensor Data Generation
        ↓
Windowing & Feature Extraction
        ↓
Model Training & Evaluation
        ↓
Damage Classification
        ↓
TFLite Deployment
```

---

## Project Objectives

The primary goals of BridgeSHM are:

1. Generate realistic bridge-response datasets using physics-based simulation.
2. Investigate the impact of operational variability on damage detection.
3. Compare traditional machine-learning and deep-learning approaches.
4. Develop an SHM workflow suitable for embedded and edge deployment.
5. Identify robust damage-sensitive features under real-world conditions.

---

## Simulation Framework

### Bridge Model

The bridge is represented as a 3-span continuous prestressed concrete girder bridge using a 3D grillage formulation.

| Parameter | Value |
|------------|---------|
| Total Length | 60 m |
| Span Layout | 3 × 20 m |
| Number of Girders | 4 |
| Deck Width | 8 m |
| Deck Element | ShellMITC4 |
| Material | Temperature-dependent concrete |
| Damping | Rayleigh damping |
| Solver | Newmark transient integration |

### Vehicle–Bridge Interaction

Traffic loading is modeled using quarter-car dynamic systems with realistic suspension and tire properties.

Supported vehicle categories:

- Passenger cars
- SUVs
- Trucks
- Heavy multi-axle trucks

The VBI implementation includes:

- ISO 8608 road roughness
- Dynamic tire–deck interaction
- Suspension dynamics
- Moving load interpolation
- Stochastic traffic generation

---

## Realistic Operational Variability

BridgeSHM intentionally incorporates uncertainty sources commonly encountered in field deployments.

| Source | Purpose |
|----------|----------|
| Traffic randomness | Simulates natural traffic patterns |
| Road roughness | Produces realistic dynamic excitation |
| Temperature effects | Modulates structural stiffness |
| Thermal gradients | Introduces thermal curvature |
| Sensor noise | Mimics instrumentation errors |
| Material variability | Represents manufacturing uncertainty |
| Solver variability | Improves dataset diversity |

---

## Damage Scenarios

Four structural conditions are simulated.

| Condition | Description |
|------------|-------------|
| Healthy | Undamaged bridge |
| Reduced Girder Stiffness | Local stiffness reduction |
| Bearing Failure | Support degradation |
| Deck Cracking | Localized deck stiffness loss |

Each condition is generated through multiple stochastic simulations to improve generalization and robustness.

---

## Dataset Overview

### Sensor Network

Seven virtual sensors are placed along the bridge girder and record:

- Vertical acceleration
- Vertical displacement
- Strain proxy measurements

### Dataset Characteristics

- 7 sensors
- 3 signal channels per sensor
- 100 Hz sampling frequency
- 20-second simulation windows
- Multiple stochastic realizations per damage condition

---

## Machine Learning Pipeline

### Feature-Based Models

- Random Forest
- XGBoost
- Support Vector Machine (SVM)
- Multi-Layer Perceptron (MLP)

### Deep Learning Models

- Per-sensor 1D CNN
- Multi-sensor CNN
- Autoencoder-based anomaly detection

### Fusion Architecture

BridgeSHM introduces a two-stage Gateway RF Fusion strategy:

1. Sensor-level binary CNNs detect healthy vs. damaged behavior.
2. A Random Forest gateway fuses outputs from all sensors to perform multiclass damage classification.

This approach demonstrated the most balanced class performance and practical deployment characteristics.

---

## Performance Summary

Models were evaluated using **run-stratified validation** to prevent data leakage between simulations and provide realistic estimates of deployment performance.

| Model             | Task                      | Accuracy  |
| ----------------- | ------------------------- | --------- |
| Random Forest     | Binary Classification     | 71.25%    |
| XGBoost*          | Multiclass Classification | 71.1%     |
| Multi-Sensor CNN  | Multiclass Classification | 47.0%     |
| Gateway RF Fusion | Multiclass Classification | **63.2%** |

* Results may be optimistic due to non-stratified evaluation.

### Key Findings

* Cross-sensor spatial features outperform individual sensor features.
* Traffic variability is the dominant source of uncertainty.
* Damage-induced frequency shifts are extremely small under operational loading.
* **Gateway RF Fusion** provides the best balance between accuracy and class-wise performance, making it the recommended deployment model.

### Key Challenge

Traffic-induced variability often masks structural damage signatures, making robust feature extraction and sensor fusion essential for reliable SHM.

---

## Repository Structure

```text
bridge_SHM/
├── digital_twin/        # Bridge simulation engine
├── preprocessing/       # Windowing and feature extraction
├── models/              # Training and evaluation pipelines
├── validation/          # Data exploration and visualization
├── modal_analysis/      # Modal analysis utilities
├── outputs/             # Generated artifacts
├── run_full_dataset.py
├── run_visualizations.py
└── requirements.txt
```


---

## Installation & Setup

### Clone the Repository

```bash
git clone https://github.com/<username>/bridge_SHM.git
cd bridge_SHM
```

### Create a Virtual Environment

Linux/macOS:

```bash
python -m venv venv
source venv/bin/activate
```

Windows:

```bash
python -m venv venv
venv\Scripts\activate
```

### Install Dependencies

Simulation dependencies:

```bash
pip install -r requirements.txt
```

---

# Running Simulations

## Generate the Full Dataset

Generate all bridge conditions and stochastic runs:

```bash
python run_full_dataset.py
```

This process will:

* Build the 3D bridge finite-element model
* Generate random traffic scenarios
* Apply vehicle–bridge interaction (VBI)
* Inject damage cases
* Simulate transient response
* Record sensor measurements
* Export datasets for ML training

Expected output:

```text
dataset_YYYYMMDD_HHMMSS/
├── dataset_healthy.csv
├── dataset_reduced_girder_stiffness.csv
├── dataset_bearing_failure.csv
├── dataset_deck_cracking.csv
├── metadata.json
└── dataset.log
```

---

## Generate Validation Visualizations

```bash
python run_visualizations.py
```

Produces:

* Time-history plots
* FFT comparisons
* PCA visualizations
* Damage analysis figures
* Modal-analysis plots

---

# Feature Extraction

Convert raw simulation outputs into machine-learning features:

```bash
python preprocessing/run_feature_pipeline.py
```

Output:

```text
3d_windowed_features.csv
```

---

## Create CNN Training Windows

Generate CNN-ready datasets:

```bash
python preprocessing/window_dataset_cnn.py
```

Output:

```text
cnn_windows/
├── sensor_SN67.npz
├── sensor_SN72.npz
├── sensor_SN77.npz
└── ...
```

---

# Training Machine Learning Models

## Random Forest

```bash
python models/train_classifier.py --model rf
```

## XGBoost

```bash
python models/train_classifier.py --model xgb
```

## Multi-Layer Perceptron (MLP)

```bash
python models/train_classifier.py --model mlp
```

## Support Vector Machine (SVM)

```bash
python models/train_classifier.py --model svm
```

Generated outputs:

```text
outputs/classifiers/
├── trained_model.pkl
├── confusion_matrix.png
├── roc_curve.png
└── metrics.json
```

---

# Training Deep Learning Models

## Binary CNN (Healthy vs Damaged)

```bash
python models/train_cnn_binary.py
```

Outputs:

```text
outputs/cnn_binary/
```

---

## Multiclass CNN

```bash
python models/train_cnn_multiclass.py
```

Outputs:

```text
outputs/cnn_multiclass/
```

---

# Training the Gateway RF Fusion Model

Train the recommended deployment architecture:

```bash
python models/train_gateway.py
```

Outputs:

```text
outputs/cnn_gateway/
├── gateway_rf.pkl
├── results.json
└── confusion_matrix.png
```

---

# Model Evaluation

Evaluate trained models:

```bash
python models/evaluate_classifier.py
```

Evaluation metrics include:

* Accuracy
* Precision
* Recall
* F1 Score
* ROC-AUC
* Confusion Matrix

---

# Modal Analysis

Run eigenvalue and mode-shape analysis:

```bash
python modal_analysis/run_modal.py
```

Run free-vibration simulations:

```bash
python modal_analysis/run_free_vibration.py
```

Extract modal features:

```bash
python modal_analysis/run_modal_features.py
```

Outputs:

```text
outputs/modal/
outputs/free_vibration/
outputs/modal_features/
```

---

# Export for ESP32 Deployment

Convert trained CNN models to TensorFlow Lite:

```bash
python models/export_tflite.py
```

Output:

```text
outputs/tflite_export/
├── model.tflite
├── metadata.json
└── labels.txt
```

---

# End-to-End Workflow

```text
run_full_dataset.py
          ↓
run_feature_pipeline.py
          ↓
train_classifier.py
          ↓
train_cnn_binary.py
          ↓
train_gateway.py
          ↓
evaluate_classifier.py
          ↓
export_tflite.py
```


## Recommended Deployment Model

Gateway RF Fusion is the preferred deployment architecture because it:

- Maintains balanced performance across classes
- Avoids the “always damaged” prediction bias
- Utilizes information from all sensors simultaneously
- Remains lightweight enough for embedded deployment

---

## Technology Stack

- Python
- OpenSeesPy
- NumPy
- Pandas
- Scikit-learn
- TensorFlow / Keras
- XGBoost
- Matplotlib

---

## Future Enhancements

- Domain adaptation using real bridge measurements
- Physics-informed neural networks
- Graph neural networks for sensor fusion
- Federated SHM learning systems
- Edge-AI optimization for low-power deployments

---

## Acknowledgements

This project combines finite-element simulation, structural dynamics, vehicle–bridge interaction modeling, and machine-learning techniques for next-generation Structural Health Monitoring research.

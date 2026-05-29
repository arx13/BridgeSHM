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

## Key Findings

### What Worked

- Spatial features consistently outperformed single-sensor temporal features.
- Sensor fusion improved damage discrimination.
- Gateway RF Fusion provided the best balance between accuracy and class recall.
- Run-stratified evaluation produced more realistic performance estimates.

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

## License

Specify the project license here.

## Acknowledgements

This project combines finite-element simulation, structural dynamics, vehicle–bridge interaction modeling, and machine-learning techniques for next-generation Structural Health Monitoring research.

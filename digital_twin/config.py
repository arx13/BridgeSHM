import numpy as np

# =========================================================
# Bridge Configuration (3D Grillage)
# =========================================================

BRIDGE_CONFIG = {
    "bridge_name": "RCC_PSC_Girder_Bridge_3D",
    "num_spans": 3,
    "span_lengths": [20.0, 20.0, 20.0],
    "deck_width": 8.0,
    "num_girders": 4,
    "girder_spacing": 2.5,
    "girder_depth": 2.0,
    "num_divisions_per_span": 10,

    # Longitudinal girder section (3D)
    "E_long": 30e9,
    "G_long": 12.5e9,
    "A_long": 0.35,
    "Iz_long": 0.08,
    "Iy_long": 0.02,
    "J_long": 0.01,

    # Transverse diaphragm section (3D)
    "E_trans": 25e9,
    "G_trans": 10.4e9,
    "A_trans": 0.20,
    "Iz_trans": 0.03,
    "Iy_trans": 0.01,
    "J_trans": 0.005,
    "diaphragm_spacing": 5.0,

    # Shell concrete deck slab
    "shell_thickness": 0.25,
    "shell_E": 30e9,
    "shell_nu": 0.2,
    "shell_rho": 2500,

    "rho": 2500,
    "mass_per_length": 875.0,
    "damping_ratio": 0.03,

    "sensor_girder": 2,
    # Sensor positions expressed as physical x-coordinates along the bridge [m]
    # 7 sensors per report: quarter/mid/quarter of each span + additional
    "sensor_x_coords": [5.0, 10.0, 15.0, 25.0, 30.0, 35.0, 45.0],
}

# =========================================================
# Simulation Configuration
# =========================================================

SIM_CONFIG = {
    "dt": 0.01,
    "total_time": 20.0,
    "g": 9.81,
    "random_seed": 42,

    "max_solver_retries": 4,
    "retry_dt_factor": 0.5,
    "rayleigh_alphaM": 0.0,
    "rayleigh_betaK": 0.0,
    "rayleigh_betaKinit": 0.0005,
    "rayleigh_betaKcomm": 0.0,
}

# =========================================================
# Vehicle-Bridge Interaction (VBI) Configuration
# =========================================================

VBI_CONFIG = {
    "iso_class": "B",
    "roughness_seed": 123,
    "roughness_dx": 0.01,

    # VBI coupling mode: "loose" (fast, default) or "tight" (research-grade, 5-10× slower)
    "vbi_mode": "loose",
    "vbi_max_iterations": 3,
    "vbi_tolerance": 1e-8,
    "vbi_relaxation": 0.5,

    "vehicle_params": {
        "light_car": {
            "Ms": 800, "Mu": 80,
            "Ks": 200e3, "Cs": 12e3, "Kt": 800e3,
        },
        "suv": {
            "Ms": 1200, "Mu": 100,
            "Ks": 250e3, "Cs": 15e3, "Kt": 1000e3,
        },
        "truck": {
            "Ms": 3000, "Mu": 250,
            "Ks": 400e3, "Cs": 25e3, "Kt": 1500e3,
        },
        "heavy_truck": {
            "Ms": 5000, "Mu": 350,
            "Ks": 500e3, "Cs": 30e3, "Kt": 2000e3,
        },
    },
}

# =========================================================
# Environmental & Noise Configuration
# =========================================================

ENV_CONFIG = {
    "thermal": {
        "alpha": 0.003,
        "T_ref": 20.0,
        "T_min": 5.0,
        "T_max": 35.0,
    },
    "noise": {
        "accel_rms_pct": 0.02,
        "disp_rms_pct": 0.005,
        "strain_rms_pct": 0.01,
        "seed_offset": 1000,
    },
}

# =========================================================
# Vehicle Library
# =========================================================

VEHICLE_LIBRARY = {
    "light_car": {
        "vehicle_class": "normal",
        "base_axle_loads_kN": [10.0, 10.0],
        "base_axle_spacing_m": [2.5],
        "speed_range_kmph": (10, 80),
        "weight_variation": 0.20,
        "spacing_variation": 0.10,
    },
    "suv": {
        "vehicle_class": "normal",
        "base_axle_loads_kN": [15.0, 15.0],
        "base_axle_spacing_m": [2.8],
        "speed_range_kmph": (10, 80),
        "weight_variation": 0.20,
        "spacing_variation": 0.10,
    },
    "truck": {
        "vehicle_class": "heavy",
        "base_axle_loads_kN": [40.0, 40.0],
        "base_axle_spacing_m": [4.0],
        "speed_range_kmph": (10, 60),
        "weight_variation": 0.15,
        "spacing_variation": 0.08,
    },
    "heavy_truck": {
        "vehicle_class": "heavy",
        "base_axle_loads_kN": [60.0, 60.0, 60.0],
        "base_axle_spacing_m": [3.5, 3.5],
        "speed_range_kmph": (10, 60),
        "weight_variation": 0.10,
        "spacing_variation": 0.08,
    },
}

# =========================================================
# Traffic Scenarios
# =========================================================

TRAFFIC_CONFIG = {
    "min_vehicles_per_run": 3,
    "max_vehicles_per_run": 10,
    "arrival_time_margin": 2.0,
    "lanes": [1, 2],
    "lane_to_girder": {1: 2, 2: 3},
    "roughness_enabled": True,
    "roughness_std": 0.015,
}

# =========================================================
# Damage Scenarios
# =========================================================

DAMAGE_CASES = {
    "healthy": {"enabled": False, "label": 0},
    "reduced_girder_stiffness": {
        "enabled": True, "target_span": 2,
        "target_girder": 2, "severity": 0.10, "label": 1,
    },
    "bearing_failure": {
        "enabled": True, "target_span": 1,
        "target_girder": 1, "severity": 0.15, "label": 2,
    },
    "deck_cracking": {
        "enabled": True, "target_span": 2,
        "target_girder": 2, "severity": 0.20, "label": 3,
    },
}

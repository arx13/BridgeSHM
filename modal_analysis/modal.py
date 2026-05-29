import openseespy.opensees as ops
import numpy as np
import pandas as pd

from digital_twin.bridge_simulation import BridgeModel3D, inject_damage_3d


def build_bridge_for_modal(bridge_config, damage_case):
    bridge = BridgeModel3D(bridge_config).build()
    if damage_case.get("enabled", False):
        inject_damage_3d(bridge, damage_case)
    return bridge


def compute_natural_frequencies(num_modes=6):
    lambdas = ops.eigen(num_modes)
    freqs = []

    for lam in lambdas:
        if lam > 0:
            omega = np.sqrt(lam)
            freq_hz = omega / (2 * np.pi)
            freqs.append(freq_hz)
        else:
            freqs.append(np.nan)

    return freqs


def extract_mode_shape(bridge_model, mode_number, girder_id=2):
    nodes = bridge_model.girder_nodes[girder_id]
    x_vals = []
    y_mode = []

    for node in nodes:
        x = bridge_model.node_coords[node][0]
        try:
            phi = ops.nodeEigenvector(node, mode_number, 2)
        except:
            phi = 0.0

        x_vals.append(x)
        y_mode.append(phi)

    x_vals = np.array(x_vals)
    y_mode = np.array(y_mode)

    max_abs = np.max(np.abs(y_mode))
    if max_abs > 1e-12:
        y_mode = y_mode / max_abs

    return x_vals, y_mode


def run_modal_analysis(bridge_config, damage_cases, num_modes=6):
    rows = []
    mode_shapes = {}

    for damage_name, damage_case in damage_cases.items():
        print(f"Running modal analysis for: {damage_name}")

        bridge = build_bridge_for_modal(bridge_config, damage_case)
        freqs = compute_natural_frequencies(num_modes=num_modes)

        for i, f in enumerate(freqs, start=1):
            rows.append({
                "damage_case": damage_name,
                "mode_number": i,
                "frequency_hz": f
            })

        mode_shapes[damage_name] = {}
        for mode_num in range(1, num_modes + 1):
            x_vals, y_mode = extract_mode_shape(bridge, mode_num, girder_id=2)
            mode_shapes[damage_name][mode_num] = {
                "x": x_vals,
                "mode_shape": y_mode
            }

        ops.wipe()

    freq_df = pd.DataFrame(rows)
    return freq_df, mode_shapes

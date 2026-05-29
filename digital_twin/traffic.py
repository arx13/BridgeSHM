import numpy as np
import pandas as pd

def generate_vehicle_instance(vehicle_name, vehicle_def, traffic_config, total_time):
    speed_min, speed_max = vehicle_def["speed_range_kmph"]
    speed_kmph = np.random.uniform(speed_min, speed_max)
    speed_mps = speed_kmph / 3.6

    weight_scale = np.random.uniform(
        1 - vehicle_def["weight_variation"],
        1 + vehicle_def["weight_variation"]
    )

    spacing_scale = np.random.uniform(
        1 - vehicle_def["spacing_variation"],
        1 + vehicle_def["spacing_variation"]
    )

    axle_loads = [w * weight_scale for w in vehicle_def["base_axle_loads_kN"]]
    axle_spacing = [s * spacing_scale for s in vehicle_def["base_axle_spacing_m"]]

    lane = np.random.choice(traffic_config["lanes"])

    entry_time = np.random.uniform(0, total_time - traffic_config["arrival_time_margin"])

    return {
        "vehicle_name": vehicle_name,
        "vehicle_class": vehicle_def["vehicle_class"],
        "speed_kmph": speed_kmph,
        "speed_mps": speed_mps,
        "weight_scale": weight_scale,
        "spacing_scale": spacing_scale,
        "axle_loads_kN": axle_loads,
        "axle_spacing_m": axle_spacing,
        "lane": lane,
        "entry_time": entry_time
    }


def generate_traffic_scenario(vehicle_library, traffic_config, total_time):
    n_vehicles = np.random.randint(
        traffic_config["min_vehicles_per_run"],
        traffic_config["max_vehicles_per_run"] + 1
    )

    vehicle_names = list(vehicle_library.keys())
    scenario = []

    for i in range(n_vehicles):
        vname = np.random.choice(vehicle_names)
        vdef = vehicle_library[vname]
        vehicle = generate_vehicle_instance(vname, vdef, traffic_config, total_time)
        vehicle["vehicle_id"] = f"V{i+1}"
        scenario.append(vehicle)

    scenario = sorted(scenario, key=lambda x: x["entry_time"])
    return scenario


def traffic_scenario_to_dataframe(scenario):
    return pd.DataFrame(scenario)
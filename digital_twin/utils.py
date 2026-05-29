import os
import numpy as np
import pandas as pd

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def flatten_list(nested):
    return [item for sublist in nested for item in sublist]

def save_dataframe(df, filepath):
    df.to_csv(filepath, index=False)

def get_total_bridge_length(span_lengths):
    return sum(span_lengths)

def nearest_node_from_x(node_map, x_target):
    closest_node = None
    min_dist = 1e9
    for node_id, x in node_map.items():
        dist = abs(x - x_target)
        if dist < min_dist:
            min_dist = dist
            closest_node = node_id
    return closest_node
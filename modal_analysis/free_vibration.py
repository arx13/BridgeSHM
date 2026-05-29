import os
import numpy as np
import pandas as pd
import openseespy.opensees as ops

from digital_twin.bridge_simulation import (
    BridgeModel3D,
    inject_damage_3d,
    SensorRecorder3D,
)

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def apply_impulse_load(bridge, magnitude=-3000.0):
    ops.timeSeries("Constant", 1)
    ops.pattern("Plain", 1, 1)

    triplets = bridge.get_sensor_triplets()

    for (_, center_node, _) in triplets:
        try:
            ops.load(center_node, 0.0, magnitude, 0.0, 0.0, 0.0, 0.0)
        except:
            pass

def setup_dynamic_analysis_3d(sim_config):
    ops.wipeAnalysis()
    ops.rayleigh(
        sim_config["rayleigh_alphaM"],
        sim_config["rayleigh_betaK"],
        sim_config["rayleigh_betaKinit"],
        sim_config["rayleigh_betaKcomm"],
    )
    ops.constraints("Plain")
    ops.numberer("RCM")
    ops.system("UmfPack")
    ops.test("NormDispIncr", 1.0e-8, 20)
    ops.algorithm("Newton")
    ops.integrator("Newmark", 0.5, 0.25)
    ops.analysis("Transient")

def run_free_vibration(
    bridge_config,
    sim_config,
    damage_case_name,
    damage_case_cfg,
    total_time=20.0,
    dt=0.01
):
    bridge = BridgeModel3D(bridge_config).build()

    if damage_case_cfg.get("enabled", False):
        inject_damage_3d(bridge, damage_case_cfg)

    recorder = SensorRecorder3D(bridge)
    recorder.initialize()

    setup_dynamic_analysis_3d(sim_config)

    apply_impulse_load(bridge)

    for _ in range(10):
        ok = ops.analyze(1, dt)
        if ok != 0:
            raise RuntimeError("Impulse excitation phase failed.")
        recorder.record_step()

    ops.remove("loadPattern", 1)

    n_steps = int(total_time / dt)

    for step in range(n_steps):
        ok = ops.analyze(1, dt)
        if ok != 0:
            print(f"Analysis failed at step {step}")
            break
        recorder.record_step()

    df = recorder.to_dataframe(
        dt=dt,
        label="free_vibration",
        vehicle_case="none",
        damage_case=damage_case_name
    )

    return df

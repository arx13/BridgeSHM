import openseespy.opensees as ops
import numpy as np
import pandas as pd
from tqdm import tqdm

from .traffic import generate_traffic_scenario, traffic_scenario_to_dataframe
from .utils import ensure_dir


# =====================================================================
# Roughness Profile (ISO 8608)
# =====================================================================

class RoughnessProfile:
    """
    Generate a 1-D road surface roughness profile using ISO 8608 PSD.

    G_d(n) = G_d(n0) * (n / n0)^{-2}

    Class B (good):     G_d(n0) =   4e-6 m^3
    Class C (average):  G_d(n0) =  16e-6 m^3
    n0 = 0.1 cycles/m  (reference spatial frequency)
    """

    CLASS_PSD = {"A": 1e-6, "B": 4e-6, "C": 16e-6, "D": 64e-6, "E": 256e-6}

    def __init__(self, iso_class, length, dx=0.01, seed=123):
        self.iso_class = iso_class.upper()
        self.length = length
        self.dx = dx
        self.seed = seed
        self._generate()

    def _generate(self):
        rng = np.random.RandomState(self.seed)
        n_points = int(self.length / self.dx) + 1
        x = np.linspace(0, self.length, n_points)
        n = np.fft.rfftfreq(n_points, d=self.dx)
        n[0] = 1e-12

        n0 = 0.1
        Gd_n0 = self.CLASS_PSD.get(self.iso_class, 4e-6)
        psd = Gd_n0 * (n / n0) ** (-2)
        amplitude = np.sqrt(psd * (1.0 / self.dx) / (2.0 * n_points))
        amplitude[0] = 0.0

        phase = rng.uniform(0, 2 * np.pi, size=len(amplitude))
        fft_profile = amplitude * np.exp(1j * phase)
        profile = np.fft.irfft(fft_profile, n=n_points)
        profile -= profile.mean()

        self.x_arr = x
        self.z_arr = profile

    def evaluate(self, x):
        return np.interp(x, self.x_arr, self.z_arr)


# =====================================================================
# Quarter-Car Axle Model (2-DOF)
# =====================================================================

class QuarterCarAxle:
    """
    2-DOF quarter-car model for a single axle.

    State vector: [zs, zs_dot, zu, zu_dot]
      zs  = sprung mass vertical displacement  (positive up)
      zu  = unsprung mass vertical displacement (positive up)

    Equations:
      Ms * zs_ddot  = -Ms*g - Ks*(zs - zu) - Cs*(zs_dot - zu_dot)
      Mu * zu_ddot  = -Mu*g - Ks*(zu - zs) - Cs*(zu_dot - zs_dot)
                       - Kt*(zu - w_contact)

      Contact force (applied to bridge, positive up):
        F = Kt * (zu - w_contact)
    """

    def __init__(self, params):
        self.Ms = params["Ms"]
        self.Mu = params["Mu"]
        self.Ks = params["Ks"]
        self.Cs = params["Cs"]
        self.Kt = params["Kt"]
        self.g = 9.81

        self.state = np.zeros(4)
        self._set_initial_condition()

    def _set_initial_condition(self):
        """Steady state on rigid ground (w_contact = 0)."""
        total_mass = self.Ms + self.Mu
        zu0 = -total_mass * self.g / self.Kt
        zs0 = zu0 - self.Ms * self.g / self.Ks
        self.state = np.array([zs0, 0.0, zu0, 0.0])

    def _ode(self, state, w_contact):
        zs, zs_dot, zu, zu_dot = state
        zs_ddot = (-self.g - self.Cs / self.Ms * (zs_dot - zu_dot)
                   - self.Ks / self.Ms * (zs - zu))
        zu_ddot = (-self.g - self.Cs / self.Mu * (zu_dot - zs_dot)
                   - self.Ks / self.Mu * (zu - zs)
                   - self.Kt / self.Mu * (zu - w_contact))
        return np.array([zs_dot, zs_ddot, zu_dot, zu_ddot])

    def compute_contact_force(self, w_contact):
        zu = self.state[2]
        return self.Kt * (zu - w_contact)

    def rk4_step(self, w_contact, dt):
        k1 = self._ode(self.state, w_contact)
        k2 = self._ode(self.state + 0.5 * dt * k1, w_contact)
        k3 = self._ode(self.state + 0.5 * dt * k2, w_contact)
        k4 = self._ode(self.state + dt * k3, w_contact)
        self.state += (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


# =====================================================================
# 3D Bridge Model (Grillage)
# =====================================================================

class BridgeModel3D:
    def __init__(self, config, E_scale=1.0):
        self.cfg = config
        self.E_scale = E_scale
        self.node_coords = {}
        self.element_info = {}
        self.girder_nodes = {}
        self.girder_elements = {}
        self.support_nodes = []
        self.next_node_id = 1
        self.next_ele_id = 1
        self.all_node_ids = []

    def reset_model(self):
        ops.wipe()
        ops.model("Basic", "-ndm", 3, "-ndf", 6)

    def build(self):
        self.reset_model()
        self._create_nodes()
        self._create_longitudinal_elements()
        self._create_diaphragms()
        self._create_shell_elements()
        self._apply_boundary_conditions()
        self._assign_masses()
        return self

    def _create_nodes(self):
        span_lengths = self.cfg["span_lengths"]
        num_div = self.cfg["num_divisions_per_span"]
        girder_spacing = self.cfg["girder_spacing"]
        num_girders = self.cfg["num_girders"]

        x_positions = [0.0]
        for span in span_lengths:
            dx = span / num_div
            for _ in range(num_div):
                x_positions.append(x_positions[-1] + dx)

        for g in range(num_girders):
            self.girder_nodes[g + 1] = []
            z = g * girder_spacing
            for x in x_positions:
                nid = self.next_node_id
                ops.node(nid, x, 0.0, z)
                self.node_coords[nid] = (x, 0.0, z)
                self.girder_nodes[g + 1].append(nid)
                self.all_node_ids.append(nid)
                self.next_node_id += 1

    def _create_longitudinal_elements(self):
        E = self.cfg["E_long"] * self.E_scale
        G = self.cfg["G_long"]
        A = self.cfg["A_long"]
        J = self.cfg["J_long"]
        Iy = self.cfg["Iy_long"]
        Iz = self.cfg["Iz_long"]

        ops.geomTransf("Linear", 1, 0, 1, 0)

        for girder_id, nodes in self.girder_nodes.items():
            self.girder_elements[girder_id] = []
            for i in range(len(nodes) - 1):
                ni, nj = nodes[i], nodes[i + 1]
                eid = self.next_ele_id
                ops.element("elasticBeamColumn", eid, ni, nj, A, E, G, J, Iy, Iz, 1)
                self.element_info[eid] = {
                    "type": "longitudinal",
                    "girder": girder_id,
                    "node_i": ni, "node_j": nj,
                }
                self.girder_elements[girder_id].append(eid)
                self.next_ele_id += 1

    def _create_diaphragms(self):
        E = self.cfg["E_trans"] * self.E_scale
        G = self.cfg["G_trans"]
        A = self.cfg["A_trans"]
        J = self.cfg["J_trans"]
        Iy = self.cfg["Iy_trans"]
        Iz = self.cfg["Iz_trans"]

        ops.geomTransf("Linear", 2, 0, 1, 0)

        num_girders = self.cfg["num_girders"]
        x_positions = [self.node_coords[n][0]
                       for n in self.girder_nodes[1]]
        spacing = self.cfg.get("diaphragm_spacing", 5.0)
        bridge_length = sum(self.cfg["span_lengths"])

        diaphragm_xs = set()
        x = 0.0
        while x <= bridge_length + 1e-6:
            diaphragm_xs.add(round(x, 6))
            x += spacing

        for i, xi in enumerate(x_positions):
            if round(xi, 6) not in diaphragm_xs:
                continue
            for g in range(1, num_girders):
                ni = self.girder_nodes[g][i]
                nj = self.girder_nodes[g + 1][i]
                eid = self.next_ele_id
                ops.element("elasticBeamColumn", eid, ni, nj,
                           A, E, G, J, Iy, Iz, 2)
                self.element_info[eid] = {
                    "type": "transverse",
                    "girder_pair": (g, g + 1),
                    "node_i": ni, "node_j": nj,
                }
                self.next_ele_id += 1

    def _create_shell_elements(self):
        shell_mat_tag = 100
        shell_sec_tag = 101
        E = self.cfg.get("shell_E", 30e9)
        nu = self.cfg.get("shell_nu", 0.2)
        rho = self.cfg.get("shell_rho", 2500)
        thickness = self.cfg.get("shell_thickness", 0.25)
        self.shell_section_tag = shell_sec_tag

        ops.nDMaterial("ElasticIsotropic", shell_mat_tag, E, nu, rho)
        ops.section("PlateFiber", shell_sec_tag, shell_mat_tag, thickness)

        num_girders = self.cfg["num_girders"]
        num_nodes_along = len(self.girder_nodes[1])

        for g in range(1, num_girders):
            for i in range(num_nodes_along - 1):
                n1 = self.girder_nodes[g][i]
                n2 = self.girder_nodes[g][i + 1]
                n3 = self.girder_nodes[g + 1][i + 1]
                n4 = self.girder_nodes[g + 1][i]

                eid = self.next_ele_id
                ops.element("ShellMITC4", eid, n1, n2, n3, n4, shell_sec_tag)
                self.element_info[eid] = {
                    "type": "shell",
                    "girder_pair": (g, g + 1),
                    "nodes": (n1, n2, n3, n4),
                }
                self.next_ele_id += 1

    def _apply_boundary_conditions(self):
        span_lengths = self.cfg["span_lengths"]
        support_xs = [0.0]
        x = 0.0
        for span in span_lengths:
            x += span
            support_xs.append(x)

        for girder_id, nodes in self.girder_nodes.items():
            for nid in nodes:
                x_pos = self.node_coords[nid][0]
                if abs(x_pos - support_xs[0]) < 1e-6:
                    ops.fix(nid, 1, 1, 1, 0, 0, 1)
                    self.support_nodes.append(nid)
                for sx in support_xs[1:]:
                    if abs(x_pos - sx) < 1e-6:
                        ops.fix(nid, 0, 1, 1, 0, 0, 1)
                        self.support_nodes.append(nid)
                        break

    def _assign_masses(self):
        m = self.cfg["mass_per_length"]
        dx = self.cfg["span_lengths"][0] / self.cfg["num_divisions_per_span"]

        for girder_id, nodes in self.girder_nodes.items():
            for i, nid in enumerate(nodes):
                factor = 0.5 if (i == 0 or i == len(nodes) - 1) else 1.0
                lumped = m * dx * factor
                ops.mass(nid, lumped, lumped, lumped, 0.0, 0.0, 0.0)

    def get_midspan_nodes(self, girder_id=2):
        mids = []
        span_starts = np.cumsum([0] + self.cfg["span_lengths"][:-1])
        span_lengths = self.cfg["span_lengths"]
        target_xs = [start + length / 2.0
                     for start, length in zip(span_starts, span_lengths)]
        candidate_nodes = self.girder_nodes[girder_id]
        for tx in target_xs:
            best = None
            best_d = 1e9
            for nid in candidate_nodes:
                d = abs(self.node_coords[nid][0] - tx)
                if d < best_d:
                    best_d = d
                    best = nid
            mids.append(best)
        return mids

    def get_sensor_triplets(self):
        triplets = []
        girder_id = self.cfg.get("sensor_girder", 2)
        target_xs = self.cfg.get("sensor_x_coords",
                                  [5.0, 10.0, 15.0, 25.0, 30.0, 35.0, 45.0])
        nodes = self.girder_nodes[girder_id]
        n_positions = [self.node_coords[n][0] for n in nodes]
        for tx in target_xs:
            idx = min(range(len(n_positions)),
                      key=lambda i: abs(n_positions[i] - tx))
            if idx <= 0 or idx >= len(nodes) - 1:
                continue
            triplets.append((nodes[idx - 1], nodes[idx], nodes[idx + 1]))
        return triplets

    def get_nearest_node(self, x, z):
        girder_spacing = self.cfg["girder_spacing"]
        g = int(round(z / girder_spacing)) + 1
        g = max(1, min(self.cfg["num_girders"], g))
        best = None
        best_d = 1e9
        for nid in self.girder_nodes[g]:
            d = abs(self.node_coords[nid][0] - x)
            if d < best_d:
                best_d = d
                best = nid
        return best

    def interpolate_vertical_deflection(self, x, z):
        girder_spacing = self.cfg["girder_spacing"]
        g = int(round(z / girder_spacing)) + 1
        g = max(1, min(self.cfg["num_girders"], g))
        nodes = self.girder_nodes[g]

        for i in range(len(nodes) - 1):
            n1, n2 = nodes[i], nodes[i + 1]
            x1 = self.node_coords[n1][0]
            x2 = self.node_coords[n2][0]
            if x1 <= x <= x2:
                if abs(x2 - x1) < 1e-12:
                    return ops.nodeDisp(n1, 2)
                r = (x - x1) / (x2 - x1)
                return ops.nodeDisp(n1, 2) * (1 - r) + ops.nodeDisp(n2, 2) * r
        last = nodes[-1]
        return ops.nodeDisp(last, 2)

    def begin_load_step(self, load_tag):
        ops.timeSeries("Constant", load_tag)
        ops.pattern("Plain", load_tag, load_tag)

    def _interpolate_load_on_girder(self, x, girder_id, force_N):
        nodes = self.girder_nodes[girder_id]
        for i in range(len(nodes) - 1):
            n1, n2 = nodes[i], nodes[i + 1]
            x1 = self.node_coords[n1][0]
            x2 = self.node_coords[n2][0]
            if x1 <= x <= x2:
                if abs(x2 - x1) < 1e-12:
                    ops.load(n1, 0.0, force_N, 0.0, 0.0, 0.0, 0.0)
                    return
                r = (x - x1) / (x2 - x1)
                ops.load(n1, 0.0, force_N * (1 - r), 0.0, 0.0, 0.0, 0.0)
                ops.load(n2, 0.0, force_N * r, 0.0, 0.0, 0.0, 0.0)
                return
        last = nodes[-1]
        ops.load(last, 0.0, force_N, 0.0, 0.0, 0.0, 0.0)

    def apply_vertical_load(self, x, z, force_N):
        girder_spacing = self.cfg["girder_spacing"]
        num_girders = self.cfg["num_girders"]
        g = int(round(z / girder_spacing)) + 1
        g = max(1, min(num_girders, g))

        weights = {g: 0.70}
        if g - 1 >= 1:
            weights[g - 1] = 0.15
        if g + 1 <= num_girders:
            weights[g + 1] = 0.15

        total_w = sum(weights.values())
        for girder_id, w in weights.items():
            weights[girder_id] = w / total_w

        for girder_id, w in weights.items():
            self._interpolate_load_on_girder(x, girder_id, force_N * w)


# =====================================================================
# Sensor Recorder (3D)
# =====================================================================

class SensorRecorder3D:
    def __init__(self, bridge_model):
        self.bridge = bridge_model
        self.accel_data = {}
        self.disp_data = {}
        self.strain_proxy_data = {}
        self.sensor_triplets = []

    def initialize(self):
        self.sensor_triplets = self.bridge.get_sensor_triplets()
        for _, center_node, _ in self.sensor_triplets:
            self.accel_data[center_node] = []
            self.disp_data[center_node] = []
            self.strain_proxy_data[center_node] = []

    def _get_vertical_disp(self, node_id):
        try:
            return ops.nodeDisp(node_id, 2)
        except Exception:
            return 0.0

    def _get_vertical_accel(self, node_id):
        try:
            return ops.nodeAccel(node_id, 2)
        except Exception:
            return 0.0

    def _estimate_curvature(self, left_node, center_node, right_node):
        xl = self.bridge.node_coords[left_node][0]
        xc = self.bridge.node_coords[center_node][0]
        xr = self.bridge.node_coords[right_node][0]
        dx1 = xc - xl
        dx2 = xr - xc
        dx = (dx1 + dx2) / 2.0
        if abs(dx) < 1e-12:
            return 0.0
        wl = self._get_vertical_disp(left_node)
        wc = self._get_vertical_disp(center_node)
        wr = self._get_vertical_disp(right_node)
        return (wl - 2.0 * wc + wr) / (dx ** 2)

    def record_step(self):
        for left_node, center_node, right_node in self.sensor_triplets:
            accel = self._get_vertical_accel(center_node)
            disp = self._get_vertical_disp(center_node)
            curvature = self._estimate_curvature(left_node, center_node, right_node)
            self.accel_data[center_node].append(accel)
            self.disp_data[center_node].append(disp)
            self.strain_proxy_data[center_node].append(curvature)

    def to_dataframe(self, dt, label, vehicle_case, damage_case):
        num_steps = len(next(iter(self.accel_data.values())))
        time = np.arange(num_steps) * dt
        rows = []
        for left_node, center_node, right_node in self.sensor_triplets:
            for i in range(num_steps):
                rows.append({
                    "time": time[i],
                    "sensor_node": center_node,
                    "acceleration": self.accel_data[center_node][i],
                    "displacement": self.disp_data[center_node][i],
                    "strain_proxy": self.strain_proxy_data[center_node][i],
                    "strain_proxy_type": "curvature_based",
                    "vehicle_case": vehicle_case,
                    "damage_case": damage_case,
                    "label": label,
                })
        return pd.DataFrame(rows)


# =====================================================================
# Environmental Noise Injection
# =====================================================================

class NoiseInjector:
    def __init__(self, env_config, run_id=0):
        self.cfg = env_config["noise"]
        self.rng = np.random.RandomState(self.cfg["seed_offset"] + run_id)

    @staticmethod
    def _pink_noise(n_samples, beta=1.0, rng=None):
        if rng is None:
            rng = np.random.RandomState()
        from numpy.fft import rfft, irfft
        white = rng.randn(n_samples)
        f = np.fft.rfftfreq(n_samples)
        f[0] = 1e-12
        white_f = rfft(white)
        pink_f = white_f / (f ** (beta / 2.0))
        pink = irfft(pink_f, n=n_samples)
        return pink / (np.std(pink) + 1e-12)

    def inject(self, df):
        accel = df["acceleration"].values.astype(np.float64)
        noise_a = self.rng.normal(0, np.std(accel) * self.cfg["accel_rms_pct"],
                                  len(accel))
        df["acceleration"] = accel + noise_a

        disp = df["displacement"].values.astype(np.float64)
        pink = self._pink_noise(len(disp), beta=1.0, rng=self.rng)
        pink *= np.std(disp) * self.cfg["disp_rms_pct"]
        df["displacement"] = disp + pink

        strain = df["strain_proxy"].values.astype(np.float64)
        noise_s = self.rng.normal(0, np.std(strain) * self.cfg["strain_rms_pct"],
                                  len(strain))
        df["strain_proxy"] = strain + noise_s

        return df


# =====================================================================
# Thermal Modulator
# =====================================================================

class ThermalModulator:
    def __init__(self, env_config):
        self.cfg = env_config["thermal"]

    def temperature_for_run(self, run_id, total_runs):
        if total_runs <= 1:
            return self.cfg["T_ref"]
        fraction = (run_id - 1) / (total_runs - 1)
        return self.cfg["T_min"] + fraction * (self.cfg["T_max"] - self.cfg["T_min"])

    def compute_E_scale(self, T):
        alpha = self.cfg["alpha"]
        T_ref = self.cfg["T_ref"]
        return 1.0 - alpha * (T - T_ref)


# =====================================================================
# Damage Injection (3D)
# =====================================================================

def inject_damage_3d(bridge_model, damage_case, E_scale=1.0):
    if not damage_case["enabled"]:
        return

    target_span = damage_case["target_span"]
    target_girder = damage_case["target_girder"]
    severity = damage_case["severity"]

    girder_elements = bridge_model.girder_elements[target_girder]
    num_spans = bridge_model.cfg["num_spans"]
    num_per_span = len(girder_elements) // num_spans

    if target_span < 1 or target_span > num_spans:
        return

    start_idx = (target_span - 1) * num_per_span
    end_idx = target_span * num_per_span
    target_eles = girder_elements[start_idx:end_idx]

    if not target_eles:
        return

    mid = len(target_eles) // 2
    window = 2
    damaged_subset = target_eles[
        max(0, mid - window): min(len(target_eles), mid + window + 1)
    ]

    E0 = bridge_model.cfg["E_long"] * E_scale
    G0 = bridge_model.cfg["G_long"]
    A = bridge_model.cfg["A_long"]
    J = bridge_model.cfg["J_long"]
    Iy = bridge_model.cfg["Iy_long"]
    Iz = bridge_model.cfg["Iz_long"]

    for eid in damaged_subset:
        info = bridge_model.element_info[eid]
        ni = info["node_i"]
        nj = info["node_j"]

        ops.remove("ele", eid)

        E_d = E0 * (1.0 - severity)
        G_d = G0 * (1.0 - severity)
        Iy_d = Iy * (1.0 - severity)
        Iz_d = Iz * (1.0 - severity)

        ops.element("elasticBeamColumn", eid, ni, nj, A, E_d, G_d, J, Iy_d, Iz_d, 1)

        bridge_model.element_info[eid]["damaged"] = True
        bridge_model.element_info[eid]["damage_severity"] = severity
        bridge_model.element_info[eid]["damage_region"] = True

    print(
        f"[Damage Injection 3D] Applied {severity * 100:.1f}% stiffness reduction "
        f"to {len(damaged_subset)} elements in span {target_span}, girder {target_girder}"
    )


# =====================================================================
# Analysis Configuration & Fallback
# =====================================================================

def configure_analysis_3d(sim_config):
    ops.wipeAnalysis()
    ops.constraints("Plain")
    ops.numberer("RCM")
    ops.system("UmfPack")
    ops.test("NormDispIncr", 1.0e-6, 100)
    ops.algorithm("Newton")
    ops.integrator("Newmark", 0.5, 0.25)
    ops.analysis("Transient")
    ops.rayleigh(
        sim_config["rayleigh_alphaM"],
        sim_config["rayleigh_betaK"],
        sim_config["rayleigh_betaKinit"],
        sim_config["rayleigh_betaKcomm"],
    )


def try_analyze_step_3d(dt, sim_config):
    ok = ops.analyze(1, dt)
    if ok == 0:
        return True

    fallback_algorithms = [
        ("NewtonLineSearch", {}),
        ("ModifiedNewton", {}),
        ("KrylovNewton", {}),
    ]

    for alg_name, kwargs in fallback_algorithms:
        try:
            ops.algorithm(alg_name, *kwargs.values())
        except Exception:
            continue
        ok = ops.analyze(1, dt)
        if ok == 0:
            ops.algorithm("Newton")
            return True

    retry_dt = dt * sim_config["retry_dt_factor"]
    ops.algorithm("Newton")
    ok = ops.analyze(1, retry_dt)
    if ok == 0:
        return True

    return False


# =====================================================================
# Tight VBI Coupling — Node State Save/Restore
# =====================================================================

def _save_node_states_vbi(node_ids):
    state = {}
    for nid in node_ids:
        d = ops.nodeDisp(nid, 2)
        v = ops.nodeVel(nid, 2)
        a = ops.nodeAccel(nid, 2)
        state[nid] = (d, v, a)
    return state


def _restore_node_states_vbi(node_ids, state):
    for nid in node_ids:
        d, v, a = state[nid]
        ops.setNodeDisp(nid, 2, d)
        ops.setNodeVel(nid, 2, v)
        ops.setNodeAccel(nid, 2, a)


# =====================================================================
# Main Simulation Entry Point
# =====================================================================

def run_simulation(bridge_config, sim_config, vehicle_library,
                   vbi_config, env_config, traffic_config,
                   damage_case_name, damage_case, run_id=1,
                   total_runs=None):
    """
    Run one 3D VBI simulation for a given damage case + run_id.

    Parameters
    ----------
    total_runs : int or None
        Total number of runs across all damage cases (for temperature cycling).
        If None, defaults to 20 for backward compatibility.

    Returns (signal_df, traffic_df) matching the schema
    of the legacy 2D pipeline.
    """
    if total_runs is None:
        total_runs = sim_config.get("total_runs", 20)
    rng = np.random.RandomState(sim_config["random_seed"] + run_id)

    # --- 1. Temperature → scale Young's modulus ---
    thermal = ThermalModulator(env_config)
    T = thermal.temperature_for_run(run_id, total_runs)
    E_scale = thermal.compute_E_scale(T)
    print(f"  T={T:.1f}°C, E_scale={E_scale:.4f}")

    # --- 2. Build 3D bridge ---
    bridge = BridgeModel3D(bridge_config, E_scale=E_scale)
    bridge.build()
    inject_damage_3d(bridge, damage_case, E_scale=E_scale)

    # --- 3. Road roughness ---
    bridge_length = sum(bridge_config["span_lengths"])
    roughness = RoughnessProfile(
        iso_class=vbi_config["iso_class"],
        length=bridge_length,
        dx=vbi_config["roughness_dx"],
        seed=vbi_config["roughness_seed"] + run_id,
    )

    # --- 4. Traffic scenario ---
    traffic_scenario = generate_traffic_scenario(
        vehicle_library, traffic_config, sim_config["total_time"]
    )
    traffic_df = traffic_scenario_to_dataframe(traffic_scenario)

    # --- 5. Create quarter-car axle models ---
    lane2z = {1: 2.5, 2: 5.0}
    axles = []
    for vehicle in traffic_scenario:
        vname = vehicle["vehicle_name"]
        params = vbi_config["vehicle_params"][vname]
        cumulative_spacing = 0.0
        for axle_idx in range(len(vehicle["axle_loads_kN"])):
            axle = QuarterCarAxle(params)
            axle.entry_time = vehicle["entry_time"]
            axle.speed_mps = vehicle["speed_mps"]
            axle.lane = vehicle["lane"]
            axle.z_contact = lane2z.get(vehicle["lane"], 2.5)
            axle.axle_offset = cumulative_spacing
            cumulative_spacing += vehicle["axle_spacing_m"][axle_idx] \
                if axle_idx < len(vehicle["axle_spacing_m"]) else 0.0
            axles.append(axle)

    # --- 6. Sensor recorder ---
    sensor_nodes = bridge.get_midspan_nodes()
    recorder = SensorRecorder3D(bridge)
    recorder.initialize()

    # --- 7. Configure analysis ---
    configure_analysis_3d(sim_config)

    dt = sim_config["dt"]
    total_time = sim_config["total_time"]
    steps = int(total_time / dt)
    label = damage_case.get("label", int(damage_case["enabled"]))
    completed_steps = 0
    all_node_ids = bridge.all_node_ids

    # VBI coupling parameters
    vbi_mode = vbi_config.get("vbi_mode", "loose")
    vbi_max_iter = min(vbi_config.get("vbi_max_iterations", 5), 3)
    vbi_tol = vbi_config.get("vbi_tolerance", 1e-8)
    omega = vbi_config.get("vbi_relaxation", 0.5)

    # --- 8. Time-stepping loop ---
    step_failed = False
    for step in tqdm(range(steps), desc=f"3D-VBI | {damage_case_name} run{run_id}"):
        if step_failed:
            break
        t = step * dt

        active_axles = []
        for axle in axles:
            local_t = t - axle.entry_time
            if local_t < 0:
                continue
            axle_x = axle.speed_mps * local_t - axle.axle_offset
            if axle_x < 0 or axle_x > bridge_length:
                continue
            active_axles.append((axle, axle_x))

        if not active_axles:
            success = try_analyze_step_3d(dt, sim_config)
            if not success:
                print(f"[WARNING] Analysis failed at step {step}, "
                      f"time={t:.2f}s, damage={damage_case_name}")
                break
            recorder.record_step()
            completed_steps += 1
            continue

        # ---- Loose coupling (fast) ----
        if vbi_mode == "loose":
            load_tag = 3000000 + step
            ops.timeSeries("Constant", load_tag)
            ops.pattern("Plain", load_tag, load_tag)

            for axle, axle_x in active_axles:
                w_bridge = bridge.interpolate_vertical_deflection(axle_x, axle.z_contact)
                z_r = roughness.evaluate(axle_x)
                w_contact = w_bridge + z_r
                axle.rk4_step(w_contact, dt)
                F_contact = axle.compute_contact_force(w_contact)
                bridge.apply_vertical_load(axle_x, axle.z_contact, -F_contact)

            success = try_analyze_step_3d(dt, sim_config)
            if not success:
                print(f"[WARNING] Analysis failed at step {step}, "
                      f"time={t:.2f}s, damage={damage_case_name}")
                break

            recorder.record_step()
            completed_steps += 1
            continue

        # ---- Tight coupling (research-grade, slower) ----
        saved_bridge_state = _save_node_states_vbi(all_node_ids)
        saved_axle_states = [a.state.copy() for a, _ in active_axles]
        current_time_val = ops.getTime()

        w_estimates = []
        for axle, axle_x in active_axles:
            w_bridge = bridge.interpolate_vertical_deflection(axle_x, axle.z_contact)
            z_r = roughness.evaluate(axle_x)
            w_estimates.append(w_bridge + z_r)

        vbi_converged = False
        for vbi_iter in range(vbi_max_iter):
            load_tag = 2000000 + step * vbi_max_iter + vbi_iter

            ops.timeSeries("Constant", load_tag)
            ops.pattern("Plain", load_tag, load_tag)

            for i, (axle, axle_x) in enumerate(active_axles):
                axle.state = saved_axle_states[i].copy()
                axle.rk4_step(w_estimates[i], dt)
                F_contact = axle.compute_contact_force(w_estimates[i])
                bridge.apply_vertical_load(axle_x, axle.z_contact, -F_contact)

            _restore_node_states_vbi(all_node_ids, saved_bridge_state)
            ops.setTime(current_time_val)
            ops.wipeAnalysis()
            configure_analysis_3d(sim_config)

            success = try_analyze_step_3d(dt, sim_config)
            if not success:
                print(f"[WARNING] Coupling analysis failed at step {step}, "
                      f"iter {vbi_iter}, damage={damage_case_name}")
                step_failed = True
                break

            errors = []
            for i, (axle, axle_x) in enumerate(active_axles):
                w_bridge = bridge.interpolate_vertical_deflection(axle_x, axle.z_contact)
                z_r = roughness.evaluate(axle_x)
                w_new = w_bridge + z_r
                errors.append(abs(w_new - w_estimates[i]))
                w_estimates[i] = omega * w_new + (1.0 - omega) * w_estimates[i]

            if max(errors) < vbi_tol:
                vbi_converged = True
                break

        for cleanup_iter in range(vbi_max_iter):
            tag = 2000000 + step * vbi_max_iter + cleanup_iter
            try:
                ops.remove("loadPattern", tag)
            except Exception:
                pass

        if step_failed:
            break

        recorder.record_step()
        completed_steps += 1

    # --- 9. Export to DataFrame ---
    df = recorder.to_dataframe(
        dt=dt, label=label,
        vehicle_case="mixed_traffic",
        damage_case=damage_case_name,
    )

    df["num_vehicles_in_run"] = len(traffic_scenario)
    df["traffic_mode"] = "stochastic_multi_vehicle"
    df["completed_steps"] = completed_steps
    df["expected_steps"] = steps
    df["completion_ratio"] = completed_steps / steps if steps > 0 else 0.0

    # --- 10. Inject sensor noise ---
    noise_engine = NoiseInjector(env_config, run_id=run_id)
    df = noise_engine.inject(df)

    # Clean up OpenSees for next run
    ops.wipe()

    return df, traffic_df

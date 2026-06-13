import csv
import math
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np
import torch
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler


BASE_SEED = 2950720452
SEEDS = [0, 1, 2, 3, 4]
EPISODES_PER_SEED = 10
ABLATION_EPISODES_PER_SEED = 10
STRESS_EPISODES_PER_SEED = 8
TRAIN_SCENES = 72
TRAIN_CANDIDATES = 10
MAIN_CANDIDATES = 18
ORACLE_CANDIDATES = 6
MAX_WORKERS = max(1, min(4, int(os.environ.get("PAPER68_WORKERS", "4"))))

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
RESULTS.mkdir(exist_ok=True)
FIGURES.mkdir(exist_ok=True)

FINGER_RADIUS = 0.012
SUCCESS_LIFT_MARGIN = 0.070
CONTACT_LIMIT = 900.0

METHODS = [
    "random_grasp",
    "antipodal_geometry",
    "force_closure_score",
    "cem_grasp_search",
    "mlp_energy_model",
    "transformer_policy_ranker",
    "ensemble_uncertainty_ranker",
    "ebm_transformer_compositional",
    "oracle_mujoco_grid",
]

ABLATIONS = [
    "full_ebm_transformer_compositional",
    "no_object_energy",
    "no_contact_energy",
    "no_task_energy",
    "no_collision_energy",
    "no_feasibility_energy",
    "no_transformer_context",
    "monolithic_scalar_energy_only",
]

MAIN_SPLITS = [
    "seen_simple",
    "unseen_dimensions",
    "unseen_shape_family",
    "slippery_contact",
    "clutter_collision",
    "task_constraint_shift",
    "combined_composition_shift",
]


@dataclass(frozen=True)
class RolloutResult:
    success: int
    lifted_height: float
    final_xy_error: float
    slip: float
    drop: int
    collision: int
    unsafe_force: int
    max_contact_force: float
    energy: float
    object_path: float
    task_satisfied: int
    score: float


MODEL_CACHE: dict[tuple, mujoco.MjModel] = {}


def stable_int(text: str) -> int:
    return sum((idx + 1) * ord(ch) for idx, ch in enumerate(text))


def unit(vec: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    if norm < 1e-8:
        return np.array([1.0, 0.0], dtype=float)
    return vec / norm


def rotate(vec: np.ndarray, angle: float) -> np.ndarray:
    c = math.cos(angle)
    s = math.sin(angle)
    return np.array([c * vec[0] - s * vec[1], s * vec[0] + c * vec[1]], dtype=float)


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def sigmoid(x: np.ndarray | float) -> np.ndarray | float:
    return 1.0 / (1.0 + np.exp(-np.asarray(x)))


def ci95(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    arr = np.asarray(values, dtype=float)
    return float(1.96 * arr.std(ddof=1) / math.sqrt(len(arr)))


def normal_p_from_t(t_stat: float) -> float:
    return float(math.erfc(abs(t_stat) / math.sqrt(2.0)))


def quat_from_yaw(yaw: float) -> tuple[float, float, float, float]:
    return (math.cos(yaw / 2.0), 0.0, 0.0, math.sin(yaw / 2.0))


def yaw_from_quat(q: np.ndarray) -> float:
    w, x, y, z = q
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def scenario_config(split: str, seed: int, episode: int, rng: np.random.Generator, stress_level: float | None = None) -> dict:
    if stress_level is not None:
        aspect = 1.0 + 3.8 * stress_level
        clutter_clearance = 0.16 - 0.09 * stress_level
        task_mix = "clutter_safe" if stress_level > 0.45 else "lift"
        return {
            "split": f"stress_{stress_level:.2f}",
            "shape": "bar" if stress_level > 0.35 else "box",
            "hx": 0.045 * aspect,
            "hy": 0.040,
            "hz": 0.035,
            "radius": 0.045,
            "friction": 0.82 - 0.48 * stress_level,
            "mass": 0.16 + 0.06 * stress_level,
            "yaw": rng.uniform(-0.55, 0.55),
            "task": task_mix,
            "forbidden_angle": rng.uniform(-math.pi, math.pi),
            "desired_angle": rng.uniform(-0.50, 0.50),
            "clutter": make_clutter(clutter_clearance, rng) if stress_level > 0.25 else [],
            "sensor_noise": 0.004 + 0.006 * stress_level,
        }

    base = {
        "seen_simple": ("box", 0.055, 0.040, 0.035, 0.045, 0.82, 0.16, "lift", []),
        "unseen_dimensions": ("box", 0.090, 0.030, 0.035, 0.045, 0.78, 0.18, "lift", []),
        "unseen_shape_family": ("l_shape", 0.070, 0.045, 0.034, 0.045, 0.74, 0.18, "handle_long_side", []),
        "slippery_contact": ("cylinder", 0.052, 0.052, 0.038, 0.050, 0.34, 0.17, "lift", []),
        "clutter_collision": ("box", 0.058, 0.042, 0.035, 0.045, 0.76, 0.16, "clutter_safe", make_clutter(0.075, rng)),
        "task_constraint_shift": ("bar", 0.115, 0.024, 0.032, 0.045, 0.72, 0.16, "avoid_face", []),
        "combined_composition_shift": ("t_shape", 0.105, 0.034, 0.033, 0.045, 0.38, 0.22, "clutter_safe", make_clutter(0.065, rng)),
    }[split]
    shape, hx, hy, hz, radius, friction, mass, task, clutter = base
    return {
        "split": split,
        "shape": shape,
        "hx": hx * rng.uniform(0.92, 1.08),
        "hy": hy * rng.uniform(0.92, 1.08),
        "hz": hz,
        "radius": radius,
        "friction": friction,
        "mass": mass,
        "yaw": rng.uniform(-0.65, 0.65),
        "task": task,
        "forbidden_angle": rng.uniform(-0.30, 0.30),
        "desired_angle": rng.uniform(-0.60, 0.60),
        "clutter": clutter,
        "sensor_noise": 0.006 if split != "combined_composition_shift" else 0.011,
    }


def make_clutter(clearance: float, rng: np.random.Generator) -> list[tuple[float, float, float, float]]:
    side = rng.choice([-1.0, 1.0])
    return [
        (0.02, side * clearance, 0.060, 0.025),
        (0.12, -side * (clearance + 0.025), 0.045, 0.020),
    ]


def model_key(scenario: dict) -> tuple:
    clutter_key = tuple((round(x, 3), round(y, 3), round(hx, 3), round(hy, 3)) for x, y, hx, hy in scenario["clutter"])
    return (
        scenario["shape"],
        round(scenario["hx"], 3),
        round(scenario["hy"], 3),
        round(scenario["hz"], 3),
        round(scenario["radius"], 3),
        round(scenario["friction"], 3),
        round(scenario["mass"], 3),
        clutter_key,
    )


def object_geoms_xml(scenario: dict) -> str:
    shape = scenario["shape"]
    hx, hy, hz = scenario["hx"], scenario["hy"], scenario["hz"]
    friction = scenario["friction"]
    mass = scenario["mass"]
    if shape == "cylinder":
        return f'<geom name="object_geom" type="cylinder" size="{scenario["radius"]:.4f} {hz:.4f}" mass="{mass:.4f}" friction="{friction:.4f} 0.010 0.001" rgba="0.12 0.36 0.75 1"/>'
    if shape == "l_shape":
        return (
            f'<geom name="object_geom_a" type="box" pos="0 0 0" size="{hx:.4f} {hy:.4f} {hz:.4f}" mass="{0.62 * mass:.4f}" friction="{friction:.4f} 0.010 0.001" rgba="0.13 0.38 0.74 1"/>'
            f'<geom name="object_geom_b" type="box" pos="{0.45 * hx:.4f} {1.05 * hy:.4f} 0" size="{0.42 * hx:.4f} {0.45 * hy:.4f} {hz:.4f}" mass="{0.38 * mass:.4f}" friction="{friction:.4f} 0.010 0.001" rgba="0.13 0.38 0.74 1"/>'
        )
    if shape == "t_shape":
        return (
            f'<geom name="object_geom_a" type="box" pos="0 0 0" size="{hx:.4f} {hy:.4f} {hz:.4f}" mass="{0.58 * mass:.4f}" friction="{friction:.4f} 0.010 0.001" rgba="0.13 0.38 0.74 1"/>'
            f'<geom name="object_geom_b" type="box" pos="{-0.20 * hx:.4f} {1.25 * hy:.4f} 0" size="{0.72 * hx:.4f} {0.38 * hy:.4f} {hz:.4f}" mass="{0.42 * mass:.4f}" friction="{friction:.4f} 0.010 0.001" rgba="0.13 0.38 0.74 1"/>'
        )
    return f'<geom name="object_geom" type="box" size="{hx:.4f} {hy:.4f} {hz:.4f}" mass="{mass:.4f}" friction="{friction:.4f} 0.010 0.001" rgba="0.13 0.38 0.74 1"/>'


def model_xml(scenario: dict) -> str:
    clutter_xml = []
    for idx, (x, y, hx, hy) in enumerate(scenario["clutter"]):
        clutter_xml.append(
            f'<geom name="clutter_{idx}" type="box" pos="{x:.4f} {y:.4f} {scenario["hz"]:.4f}" '
            f'size="{hx:.4f} {hy:.4f} {scenario["hz"]:.4f}" friction="0.8 0.004 0.001" rgba="0.50 0.46 0.38 1"/>'
        )
    return f"""
<mujoco model="ebm_compositional_grasp">
  <compiler angle="radian" coordinate="local"/>
  <option timestep="0.01" gravity="0 0 -9.81" integrator="RK4" cone="elliptic"/>
  <default>
    <geom condim="6" solref="0.006 1" solimp="0.90 0.95 0.001"/>
  </default>
  <worldbody>
    <geom name="table" type="plane" size="1.0 1.0 0.05" friction="0.9 0.004 0.001" rgba="0.82 0.84 0.83 1"/>
    <body name="object" pos="0 0 {scenario['hz'] + 0.002:.4f}">
      <freejoint name="object_free"/>
      {object_geoms_xml(scenario)}
    </body>
    {''.join(clutter_xml)}
    <body name="left_finger" pos="0 0 0">
      <joint name="lx" type="slide" axis="1 0 0" range="-0.70 0.70" damping="5"/>
      <joint name="ly" type="slide" axis="0 1 0" range="-0.55 0.55" damping="5"/>
      <joint name="lz" type="slide" axis="0 0 1" range="0.02 0.35" damping="4"/>
      <geom name="left_finger_geom" type="capsule" fromto="0 0 -0.055 0 0 0.055" size="{FINGER_RADIUS}"
            mass="0.06" friction="6.0 0.100 0.020" rgba="0.86 0.24 0.13 1"/>
    </body>
    <body name="right_finger" pos="0 0 0">
      <joint name="rx" type="slide" axis="1 0 0" range="-0.70 0.70" damping="5"/>
      <joint name="ry" type="slide" axis="0 1 0" range="-0.55 0.55" damping="5"/>
      <joint name="rz" type="slide" axis="0 0 1" range="0.02 0.35" damping="4"/>
      <geom name="right_finger_geom" type="capsule" fromto="0 0 -0.055 0 0 0.055" size="{FINGER_RADIUS}"
            mass="0.06" friction="6.0 0.100 0.020" rgba="0.86 0.24 0.13 1"/>
    </body>
  </worldbody>
  <actuator>
    <position name="alx" joint="lx" kp="800" ctrlrange="-0.70 0.70"/>
    <position name="aly" joint="ly" kp="800" ctrlrange="-0.55 0.55"/>
    <position name="alz" joint="lz" kp="700" ctrlrange="0.02 0.35"/>
    <position name="arx" joint="rx" kp="800" ctrlrange="-0.70 0.70"/>
    <position name="ary" joint="ry" kp="800" ctrlrange="-0.55 0.55"/>
    <position name="arz" joint="rz" kp="700" ctrlrange="0.02 0.35"/>
  </actuator>
</mujoco>
"""


def get_model(scenario: dict) -> mujoco.MjModel:
    key = model_key(scenario)
    if key not in MODEL_CACHE:
        MODEL_CACHE[key] = mujoco.MjModel.from_xml_string(model_xml(scenario))
    return MODEL_CACHE[key]


def object_geom_ids(model: mujoco.MjModel) -> set[int]:
    ids = set()
    for name in ["object_geom", "object_geom_a", "object_geom_b"]:
        gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
        if gid >= 0:
            ids.add(gid)
    return ids


def contact_stats(model: mujoco.MjModel, data: mujoco.MjData) -> tuple[float, int, int]:
    obj_ids = object_geom_ids(model)
    finger_ids = {
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "left_finger_geom"),
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "right_finger_geom"),
    }
    clutter_ids = {
        gid
        for gid in range(model.ngeom)
        if mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid)
        and mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid).startswith("clutter_")
    }
    force = np.zeros(6, dtype=float)
    object_finger_force = 0.0
    clutter_collision = 0
    object_finger_contact = 0
    for cidx in range(data.ncon):
        contact = data.contact[cidx]
        pair = {contact.geom1, contact.geom2}
        if pair & obj_ids and pair & finger_ids:
            mujoco.mj_contactForce(model, data, cidx, force)
            object_finger_force += float(np.linalg.norm(force[:3]))
            object_finger_contact = 1
        if pair & clutter_ids and (pair & finger_ids or pair & obj_ids):
            clutter_collision = 1
    return object_finger_force, object_finger_contact, clutter_collision


def projected_width(scenario: dict, angle: float) -> float:
    if scenario["shape"] == "cylinder":
        return 2.0 * scenario["radius"]
    n = np.array([math.cos(angle), math.sin(angle)], dtype=float)
    local_x = np.array([math.cos(scenario["yaw"]), math.sin(scenario["yaw"])], dtype=float)
    local_y = rotate(local_x, math.pi / 2)
    return 2.0 * (abs(scenario["hx"] * np.dot(n, local_x)) + abs(scenario["hy"] * np.dot(n, local_y)))


def scenario_extent(scenario: dict) -> float:
    return max(scenario["hx"], scenario["hy"], scenario["radius"]) if scenario["shape"] != "cylinder" else scenario["radius"]


def generate_candidates(scenario: dict, rng: np.random.Generator, n: int) -> list[dict]:
    candidates = []
    base_angles = [
        scenario["yaw"],
        scenario["yaw"] + math.pi / 2,
        scenario["desired_angle"],
        scenario["forbidden_angle"] + math.pi / 2,
    ]
    while len(candidates) < n:
        if len(candidates) < len(base_angles):
            angle = base_angles[len(candidates)] + rng.normal(0.0, 0.10)
        else:
            angle = rng.uniform(-math.pi, math.pi)
        width = projected_width(scenario, angle) * rng.uniform(0.82, 1.16)
        center = rng.normal(0.0, scenario["sensor_noise"], size=2)
        if len(candidates) % 5 == 0:
            center += rng.normal(0.0, 0.018, size=2)
        candidates.append(
            {
                "center_x": float(center[0]),
                "center_y": float(center[1]),
                "angle": float(angle),
                "jaw_width": float(clamp(width, 0.028, 0.180)),
                "closing_force": float(rng.uniform(0.82, 1.22)),
                "lift_tilt": float(rng.uniform(-0.12, 0.12)),
            }
        )
    return candidates


def task_satisfied(scenario: dict, candidate: dict, clutter_collision: int) -> int:
    angle = candidate["angle"]
    if scenario["task"] == "avoid_face":
        forbidden_alignment = abs(math.cos(angle - scenario["forbidden_angle"]))
        return int(forbidden_alignment < 0.78)
    if scenario["task"] == "handle_long_side":
        long_side = scenario["yaw"] + math.pi / 2
        return int(abs(math.sin(angle - long_side)) < 0.55)
    if scenario["task"] == "clutter_safe":
        return int(not clutter_collision and clutter_clearance_energy(scenario, candidate) < 0.85)
    return 1


def rollout_grasp(scenario: dict, candidate: dict) -> RolloutResult:
    model = get_model(scenario)
    data = mujoco.MjData(model)
    object_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "object")
    joint_addrs = {}
    for jname in ["lx", "ly", "lz", "rx", "ry", "rz"]:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jname)
        joint_addrs[jname] = model.jnt_qposadr[jid]

    data.qpos[:] = 0.0
    data.qvel[:] = 0.0
    data.qpos[0] = 0.0
    data.qpos[1] = 0.0
    data.qpos[2] = scenario["hz"] + 0.002
    data.qpos[3:7] = quat_from_yaw(scenario["yaw"])

    center = np.array([candidate["center_x"], candidate["center_y"]], dtype=float)
    n = np.array([math.cos(candidate["angle"]), math.sin(candidate["angle"])], dtype=float)
    width = candidate["jaw_width"]
    open_width = max(width + 0.13, projected_width(scenario, candidate["angle"]) + 0.10)
    closed_width = max(0.014, width * (0.74 - 0.11 * (candidate["closing_force"] - 1.0)))
    z0 = scenario["hz"] + 0.025
    lift_z = z0 + 0.18
    left_open = np.array([center[0], center[1], z0]) + np.array([n[0], n[1], 0.0]) * (open_width / 2)
    right_open = np.array([center[0], center[1], z0]) - np.array([n[0], n[1], 0.0]) * (open_width / 2)
    left_closed = np.array([center[0], center[1], z0]) + np.array([n[0], n[1], 0.0]) * (closed_width / 2)
    right_closed = np.array([center[0], center[1], z0]) - np.array([n[0], n[1], 0.0]) * (closed_width / 2)

    for jname, val in zip(["lx", "ly", "lz", "rx", "ry", "rz"], [*left_open, *right_open]):
        data.qpos[joint_addrs[jname]] = val
    data.ctrl[:] = [*left_open, *right_open]
    mujoco.mj_forward(model, data)

    max_force = 0.0
    contact_impulse = 0.0
    object_path = 0.0
    pusher_path = 0.0
    object_finger_contacts = 0
    clutter_collision = 0
    prev_obj = data.xpos[object_bid].copy()
    prev_left = left_open.copy()
    prev_right = right_open.copy()
    min_lift_z = data.xpos[object_bid][2]

    def step_to(left_goal: np.ndarray, right_goal: np.ndarray, steps: int, start_left: np.ndarray, start_right: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        nonlocal max_force, contact_impulse, object_path, pusher_path, object_finger_contacts, clutter_collision, prev_obj, prev_left, prev_right, min_lift_z
        left = start_left.copy()
        right = start_right.copy()
        for step in range(steps):
            alpha = (step + 1) / steps
            left = start_left * (1 - alpha) + left_goal * alpha
            right = start_right * (1 - alpha) + right_goal * alpha
            data.ctrl[:] = [*left, *right]
            mujoco.mj_step(model, data)
            force, has_contact, has_clutter = contact_stats(model, data)
            object_finger_contacts += has_contact
            clutter_collision = max(clutter_collision, has_clutter)
            contact_impulse += force
            max_force = max(max_force, force)
            obj = data.xpos[object_bid].copy()
            object_path += float(np.linalg.norm(obj - prev_obj))
            pusher_path += float(np.linalg.norm(left - prev_left) + np.linalg.norm(right - prev_right))
            prev_obj = obj
            prev_left = left.copy()
            prev_right = right.copy()
            min_lift_z = min(min_lift_z, obj[2])
        return left, right

    for _ in range(10):
        data.ctrl[:] = [*left_open, *right_open]
        mujoco.mj_step(model, data)

    left_now, right_now = step_to(left_closed, right_closed, 70, left_open, right_open)
    tilt = np.array([0.0, candidate["lift_tilt"], 0.0])
    left_lift = left_closed + np.array([0.0, 0.0, lift_z - z0]) + tilt
    right_lift = right_closed + np.array([0.0, 0.0, lift_z - z0]) + tilt
    left_now, right_now = step_to(left_lift, right_lift, 92, left_now, right_now)
    step_to(left_lift, right_lift, 18, left_now, right_now)

    final_pos = data.xpos[object_bid].copy()
    lifted_height = float(final_pos[2] - (scenario["hz"] + 0.002))
    final_xy_error = float(np.linalg.norm(final_pos[:2] - center))
    slip = final_xy_error + max(0.0, 0.05 - lifted_height)
    drop = int(lifted_height < SUCCESS_LIFT_MARGIN)
    unsafe_force = int(max_force > CONTACT_LIMIT)
    energy = pusher_path + 0.0006 * contact_impulse
    task_ok = task_satisfied(scenario, candidate, clutter_collision)
    success = int(lifted_height > SUCCESS_LIFT_MARGIN and final_xy_error < 0.105 and object_finger_contacts > 8 and not unsafe_force and task_ok)
    score = success - 0.55 * drop - 0.35 * clutter_collision - 0.20 * unsafe_force - 0.25 * min(1.0, slip / 0.12) - 0.08 * min(1.0, energy / 1.0)
    return RolloutResult(
        success=success,
        lifted_height=lifted_height,
        final_xy_error=final_xy_error,
        slip=slip,
        drop=drop,
        collision=clutter_collision,
        unsafe_force=unsafe_force,
        max_contact_force=max_force,
        energy=energy,
        object_path=object_path,
        task_satisfied=task_ok,
        score=float(score),
    )


def object_energy(scenario: dict, candidate: dict) -> float:
    width = projected_width(scenario, candidate["angle"])
    center_err = math.hypot(candidate["center_x"], candidate["center_y"])
    width_err = abs(candidate["jaw_width"] - 0.92 * width) / max(width, 1e-4)
    aspect = scenario["hx"] / max(scenario["hy"], 1e-4)
    angle_pref = 0.15 * abs(math.sin(candidate["angle"] - scenario["yaw"])) if aspect > 1.7 else 0.0
    return 1.7 * center_err + width_err + angle_pref


def contact_energy(scenario: dict, candidate: dict) -> float:
    width = projected_width(scenario, candidate["angle"])
    squeeze = (width - candidate["jaw_width"]) / max(width, 1e-4)
    low_squeeze = max(0.0, 0.08 - squeeze)
    high_squeeze = max(0.0, squeeze - 0.28)
    friction_penalty = max(0.0, 0.52 - scenario["friction"]) * (0.9 + 1.5 * low_squeeze)
    return 2.0 * low_squeeze + 1.1 * high_squeeze + friction_penalty + 0.10 * abs(candidate["lift_tilt"])


def task_energy(scenario: dict, candidate: dict) -> float:
    if scenario["task"] == "avoid_face":
        return max(0.0, abs(math.cos(candidate["angle"] - scenario["forbidden_angle"])) - 0.62)
    if scenario["task"] == "handle_long_side":
        long_side = scenario["yaw"] + math.pi / 2
        return abs(math.sin(candidate["angle"] - long_side))
    if scenario["task"] == "clutter_safe":
        return 0.60 * clutter_clearance_energy(scenario, candidate)
    return 0.05 * abs(math.sin(candidate["angle"] - scenario["desired_angle"]))


def clutter_clearance_energy(scenario: dict, candidate: dict) -> float:
    if not scenario["clutter"]:
        return 0.0
    center = np.array([candidate["center_x"], candidate["center_y"]], dtype=float)
    n = np.array([math.cos(candidate["angle"]), math.sin(candidate["angle"])], dtype=float)
    width = max(candidate["jaw_width"], projected_width(scenario, candidate["angle"])) + 0.06
    finger_points = [center + n * width / 2, center - n * width / 2]
    risks = []
    for cx, cy, hx, hy in scenario["clutter"]:
        for p in finger_points:
            dx = max(0.0, abs(p[0] - cx) - hx)
            dy = max(0.0, abs(p[1] - cy) - hy)
            dist = math.hypot(dx, dy)
            risks.append(max(0.0, 0.055 - dist) / 0.055)
    return max(risks) if risks else 0.0


def feasibility_energy(scenario: dict, candidate: dict) -> float:
    width = candidate["jaw_width"]
    reach = math.hypot(candidate["center_x"], candidate["center_y"])
    jaw_penalty = max(0.0, width - 0.16) + max(0.0, 0.030 - width)
    force_penalty = max(0.0, candidate["closing_force"] - 1.16) * (0.55 + 0.6 * max(0.0, 0.45 - scenario["friction"]))
    return 1.8 * reach + 4.5 * jaw_penalty + force_penalty


def analytic_components(scenario: dict, candidate: dict) -> dict:
    return {
        "object_energy": object_energy(scenario, candidate),
        "contact_energy": contact_energy(scenario, candidate),
        "task_energy": task_energy(scenario, candidate),
        "collision_energy": clutter_clearance_energy(scenario, candidate),
        "feasibility_energy": feasibility_energy(scenario, candidate),
    }


def feature_vector(scenario: dict, candidate: dict) -> np.ndarray:
    comps = analytic_components(scenario, candidate)
    shape_ids = {
        "box": [1, 0, 0, 0],
        "cylinder": [0, 1, 0, 0],
        "bar": [0, 0, 1, 0],
        "l_shape": [0, 0, 0, 1],
        "t_shape": [0, 0, 0, 1],
    }[scenario["shape"]]
    task_ids = {
        "lift": [1, 0, 0, 0],
        "clutter_safe": [0, 1, 0, 0],
        "avoid_face": [0, 0, 1, 0],
        "handle_long_side": [0, 0, 0, 1],
    }[scenario["task"]]
    width = projected_width(scenario, candidate["angle"])
    return np.asarray(
        [
            scenario["hx"],
            scenario["hy"],
            scenario["hz"],
            scenario["radius"],
            scenario["friction"],
            scenario["mass"],
            math.sin(scenario["yaw"]),
            math.cos(scenario["yaw"]),
            float(len(scenario["clutter"])),
            candidate["center_x"],
            candidate["center_y"],
            math.sin(candidate["angle"]),
            math.cos(candidate["angle"]),
            candidate["jaw_width"],
            candidate["closing_force"],
            candidate["lift_tilt"],
            width,
            abs(candidate["jaw_width"] - width) / max(width, 1e-4),
            *shape_ids,
            *task_ids,
            comps["object_energy"],
            comps["contact_energy"],
            comps["task_energy"],
            comps["collision_energy"],
            comps["feasibility_energy"],
        ],
        dtype=np.float32,
    )


class TinyTransformerRanker(torch.nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.proj = torch.nn.Linear(input_dim, 32)
        layer = torch.nn.TransformerEncoderLayer(d_model=32, nhead=4, dim_feedforward=64, dropout=0.05, batch_first=True)
        self.encoder = torch.nn.TransformerEncoder(layer, num_layers=1)
        self.out = torch.nn.Linear(32, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = torch.relu(self.proj(x))
        return self.out(self.encoder(z)).squeeze(-1)


def run_training_rollout_task(task: tuple) -> dict:
    scenario, candidate, scene_id, cand_id = task
    result = rollout_grasp(scenario, candidate)
    row = {
        "scene_id": scene_id,
        "candidate_id": cand_id,
        "success": result.success,
        "score": f"{result.score:.5f}",
        "lifted_height": f"{result.lifted_height:.5f}",
        "slip": f"{result.slip:.5f}",
        "collision": result.collision,
        "unsafe_force": result.unsafe_force,
    }
    row.update({f"f{i}": f"{v:.6f}" for i, v in enumerate(feature_vector(scenario, candidate))})
    return row


def run_rollout_eval_task(task: dict) -> dict:
    method = task["method"]
    candidates = task["candidates"]
    scenario = task["scenario"]
    if method == "oracle_mujoco_grid":
        best = None
        best_idx = -1
        for idx, candidate in enumerate(candidates):
            result = rollout_grasp(scenario, candidate)
            if best is None or result.score > best.score:
                best = result
                best_idx = idx
        result = best
        chosen = candidates[best_idx]
        sampled = len(candidates)
    else:
        chosen = candidates[0]
        result = rollout_grasp(scenario, chosen)
        sampled = 1
    comps = analytic_components(scenario, chosen)
    return {
        "method": method.replace("ablation:", ""),
        "split": task["split"],
        "seed": task["seed"],
        "episode": task["episode"],
        "success": result.success,
        "lifted_height": f"{result.lifted_height:.5f}",
        "final_xy_error": f"{result.final_xy_error:.5f}",
        "slip": f"{result.slip:.5f}",
        "drop": result.drop,
        "collision": result.collision,
        "unsafe_force": result.unsafe_force,
        "max_contact_force": f"{result.max_contact_force:.5f}",
        "energy": f"{result.energy:.5f}",
        "task_satisfied": result.task_satisfied,
        "rollout_score": f"{result.score:.5f}",
        "sampled_candidates": sampled,
        "object_energy": f"{comps['object_energy']:.5f}",
        "contact_energy": f"{comps['contact_energy']:.5f}",
        "task_energy": f"{comps['task_energy']:.5f}",
        "collision_energy": f"{comps['collision_energy']:.5f}",
        "feasibility_energy": f"{comps['feasibility_energy']:.5f}",
        "shape": scenario["shape"],
        "task": scenario["task"],
        "friction": f"{scenario['friction']:.4f}",
        "jaw_width": f"{chosen['jaw_width']:.5f}",
        "angle": f"{chosen['angle']:.5f}",
    }


def run_tasks(tasks: list, worker_fn) -> list[dict]:
    if MAX_WORKERS == 1:
        return [worker_fn(task) for task in tasks]
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        return list(executor.map(worker_fn, tasks, chunksize=4))


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    tmp = path.with_suffix(".partial.csv")
    with tmp.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)


def build_training_data() -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    tasks = []
    scene_meta = []
    rng = np.random.default_rng(BASE_SEED + 77)
    train_splits = ["seen_simple", "unseen_dimensions", "slippery_contact", "task_constraint_shift", "clutter_collision"]
    for scene_id in range(TRAIN_SCENES):
        split = train_splits[scene_id % len(train_splits)]
        scenario = scenario_config(split, scene_id % 5, scene_id, rng)
        candidates = generate_candidates(scenario, rng, TRAIN_CANDIDATES)
        scene_meta.append((scenario, candidates))
        for cand_id, candidate in enumerate(candidates):
            tasks.append((scenario, candidate, scene_id, cand_id))
    rows = run_tasks(tasks, run_training_rollout_task)
    write_csv(RESULTS / "training_rollouts.csv", rows)
    fcols = [c for c in rows[0] if c.startswith("f")]
    X = np.asarray([[float(r[c]) for c in fcols] for r in rows], dtype=np.float32)
    y = np.asarray([int(r["success"]) for r in rows], dtype=np.int64)
    scene_ids = np.asarray([int(r["scene_id"]) for r in rows], dtype=np.int64)
    summary = {
        "rows": len(rows),
        "success_rate": float(y.mean()),
        "feature_dim": X.shape[1],
        "train_scenes": TRAIN_SCENES,
        "candidates_per_scene": TRAIN_CANDIDATES,
    }
    return X, y, scene_ids, summary


def train_models(X: np.ndarray, y: np.ndarray, scene_ids: np.ndarray) -> dict:
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    class_weight = "balanced" if 0 < y.mean() < 1 else None
    mlp = MLPClassifier(hidden_layer_sizes=(32, 16), activation="relu", alpha=1e-3, learning_rate_init=2e-3, max_iter=450, random_state=11)
    mlp.fit(Xs, y)
    rf_models = []
    for idx in range(3):
        rf = RandomForestClassifier(n_estimators=80, max_depth=6, min_samples_leaf=4, class_weight=class_weight, random_state=BASE_SEED % 10000 + idx)
        rf.fit(Xs, y)
        rf_models.append(rf)
    mono = LogisticRegression(max_iter=500, class_weight=class_weight, random_state=22)
    mono.fit(Xs[:, -5:], y)
    hgb = HistGradientBoostingClassifier(max_iter=110, max_leaf_nodes=15, learning_rate=0.06, random_state=33)
    hgb.fit(Xs, y)

    torch.manual_seed(42)
    scene_count = int(scene_ids.max()) + 1
    seqs = []
    labels = []
    for sid in range(scene_count):
        mask = scene_ids == sid
        seqs.append(Xs[mask])
        labels.append(y[mask])
    Xseq = torch.tensor(np.stack(seqs), dtype=torch.float32)
    yseq = torch.tensor(np.stack(labels), dtype=torch.float32)
    transformer = TinyTransformerRanker(Xseq.shape[-1])
    opt = torch.optim.AdamW(transformer.parameters(), lr=2e-3, weight_decay=1e-4)
    pos_weight = torch.tensor([(len(y) - y.sum()) / max(1, y.sum())], dtype=torch.float32)
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    transformer.train()
    for _ in range(55):
        opt.zero_grad(set_to_none=True)
        logits = transformer(Xseq)
        loss = loss_fn(logits, yseq)
        loss.backward()
        opt.step()
    transformer.eval()
    with torch.no_grad():
        train_prob = torch.sigmoid(transformer(Xseq)).numpy().reshape(-1)
    torch_acc = float(((train_prob > 0.5).astype(int) == y).mean())

    write_csv(
        RESULTS / "training_summary.csv",
        [
            {
                "training_rows": len(y),
                "success_rate": f"{float(y.mean()):.4f}",
                "mlp_train_accuracy": f"{float(mlp.score(Xs, y)):.4f}",
                "transformer_train_accuracy": f"{torch_acc:.4f}",
                "mono_train_accuracy": f"{float(mono.score(Xs[:, -5:], y)):.4f}",
            }
        ],
    )
    return {"scaler": scaler, "mlp": mlp, "ensemble": rf_models, "mono": mono, "hgb": hgb, "transformer": transformer}


def predict_transformer(models: dict, X: np.ndarray) -> np.ndarray:
    Xs = models["scaler"].transform(X)
    with torch.no_grad():
        logits = models["transformer"](torch.tensor(Xs[None, :, :], dtype=torch.float32)).numpy()[0]
    return np.asarray(sigmoid(logits), dtype=float)


def learned_scores(models: dict, X: np.ndarray, model_name: str) -> np.ndarray:
    Xs = models["scaler"].transform(X)
    if model_name == "mlp":
        return models["mlp"].predict_proba(Xs)[:, 1]
    if model_name == "transformer":
        return predict_transformer(models, X)
    if model_name == "ensemble":
        probs = np.stack([m.predict_proba(Xs)[:, 1] for m in models["ensemble"]], axis=0)
        return probs.mean(axis=0) - 0.35 * probs.std(axis=0)
    if model_name == "mono":
        return models["mono"].predict_proba(Xs[:, -5:])[:, 1]
    return models["hgb"].predict_proba(Xs)[:, 1]


def analytic_score(scenario: dict, candidate: dict, kind: str) -> float:
    comps = analytic_components(scenario, candidate)
    if kind == "antipodal_geometry":
        return -(1.20 * comps["object_energy"] + 0.20 * comps["feasibility_energy"] + 0.20 * comps["collision_energy"])
    if kind == "force_closure_score":
        return -(0.75 * comps["object_energy"] + 1.35 * comps["contact_energy"] + 0.25 * comps["task_energy"] + 0.15 * comps["feasibility_energy"])
    if kind == "cem_grasp_search":
        return -(0.65 * comps["object_energy"] + 1.00 * comps["contact_energy"] + 0.75 * comps["task_energy"] + 0.60 * comps["collision_energy"] + 0.25 * comps["feasibility_energy"])
    return -sum(comps.values())


def ebm_score(scenario: dict, candidate: dict, transformer_prob: float, ablation: str | None = None, mono_prob: float | None = None) -> float:
    comps = analytic_components(scenario, candidate)
    weights = {
        "object_energy": 0.60,
        "contact_energy": 0.95,
        "task_energy": 0.70,
        "collision_energy": 0.75,
        "feasibility_energy": 0.45,
    }
    if ablation == "no_object_energy":
        weights["object_energy"] = 0.0
    if ablation == "no_contact_energy":
        weights["contact_energy"] = 0.0
    if ablation == "no_task_energy":
        weights["task_energy"] = 0.0
    if ablation == "no_collision_energy":
        weights["collision_energy"] = 0.0
    if ablation == "no_feasibility_energy":
        weights["feasibility_energy"] = 0.0
    energy = sum(weights[k] * comps[k] for k in weights)
    context = transformer_prob
    if ablation == "no_transformer_context":
        context = 0.50
    if ablation == "monolithic_scalar_energy_only":
        context = mono_prob if mono_prob is not None else 0.50
        energy = 0.0
    return 1.15 * context - energy


def choose_candidates(method: str, scenario: dict, candidates: list[dict], models: dict, rng: np.random.Generator) -> list[dict]:
    if method == "random_grasp":
        return [candidates[int(rng.integers(0, len(candidates)))]]
    X = np.stack([feature_vector(scenario, c) for c in candidates])
    if method in {"antipodal_geometry", "force_closure_score", "cem_grasp_search"}:
        scores = np.asarray([analytic_score(scenario, c, method) for c in candidates])
    elif method == "mlp_energy_model":
        scores = learned_scores(models, X, "mlp")
    elif method == "transformer_policy_ranker":
        scores = learned_scores(models, X, "transformer")
    elif method == "ensemble_uncertainty_ranker":
        scores = learned_scores(models, X, "ensemble")
    elif method == "oracle_mujoco_grid":
        analytic = np.asarray([analytic_score(scenario, c, "cem_grasp_search") for c in candidates])
        top = np.argsort(analytic)[-ORACLE_CANDIDATES:]
        return [candidates[int(i)] for i in top]
    else:
        ablation = None
        if method.startswith("ablation:"):
            ablation = method.split(":", 1)[1].replace("full_ebm_transformer_compositional", "")
            if ablation == "":
                ablation = None
        tprob = learned_scores(models, X, "transformer")
        mono = learned_scores(models, X, "mono")
        scores = np.asarray([ebm_score(scenario, c, tprob[idx], ablation, mono[idx]) for idx, c in enumerate(candidates)])
    return [candidates[int(np.argmax(scores))]]


def make_eval_tasks(methods: list[str], splits: list[str], models: dict, episodes_per_seed: int, stress_level: float | None = None) -> list[dict]:
    tasks = []
    for method in methods:
        for split in splits:
            for seed in SEEDS:
                for episode in range(episodes_per_seed):
                    rng = np.random.default_rng(BASE_SEED + stable_int(method) + stable_int(split) * 11 + seed * 1009 + episode * 7919)
                    scenario = scenario_config(split, seed, episode, rng, stress_level=stress_level)
                    candidates = generate_candidates(scenario, rng, MAIN_CANDIDATES)
                    chosen = choose_candidates(method, scenario, candidates, models, rng)
                    tasks.append({"method": method, "split": split if stress_level is None else f"stress_{stress_level:.2f}", "seed": seed, "episode": episode, "scenario": scenario, "candidates": chosen})
    return tasks


def summarize(rows: list[dict], group_keys: list[str]) -> list[dict]:
    grouped: dict[tuple, list[dict]] = {}
    for row in rows:
        key = tuple(row[k] for k in group_keys)
        grouped.setdefault(key, []).append(row)
    output = []
    for key, group in sorted(grouped.items()):
        success = [float(r["success"]) for r in group]
        slip = [float(r["slip"]) for r in group]
        drop = [float(r["drop"]) for r in group]
        collision = [float(r["collision"]) for r in group]
        force = [float(r["unsafe_force"]) for r in group]
        energy = [float(r["energy"]) for r in group]
        sampled = [float(r["sampled_candidates"]) for r in group]
        out = {k: v for k, v in zip(group_keys, key)}
        out.update(
            {
                "mean_success": f"{float(np.mean(success)):.4f}",
                "ci95_success": f"{ci95(success):.4f}",
                "mean_slip": f"{float(np.mean(slip)):.4f}",
                "ci95_slip": f"{ci95(slip):.4f}",
                "drop_rate": f"{float(np.mean(drop)):.4f}",
                "collision_rate": f"{float(np.mean(collision)):.4f}",
                "unsafe_force_rate": f"{float(np.mean(force)):.4f}",
                "mean_energy": f"{float(np.mean(energy)):.4f}",
                "sampled_candidates": f"{float(np.mean(sampled)):.2f}",
                "episodes": len(group),
                "seeds": len({r["seed"] for r in group}),
            }
        )
        output.append(out)
    return output


def seed_metrics(rows: list[dict]) -> list[dict]:
    return summarize(rows, ["method", "split", "seed"])


def pairwise_stats(seed_rows: list[dict], split: str = "combined_composition_shift") -> list[dict]:
    proposed = "ebm_transformer_compositional"
    metric_map = {
        (r["method"], r["split"], r["seed"]): float(r["mean_success"])
        for r in seed_rows
        if r["split"] == split
    }
    rows = []
    for method in METHODS:
        if method == proposed:
            continue
        diffs = []
        for seed in SEEDS:
            p_key = (proposed, split, seed)
            b_key = (method, split, seed)
            if p_key in metric_map and b_key in metric_map:
                diffs.append(metric_map[p_key] - metric_map[b_key])
        if not diffs:
            continue
        mean_diff = float(np.mean(diffs))
        sd = float(np.std(diffs, ddof=1)) if len(diffs) > 1 else 0.0
        t_stat = mean_diff / (sd / math.sqrt(len(diffs)) + 1e-9)
        rows.append(
            {
                "split": split,
                "baseline": method,
                "mean_success_diff_vs_ebm": f"{mean_diff:.4f}",
                "paired_t_approx": f"{t_stat:.4f}",
                "normal_approx_p": f"{normal_p_from_t(t_stat):.4f}",
                "seeds": len(diffs),
            }
        )
    return rows


def plot_success(metrics: list[dict], path: Path) -> None:
    selected = ["force_closure_score", "cem_grasp_search", "mlp_energy_model", "transformer_policy_ranker", "ensemble_uncertainty_ranker", "ebm_transformer_compositional", "oracle_mujoco_grid"]
    x = np.arange(len(MAIN_SPLITS))
    width = 0.10
    fig, ax = plt.subplots(figsize=(12, 5))
    for idx, method in enumerate(selected):
        vals = []
        for split in MAIN_SPLITS:
            match = [r for r in metrics if r["method"] == method and r["split"] == split]
            vals.append(float(match[0]["mean_success"]) if match else 0.0)
        ax.bar(x + (idx - len(selected) / 2) * width + width / 2, vals, width, label=method.replace("_", " "))
    ax.set_ylabel("Grasp success rate")
    ax.set_ylim(0.0, 1.0)
    ax.set_xticks(x)
    ax.set_xticklabels([s.replace("_", "\n") for s in MAIN_SPLITS], fontsize=8)
    ax.legend(fontsize=7, ncol=2)
    ax.set_title("MuJoCo compositional grasping success by split")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_ablation(metrics: list[dict], path: Path) -> None:
    vals = [(r["method"], float(r["mean_success"]), float(r["ci95_success"])) for r in metrics if r["split"] == "combined_composition_shift"]
    vals.sort(key=lambda item: item[1], reverse=True)
    fig, ax = plt.subplots(figsize=(10, 4.8))
    x = np.arange(len(vals))
    ax.bar(x, [v[1] for v in vals], yerr=[v[2] for v in vals], color="#70543e")
    ax.set_xticks(x)
    ax.set_xticklabels([v[0].replace("_", "\n") for v in vals], fontsize=8)
    ax.set_ylabel("Combined-shift success")
    ax.set_ylim(0.0, 1.0)
    ax.set_title("Compositional energy ablations")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_stress(stress_metrics: list[dict], path: Path) -> None:
    selected = ["cem_grasp_search", "mlp_energy_model", "transformer_policy_ranker", "ensemble_uncertainty_ranker", "ebm_transformer_compositional"]
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    for method in selected:
        xs, ys = [], []
        for row in stress_metrics:
            if row["method"] == method:
                xs.append(float(row["stress_level"]))
                ys.append(float(row["mean_success"]))
        order = np.argsort(xs)
        ax.plot(np.asarray(xs)[order], np.asarray(ys)[order], marker="o", label=method.replace("_", " "))
    ax.set_xlabel("Composition stress")
    ax.set_ylabel("Success rate")
    ax.set_ylim(0.0, 1.0)
    ax.legend(fontsize=8)
    ax.set_title("Stress sweep: friction + clutter + aspect ratio + task constraints")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_safety(metrics: list[dict], path: Path) -> None:
    selected = ["force_closure_score", "cem_grasp_search", "mlp_energy_model", "transformer_policy_ranker", "ebm_transformer_compositional"]
    x = np.arange(len(MAIN_SPLITS))
    fig, ax = plt.subplots(figsize=(10, 4.8))
    for method in selected:
        vals = []
        for split in MAIN_SPLITS:
            match = [r for r in metrics if r["method"] == method and r["split"] == split]
            vals.append(float(match[0]["collision_rate"]) + float(match[0]["unsafe_force_rate"]) if match else 0.0)
        ax.plot(x, vals, marker="o", label=method.replace("_", " "))
    ax.set_ylabel("Collision + unsafe-force rate")
    ax.set_xticks(x)
    ax.set_xticklabels([s.replace("_", "\n") for s in MAIN_SPLITS], fontsize=8)
    ax.legend(fontsize=8)
    ax.set_title("Safety failures by split")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def make_negative_cases() -> list[dict]:
    return [
        {
            "case": "deformable_object",
            "expected_behavior": "contact energy should adapt to changing geometry",
            "observed_failure_mode": "rigid-body MuJoCo energies cannot represent deformation or post-contact shape change",
            "submission_implication": "needs deformable benchmark before claiming general compositional grasping",
        },
        {
            "case": "occluded_geometry",
            "expected_behavior": "transformer context should infer hidden geometry from scene cues",
            "observed_failure_mode": "candidate features assume known object extent; occlusion can make all rankers overconfident",
            "submission_implication": "requires perception uncertainty and real visual input",
        },
        {
            "case": "adversarial_low_friction_coating",
            "expected_behavior": "contact energy should penalize slip",
            "observed_failure_mode": "without tactile probing, learned rankers confuse shape-compatible grasps with stable grasps",
            "submission_implication": "needs tactile or material sensing for deployment claims",
        },
        {
            "case": "semantic_task_ambiguity",
            "expected_behavior": "task energy should choose the correct functional contact",
            "observed_failure_mode": "all physical energies can be satisfied while the semantic instruction is wrong",
            "submission_implication": "scope is physical grasp selection, not language grounding",
        },
    ]


def main() -> None:
    X, y, scene_ids, train_summary = build_training_data()
    models = train_models(X, y, scene_ids)

    main_tasks = make_eval_tasks(METHODS, MAIN_SPLITS, models, EPISODES_PER_SEED)
    raw_rows = run_tasks(main_tasks, run_rollout_eval_task)
    write_csv(RESULTS / "ebm_grasping_raw.csv", raw_rows)
    seed_rows = seed_metrics(raw_rows)
    write_csv(RESULTS / "raw_seed_metrics.csv", seed_rows)
    metrics = summarize(raw_rows, ["method", "split"])
    write_csv(RESULTS / "ebm_grasping_metrics.csv", metrics)
    write_csv(RESULTS / "metrics.csv", metrics)
    pairwise = pairwise_stats(seed_rows)
    write_csv(RESULTS / "ebm_grasping_pairwise.csv", pairwise)
    write_csv(RESULTS / "pairwise_stats.csv", pairwise)

    ablation_methods = [f"ablation:{name}" for name in ABLATIONS]
    ablation_tasks = make_eval_tasks(ablation_methods, ["combined_composition_shift"], models, ABLATION_EPISODES_PER_SEED)
    ablation_rows = run_tasks(ablation_tasks, run_rollout_eval_task)
    write_csv(RESULTS / "ebm_grasping_ablation_raw.csv", ablation_rows)
    ablation_metrics = summarize(ablation_rows, ["method", "split"])
    write_csv(RESULTS / "ebm_grasping_ablation.csv", ablation_metrics)
    write_csv(RESULTS / "ablation_metrics.csv", ablation_metrics)

    stress_methods = ["cem_grasp_search", "mlp_energy_model", "transformer_policy_ranker", "ensemble_uncertainty_ranker", "ebm_transformer_compositional"]
    stress_rows = []
    for level in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
        stress_tasks = make_eval_tasks(stress_methods, ["stress_sweep"], models, STRESS_EPISODES_PER_SEED, stress_level=level)
        stress_rows.extend(run_tasks(stress_tasks, run_rollout_eval_task))
    stress_metrics = summarize(stress_rows, ["method", "split"])
    stress_output = []
    for row in stress_metrics:
        out = dict(row)
        out["stress_level"] = out["split"].replace("stress_", "")
        stress_output.append(out)
    write_csv(RESULTS / "stress_sweep.csv", stress_output)
    write_csv(FIGURES / "stress_curve_data.csv", stress_output)

    write_csv(RESULTS / "negative_cases.csv", make_negative_cases())
    plot_success(metrics, FIGURES / "ebm_grasping_success_by_split.png")
    plot_ablation(ablation_metrics, FIGURES / "ebm_grasping_ablation_success.png")
    plot_stress(stress_output, FIGURES / "ebm_grasping_stress_sweep.png")
    plot_safety(metrics, FIGURES / "ebm_grasping_safety_failures.png")

    combined = {r["method"]: r for r in metrics if r["split"] == "combined_composition_shift"}
    ablation_combined = {r["method"]: r for r in ablation_metrics if r["split"] == "combined_composition_shift"}
    proposed = combined["ebm_transformer_compositional"]
    best_non_oracle = max(
        (r for m, r in combined.items() if m not in {"ebm_transformer_compositional", "oracle_mujoco_grid"}),
        key=lambda r: float(r["mean_success"]),
    )
    terminal = "STRONG_REVISE"
    reason = "EBM has real MuJoCo evidence but still needs hardware/public benchmark and deeper manual related work"
    if float(proposed["mean_success"]) <= float(best_non_oracle["mean_success"]) + 0.025:
        terminal = "KILL_ARCHIVE"
        reason = "EBM compositional ranker is matched or beaten by a non-oracle baseline on combined composition shift"
    mono = ablation_combined.get("monolithic_scalar_energy_only")
    full = ablation_combined.get("full_ebm_transformer_compositional")
    if mono and full and float(mono["mean_success"]) >= float(full["mean_success"]) - 0.025:
        terminal = "KILL_ARCHIVE"
        reason = "monolithic scalar energy ablation matches the full compositional EBM"

    with (RESULTS / "summary.txt").open("w", encoding="utf-8") as handle:
        handle.write("Paper 68 real MuJoCo EBM transformer compositional grasping rebuild\n")
        handle.write(f"Seeds: {SEEDS}; episodes per seed: {EPISODES_PER_SEED}; workers: {MAX_WORKERS}\n")
        handle.write(
            "Training rows: {rows}; training success rate: {success_rate:.4f}; feature dim: {feature_dim}\n".format(**train_summary)
        )
        handle.write("Main rows: %d; ablation rows: %d; stress rows: %d\n" % (len(raw_rows), len(ablation_rows), len(stress_rows)))
        handle.write(f"Terminal decision: {terminal}\n")
        handle.write(f"Terminal reason: {reason}\n")
        handle.write("\nCombined-composition-shift main results:\n")
        for method in METHODS:
            row = combined[method]
            handle.write(
                f"- {method}: success={row['mean_success']} ci95={row['ci95_success']} "
                f"slip={row['mean_slip']} drop={row['drop_rate']} collision={row['collision_rate']} "
                f"unsafe={row['unsafe_force_rate']} energy={row['mean_energy']} sampled={row['sampled_candidates']}\n"
            )
        handle.write("\nCombined-composition-shift ablations:\n")
        for method, row in sorted(ablation_combined.items()):
            handle.write(
                f"- {method}: success={row['mean_success']} ci95={row['ci95_success']} "
                f"collision={row['collision_rate']} unsafe={row['unsafe_force_rate']} sampled={row['sampled_candidates']}\n"
            )
        handle.write("\nPairwise combined-shift comparisons vs ebm_transformer_compositional:\n")
        for row in pairwise:
            handle.write(
                f"- {row['baseline']}: diff={row['mean_success_diff_vs_ebm']} "
                f"t={row['paired_t_approx']} p={row['normal_approx_p']}\n"
            )

    print(f"wrote Paper 68 MuJoCo evidence to {RESULTS}")


if __name__ == "__main__":
    main()

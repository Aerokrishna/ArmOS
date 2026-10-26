#!/usr/bin/env python3
"""
Closed-loop validation of the admittance controller (armos_control/admittance.py) in MuJoCo.

Everything runs in one process, on the same topics as the real FR3:

  CyRoSimulator   dual_franka config (torque actuation, wrist F/T sensors from
                  dual_panda.xml, gravity on) + a table under the TCP, added here in memory only
  ArmosBridge     sim_bridge.py: /left/{current_pose, franka_robot_state_broadcaster/...}
                  out, /left/target_pose in (crisp's Cartesian impedance), /clock
  Controller      admittance.py with config/admittance.yaml (the FR3 parameters)
  Scenario        this file -> /left/admittance/{control_mode,target_pose,target_wrench}

--axes tcp|base selects the controller's axes. The scenario is written in base-frame
axes; for tcp it is mapped onto the tool axes (the tool points down, axis-aligned).

Phases (sim time, ~wall time):
  settle    0-2 s   all axes pose, hold start pose (controller tares the F/T here)
  comply    2-7 s   x = force axis, target 0 N; external +x push on the hand
                    for 2.5 s, then released -> moves at ~min(F/D_force, cap), then stops
  pose      7-16 s  all axes pose; step to offset position + rotated orientation
  approach 16-21 s  all axes pose; gripper pointing down above the table
  press    21-40 s  z = force axis with target Fz, others pose -> approach at
                    ~min(|Fz|/D_force, cap), touch without bouncing, hold the force

The controller runs on sim time (use_sim_time + /clock from the bridge), so its
dt matches the physics even if the machine can't keep real time. Before t=0 the
sim runs 3 s with the bridge publishing: the bridge holds q_home in joint
impedance and the controller tares its wrench (2 s) and starts holding the pose.

Pass/fail uses MuJoCo ground truth (site pose, table contact normal force), not
the controller's own telemetry.

The bridge publishes the wrist F/T like Franka's external wrench estimate
(payload removed, base frame, robot-on-environment); the controller tares,
filters and moves it to the TCP like on the robot.

Run (venv active, distinct ROS_DOMAIN_ID for parallel runs):
  source /opt/ros/humble/setup.bash
  source ~/armos_ws/install/setup.bash
  python test/test_admittance_sim.py                          # viewer + plot, tcp axes
  python test/test_admittance_sim.py --headless --axes base   # smoke test, base axes
"""

import argparse
import csv
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import mujoco
import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.signals import SignalHandlerOptions
from geometry_msgs.msg import PoseStamped, WrenchStamped
from rclpy.parameter import Parameter
from std_msgs.msg import UInt8MultiArray

from armos_control import sim_bridge as armos_ros_sim
from armos_control.sim_bridge import ArmosBridge
from armos_control.admittance import AdmittanceController
from armos_sim.model.actuation import swap_to_motors
from armos_sim.runtime import simulator as sim_module
from armos_sim import CyRoSimulator

PARAMS_FILE = Path(__file__).resolve().parents[1] / "config" / "admittance.yaml"

ARM = "left"
HAND_BODY = "left_hand"
TABLE = "admittance_test_table"


# --------------------------------------------------------------------------- #
# Small SO(3) helpers                                                         #
# --------------------------------------------------------------------------- #
def rot_x(deg):
    a = np.radians(deg)
    return np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]])


def rot_z(deg):
    a = np.radians(deg)
    return np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])


def rot_angle_deg(R_a, R_b):
    c = np.clip((np.trace(R_a.T @ R_b) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(c)))


def rot_to_quat_wxyz(R):
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.asarray(R, dtype=float).reshape(9))
    return q


# --------------------------------------------------------------------------- #
# Simulator + table (user XML/YAML untouched)                                 #
# --------------------------------------------------------------------------- #
def _augment_model(scene_path, arm_configs, arm_cfg, table_drop, loaded_model):
    spec = mujoco.MjSpec.from_file(str(scene_path))

    # Same actuation and physics as the loader's model: torque arms on motors, the YAML timestep.
    raw = spec.compile()
    for cfg in arm_configs.values():
        if cfg.actuation == "torque":
            servos = sim_module._RobotLoader._actuators_for_joints(raw, cfg.joint_names)
            swap_to_motors(spec, raw, cfg.joint_names, servos)
    spec.option.timestep = loaded_model.opt.timestep

    # Place the table relative to the TCP at q_home.
    m = spec.compile()
    d = mujoco.MjData(m)
    for name, q in zip(arm_cfg.joint_names, arm_cfg.q_home):
        d.qpos[m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, name)]] = q
    mujoco.mj_forward(m, d)
    tcp = d.site_xpos[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, arm_cfg.ee_site)].copy()

    top = tcp[2] - table_drop
    center = np.array([tcp[0] + 0.05, tcp[1], top - 0.02])
    spec.worldbody.add_geom(name=TABLE, type=mujoco.mjtGeom.mjGEOM_BOX,
                            size=[0.15, 0.15, 0.02], pos=center,
                            rgba=[0.55, 0.42, 0.3, 1.0])
    return spec.compile(), center, top


def build_sim(table_drop):
    """Construct CyRoSimulator through its normal YAML path, swapping in the model with the table."""
    original = sim_module._RobotLoader.from_yaml
    info = {}

    def patched(config_path):
        scene_path, arm_configs, aux = original(config_path)
        if not (arm_configs[ARM].wrench_force_sensor and arm_configs[ARM].wrench_torque_sensor):
            raise RuntimeError(f"config has no wrench sensors for arm '{ARM}'")
        model, center, top = _augment_model(scene_path, arm_configs, arm_configs[ARM], table_drop,
                                            aux["_preloaded_model"])
        aux["_preloaded_model"] = model
        info.update(table_center=center, table_top=top)
        return scene_path, arm_configs, aux

    sim_module._RobotLoader.from_yaml = staticmethod(patched)
    try:
        config = armos_ros_sim._resolve_runtime_config(armos_ros_sim._resolve_config_path(None))
        sim = CyRoSimulator(config_path=config)
    finally:
        sim_module._RobotLoader.from_yaml = staticmethod(original)
    return sim, info


# --------------------------------------------------------------------------- #
# Scenario                                                                    #
# --------------------------------------------------------------------------- #
@dataclass
class Phase:
    name: str
    t0: float
    t1: float
    mask: list                      # 1 = pose axis, 0 = force axis (base-frame axes)
    pos: np.ndarray                 # target position, base frame
    rot: np.ndarray                 # target rotation, base frame
    wrench: np.ndarray = field(default_factory=lambda: np.zeros(6))   # base-frame axes, robot-on-env
    push: np.ndarray = field(default_factory=lambda: np.zeros(3))   # world N on hand
    push_window: tuple = (0.0, 0.0)                                  # sim time

    def push_at(self, t):
        return self.push if self.push_window[0] <= t < self.push_window[1] else np.zeros(3)


def make_phases(p0, R0, R_base, base_world, table_center, push_n, fz):
    table_xy_base = R_base.T @ (table_center - base_world)
    p_above = np.array([table_xy_base[0], table_xy_base[1], p0[2]])
    press_wrench = np.array([0.0, 0.0, fz, 0.0, 0.0, 0.0])
    return [
        Phase("settle", 0.0, 2.0, [1] * 6, p0, R0),
        Phase("comply", 2.0, 7.0, [0, 1, 1, 1, 1, 1], p0, R0,
              push=np.array([push_n, 0.0, 0.0]), push_window=(2.0, 4.5)),
        Phase("pose", 7.0, 16.0, [1] * 6, p0 + np.array([0.0, -0.08, 0.06]),
              rot_z(20.0) @ rot_x(15.0) @ R0),
        Phase("approach", 16.0, 21.0, [1] * 6, p_above, R0),
        Phase("press", 21.0, 40.0, [1, 1, 0, 1, 1, 1], p_above, R0, wrench=press_wrench),
    ]


def axis_map(R0, axes):
    """Base axis i -> (law axis k, sign): the scenario's base-frame masks / wrenches in the law's axes.

    tcp: the tool must be axis-aligned with the base (R0 = TCP orientation at the start).
    """
    if axes == "base":
        return [(i, 1.0) for i in range(3)]
    out = []
    for i in range(3):
        k = int(np.argmax(np.abs(R0[i, :])))
        if abs(R0[i, k]) < np.cos(np.radians(10.0)):
            raise RuntimeError("tcp axes: the start TCP is not axis-aligned with the base")
        out.append((k, float(np.sign(R0[i, k]))))
    return out


class ScenarioDriver(Node):
    def __init__(self, namespace, amap):
        super().__init__("admittance_test_driver", namespace=namespace)
        self._amap = amap
        self._mode = self.create_publisher(UInt8MultiArray, "admittance/control_mode", 10)
        self._pose = self.create_publisher(PoseStamped, "admittance/target_pose", 10)
        self._wrench = self.create_publisher(WrenchStamped, "admittance/target_wrench", 10)

    def to_law(self, mask, wrench):
        m, w = [1] * 6, np.zeros(6)
        for i, (k, sgn) in enumerate(self._amap):
            m[k], m[3 + k] = int(mask[i]), int(mask[3 + i])
            w[k], w[3 + k] = sgn * wrench[i], sgn * wrench[3 + i]
        return m, w

    def publish(self, ph: Phase):
        mask, wrench = self.to_law(ph.mask, ph.wrench)
        self._mode.publish(UInt8MultiArray(data=mask))
        pose = PoseStamped()
        pose.header.frame_id = "base"
        p = pose.pose
        p.position.x, p.position.y, p.position.z = (float(v) for v in ph.pos)
        w, x, y, z = rot_to_quat_wxyz(ph.rot)
        p.orientation.w, p.orientation.x = float(w), float(x)
        p.orientation.y, p.orientation.z = float(y), float(z)
        self._pose.publish(pose)
        wr = WrenchStamped()   # empty frame_id: the controller's axes
        wr.wrench.force.x, wr.wrench.force.y, wr.wrench.force.z = (float(v) for v in wrench[:3])
        wr.wrench.torque.x, wr.wrench.torque.y, wr.wrench.torque.z = (float(v) for v in wrench[3:])
        self._wrench.publish(wr)


# --------------------------------------------------------------------------- #
# Ground truth + visualisation                                                #
# --------------------------------------------------------------------------- #
class Truth:
    def __init__(self, sim):
        self.m, self.d = sim._mj.model, sim._mj.data
        cfg = sim.arm_config(ARM)
        self.R_base, self.base = cfg.base_world_rot, cfg.base_world
        self.ee_site = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_SITE, cfg.ee_site)
        self.hand = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, HAND_BODY)
        self.table = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_GEOM, TABLE)
        root = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, cfg.start_body)
        self.arm_bodies = {b for b in range(self.m.nbody) if self.m.body_rootid[b] == root}
        self._f6 = np.zeros(6)

    def tcp_base(self):
        p = self.d.site_xpos[self.ee_site]
        R = self.d.site_xmat[self.ee_site].reshape(3, 3)
        return self.R_base.T @ (p - self.base), self.R_base.T @ R

    def to_world(self, p, R):
        return self.R_base @ p + self.base, self.R_base @ R

    def table_normal_force(self):
        total = 0.0
        for i in range(self.d.ncon):
            c = self.d.contact[i]
            if self.table not in (c.geom1, c.geom2):
                continue
            other = c.geom2 if c.geom1 == self.table else c.geom1
            if self.m.geom_bodyid[other] in self.arm_bodies:
                mujoco.mj_contactForce(self.m, self.d, i, self._f6)
                total += self._f6[0]
        return total


AXIS_RGB = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32)


def _add_connector(scn, gtype, width, a, b, rgba):
    if scn.ngeom >= scn.maxgeom:
        return
    g = scn.geoms[scn.ngeom]
    mujoco.mjv_initGeom(g, gtype, np.zeros(3), np.zeros(3), np.zeros(9),
                        np.asarray(rgba, dtype=np.float32))
    mujoco.mjv_connector(g, gtype, width, np.asarray(a, float), np.asarray(b, float))
    scn.ngeom += 1


def draw_scene(scn, frames, arrows):
    scn.ngeom = 0
    for p, R, alpha, length, width in frames:
        for k in range(3):
            _add_connector(scn, mujoco.mjtGeom.mjGEOM_CAPSULE, width, p, p + length * R[:, k],
                           [*AXIS_RGB[k], alpha])
    for a, b, rgba in arrows:
        _add_connector(scn, mujoco.mjtGeom.mjGEOM_ARROW, 0.01, a, b, rgba)


# --------------------------------------------------------------------------- #
# Evaluation                                                                  #
# --------------------------------------------------------------------------- #
def evaluate(log, phases, fz, d_force, v_cap, push_n, sim_modes):
    t = log["t"]
    pos = log["pos"]
    results = []

    def sel(t0, t1):
        return (t >= t0) & (t < t1)

    def check(name, ok, detail):
        results.append((name, bool(ok), detail))

    def vel(mask):
        idx = np.flatnonzero(mask)
        if idx.size < 3:
            return np.zeros((1, 3))
        return np.gradient(pos[idx], t[idx], axis=0)

    ph = {p.name: p for p in phases}

    age = log["telem_age"][t >= 0.5]
    check("harness: controller saw fresh telemetry", np.nanmax(age) < 0.05,
          f"max age {np.nanmax(age) * 1e3:.0f} ms of sim time (< 50)")
    check("sim arm on crisp's Cartesian impedance", sim_modes == {"CARTESIAN_IMPEDANCE"},
          f"sim modes seen: {sorted(sim_modes)}")

    s = ph["settle"]
    m = sel(s.t1 - 0.5, s.t1)
    drift = np.linalg.norm(pos[m] - s.pos, axis=1).max()
    check("settle: holds start pose", drift < 0.002, f"max drift {drift * 1e3:.1f} mm (< 2)")

    c = ph["comply"]
    on = sel(*c.push_window)
    i_on, i_off = np.flatnonzero(on)[[0, -1]]
    dx = pos[i_off, 0] - pos[i_on, 0]
    off_axis = np.abs(pos[on][:, 1:] - pos[i_on, 1:]).max()
    rot_drift = max(rot_angle_deg(log["R"][i_on], log["R"][i]) for i in np.flatnonzero(on))
    expected = min(push_n / d_force, v_cap) * (t[i_off] - t[i_on])
    check("comply: moves along +x at ~min(F/D, cap)", 0.6 * expected < dx < 1.3 * expected,
          f"dx {dx * 1e3:+.1f} mm, expected {expected * 1e3:+.1f} (x0.6..1.3)")
    check("comply: off-axis position held", off_axis < 0.005,
          f"max |dy|,|dz| {off_axis * 1e3:.1f} mm (< 5)")
    check("comply: orientation held", rot_drift < 2.0, f"max {rot_drift:.2f} deg (< 2)")
    after = sel(c.t1 - 1.5, c.t1)
    vx_after = np.abs(vel(after)[:, 0]).mean()
    back = pos[i_off, 0] - pos[after][-1, 0]
    check("comply: stops after release", vx_after < 0.003, f"mean |vx| {vx_after * 1e3:.1f} mm/s (< 3)")
    check("comply: no spring-back on force axis", back < 0.005,
          f"x moved back {back * 1e3:+.1f} mm (< 5)")

    for name in ("pose", "approach"):
        p = ph[name]
        m = sel(p.t1 - 1.0, p.t1)
        perr = np.linalg.norm(pos[m] - p.pos, axis=1).mean()
        rerr = np.mean([rot_angle_deg(p.rot, R) for R in log["R"][m]])
        check(f"{name}: position reached", perr < 0.005, f"err {perr * 1e3:.1f} mm (< 5)")
        check(f"{name}: orientation reached", rerr < 2.0, f"err {rerr:.2f} deg (< 2)")
        f_free = np.linalg.norm(log["wrench"][m][:, :3], axis=1).max()
        check(f"{name}: free-space wrench ~0 (gravity comp)", f_free < 0.5,
              f"max |F| {f_free:.2f} N (< 0.5)")

    p = ph["press"]
    whole = sel(p.t0, p.t1)
    contact = log["contact"]
    hit = np.flatnonzero(whole & (contact > 0.5))
    check("press: makes contact with the table", hit.size > 0,
          f"first contact at t={t[hit[0]]:.2f} s" if hit.size else "never touched")
    target = abs(fz)
    if hit.size:
        after_touch = whole & (t >= t[hit[0]] + 0.5)
        peak = contact[whole].max()
        low = contact[after_touch].min()
        check("press: no overshoot on touch-down", peak < 1.5 * target + 1.0,
              f"peak {peak:.2f} N (< {1.5 * target + 1.0:.1f})")
        check("press: no bounce / lift-off", low > 0.5 * target,
              f"min after touch {low:.2f} N (> {0.5 * target:.1f})")
    # Steady-state window: last 4 s, but not earlier than 2 s after touch-down.
    w0 = max(p.t1 - 4.0, t[hit[0]] + 2.0) if hit.size else p.t1 - 4.0
    m = sel(w0, p.t1)
    check("press: >= 1 s of steady contact to evaluate", p.t1 - w0 >= 1.0,
          f"window {p.t1 - w0:.1f} s")
    f_ctrl = -log["wrench"][m][:, 2].mean()
    check("press: controller wrench matches truth", abs(f_ctrl - contact[m].mean()) < 0.5,
          f"controller {f_ctrl:.2f} N vs table {contact[m].mean():.2f} N (+-0.5)")
    mean_f, std_f = contact[m].mean(), contact[m].std()
    check("press: mean contact force", abs(mean_f - target) < 0.2 * target + 0.5,
          f"{mean_f:.2f} N vs target {target:.2f} N (+-{0.2 * target + 0.5:.1f})")
    check("press: contact force steady", std_f < 1.0, f"std {std_f:.2f} N (< 1)")
    xy = np.linalg.norm(pos[m][:, :2] - p.pos[:2], axis=1).max()
    tilt = max(rot_angle_deg(p.rot, R) for R in log["R"][m])
    check("press: xy held", xy < 0.005, f"max {xy * 1e3:.1f} mm (< 5)")
    check("press: gripper kept pointing down", tilt < 3.0, f"max {tilt:.2f} deg (< 3)")
    return results


def save_outputs(log, phases, out_dir, show):
    out_dir.mkdir(parents=True, exist_ok=True)
    t = log["t"]
    with open(out_dir / "log.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["t", "phase", "x", "y", "z", "tx", "ty", "tz", "rot_err_deg",
                    "vx_virt", "vy_virt", "vz_virt", "fx_ctrl", "fy_ctrl", "fz_ctrl",
                    "contact_n", "push_x"])
        for i in range(len(t)):
            w.writerow([f"{t[i]:.3f}", log["phase"][i], *np.round(log["pos"][i], 5),
                        *np.round(log["tgt"][i], 5), f"{log['rot_err'][i]:.3f}",
                        *np.round(log["cmd"][i][:3], 5), *np.round(log["wrench"][i][:3], 4),
                        f"{log['contact'][i]:.4f}", f"{log['push'][i][0]:.2f}"])

    import matplotlib
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(5, 1, figsize=(11, 13), sharex=True)
    for k, lab in enumerate("xyz"):
        ax[0].plot(t, log["pos"][:, k], color=f"C{k}", label=f"{lab} actual")
        ax[0].plot(t, log["tgt"][:, k], color=f"C{k}", ls="--", lw=1, label=f"{lab} target")
        if log["virt"] is not None:
            ax[0].plot(t, log["virt"][:, k], color=f"C{k}", ls=":", lw=1)
    ax[0].set_ylabel("TCP pos [m]\n(-- target, : virtual)")
    ax[0].legend(ncol=6, fontsize=8, loc="upper right")
    ax[1].plot(t, log["rot_err"], color="C4")
    ax[1].set_ylabel("rot err vs target [deg]")
    for k, lab in enumerate("xyz"):
        ax[2].plot(t, log["cmd"][:, k], label=f"v{lab} virtual")
    ax[2].set_ylabel("virtual vel [m/s]")
    ax[2].legend(ncol=3, fontsize=8)
    for k, lab in enumerate("xyz"):
        ax[3].plot(t, log["wrench"][:, k], label=f"F{lab} (controller, base, on env)")
    ax[3].plot(t, log["push"][:, 0], "k--", lw=1, label="external push x (world)")
    ax[3].set_ylabel("wrench [N]")
    ax[3].legend(ncol=4, fontsize=8)
    ax[4].plot(t, log["contact"], color="C3", label="table normal force (truth)")
    ax[4].plot(t, log["fz_target"], "k--", lw=1, label="|Fz target|")
    ax[4].set_ylabel("contact [N]")
    ax[4].set_xlabel("sim time [s]")
    ax[4].legend(fontsize=8)
    for a in ax:
        for i, ph in enumerate(phases):
            a.axvspan(ph.t0, ph.t1, color="0.5" if i % 2 else "0.85", alpha=0.15, lw=0)
        a.grid(alpha=0.3)
    for ph in phases:
        ax[0].text((ph.t0 + ph.t1) / 2, 1.02, ph.name, transform=ax[0].get_xaxis_transform(),
                   ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_dir / "plot.png", dpi=110)
    if show:
        plt.show()


# --------------------------------------------------------------------------- #
# Main                                                                        #
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--headless", action="store_true", help="no viewer, no plot window")
    ap.add_argument("--push", type=float, default=8.0, help="external push in comply phase [N]")
    ap.add_argument("--fz", type=float, default=-5.0,
                    help="target Fz in press phase, in the controller's wrench convention [N]")
    ap.add_argument("--table-drop", type=float, default=0.06,
                    help="table top distance below the home TCP [m]")
    ap.add_argument("--axes", choices=("tcp", "base"), default="tcp", help="controller axes")
    ap.add_argument("--out", type=Path, default=None, help="output dir for log.csv/plot.png")
    args = ap.parse_args()
    out_dir = args.out or Path.cwd() / f"admittance_test_{time.strftime('%Y%m%d_%H%M%S')}"

    # Ctrl-C is handled here (clean ROS/viewer shutdown), not by rclpy.
    rclpy.init(args=["--ros-args", "-p", "use_sim_time:=true", "--params-file", str(PARAMS_FILE)],
               signal_handler_options=SignalHandlerOptions.NO)
    sim, info = build_sim(args.table_drop)
    truth = Truth(sim)

    namespace = f"/{ARM}" if len(sim.arm_ids) > 1 else "/"
    bridge = ArmosBridge(sim)
    ctrl = AdmittanceController(namespace=namespace, parameter_overrides=[
        Parameter("axes", value=args.axes),
        Parameter("tcp_frame", value=sim.arm_config(ARM).ee_site),
    ])
    p_start, R_start = truth.tcp_base()
    driver = ScenarioDriver(namespace, axis_map(R_start, args.axes))
    executor = SingleThreadedExecutor()
    for n in (bridge, ctrl, driver):
        executor.add_node(n)
    spin = threading.Thread(target=executor.spin, daemon=True)
    spin.start()

    dt = sim._dt
    # Bridge holds q_home; the controller tares (2 s) and starts holding the measured pose.
    wall = time.perf_counter()
    tick = 0
    while sim.sim_time < 3.0:
        telem = sim.step()
        for cb in sim._callbacks:
            cb(tick, sim.sim_time, telem)
        tick += 1
        time.sleep(max(0.0, wall + sim.sim_time - time.perf_counter()))
    if not ctrl.wrench.tared or ctrl.law.virtual_pos is None:
        print("[test] controller not running after 3 s (tare not done?)")
        return 2
    t_start = sim.sim_time

    p0, R0 = truth.tcp_base()
    phases = make_phases(p0, R0, truth.R_base, truth.base, info["table_center"], args.push, args.fz)
    t_end = phases[-1].t1
    print(f"[test] axes {args.axes} | start TCP (base) {np.round(p0, 4)} | table top "
          f"z={info['table_top']:.3f} m (world) | push={args.push} N | fz={args.fz} N")

    viewer = None
    if not args.headless:
        import mujoco.viewer
        viewer = mujoco.viewer.launch_passive(truth.m, truth.d, show_left_ui=False,
                                              show_right_ui=False)
        viewer.cam.azimuth, viewer.cam.elevation, viewer.cam.distance = 150, -25, 1.4
        viewer.cam.lookat[:] = truth.base + np.array([0.35, 0.0, 0.4])
        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = True
        truth.m.vis.map.force = 0.02

    keys = ("t", "phase", "pos", "R", "tgt", "rot_err", "cmd", "virt", "wrench",
            "contact", "push", "fz_target", "telem_age")
    log = {k: [] for k in keys}
    sim_modes = set()
    tick0 = tick
    prev, last_pub = None, -1.0
    wall0 = time.perf_counter()
    try:
        while sim.sim_time - t_start < t_end:
            if viewer is not None and not viewer.is_running():
                print("[test] viewer closed, aborting")
                break
            t = sim.sim_time - t_start
            ph = next(p for p in phases if p.t0 <= t < p.t1) if t < t_end else phases[-1]
            if ph is not prev:
                print(f"[test] t={t:5.2f}s  phase -> {ph.name}")
            if ph is not prev or t - last_pub >= 0.1:   # clients stream targets (force_target_timeout)
                driver.publish(ph)
                last_pub = t
            prev = ph

            push = ph.push_at(t)
            truth.d.xfrc_applied[truth.hand, :3] = push

            telem = sim.step()
            for cb in sim._callbacks:
                cb(tick, sim.sim_time, telem)
            sim_modes.add(telem.control_mode[ARM])

            pos, R = truth.tcp_base()
            law = ctrl.law
            virt = None if law.virtual_pos is None else (law.virtual_pos.copy(), law.virtual_R.copy())
            log["t"].append(sim.sim_time - t_start)
            log["phase"].append(ph.name)
            log["pos"].append(pos)
            log["R"].append(R)
            log["tgt"].append(ph.pos)
            log["rot_err"].append(rot_angle_deg(ph.rot, R))
            log["cmd"].append(np.array(law.virtual_vel, dtype=float))
            log["virt"].append(np.full(3, np.nan) if virt is None else virt[0])
            w = ctrl.curr_wrench
            if w is not None and args.axes == "tcp":   # law axes -> base, for the checks
                w = np.concatenate((ctrl.curr_R @ w[:3], ctrl.curr_R @ w[3:]))
            log["wrench"].append(np.full(6, np.nan) if w is None else np.array(w, dtype=float))
            log["contact"].append(truth.table_normal_force())
            log["push"].append(push.copy())
            log["fz_target"].append(abs(ph.wrench[2]))
            log["telem_age"].append(sim.sim_time - ctrl.pose_time)

            if viewer is not None and tick % 4 == 0:
                p_w, R_w = truth.to_world(ph.pos, ph.rot)
                a_w, Ra_w = truth.to_world(pos, R)
                frames = [(p_w, R_w, 1.0, 0.09, 0.004), (a_w, Ra_w, 0.6, 0.05, 0.0025)]
                if virt is not None:
                    v_w, Rv_w = truth.to_world(*virt)
                    frames.append((v_w, Rv_w, 0.3, 0.07, 0.003))
                arrows = []
                if np.any(push):
                    hand_p = truth.d.xpos[truth.hand]
                    arrows.append((hand_p - 0.02 * push, hand_p, [1.0, 0.1, 0.1, 1.0]))
                with viewer.lock():
                    draw_scene(viewer.user_scn, frames, arrows)
                viewer.set_texts([(mujoco.mjtFontScale.mjFONTSCALE_150, mujoco.mjtGridPos.mjGRID_TOPLEFT,
                                   f"{ph.name}  t={t:5.1f}s",
                                   f"mask {ph.mask}  contact {log['contact'][-1]:.1f} N")])
                viewer.sync()

            tick += 1
            sleep = wall0 + (tick - tick0) * dt - time.perf_counter()
            if sleep > 0:
                time.sleep(sleep)
    except KeyboardInterrupt:
        print("[test] interrupted")
    finally:
        executor.shutdown(timeout_sec=1.0)
        spin.join(timeout=2.0)
        for n in (bridge, ctrl, driver):
            n.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        if viewer is not None:
            viewer.close()

    lag = (time.perf_counter() - wall0) - (sim.sim_time - t_start)
    for k in keys:
        if k != "phase":
            log[k] = np.array(log[k])
    if len(log["t"]) == 0 or log["t"][-1] < t_end - dt:
        print("[test] run incomplete, no evaluation")
        return 2

    results = evaluate(log, phases, args.fz, float(ctrl.law.D_force[0]), ctrl.law.cfg.max_force_lin_vel,
                       args.push, sim_modes)
    print(f"\n=== admittance validation, {args.axes} axes (wall lag {lag:+.2f}s) ===")
    for name, ok, detail in results:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name:42s} {detail}")
    n_fail = sum(not ok for _, ok, _ in results)
    print(f"  {len(results) - n_fail}/{len(results)} passed")

    print(f"  log + plot: {out_dir}")
    save_outputs(log, phases, out_dir, show=not args.headless)
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())

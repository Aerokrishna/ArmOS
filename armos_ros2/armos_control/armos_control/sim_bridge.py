#!/usr/bin/env python3
"""
ArmosBridge — ROS2 <-> CyRoSimulator bridge, following the same architecture
as UDPInterface (sim_cyro/udp_interface.py):

  - Main thread   : sim.run(headless=False) -> owns physics + MuJoCo viewer.
  - Background    : rclpy.spin(node)         -> owns ROS2 I/O (like _cmd_loop).
  - No lock, no mailbox: subscriber callbacks write straight into `self._sim`,
    exactly like UDPInterface._dispatch() does. Telemetry is published from
    inside sim.on_step(), exactly like UDPInterface._on_sim_step().

Per-arm topics/services; arm identity is inferred from endpoint name
(e.g. /armos/<arm_id>/joint_control), not from a message field.

Mode model:
  1. The per-message `mode` field selects the concrete CyRoSimulator
      ControlMode, same mapping UDPInterface._dispatch uses.
  2. Per arm, set_mode is called only when the requested ControlMode changes
      (including the first received command), then retained until changed.

Safety-relevant deviation from UDPInterface._dispatch's literal statement
order: UDP calls `sim.set_mode(...)` *before* writing the target (e.g.
set_tcp_pose). That's two separate Python statements, so there's a small
GIL-preemption window where the sim thread could run a tick already in the
new mode but before the new target lands — reading whatever the previous
target was. Here the target is written *first*, `set_mode` *last*, which
closes that specific window: set_joint_pos/set_tcp_pose/etc. don't depend on
which mode is currently active, so writing them early is safe, and by the
time set_mode flips, the target is already fresh.

Actuation: the per-message `mode` maps onto the arm's actuation. Torque arms
(the default for fr3.yaml / dual_franka.yaml): joint mode 0 -> JOINT_IMPEDANCE,
2 -> JOINT_TORQUE, cart mode 0 -> CARTESIAN_IMPEDANCE; the velocity modes are
position-only and are refused with a warning.

Robot-compatible interface (same topics as the real FR3 under crisp + franka_ros2,
so armos_control's admittance node and franka_custom_control's scripts run
unchanged on the sim). Per arm, under `<namespace>` for a single-arm config and
`<namespace>/<arm_id>` for several arms:
  publishes  current_pose                    PoseStamped   TCP (ee_site) in `base`
             joint_states                    JointState
             franka_robot_state_broadcaster/external_wrench_in_base_frame
                                             WrenchStamped wrist F/T, payload removed,
                                                           robot-on-environment, base frame,
                                                           torque about the sensor origin
             franka_robot_state_broadcaster/current_pose
                                             PoseStamped   the F/T sensor origin (torque point)
  subscribes target_pose                     PoseStamped   -> CARTESIAN_IMPEDANCE (torque arms,
                                                           crisp's gains) or TASK_POSE
and /clock (sim time) when publish_clock is true. crisp's target_wrench,
target_joint and target_stiffness are not emulated yet.

Cartesian impedance gains: crisp cartesian_controller's parameter names, so a client sets them the same
way on the sim (this node) and the robot (the controller node): task.k_pos_{x,y,z} [N/m],
task.k_rot_{x,y,z} [Nm/rad], task.d_pos_*, task.d_rot_* (negative = 2 sqrt(k), as crisp). Defaults:
franka_custom_control's crisp_controllers.yaml. Changes apply live to every torque arm.

ASSUMPTIONS — confirm against your actual generated interfaces and
sim_types.py; each is marked inline with `# ASSUMPTION:`.
"""

import threading
import time
from functools import partial
from pathlib import Path

import numpy as np
import rclpy
import yaml
from ament_index_python.packages import get_package_share_directory
from builtin_interfaces.msg import Time
from geometry_msgs.msg import PoseStamped, WrenchStamped
from rclpy.node import Node
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import JointState

# ASSUMPTION: message/service package name — adjust to your actual package.
from armos_interfaces.msg import ArmosCartControl, ArmosJointControl, ArmosTelemetry
from armos_interfaces.srv import ArmosSetGripper

from armos_core.math.so3 import euler_xyz_to_rot, quat_normalize, quat_to_rot, rot_to_euler_xyz, rot_to_quat
from armos_sim import ASSETS_DIR, ControlMode, CyRoSimulator, Telemetry

# ASSUMPTION: ControlMode members match sim_cyro.types usage seen in
# udp_interface.py (JOINT_POS, JOINT_VEL, JOINT_TORQUE, TASK_POSE, TASK_VEL).
_JOINT_SUBMODE = {
    0: ControlMode.JOINT_POS,
    1: ControlMode.JOINT_VEL,
    2: ControlMode.JOINT_TORQUE,
}
_CART_SUBMODE = {
    0: ControlMode.TASK_POSE,
    1: ControlMode.TASK_VEL,
}
# Torque-actuated arms: the same message modes on the torque laws (no velocity modes).
_JOINT_SUBMODE_TORQUE = {
    0: ControlMode.JOINT_IMPEDANCE,
    2: ControlMode.JOINT_TORQUE,
}
_CART_SUBMODE_TORQUE = {
    0: ControlMode.CARTESIAN_IMPEDANCE,
}
# crisp_controllers.yaml (franka_custom_control) cartesian_controller task gains, crisp parameter names.
_GAIN_NAMES = [f"task.{k}_{a}_{x}" for k in ("k", "d") for a in ("pos", "rot") for x in "xyz"]
_GAIN_DEFAULTS = dict(zip(_GAIN_NAMES, [1500.0, 1500.0, 500.0, 30.0, 30.0, 30.0, 80.0, 80.0, 30.0, 2.0, 2.0, 2.0]))
_DEFAULT_CONFIG_NAME = "dual_franka.yaml"

def _stamp(t: float) -> Time:
    sec, nanosec = divmod(int(round(t * 1e9)), 1_000_000_000)
    return Time(sec=sec, nanosec=nanosec)


class ArmosBridge(Node):
    def __init__(self, sim: CyRoSimulator, **kwargs):
        super().__init__("armos_bridge", **kwargs)
        self._sim = sim
        self._active_mode = {arm_id: None for arm_id in sim.arm_ids}
        self._torque = {
            arm_id: sim.arm_config(arm_id).actuation == "torque" for arm_id in sim.arm_ids
        }
        namespace = self.declare_parameter("namespace", "").value.strip("/")
        self._base_frame = self.declare_parameter("base_frame", "base").value
        self._publish_clock = self.declare_parameter("publish_clock", True).value
        self._gains = {n: float(self.declare_parameter(n, _GAIN_DEFAULTS[n]).value) for n in _GAIN_NAMES}
        kp, kd = self._gain_arrays(self._gains)

        # --- per-arm topics/services; arm is inferred from endpoint name ---
        self._joint_sub = {}
        self._cart_sub = {}
        self._telem_pub = {}
        self._gripper_srv = {}
        for arm_id in sim.arm_ids:
            base = f"/armos/{arm_id}"
            self._gripper_srv[arm_id] = self.create_service(
                ArmosSetGripper,
                f"{base}/setgripper",
                partial(self._on_set_gripper, arm_id),
            )
            self._joint_sub[arm_id] = self.create_subscription(
                ArmosJointControl,
                f"{base}/joint_control",
                partial(self._on_joint_cmd, arm_id),
                10,
            )
            self._cart_sub[arm_id] = self.create_subscription(
                ArmosCartControl,
                f"{base}/cart_control",
                partial(self._on_cart_cmd, arm_id),
                10,
            )
            
            self._telem_pub[arm_id] = self.create_publisher(
                ArmosTelemetry,
                f"{base}/telemetry",
                10,
            )

        # --- robot-compatible (crisp / franka_ros2) topics, per arm namespace ---
        self._robot_ns = {}
        self._robot_pubs = {}
        for arm_id in sim.arm_ids:
            parts = [namespace] + ([arm_id] if len(sim.arm_ids) > 1 else [])
            ns = "/" + "/".join(x for x in parts if x)
            pre = ns.rstrip("/")
            self._robot_ns[arm_id] = ns
            self._robot_pubs[arm_id] = {
                "pose": self.create_publisher(PoseStamped, f"{pre}/current_pose", 10),
                "joints": self.create_publisher(JointState, f"{pre}/joint_states", 10),
                "wrench": self.create_publisher(
                    WrenchStamped,
                    f"{pre}/franka_robot_state_broadcaster/external_wrench_in_base_frame", 10),
                "wrench_point": self.create_publisher(
                    PoseStamped, f"{pre}/franka_robot_state_broadcaster/current_pose", 10),
            }
            self.create_subscription(PoseStamped, f"{pre}/target_pose",
                                     partial(self._on_robot_target_pose, arm_id), 10)
        self._clock_pub = self.create_publisher(Clock, "/clock", 10) if self._publish_clock else None

        # --- startup: hold the YAML home configuration ---
        for arm_id in sim.arm_ids:
            cfg = sim.arm_config(arm_id)
            if self._torque[arm_id]:
                sim.set_impedance_gains(arm_id, kp=kp, kd=kd, space="cartesian")
                start = ControlMode.JOINT_IMPEDANCE
            else:
                start = ControlMode.JOINT_POS
            sim.set_joint_pos(arm_id, cfg.q_home)   # target first
            self._maybe_set_mode(arm_id, start)       # mode last

        sim.on_step(self._publish_telemetry)  # telemetry published in the sim's own loop
        self.add_on_set_parameters_callback(self._on_set_parameters)

        self.get_logger().info(
            f"ArmosBridge ready | arms={sim.arm_ids} | torque={self._torque} | "
            f"holding q_home | per-arm endpoints=/armos/<arm_id>/* and "
            f"{sorted(self._robot_ns.values())} (current_pose, joint_states, target_pose, ...)"
        )

    # ------------------------------------------------------------------ #
    # ROS2 callbacks (background thread) — write straight into sim, no   #
    # lock/mailbox, matching UDPInterface._dispatch's direct-write model #
    # ------------------------------------------------------------------ #
    def _on_joint_cmd(self, arm_id: str, msg: ArmosJointControl):
        submodes = _JOINT_SUBMODE_TORQUE if self._torque[arm_id] else _JOINT_SUBMODE
        if msg.mode not in submodes:
            self.get_logger().warn(
                f"[{arm_id}] joint mode {msg.mode} not available "
                f"({'torque' if self._torque[arm_id] else 'position'} actuation)",
                throttle_duration_sec=2.0)
            return

        ndof = self._sim.ndof(arm_id)
        q = np.array(
            [msg.joint1, msg.joint2, msg.joint3, msg.joint4,
             msg.joint5, msg.joint6, msg.joint7],
            dtype=float,
        )[:ndof]  # ASSUMPTION: extra fields ignored for <7-DOF arms

        if msg.mode == 0:
            self._sim.set_joint_pos(arm_id, q)
        elif msg.mode == 1:
            self._sim.set_joint_vel(arm_id, q)
        elif msg.mode == 2:
            self._sim.set_joint_torque(arm_id, q)
        self._maybe_set_mode(arm_id, submodes[msg.mode])

    def _on_cart_cmd(self, arm_id: str, msg: ArmosCartControl):
        submodes = _CART_SUBMODE_TORQUE if self._torque[arm_id] else _CART_SUBMODE
        if msg.mode not in submodes:
            self.get_logger().warn(
                f"[{arm_id}] cart mode {msg.mode} not available "
                f"({'torque' if self._torque[arm_id] else 'position'} actuation)",
                throttle_duration_sec=2.0)
            return

        if msg.mode == 0:  # pose: x,y,z = position, a,b,c = euler orientation
            pos = np.array([msg.x, msg.y, msg.z], dtype=float)
            rot = euler_xyz_to_rot((msg.a, msg.b, msg.c))
            self._sim.set_tcp_pose(arm_id, pos, rot)
        else:  # velocity: x,y,z = linear vel, a,b,c = angular vel
            lin = np.array([msg.x, msg.y, msg.z], dtype=float)
            ang = np.array([msg.a, msg.b, msg.c], dtype=float)
            self._sim.set_tcp_vel(arm_id, lin, ang)
        self._maybe_set_mode(arm_id, submodes[msg.mode])

    def _on_robot_target_pose(self, arm_id: str, msg: PoseStamped):
        """crisp cartesian_controller target: TCP pose in `base` -> Cartesian impedance (torque arms)."""
        if msg.header.frame_id and msg.header.frame_id != self._base_frame:
            self.get_logger().warn(
                f"[{arm_id}] target_pose frame '{msg.header.frame_id}' != '{self._base_frame}', "
                "ignored", throttle_duration_sec=2.0)
            return
        q, o = msg.pose.position, msg.pose.orientation
        pos = np.array([q.x, q.y, q.z], dtype=float)
        quat = np.array([o.w, o.x, o.y, o.z], dtype=float)
        if not (np.all(np.isfinite(pos)) and np.all(np.isfinite(quat))) or np.linalg.norm(quat) < 1e-8:
            self.get_logger().warn(f"[{arm_id}] invalid target_pose ignored", throttle_duration_sec=2.0)
            return
        self._sim.set_tcp_pose(arm_id, pos, quat_to_rot(quat_normalize(quat)))   # target first
        self._maybe_set_mode(
            arm_id,
            ControlMode.CARTESIAN_IMPEDANCE if self._torque[arm_id] else ControlMode.TASK_POSE,
        )

    @staticmethod
    def _gain_arrays(g: dict) -> tuple[np.ndarray, np.ndarray]:
        """crisp task.* gains -> (kp, kd) (6,) each; a negative damping is 2 sqrt(k)."""
        kp = np.array([g[f"task.k_{a}_{x}"] for a in ("pos", "rot") for x in "xyz"])
        kd = np.array([g[f"task.d_{a}_{x}"] for a in ("pos", "rot") for x in "xyz"])
        return kp, np.where(kd < 0.0, 2.0 * np.sqrt(kp), kd)

    def _on_set_parameters(self, params):
        """Live crisp-style gain changes -> every torque arm's Cartesian impedance."""
        from rcl_interfaces.msg import SetParametersResult
        new = dict(self._gains)
        for prm in params:
            if prm.name in new:
                try:
                    new[prm.name] = float(prm.value)
                except (TypeError, ValueError):
                    return SetParametersResult(successful=False, reason=f"{prm.name} must be a number")
                if prm.name.startswith("task.k_") and new[prm.name] < 0.0:
                    return SetParametersResult(successful=False, reason=f"{prm.name} must be >= 0")
        if new != self._gains:
            self._gains = new
            kp, kd = self._gain_arrays(new)
            for arm_id in self._sim.arm_ids:
                if self._torque[arm_id]:
                    self._sim.set_impedance_gains(arm_id, kp=kp, kd=kd, space="cartesian")
            self.get_logger().info(f"Cartesian impedance gains: kp={kp.round(1).tolist()} kd={kd.round(2).tolist()}")
        return SetParametersResult(successful=True)

    def _on_set_gripper(
        self,
        arm_id: str,
        request: ArmosSetGripper.Request,
        response: ArmosSetGripper.Response,
    ):
        position = request.position
        try:
            self._sim.set_gripper(arm_id, position)
        except (KeyError, ValueError) as exc:
            response.success = False
            response.message = str(exc)
            return response

        response.success = True
        response.message = "Gripper position set"
        return response

    # ------------------------------------------------------------------ #
    # Sim-thread hook — called from inside sim._tick() via on_step,      #
    # same as UDPInterface._on_sim_step. Loops over arms every tick.     #
    # ------------------------------------------------------------------ #
    def _publish_telemetry(self, tick: int, t: float, telem: Telemetry):
        for arm_id in self._sim.arm_ids:
            arm_t = self._get_arm_telem(telem, arm_id)
            if arm_t is None:
                continue
            msg = ArmosTelemetry()
            msg.timestamp = float(telem.timestamp)
            msg.joint_pos = self._float_array(arm_t["joint_pos"])
            msg.joint_vel = self._float_array(arm_t["joint_vel"])
            msg.joint_torque = self._float_array(arm_t["joint_torque"])
            msg.wrench = self._float_array(arm_t["wrench"])
            msg.tcp_pose = self._pose_to_xyzabc(arm_t["tcp_pose"])
            msg.tcp_twist = self._float_array(arm_t["tcp_twist"])
            msg.manipulability = float(arm_t["manipulability"])
            msg.singularity_distance = float(arm_t["singularity_distance"])
            self._telem_pub[arm_id].publish(msg)
            self._publish_robot_state(arm_id, t, telem)
        if self._clock_pub is not None:
            self._clock_pub.publish(Clock(clock=_stamp(t)))

    def _publish_robot_state(self, arm_id: str, t: float, telem: Telemetry):
        """crisp / franka_ros2 feedback topics of one arm."""
        pubs = self._robot_pubs[arm_id]
        stamp = _stamp(t)
        pubs["pose"].publish(self._pose_stamped(stamp, telem.tcp_pose[arm_id]))

        js = JointState()
        js.header.stamp = stamp
        js.name = list(self._sim.arm_config(arm_id).joint_names)
        js.position = self._float_array(telem.joint_pos[arm_id])
        js.velocity = self._float_array(telem.joint_vel[arm_id])
        js.effort = self._float_array(telem.joint_torque[arm_id])
        pubs["joints"].publish(js)

        ext = self._sim.external_wrench(arm_id)
        if ext is None:
            return
        wrench, point = ext
        w = WrenchStamped()
        w.header.stamp, w.header.frame_id = stamp, self._base_frame
        f, tq = w.wrench.force, w.wrench.torque
        f.x, f.y, f.z, tq.x, tq.y, tq.z = (float(v) for v in wrench)
        pubs["wrench"].publish(w)
        T = np.eye(4)
        T[:3, 3] = point
        pubs["wrench_point"].publish(self._pose_stamped(stamp, T))

    def _pose_stamped(self, stamp: Time, T) -> PoseStamped:
        T = np.asarray(T, dtype=float).reshape(4, 4)
        msg = PoseStamped()
        msg.header.stamp, msg.header.frame_id = stamp, self._base_frame
        msg.pose.position.x, msg.pose.position.y, msg.pose.position.z = (float(v) for v in T[:3, 3])
        w, x, y, z = rot_to_quat(T[:3, :3])
        o = msg.pose.orientation
        o.w, o.x, o.y, o.z = float(w), float(x), float(y), float(z)
        return msg

    # ------------------------------------------------------------------ #
    # Helpers                                                            #
    # ------------------------------------------------------------------ #
    def _get_arm_telem(self, telem: Telemetry, arm_id: str):
        if arm_id not in telem.joint_pos:
            return None

        return {
            "joint_pos": telem.joint_pos[arm_id],
            "joint_vel": telem.joint_vel.get(arm_id, np.zeros(0)),
            "joint_torque": telem.joint_torque.get(arm_id, np.zeros(0)),
            "wrench": telem.wrench.get(arm_id, np.zeros(6)),
            "tcp_pose": telem.tcp_pose.get(arm_id, np.zeros((4, 4))),
            "tcp_twist": telem.tcp_twist.get(arm_id, np.zeros(6)),
            "manipulability": telem.manipulability.get(arm_id, 0.0),
            "singularity_distance": telem.singularity_distance.get(arm_id, 0.0),
        }

    @staticmethod
    def _float_array(value) -> list[float]:
        return np.asarray(value, dtype=float).reshape(-1).tolist()

    @staticmethod
    def _pose_to_xyzabc(pose) -> list[float]:
        """Homogeneous pose -> [x, y, z, a, b, c], R = Rx(a) @ Ry(b) @ Rz(c) (radians)."""
        transform = np.asarray(pose, dtype=float).reshape(4, 4)
        return [float(v) for v in transform[:3, 3]] + [
            float(v) for v in rot_to_euler_xyz(transform[:3, :3])
        ]

    def _maybe_set_mode(self, arm_id: str, requested_mode: ControlMode):
        # The sim may have switched on its own (a diverging arm is held in JOINT_IMPEDANCE).
        if self._active_mode.get(arm_id) == requested_mode and self._sim.mode(arm_id) == requested_mode:
            return
        self._sim.set_mode(arm_id, requested_mode)
        self._active_mode[arm_id] = requested_mode

    def _current_joint_pos(self, telem: Telemetry, arm_id: str) -> np.ndarray:
        arm_t = self._get_arm_telem(telem, arm_id)
        if arm_t is not None:
            return np.asarray(arm_t["joint_pos"], dtype=float).copy()
        return self._sim.arm_config(arm_id).q_home.copy()  # fallback

def _resolve_config_path(config_path: str | None = None) -> Path:
    candidates = []

    if config_path:
        requested_path = Path(config_path).expanduser()
        if requested_path.is_absolute():
            candidates.append(requested_path)
        else:
            candidates.extend((Path.cwd() / requested_path, requested_path))

    try:
        share_dir = Path(get_package_share_directory("armos_control"))
        candidates.append(share_dir / "config" / (config_path or _DEFAULT_CONFIG_NAME))
    except Exception:
        pass

    script_path = Path(__file__).resolve()
    for base in [script_path.parent, *script_path.parents[:6]]:
        candidates.append(base / "config" / (config_path or _DEFAULT_CONFIG_NAME))

    for path in candidates:
        if path.exists():
            return path

    raise FileNotFoundError(
        f"Could not locate configuration file '{config_path or _DEFAULT_CONFIG_NAME}'. "
        "Pass an absolute path or a path relative to the current directory."
    )


def _resolve_runtime_config(config_path: Path, frictionloss_scale: float = -1.0) -> Path:
    with open(config_path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh) or {}
    if frictionloss_scale >= 0.0:   # override of physics.frictionloss_scale (joint friction)
        cfg.setdefault("physics", {})["frictionloss_scale"] = float(frictionloss_scale)

    scene_name = cfg.get("scene_xml_path")
    robot_name = cfg.get("robot_xml_path")
    if not scene_name or not robot_name:
        raise ValueError(
            f"Configuration {config_path} must define scene_xml_path and robot_xml_path"
        )

    asset_dirs = [config_path.parent, ASSETS_DIR]
    if cfg.get("assets_dir"):
        asset_dirs.append(ASSETS_DIR / cfg["assets_dir"])

    def find_asset(name: str) -> Path | None:
        requested = Path(name).expanduser()
        candidates = [requested] if requested.is_absolute() else [
            asset_dir / requested for asset_dir in asset_dirs
        ]
        return next((candidate.resolve() for candidate in candidates if candidate.is_file()), None)

    scene_xml = find_asset(scene_name)
    robot_xml = find_asset(robot_name)
    if scene_xml is None or robot_xml is None:
        raise FileNotFoundError(
            f"MuJoCo XML assets from {config_path} were not found: "
            f"scene_xml_path={scene_name!r}, robot_xml_path={robot_name!r}"
        )

    cfg["scene_xml_path"] = str(scene_xml)
    cfg["robot_xml_path"] = str(robot_xml)

    runtime_path = config_path.with_name(f"{config_path.stem}_runtime.yaml")
    with open(runtime_path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(cfg, fh, sort_keys=False)

    return runtime_path


def main():
    rclpy.init()

    config_node = rclpy.create_node("armos_config")
    config_node.declare_parameter("config_path", _DEFAULT_CONFIG_NAME)
    config_node.declare_parameter("headless", False)
    config_node.declare_parameter("frictionloss_scale", -1.0)   # < 0: as in the config
    config_name = config_node.get_parameter("config_path").value
    headless = bool(config_node.get_parameter("headless").value)
    friction = float(config_node.get_parameter("frictionloss_scale").value)
    config_node.destroy_node()

    config_path = _resolve_config_path(str(config_name))
    runtime_config = _resolve_runtime_config(config_path, friction)
    sim = CyRoSimulator(config_path=runtime_config)
    node = ArmosBridge(sim)

    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()

    try:
        time.sleep(1)  # let OS scheduler settle before launching the viewer
        sim.run(headless=headless)  # blocks main thread; opens the MuJoCo viewer unless headless
    except KeyboardInterrupt:
        pass
    finally:
        sim.close()
        rclpy.try_shutdown()   # rclpy's SIGINT handler may have shut the context down already
        spin_thread.join(timeout=1.0)


if __name__ == "__main__":
    main()
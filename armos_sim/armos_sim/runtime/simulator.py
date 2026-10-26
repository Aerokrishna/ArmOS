"""CyRoSimulator — high-level API for the armos MuJoCo simulation."""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Union

import mujoco
import numpy as np

from ..constants import _DEFAULT_DECIMATION, _VIEWER_HZ
from ..model.loader import _RobotLoader
from ..model.mj_model import _MuJoCoModel
from ..modes.dispatcher import _ArmController, make_controller
from ..modes.torque import _TorqueArmController
from ..plant import ArmPlant, make_plants
from ..types import ArmConfig, ControlMode, Telemetry, ToolParams
from .pinocchio_sync import _PinocchioSync
from .telemetry import _TelemetryCollector
from .viewer_frames import (
    draw_all_frames,
    poses_equal,
    read_object_pose,
    resolve_object_ref,
)

logger = logging.getLogger(__name__)

_BAD_STATE_WARNINGS = tuple(
    int(w) for w in (mujoco.mjtWarning.mjWARN_BADQPOS, mujoco.mjtWarning.mjWARN_BADQVEL,
                     mujoco.mjtWarning.mjWARN_BADQACC)
)


def _resolve_config_path(config_path) -> Path:
    """Resolve config path; use bundled default if None."""
    if config_path is not None:
        return Path(config_path)
    default = Path(__file__).resolve().parents[1] / "assets" / "franka_fr3" / "fr3.yaml"
    if not default.exists():
        raise FileNotFoundError(
            f"Default config not found at {default}. "
            "Install the package properly or pass config_path explicitly."
        )
    return default


class CyRoSimulator:
    """
    High-level simulator for one or more robot arms.

    Parameters
    ----------
    config_path : str | Path, optional
        Path to YAML config. Defaults to the bundled FR3 config (franka_fr3/fr3.yaml).
    xml_path : str | Path, optional
        Path to MJCF scene (programmatic use).
    arm_configs : dict[str, ArmConfig], optional
        Pre-built arm configs (paired with xml_path).
    actuation : "position" | "torque", optional
        Override the actuation of every arm (default: per arm from the YAML, else "position").
        Torque arms run JOINT_TORQUE / JOINT_IMPEDANCE / CARTESIAN_IMPEDANCE / CARTESIAN_ADMITTANCE.
    timestep : float, optional
        Physics timestep [s] (default: YAML `physics.timestep`, else the MJCF).
    control_decimation : int, optional
        Physics steps per control tick (default: YAML `physics.decimation.control`, else 2).
        Position modes, telemetry, `on_step` callbacks and `step()` run once per tick.
    admittance_decimation : int, optional
        Physics steps per CARTESIAN_ADMITTANCE outer-loop update (default: YAML
        `physics.decimation.admittance`, else the control decimation).
        Torque laws (JOINT_TORQUE, impedance, the admittance inner loop) run every physics step.
    telemetry_stride : int
        Collect telemetry every N control ticks (1 = every tick).
    realtime_factor : float
        Pacing of `run()`: 1.0 = real time, 0 = as fast as possible.
    """

    def __init__(
        self,
        config_path=None,
        *,
        xml_path=None,
        arm_configs: Optional[Dict[str, ArmConfig]] = None,
        actuation: Optional[str] = None,
        timestep: Optional[float] = None,
        control_decimation: Optional[int] = None,
        admittance_decimation: Optional[int] = None,
        telemetry_stride: int = 1,
        realtime_factor: float = 1.0,
    ):
        if xml_path is not None and arm_configs is not None:
            self._xml_path = Path(xml_path)
            self._arm_configs = arm_configs
            self._aux_config = dict(_RobotLoader._DEFAULT_AUX)
        else:
            cfg_path = _resolve_config_path(config_path)
            dec = {k: v for k, v in (("control", control_decimation), ("admittance", admittance_decimation))
                   if v is not None}
            overrides = {
                k: v for k, v in (("actuation", actuation), ("timestep", timestep), ("decimation", dec or None))
                if v is not None
            }
            self._xml_path, self._arm_configs, self._aux_config = _RobotLoader.from_yaml(
                cfg_path, **overrides
            )

        preloaded = self._aux_config.pop("_preloaded_model", None)
        dec_cfg = (self._aux_config.pop("physics", None) or {}).get("decimation") or {}
        control_dec = control_decimation or dec_cfg.get("control") or _DEFAULT_DECIMATION
        self._adm_decimation = int(admittance_decimation or dec_cfg.get("admittance") or control_dec)
        self._mj = _MuJoCoModel(self._xml_path, preloaded_model=preloaded, n_substeps=control_dec)
        self._pin_sync = _PinocchioSync(self._arm_configs, self._mj)
        self._plants: Dict[str, ArmPlant] = make_plants(
            self._mj.model, self._mj.data, self._arm_configs, self._aux_config.get("grippers")
        )
        self._telemetry_stride = max(1, int(telemetry_stride))
        self.realtime_factor = float(realtime_factor)

        self._controllers: Dict[str, Union[_ArmController, _TorqueArmController]] = {
            arm_id: make_controller(cfg, self._mj, self._pin_sync, self._plants.get(arm_id))
            for arm_id, cfg in self._arm_configs.items()
        }
        self._position_arms = [a for a in self._arm_configs if a not in self._plants]
        self._torque_arms = list(self._plants)

        self._telem = _TelemetryCollector(
            self._mj,
            self._arm_configs,
            self._pin_sync,
            self._controllers,
            self._aux_config,
        )

        # Control tick period: integration step of the position modes and pacing of run().
        self._dt = self._mj.model.opt.timestep * self._mj.n_substeps
        self._viewer_stride = max(1, round(1.0 / (_VIEWER_HZ * self._dt)))
        self._tick_count = 0
        self._physics_steps = 0
        self._last_finite_qpos = self._mj.data.qpos.copy()
        self._last_finite_ctrl = self._mj.data.ctrl.copy()
        self._last_finite_time = 0.0
        self._sim_time = 0.0
        self._running = False
        self._callbacks: list[Callable[[int, float, Telemetry], None]] = []
        self._pre_tick_callbacks: list[Callable[[], None]] = []
        self._last_telem: Optional[Telemetry] = None
        self._object_frames: list[str] = []
        self._custom_frames: dict[str, np.ndarray] = {}
        self._frames_redraw_pending = False
        self._object_frame_ref_cache: dict[str, tuple] = {}
        self._drawn_object_pose_cache: dict[str, tuple] = {}
        self._frames_lock = threading.Lock()

        self._reset_to_home()
        logger.info(
            "CyRoSimulator ready | arms=%s | torque arms=%s | physics dt=%.4f s | "
            "control every %d steps (%.4f s) | admittance every %d steps | Pinocchio=%s | Pink=%s",
            list(self._arm_configs.keys()),
            self._torque_arms,
            self._mj.model.opt.timestep,
            self._mj.n_substeps,
            self._dt,
            self._adm_decimation,
            self._pin_sync.available,
            {a: getattr(self._controllers[a], "_pink", None) is not None for a in self._position_arms},
        )

    @property
    def sim_time(self) -> float:
        return self._sim_time

    @property
    def arm_ids(self) -> List[str]:
        return list(self._arm_configs.keys())

    def ndof(self, arm_id: str) -> int:
        return self._arm_configs[arm_id].ndof

    def arm_config(self, arm_id: str) -> ArmConfig:
        self._check_arm(arm_id)
        return self._arm_configs[arm_id]

    def set_tool_params(self, arm_id: str, tool: ToolParams) -> None:
        self._check_arm(arm_id)
        self._arm_configs[arm_id].tool = tool

    def set_mode(self, arm_id: str, mode: ControlMode) -> None:
        self._check_arm(arm_id)
        self._controllers[arm_id].set_mode(mode)

    def set_joint_pos(self, arm_id: str, q) -> None:
        self._check_arm(arm_id)
        self._controllers[arm_id].set_joint_pos(np.asarray(q, dtype=float))

    def set_joint_vel(self, arm_id: str, qdot) -> None:
        self._check_arm(arm_id)
        self._controllers[arm_id].set_joint_vel(np.asarray(qdot, dtype=float))

    def set_joint_torque(self, arm_id: str, tau) -> None:
        self._check_arm(arm_id)
        self._controllers[arm_id].set_joint_torque(
            np.asarray(tau, dtype=float)
        )

    def set_flange_pose(self, arm_id: str, pos, rot=None) -> None:
        self._check_arm(arm_id)
        rot = np.eye(3) if rot is None else np.asarray(rot, dtype=float)
        self._controllers[arm_id].set_flange_pose(np.asarray(pos, dtype=float), rot)

    def set_tcp_pose(self, arm_id: str, pos, rot=None) -> None:
        self._check_arm(arm_id)
        rot = np.eye(3) if rot is None else np.asarray(rot, dtype=float)
        self._controllers[arm_id].set_tcp_pose(np.asarray(pos, dtype=float), rot)

    def set_flange_vel(self, arm_id: str, linear, angular) -> None:
        self._check_arm(arm_id)
        self._controllers[arm_id].set_flange_vel(
            np.asarray(linear, dtype=float),
            np.asarray(angular, dtype=float),
        )

    def set_tcp_vel(self, arm_id: str, linear, angular) -> None:
        self._check_arm(arm_id)
        self._controllers[arm_id].set_tcp_vel(
            np.asarray(linear, dtype=float),
            np.asarray(angular, dtype=float),
        )

    def set_gripper_joints(self, arm_id: str, q) -> None:
        """Set dg3f gripper joint angles for 'left' or 'right' arm.

        q : array of 12 angles [rad], ordered finger1→finger2→finger3,
            each finger: [j1, j2, j3, j4] (proximal-to-distal).
        """
        dg3f = self._aux_config.get("dg3f_grippers", {})
        act_names = dg3f.get(arm_id)
        if not act_names:
            raise ValueError(f"No dg3f gripper configured for arm '{arm_id}'")
        q = np.asarray(q, dtype=float)
        if q.shape != (len(act_names),):
            raise ValueError(
                f"Expected {len(act_names)} joint angles, got {q.shape}"
            )
        for name, val in zip(act_names, q):
            self._mj.set_actuator(name, val)

    def set_gripper(self, arm_id: str, position: float) -> None:
        # Set gripper position. Arm must be 'left' or 'right': 0=closed, 1=open
        if arm_id not in ("left", "right"):
            raise ValueError(f"Gripper only on left/right arms, not '{arm_id}'")
        act = self._aux_config["grippers"].get(arm_id)
        if not act:
            raise ValueError(f"No gripper configured for arm '{arm_id}'")
        
        open_pos = self._aux_config["gripper_open"]
        closed_pos = self._aux_config["gripper_closed"]
        pos = np.clip(float(position), 0.0, 1.0)
        target = closed_pos + pos * (open_pos - closed_pos)

        self._mj.set_actuator(act, target)

    def set_actuator(self, actuator_name: str, value: float) -> None:
        self._mj.set_actuator(actuator_name, float(value))

    # ------------------------------------------------------------------ torque-actuated arms
    def plant(self, arm_id: str) -> ArmPlant:
        """The `ArmPlant` of a torque-actuated arm (views into the live MjData)."""
        self._check_torque_arm(arm_id)
        return self._plants[arm_id]

    def mode(self, arm_id: str) -> ControlMode:
        """Active `ControlMode` of an arm (it can change on its own: a diverging arm is held in JOINT_IMPEDANCE)."""
        self._check_arm(arm_id)
        return self._controllers[arm_id].mode

    def set_impedance_gains(self, arm_id: str, kp=None, kd=None, space: str = "cartesian") -> None:
        """Impedance gains; kd None = 2 sqrt(kp). space: "cartesian" (6,), "joint" (n,) or
        "admittance_inner" (6,, the CARTESIAN_ADMITTANCE inner loop)."""
        self._check_torque_arm(arm_id)
        self._controllers[arm_id].set_impedance_gains(kp, kd, space)

    def set_admittance_target(self, arm_id: str, pos=None, rot=None, wrench=None, mask=None) -> None:
        """CARTESIAN_ADMITTANCE targets: TCP pose (base frame), wrench [f, tau] (base frame,
        robot-on-environment; pressing down with 5 N is f_z = -5) and per-axis mask
        [x, y, z, roll, pitch, yaw] (1 = pose, 0 = force). None leaves a field unchanged."""
        self._check_torque_arm(arm_id)
        self._controllers[arm_id].set_admittance_target(pos, rot, wrench, mask)

    def tare(self, arm_id: str) -> None:
        """Re-estimate the arm's F/T bias (admittance holds still meanwhile)."""
        self._check_torque_arm(arm_id)
        self._controllers[arm_id].tare()

    def external_wrench(self, arm_id: str):
        """Wrist F/T as an external wrench estimate (like Franka's O_F_ext_hat_K): (wrench, point) with
        the payload-compensated [f, tau] the robot exerts on the environment in the arm base frame,
        torque about `point` (sensor origin, base frame). None for position arms or without a sensor."""
        self._check_arm(arm_id)
        if arm_id not in self._plants:
            return None
        return self._controllers[arm_id].external_wrench()

    def get_telemetry(self) -> Telemetry:
        if self._last_telem is None:
            mujoco.mj_kinematics(self._mj.model, self._mj.data)
            self._pin_sync.sync()
            self._last_telem = self._telem.collect(self._sim_time)
        return self._last_telem

    def show_object_frame(self, object_name: str) -> None:
        """Register an RGB axis frame on a named body, site, or geom.

        Drawn in the passive viewer loop at viewer refresh rate (~20 Hz).
        """
        if not object_name:
            return
        ref = resolve_object_ref(self._mj.model, object_name)
        if ref is None:
            logger.debug("Object frame not found in model: %r", object_name)
            return
        with self._frames_lock:
            if object_name not in self._object_frames:
                self._object_frames.append(object_name)
                self._object_frame_ref_cache[object_name] = ref
                self._frames_redraw_pending = True

    def remove_object_frame(self, object_name: str) -> None:
        """Stop drawing a named object frame overlay."""
        with self._frames_lock:
            if object_name in self._object_frames:
                self._object_frames.remove(object_name)
                self._object_frame_ref_cache.pop(object_name, None)
                self._drawn_object_pose_cache.pop(object_name, None)
                self._frames_redraw_pending = True

    def show_custom_frame(self, name: str, pose: np.ndarray) -> None:
        """Register a named 4x4 pose overlay in the MuJoCo world frame.

        Drawn in the passive viewer loop at viewer refresh rate (~20 Hz).
        Re-sending the same pose is a no-op.
        """
        if not name:
            raise ValueError("Frame name must be non-empty")
        T = np.asarray(pose, dtype=float)
        if T.shape != (4, 4):
            raise ValueError(f"Frame pose must be 4x4, got {T.shape}")
        if not np.all(np.isfinite(T)):
            raise ValueError("Frame pose must contain only finite values")
        with self._frames_lock:
            existing = self._custom_frames.get(name)
            if existing is not None and poses_equal(
                existing[:3, 3], existing[:3, :3], T[:3, 3], T[:3, :3],
            ):
                return
            self._custom_frames[name] = T.copy()
            self._frames_redraw_pending = True

    def remove_custom_frame(self, name: str) -> None:
        """Remove a named custom frame overlay."""
        with self._frames_lock:
            removed = self._custom_frames.pop(name, None)
            if removed is not None:
                self._frames_redraw_pending = True

    def _object_poses_changed(
        self,
        object_frames: list[str],
        ref_cache: dict[str, tuple[int, int]],
    ) -> bool:
        """Return True when any tracked object frame pose differs from the draw cache."""
        for name in object_frames:
            ref = ref_cache.get(name)
            if ref is None:
                changed = True
                return changed
            pos, rmat = read_object_pose(self._mj.data, ref[0], ref[1])
            cached = self._drawn_object_pose_cache.get(name)
            if cached is None:
                changed = True
                return changed
            cached_pos, cached_rmat = cached
            if not poses_equal(pos, rmat, cached_pos, cached_rmat):
                changed = True
                return changed
        changed = False
        return changed

    def _update_drawn_object_pose_cache(self, data, object_frames: list[str]) -> None:
        """Refresh the drawn-pose cache after writing object frame overlays."""
        for name in object_frames:
            ref = self._object_frame_ref_cache.get(name)
            if ref is None:
                continue
            pos, rmat = read_object_pose(data, ref[0], ref[1])
            self._drawn_object_pose_cache[name] = (pos.copy(), rmat.copy())

    def on_step(
        self,
        fn: Callable[[int, float, Telemetry], None],
    ) -> None:
        self._callbacks.append(fn)

    def before_tick(self, fn: Callable[[], None]) -> None:
        """Register a callback invoked at the very start of each tick, before
        the control loop writes data.ctrl.  Use this to forward viewer slider
        values (data.ctrl set by the passive viewer between ticks) into the
        joint-position targets before ctrl.compute() runs."""
        self._pre_tick_callbacks.append(fn)

    def step(self) -> Telemetry:
        self._tick()
        return self._last_telem  # type: ignore[return-value]

    def run(
        self,
        duration: Optional[float] = None,
        headless: bool = False,
    ) -> None:
        self._running = True
        if headless:
            self._loop_headless(duration)
        else:
            self._loop_viewer(duration)

    def stop(self) -> None:
        self._running = False

    def close(self) -> None:
        self._running = False

    def __enter__(self) -> "CyRoSimulator":
        return self

    def __exit__(self, *_) -> None:
        self.close()

    def reset(self) -> None:
        self._reset_to_home()
        self._sim_time = 0.0
        self._tick_count = 0
        self._physics_steps = 0
        self._last_telem = None

    def _tick(self) -> None:
        for cb in self._pre_tick_callbacks:
            cb()

        dt = self._dt
        t = self._sim_time
        m, d = self._mj.model, self._mj.data

        if self._position_arms:
            self._pin_sync.sync()
        self._mj.clear_applied()

        for arm_id in self._position_arms:
            ctrl = self._controllers[arm_id]
            cfg = self._arm_configs[arm_id]
            q_ref, tau_ff = ctrl.compute(dt)
            self._mj.set_ctrl_pos(cfg.actuator_names, q_ref)
            if np.any(tau_ff != 0.0):
                self._mj.add_ctrl_torque(cfg.joint_names, tau_ff)

        if not self._torque_arms:
            self._mj.physics_step()
            self._physics_steps += self._mj.n_substeps
        else:
            # Torque laws see the state at the start of each physics step (no one-step lag).
            adm_dt = self._adm_decimation * m.opt.timestep
            for _ in range(self._mj.n_substeps):
                mujoco.mj_step1(m, d)
                outer = self._physics_steps % self._adm_decimation == 0
                for arm_id in self._torque_arms:
                    ctrl = self._controllers[arm_id]
                    if outer:
                        ctrl.update_outer(adm_dt)
                    np.copyto(self._plants[arm_id].ctrl_arm, ctrl.torque())
                mujoco.mj_step2(m, d)
                self._physics_steps += 1
        self._sim_time += dt
        self._tick_count += 1

        # MuJoCo resets MjData to qpos0 on a NaN/huge qpos, qvel or qacc (and only warns), so a
        # divergence shows up either as non-finite qpos or as a bad-state warning.
        if any(d.warning[w].number for w in _BAD_STATE_WARNINGS) or not np.all(np.isfinite(d.qpos)):
            for w in _BAD_STATE_WARNINGS:
                d.warning[w].number = 0
            logger.warning("Simulation diverged at t=%.4f — restoring the last finite state.", t)
            np.copyto(d.qpos, self._last_finite_qpos)
            np.copyto(d.ctrl, self._last_finite_ctrl)
            d.time = self._last_finite_time
            for arm_id in self._position_arms:
                ctrl = self._controllers[arm_id]
                cfg = self._arm_configs[arm_id]
                for i, jname in enumerate(cfg.joint_names):
                    d.qpos[m.jnt_qposadr[self._mj.joint_id(jname)]] = ctrl._q_ref[i]
            d.qvel[:] = 0.0
            d.qacc[:] = 0.0
            d.qacc_warmstart[:] = 0.0
            mujoco.mj_forward(m, d)
            for arm_id in self._torque_arms:
                self._controllers[arm_id].hold()
        else:
            self._save_finite_state()

        if self._tick_count % self._telemetry_stride == 0 or self._last_telem is None:
            # Refresh kinematics from the post-step state before collecting telemetry.
            if self._torque_arms:
                mujoco.mj_kinematics(m, d)
            self._pin_sync.sync()
            self._last_telem = self._telem.collect(t)

    def _loop_headless(self, duration: Optional[float]) -> None:
     
        tick = 0
        while self._running:
            if duration is not None and self._sim_time >= duration:
                break
            t0 = time.perf_counter()
            self._tick()
            
            if self._callbacks and self._last_telem:
                for _cb in self._callbacks:
                    _cb(tick, self._sim_time, self._last_telem)
            if self.realtime_factor > 0:
                sleep_t = self._dt / self.realtime_factor - (time.perf_counter() - t0)
                if sleep_t > 0:
                    time.sleep(sleep_t)
            tick += 1
        self._running = False

    def _loop_viewer(self, duration: Optional[float]) -> None:
        import mujoco.viewer

        with mujoco.viewer.launch_passive(
            self._mj.model,
            self._mj.data,
            show_left_ui=False,
            show_right_ui=False,
        ) as viewer:
            viewer.cam.azimuth = 135
            viewer.cam.elevation = -20
            viewer.cam.distance = 2.5
            viewer.cam.lookat = np.array([0.0, 0.0, 0.6])
            tick = 0
            while viewer.is_running() and self._running:
                if duration is not None and self._sim_time >= duration:
                    break
                t0 = time.perf_counter()
                self._tick()
                
                if self._callbacks and self._last_telem:
                    for _cb in self._callbacks:
                        _cb(tick, self._sim_time, self._last_telem)
                if tick % self._viewer_stride == 0:
                    with self._frames_lock:
                        object_frames = list(self._object_frames)
                        custom_frames = dict(self._custom_frames)
                        redraw_pending = self._frames_redraw_pending
                        ref_cache = dict(self._object_frame_ref_cache)
                    if object_frames or custom_frames:
                        if redraw_pending or self._object_poses_changed(
                            object_frames, ref_cache,
                        ):
                            draw_all_frames(
                                viewer,
                                self._mj.model,
                                self._mj.data,
                                object_frames,
                                custom_frames,
                                ref_cache,
                            )
                            with self._frames_lock:
                                self._update_drawn_object_pose_cache(
                                    self._mj.data, object_frames,
                                )
                                self._frames_redraw_pending = False
                    elif redraw_pending:
                        viewer.user_scn.ngeom = 0
                        with self._frames_lock:
                            self._frames_redraw_pending = False
                    viewer.sync()
                if self.realtime_factor > 0:
                    sleep_t = self._dt / self.realtime_factor - (time.perf_counter() - t0)
                    if sleep_t > 0:
                        time.sleep(sleep_t)
                tick += 1

    def _reset_to_home(self) -> None:
        for arm_id, cfg in self._arm_configs.items():
            ctrl = self._controllers[arm_id]
            for i, jname in enumerate(cfg.joint_names):
                adr = self._mj.model.jnt_qposadr[self._mj.joint_id(jname)]
                self._mj.data.qpos[adr] = cfg.q_home[i]
                self._mj.data.qvel[self._mj.model.jnt_dofadr[self._mj.joint_id(jname)]] = 0.0
            if arm_id in self._plants:
                self._plants[arm_id].ctrl_arm[:] = 0.0
            else:
                ctrl._q_ref = cfg.q_home.copy()
                self._mj.set_ctrl_pos(cfg.actuator_names, cfg.q_home)
        # Close grippers and zero camera yaw
        for arm_id in ("left", "right"):
            if arm_id in self._aux_config.get("grippers", {}):
                self.set_gripper(arm_id, 0.0)
        for arm_id, act_names in self._aux_config.get("dg3f_grippers", {}).items():
            for name in act_names:
                self._mj.set_actuator(name, 0.0)
        mujoco.mj_forward(self._mj.model, self._mj.data)
        for arm_id in self._torque_arms:
            self._controllers[arm_id].reset()  # JOINT_IMPEDANCE holding q_home
        self._save_finite_state()

    def _save_finite_state(self) -> None:
        d = self._mj.data
        np.copyto(self._last_finite_qpos, d.qpos)
        np.copyto(self._last_finite_ctrl, d.ctrl)
        self._last_finite_time = d.time

    def _check_torque_arm(self, arm_id: str) -> None:
        self._check_arm(arm_id)
        if arm_id not in self._plants:
            raise ValueError(f"Arm '{arm_id}' is position-actuated (load with actuation='torque')")

    def _check_arm(self, arm_id: str) -> None:
        if arm_id not in self._arm_configs:
            raise ValueError(
                f"Unknown arm '{arm_id}'. Valid: {self.arm_ids}"
            )

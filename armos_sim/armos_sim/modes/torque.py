"""Per-arm controller for torque-actuated arms.

Modes (all laws from armos_core):
    JOINT_TORQUE          tau = commanded torque + qfrc_bias
    JOINT_IMPEDANCE       JointImpedance, target from set_joint_pos
    CARTESIAN_IMPEDANCE   CartesianImpedance, target from set_tcp_pose / set_flange_pose
    CARTESIAN_ADMITTANCE  WrenchCompensator + HybridAdmittance (outer, once per control tick) -> virtual pose
                          -> stiff CartesianImpedance (inner, every physics tick). The inner loop is stiff
                          (3000 N/m, 100 Nm/rad, as mujoco_rl_bed's AxisCompliance) so the TCP stays on the
                          virtual frame despite joint friction; a soft inner loop lags it by several mm.

Targets and poses given to this class are in the arm base frame (like the position modes and the
telemetry); the laws run in the world frame on the plant's TCP site (`ArmConfig.ee_site`).

A target set while another mode is active is kept and applied when its mode is entered (the bridge
writes the target first, then switches the mode). Entering a mode without a pending target holds the
current state.
"""

from __future__ import annotations

import logging
from typing import Optional

import mujoco
import numpy as np

from armos_core.control import (
    CartesianImpedanceCfg,
    HybridAdmittance,
    HybridAdmittanceCfg,
    JointImpedanceCfg,
    WrenchCompensator,
    WrenchCompensatorCfg,
)
from armos_core.math.se3 import inv_T, make_T
from armos_core.math.so3 import quat_to_rot, rot_to_quat

from ..plant import ArmPlant
from ..types import TORQUE_MODES, ArmConfig, ControlMode

logger = logging.getLogger(__name__)


class _TorqueArmController:
    """Mode switching and targets for one torque-actuated arm."""

    def __init__(self, cfg: ArmConfig, plant: ArmPlant):
        self._cfg = cfg
        self._plant = plant
        n = cfg.ndof

        jcfg = JointImpedanceCfg()
        if len(jcfg.kp) != n:  # defaults are for 7-DOF Franka arms
            jcfg = JointImpedanceCfg(kp=(300.0,) * n, kd=None)
        self._joint = jcfg.build(plant)
        self._cart = CartesianImpedanceCfg().build(plant)
        self._cart_adm = CartesianImpedanceCfg(kp=(3000.0,) * 3 + (100.0,) * 3).build(plant)
        self._adm = HybridAdmittance(HybridAdmittanceCfg())
        self._laws = {"cartesian": self._cart, "joint": self._joint, "admittance_inner": self._cart_adm}
        # Gains given by set_impedance_gains, re-applied when their law is reset on mode entry (a law's
        # reset restores its config gains).
        self._gains: dict[str, tuple] = {}

        # Frames: world <- arm base, TCP <- ee_site (tool), sensor <- TCP.
        self._T_wb = make_T(cfg.base_world, cfg.base_world_rot)
        self._T_bw = inv_T(self._T_wb)
        tl = cfg.tool.tcp_location
        self._T_site_tcp = make_T(tl[:3], quat_to_rot(tl[3:]))
        self._T_tcp_site = inv_T(self._T_site_tcp)
        R_tcp_sensor = np.eye(3)
        if plant.has_wrench_sensor:
            # Fixed sensor-in-TCP rotation, read from a scratch MjData at qpos0.
            d = mujoco.MjData(plant.model)
            mujoco.mj_kinematics(plant.model, d)
            R_w_site = d.site_xmat[plant.h.tcp_site_id].reshape(3, 3)
            R_w_ft = d.site_xmat[plant.h.ft_site_id].reshape(3, 3)
            R_tcp_sensor = (R_w_site @ self._T_site_tcp[:3, :3]).T @ R_w_ft
        g_base = self._T_bw[:3, :3] @ plant.model.opt.gravity
        self._wc = WrenchCompensator(WrenchCompensatorCfg(
            gravity=g_base, sensor_in_tcp_quat=rot_to_quat(R_tcp_sensor),
        ))

        self._mode = ControlMode.JOINT_IMPEDANCE
        self._tau_cmd = np.zeros(n)
        self._tau = np.zeros(n)
        self._pending_q: Optional[np.ndarray] = None
        self._pending_site_T: Optional[np.ndarray] = None   # world-frame ee_site target
        self._adm_pose_set = False                           # admittance pose target given for next entry

    # ------------------------------------------------------------------ mode
    @property
    def mode(self) -> ControlMode:
        return self._mode

    @property
    def plant(self) -> ArmPlant:
        return self._plant

    def set_mode(self, mode: ControlMode) -> None:
        if mode not in TORQUE_MODES:
            raise ValueError(
                f"Arm '{self._cfg.arm_id}' is torque-actuated; {mode.name} needs position actuation. "
                f"Valid: {sorted(m.name for m in TORQUE_MODES)}"
            )
        if mode == self._mode:
            return
        logger.debug("Mode switch: %s → %s", self._mode.name, mode.name)
        self._mode = mode
        self._enter(mode)

    def _enter(self, mode: ControlMode) -> None:
        p = self._plant
        if mode == ControlMode.JOINT_IMPEDANCE:
            self._joint.reset(p)
            self._restore_gains("joint")
            if self._pending_q is not None:
                self._joint.set_target(q=self._pending_q)
                self._pending_q = None
        elif mode == ControlMode.CARTESIAN_IMPEDANCE:
            self._cart.reset(p)
            self._restore_gains("cartesian")
            if self._pending_site_T is not None:
                self._set_cart_target(self._pending_site_T)
                self._pending_site_T = None
        elif mode == ControlMode.CARTESIAN_ADMITTANCE:
            self._cart_adm.reset(p)      # hold while the F/T tare collects
            self._restore_gains("admittance_inner")
            self._wc.reset()
            # The target becomes the current pose after the tare unless one was given for this entry.
            self._adm.clear(keep_target=self._adm_pose_set)
            self._adm_pose_set = False

    def hold(self) -> None:
        """Switch to JOINT_IMPEDANCE holding the current configuration (no pending target)."""
        self._pending_q = None
        self._mode = ControlMode.JOINT_IMPEDANCE
        self._joint.reset(self._plant)
        self._restore_gains("joint")

    # ------------------------------------------------------------------ targets
    def set_joint_pos(self, q: np.ndarray) -> None:
        q = np.clip(q, self._cfg.q_lo, self._cfg.q_hi)
        if self._mode == ControlMode.JOINT_IMPEDANCE:
            self._joint.set_target(q=q)
        else:
            self._pending_q = q

    def set_joint_torque(self, tau: np.ndarray) -> None:
        self._tau_cmd = np.asarray(tau, dtype=float).copy()

    def set_tcp_pose(self, pos: np.ndarray, rot: np.ndarray) -> None:
        self._set_site_target(self._T_wb @ make_T(pos, rot) @ self._T_tcp_site)

    def set_flange_pose(self, pos: np.ndarray, rot: np.ndarray) -> None:
        self._set_site_target(self._T_wb @ make_T(pos, rot))

    def _set_site_target(self, T_w_site: np.ndarray) -> None:
        if self._mode == ControlMode.CARTESIAN_IMPEDANCE:
            self._set_cart_target(T_w_site)
        else:
            self._pending_site_T = T_w_site

    def _set_cart_target(self, T_w_site: np.ndarray) -> None:
        self._cart.set_target(pos=T_w_site[:3, 3], quat=rot_to_quat(T_w_site[:3, :3]))

    def set_joint_vel(self, qdot: np.ndarray) -> None:
        raise ValueError(f"Arm '{self._cfg.arm_id}' is torque-actuated: no joint velocity mode")

    def set_flange_vel(self, linear: np.ndarray, angular: np.ndarray) -> None:
        raise ValueError(f"Arm '{self._cfg.arm_id}' is torque-actuated: no Cartesian velocity mode")

    set_tcp_vel = set_flange_vel

    def set_impedance_gains(self, kp=None, kd=None, space: str = "cartesian") -> None:
        """Stiffness/damping of an impedance law; kd None = 2 sqrt(kp).

        space: "cartesian" (6,), "joint" (n,) or "admittance_inner" (6,, the CARTESIAN_ADMITTANCE inner loop).
        The gains are kept when the law is reset on mode entry.
        """
        kp = None if kp is None else np.asarray(kp, dtype=float)
        kd = None if kd is None else np.asarray(kd, dtype=float)
        self._laws[space].set_target(kp=kp, kd=kd)
        old_kp, old_kd = self._gains.get(space, (None, None))
        self._gains[space] = (kp if kp is not None else old_kp, kd if kd is not None else old_kd)

    def _restore_gains(self, space: str) -> None:
        kp, kd = self._gains.get(space, (None, None))
        if kp is not None or kd is not None:
            self._laws[space].set_target(kp=kp, kd=kd)

    def set_admittance_target(self, pos=None, rot=None, wrench=None, mask=None) -> None:
        """TCP pose target (base frame), target wrench [f, tau] (base frame, robot-on-environment) and
        per-axis mask (1 = pose, 0 = force) for CARTESIAN_ADMITTANCE. None leaves a field unchanged."""
        if pos is not None:
            R = np.eye(3) if rot is None else np.asarray(rot, dtype=float)
            self._adm.set_target_pose(np.asarray(pos, dtype=float), R)
            self._adm_pose_set = self._mode != ControlMode.CARTESIAN_ADMITTANCE
        if wrench is not None:
            self._adm.set_target_wrench(wrench)
        if mask is not None:
            self._adm.set_mask(mask)

    def tare(self) -> None:
        """Re-estimate the F/T bias (the admittance holds still until done)."""
        self._wc.reset()

    def external_wrench(self) -> Optional[tuple[np.ndarray, np.ndarray]]:
        """The wrist F/T reading as an external wrench estimate, like Franka's O_F_ext_hat_K.

        Returns:
            (wrench, point): payload-compensated [f, tau] the robot exerts on the environment, arm base
            frame, torque about `point` (the F/T sensor origin, base frame); None without a sensor.
        """
        p = self._plant
        if not p.has_wrench_sensor:
            return None
        R_sensor = self._wc.sensor_rotation(self._tcp_in_base()[:3, :3])
        wrench = self._wc.compensate(np.array(p.wrist_wrench(), dtype=float), R_sensor)
        point = self._T_bw[:3, :3] @ p.data.site_xpos[p.h.ft_site_id] + self._T_bw[:3, 3]
        return wrench, point

    # ------------------------------------------------------------------ loop
    def _tcp_in_base(self) -> np.ndarray:
        p = self._plant
        return self._T_bw @ make_T(p.ee_pos, p.ee_rot.reshape(3, 3)) @ self._T_site_tcp

    def update_outer(self, dt: float) -> None:
        """Once per control tick, after `mj_step1`: the admittance outer loop."""
        if self._mode != ControlMode.CARTESIAN_ADMITTANCE:
            return
        T_tcp = self._tcp_in_base()
        wrench = self._wc.update(self._plant.wrist_wrench(), T_tcp[:3, :3], dt)
        if wrench is None:   # taring: the inner loop holds the pose from mode entry
            return
        if self._adm.virtual_pos is None:
            self._adm.reset(T_tcp[:3, 3], T_tcp[:3, :3])
        R_axes = T_tcp[:3, :3] if self._adm.axes == "tcp" else None
        if R_axes is not None:   # base-frame wrench -> TCP axes
            wrench = np.concatenate((R_axes.T @ wrench[:3], R_axes.T @ wrench[3:]))
        pos, R = self._adm.step(wrench, dt, T_tcp[:3, 3], T_tcp[:3, :3])
        T_site = self._T_wb @ make_T(pos, R) @ self._T_tcp_site
        self._cart_adm.set_target(pos=T_site[:3, 3], quat=rot_to_quat(T_site[:3, :3]))

    def torque(self) -> np.ndarray:
        """Every physics tick, after `mj_step1`: joint torques for the active mode."""
        p = self._plant
        if self._mode == ControlMode.JOINT_IMPEDANCE:
            return self._joint.torque(p)
        if self._mode == ControlMode.CARTESIAN_IMPEDANCE:
            return self._cart.torque(p)
        if self._mode == ControlMode.CARTESIAN_ADMITTANCE:
            return self._cart_adm.torque(p)
        np.add(self._tau_cmd, p.qfrc_bias, out=self._tau)   # JOINT_TORQUE
        return self._tau

    def reset(self) -> None:
        """After a simulator reset: hold the current configuration, drop pending targets."""
        self._pending_q = None
        self._pending_site_T = None
        self._tau_cmd[:] = 0.0
        self._adm = HybridAdmittance(self._adm.cfg)
        self._wc.reset()
        self.hold()

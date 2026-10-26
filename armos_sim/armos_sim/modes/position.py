"""Position-actuated arms: the per-mode joint references (servo setpoints) and the Pink IK solver.

Each `*_ref` function returns the new joint reference for one control tick; mode switching, command
storage and blending live in `modes.dispatcher._ArmController`. Pinocchio and Pink are imported on
first use (`_PinkSolver.create`), not with armos_sim.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from ..runtime.pinocchio_sync import _PinocchioSync
from ..types import ArmConfig, _JointCmd, _TaskCmd

from armos_core.math.so3 import quat_to_rot

logger = logging.getLogger(__name__)


class _PinkSolver:
    """Wraps a Pink QP-based differential IK for one arm."""

    def __init__(self, cfg: ArmConfig, pin_model, pin, pink):
        self._cfg = cfg
        self._model = pin_model
        self._pin = pin
        self._pink = pink
        # Own pin.Data: Pink's Configuration.update overwrites the kinematics stored in it.
        self._config = pink.Configuration(pin_model, pin_model.createData(), pin.neutral(pin_model))
        # Gains < 1: with deadbeat gains (1.0) the frame and posture tasks
        # fight in the null space and q_ref flips between two solutions every
        # tick (100 Hz chatter on the Panda wrist joints at q_home).
        self._ee_task = pink.tasks.FrameTask(
            f"{cfg.arm_id}_ee",
            position_cost=1.0,
            orientation_cost=0.5,
            gain=0.5,
        )
        self._posture_task = pink.tasks.PostureTask(cost=1e-3, gain=0.05)
        self._posture_task.set_target(np.zeros(pin_model.nq))

    @classmethod
    def create(cls, cfg: ArmConfig, pin_model) -> Optional["_PinkSolver"]:
        """Import Pinocchio + Pink and build the solver; None when either is unavailable."""
        try:
            import pinocchio as pin
            import pink
            import pink.tasks  # noqa: F401
        except Exception as exc:  # ImportError, or a numpy ABI mismatch
            logger.info("Pink unavailable (%s): TASK_POSE holds position.", exc)
            return None
        return cls(cfg, pin_model, pin, pink)

    def se3(self, rot: np.ndarray, pos: np.ndarray):
        return self._pin.SE3(rot, pos)

    def solve(self, q: np.ndarray, target_SE3, dt: float) -> np.ndarray:
        ndof = self._cfg.ndof
        q_safe = np.clip(
            np.nan_to_num(q, nan=0.0, posinf=2.8, neginf=-2.8),
            self._cfg.q_lo,
            self._cfg.q_hi,
        )
        q_pin = np.zeros(self._model.nq)
        q_pin[:ndof] = q_safe
        self._config.update(q_pin)
        self._ee_task.set_target(target_SE3)
        try:
            vel = self._pink.solve_ik(
                self._config,
                [self._ee_task, self._posture_task],
                dt,
                solver="quadprog",
                safety_break=False,
            )
        except Exception as exc:
            logger.warning("Pink IK failed (%s) — holding position.", exc)
            return np.zeros(ndof)
        return vel[:ndof]


# ---------------------------------------------------------------------- per-mode references
def safe_q(cfg: ArmConfig, q: np.ndarray) -> np.ndarray:
    return np.clip(
        np.nan_to_num(q, nan=0.0, posinf=2.8, neginf=-2.8),
        cfg.q_lo,
        cfg.q_hi,
    )


def tcp_in_flange(cfg: ArmConfig) -> np.ndarray:
    """4×4 SE3 of TCP in flange frame, from tool params."""
    tl = cfg.tool.tcp_location
    T = np.eye(4)
    T[:3, :3] = quat_to_rot(tl[3:])
    T[:3, 3] = tl[:3]
    return T


def joint_pos_ref(cmd: Optional[_JointCmd], q_ref: np.ndarray) -> np.ndarray:
    """JOINT_POS: the commanded position, else hold the reference."""
    if cmd and cmd.position is not None:
        return cmd.position
    return q_ref.copy()


def joint_vel_ref(cfg: ArmConfig, cmd: Optional[_JointCmd], q_ref: np.ndarray, dt: float) -> np.ndarray:
    """JOINT_VEL: integrate the commanded velocity into the reference."""
    if cmd and cmd.velocity is not None:
        return np.clip(q_ref + cmd.velocity * dt, cfg.q_lo, cfg.q_hi)
    return q_ref.copy()


def task_pose_ref(
    cfg: ArmConfig,
    pink: Optional[_PinkSolver],
    cmd: Optional[_TaskCmd],
    q: np.ndarray,
    q_ref: np.ndarray,
    dt: float,
) -> np.ndarray:
    """TASK_POSE: one Pink IK step towards the commanded flange / TCP pose (arm base frame)."""
    if not (pink and cmd):
        if not pink:
            logger.warning(
                "TASK_POSE requires Pinocchio + Pink. "
                "Falling back to holding position."
            )
        return q.copy()
    # Integrate IK from the reference, not the measured q (same as
    # TASK_VEL). Feeding measured q back into the servo setpoint makes the
    # position actuators chatter at the control-rate Nyquist frequency.
    q_safe = safe_q(cfg, q_ref)

    if cmd.use_tcp:
        # TCP target → flange target: T_flange = T_tcp @ inv(T_tcp_in_flange)
        T_tcp = np.eye(4)
        T_tcp[:3, :3] = cmd.orientation
        T_tcp[:3, 3] = cmd.position
        T_tf = tcp_in_flange(cfg)
        R_tf = T_tf[:3, :3]
        T_tf_inv = np.eye(4)
        T_tf_inv[:3, :3] = R_tf.T
        T_tf_inv[:3, 3] = -R_tf.T @ T_tf[:3, 3]
        T_flange = T_tcp @ T_tf_inv
        target = pink.se3(T_flange[:3, :3], T_flange[:3, 3])
    else:
        target = pink.se3(cmd.orientation, cmd.position)
    qdot = pink.solve(q_safe, target, dt)
    return np.clip(
        q_safe + qdot * dt,
        cfg.q_lo,
        cfg.q_hi,
    )


def task_vel_ref(
    cfg: ArmConfig,
    sync: _PinocchioSync,
    cmd: Optional[_TaskCmd],
    q_ref: np.ndarray,
    dt: float,
) -> np.ndarray:
    """TASK_VEL: damped least-squares map of the commanded flange / TCP twist, integrated into the reference."""
    if not cmd:
        return q_ref.copy()

    omega = cmd.angular_vel
    if cmd.use_tcp:
        # TCP twist → flange twist: v_flange = v_tcp - ω × (R_flange @ p_tcp)
        p_tcp_local = cfg.tool.tcp_location[:3]
        R_flange = sync.get_flange_pose(cfg.arm_id)[:3, :3]
        v_flange = cmd.linear_vel - np.cross(omega, R_flange @ p_tcp_local)
    else:
        v_flange = cmd.linear_vel

    twist = np.concatenate([v_flange, omega])
    J = sync.get_jacobian(cfg.arm_id)
    if not np.all(np.isfinite(J)):
        return q_ref.copy()
    lam = 0.05
    J_dls = J.T @ np.linalg.inv(J @ J.T + lam * np.eye(6))
    qdot_cmd = J_dls @ twist
    speed = np.linalg.norm(qdot_cmd)
    if speed > 1.0:
        qdot_cmd /= speed
    return np.clip(
        q_ref + qdot_cmd * dt,
        cfg.q_lo,
        cfg.q_hi,
    )

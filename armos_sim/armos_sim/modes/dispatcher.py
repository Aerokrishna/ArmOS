"""Per-arm mode dispatch.

`make_controller` picks an arm's controller from its actuation:
    position → `_ArmController` (below): mode switching with a blended joint reference, commands, and the
               per-mode references from `modes.position`.
    torque   → `modes.torque._TorqueArmController`: the armos_core torque laws.
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple, Union

import numpy as np

from ..constants import _BLEND_TIME
from ..model.mj_model import _MuJoCoModel
from ..plant import ArmPlant
from ..runtime.pinocchio_sync import _PinocchioSync
from ..types import POSITION_MODES, ArmConfig, ControlMode, _JointCmd, _TaskCmd
from . import position
from .position import _PinkSolver
from .torque import _TorqueArmController

logger = logging.getLogger(__name__)


class _ModeLatch:
    """Blends joint reference from old to new mode over _BLEND_TIME seconds."""

    def __init__(self, ndof: int):
        self._mode = ControlMode.JOINT_POS
        # Start with the blend finished: there is no previous mode to blend from, and a
        # zero latch would pull the reference towards q = 0 until the first command.
        self._blend_t = _BLEND_TIME
        self._q_latch = np.zeros(ndof)
        self._qdot_latch = np.zeros(ndof)

    @property
    def mode(self) -> ControlMode:
        return self._mode

    @property
    def q_latch(self) -> np.ndarray:
        return self._q_latch

    def switch(
        self,
        new_mode: ControlMode,
        q_now: np.ndarray,
        qdot_now: np.ndarray,
    ) -> None:
        if new_mode == self._mode:
            return
        logger.debug("Mode switch: %s → %s", self._mode.name, new_mode.name)
        self._q_latch = q_now.copy()
        self._qdot_latch = qdot_now.copy()
        self._blend_t = 0.0
        self._mode = new_mode

    def alpha(self, dt: float) -> float:
        self._blend_t += dt
        return min(1.0, self._blend_t / _BLEND_TIME)


class _ArmController:
    """
    Unified mode-switching controller for a single position-actuated arm.

    Joint-space modes work without Pinocchio or Pink.
    TASK_POSE needs both; TASK_VEL uses Pinocchio when available.
    """

    def __init__(
        self,
        cfg: ArmConfig,
        mj: _MuJoCoModel,
        pin_sync: _PinocchioSync,
        pink_solver: Optional[_PinkSolver],
    ):
        self._cfg = cfg
        self._mj = mj
        self._sync = pin_sync
        self._pink = pink_solver
        self._latch = _ModeLatch(cfg.ndof)
        self._q_ref = cfg.q_home.copy()
        self._joint_cmd: Optional[_JointCmd] = None
        self._task_cmd: Optional[_TaskCmd] = None

    @property
    def mode(self) -> ControlMode:
        return self._latch.mode

    def set_mode(self, mode: ControlMode) -> None:
        if mode not in POSITION_MODES:
            raise ValueError(
                f"Arm '{self._cfg.arm_id}' is position-actuated; {mode.name} needs torque actuation "
                "(load the config with actuation='torque')."
            )
        q = self._sync.get_q(self._cfg.arm_id)
        qdot = self._sync.get_qdot(self._cfg.arm_id)
        prev_mode = self._latch.mode
        self._latch.switch(mode, q, qdot)
        if mode != prev_mode and np.all(np.isfinite(q)):
            self._q_ref = np.clip(q, self._cfg.q_lo, self._cfg.q_hi)

    def set_joint_pos(self, q: np.ndarray) -> None:
        self._joint_cmd = _JointCmd(
            position=np.clip(q, self._cfg.q_lo, self._cfg.q_hi)
        )

    def set_joint_vel(self, qdot: np.ndarray) -> None:
        self._joint_cmd = _JointCmd(velocity=np.asarray(qdot, dtype=float))

    def set_joint_torque(self, tau: np.ndarray) -> None:
        self._joint_cmd = _JointCmd(torque=np.asarray(tau, dtype=float))

    def set_flange_pose(self, pos: np.ndarray, rot: np.ndarray) -> None:
        cmd = _TaskCmd()
        cmd.position = np.asarray(pos)
        cmd.orientation = np.asarray(rot)
        cmd.use_tcp = False
        self._task_cmd = cmd

    def set_tcp_pose(self, pos: np.ndarray, rot: np.ndarray) -> None:
        cmd = _TaskCmd()
        cmd.position = np.asarray(pos)
        cmd.orientation = np.asarray(rot)
        cmd.use_tcp = True
        self._task_cmd = cmd

    def set_flange_vel(self, linear: np.ndarray, angular: np.ndarray) -> None:
        cmd = _TaskCmd()
        cmd.linear_vel = np.asarray(linear)
        cmd.angular_vel = np.asarray(angular)
        cmd.use_tcp = False
        self._task_cmd = cmd

    def set_tcp_vel(self, linear: np.ndarray, angular: np.ndarray) -> None:
        cmd = _TaskCmd()
        cmd.linear_vel = np.asarray(linear)
        cmd.angular_vel = np.asarray(angular)
        cmd.use_tcp = True
        self._task_cmd = cmd

    def compute(self, dt: float) -> Tuple[np.ndarray, np.ndarray]:
        """Returns (q_ref, tau_feedforward) for this tick."""
        cfg = self._cfg
        q_raw = self._sync.get_q(cfg.arm_id)
        bias_raw = self._mj.get_qfrc_bias(cfg.joint_names)

        q = np.clip(
            np.where(np.isfinite(q_raw), q_raw, self._q_ref),
            cfg.q_lo,
            cfg.q_hi,
        )
        # Gravity compensation in every mode, so position control does not fight static load.
        tau_ff = np.clip(
            np.where(np.isfinite(bias_raw), bias_raw, 0.0),
            -500.0,
            500.0,
        )

        mode = self._latch.mode
        if mode == ControlMode.JOINT_POS:
            q_new = position.joint_pos_ref(self._joint_cmd, self._q_ref)
        elif mode == ControlMode.JOINT_VEL:
            q_new = position.joint_vel_ref(cfg, self._joint_cmd, self._q_ref, dt)
        elif mode == ControlMode.JOINT_TORQUE:
            q_new = q.copy()
            if self._joint_cmd and self._joint_cmd.torque is not None:
                tau_ff = tau_ff + self._joint_cmd.torque
        elif mode == ControlMode.TASK_POSE:
            q_new = position.task_pose_ref(cfg, self._pink, self._task_cmd, q, self._q_ref, dt)
        elif mode == ControlMode.TASK_VEL:
            q_new = position.task_vel_ref(cfg, self._sync, self._task_cmd, self._q_ref, dt)
        else:
            q_new = self._q_ref.copy()

        a = self._latch.alpha(dt)
        if a < 1.0:
            q_new = (1.0 - a) * self._latch.q_latch + a * q_new

        self._q_ref = q_new.copy()
        return q_new, tau_ff


def make_controller(
    cfg: ArmConfig,
    mj: _MuJoCoModel,
    pin_sync: _PinocchioSync,
    plant: Optional[ArmPlant],
) -> Union[_ArmController, _TorqueArmController]:
    """The controller for one arm: torque laws when it has a plant (torque actuation), else position modes."""
    if plant is not None:
        return _TorqueArmController(cfg, plant)
    pin_model = pin_sync._pin_models.get(cfg.arm_id)
    pink = _PinkSolver.create(cfg, pin_model) if pin_model is not None else None
    return _ArmController(cfg, mj, pin_sync, pink)

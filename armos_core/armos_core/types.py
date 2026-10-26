"""Shared types: arm handles, the plant interface the control laws read, robot state."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import numpy as np

if TYPE_CHECKING:
    import mujoco


@dataclass(frozen=True)
class ArmHandles:
    """Ids and index ranges of one arm, resolved once from a compiled MuJoCo model.

    Field names follow `mujoco_rl_bed.sim.scene.SceneHandles`, so either record can drive
    `MujocoArm` and the controllers.

    Attributes:
        arm_joint_names: Arm joint names in kinematic order, length n_arm.
        n_arm: Number of arm joints.
        arm_qpos: Contiguous slice of `data.qpos` for the arm.
        arm_dof: Contiguous slice of `data.qvel` / `qfrc_*` / Jacobian columns for the arm.
        arm_ctrl: Contiguous slice of `data.ctrl` for the arm actuators.
        torque_limits: Per-joint torque limits [Nm], shape (n_arm,).
        q_home: Home configuration [rad], shape (n_arm,).
        tcp_site_id: Site id of the TCP (the pose the Cartesian laws control).
        ee_body_id: Body id carrying the TCP site.
        ft_site_id: Site id of the F/T sensor frame, or -1.
        ft_force: Slice of `data.sensordata` with the force (site frame) [N], or None.
        ft_torque: Slice of `data.sensordata` with the torque (site frame) [Nm], or None.
        joint_ids: Name -> id for every named joint.
        gripper_act_id: Gripper actuator id, or -1.
        gripper_ctrlrange: Gripper ctrl range (lo, hi), shape (2,).
    """

    arm_joint_names: tuple[str, ...]
    n_arm: int
    arm_qpos: slice
    arm_dof: slice
    arm_ctrl: slice
    torque_limits: np.ndarray
    q_home: np.ndarray
    tcp_site_id: int
    ee_body_id: int = -1
    ft_site_id: int = -1
    ft_force: slice | None = None
    ft_torque: slice | None = None
    joint_ids: dict[str, int] = field(default_factory=dict)
    gripper_act_id: int = -1
    gripper_ctrlrange: np.ndarray = field(default_factory=lambda: np.zeros(2))


@runtime_checkable
class PlantView(Protocol):
    """What the control laws in `armos_core.control` read from a plant.

    Implemented by `armos_core.kinematics.mujoco_kin.MujocoArm` (and so by `armos_sim`'s `ArmPlant`)
    and by `mujoco_rl_bed`'s `FrankaPlant`. Arrays are views or plant-owned buffers: copy them to keep
    a snapshot. Positions and orientations are in the world frame.
    """

    model: "mujoco.MjModel"
    data: "mujoco.MjData"
    h: ArmHandles
    n: int

    @property
    def q(self) -> np.ndarray: ...

    @property
    def qd(self) -> np.ndarray: ...

    @property
    def ee_pos(self) -> np.ndarray: ...

    @property
    def ee_rot(self) -> np.ndarray: ...

    @property
    def qfrc_bias(self) -> np.ndarray: ...

    @property
    def torque_limits(self) -> np.ndarray: ...

    @property
    def dt(self) -> float: ...

    def ee_quat(self) -> np.ndarray: ...

    def jacobian(self) -> np.ndarray: ...

    def ee_vel(self, J: np.ndarray | None = None) -> np.ndarray: ...

    def wrist_wrench(self, world_frame: bool = False) -> np.ndarray: ...


@dataclass
class RobotState:
    """Snapshot of one arm, as the outer-loop controllers and ROS nodes exchange it.

    Attributes:
        t: Time [s].
        q, dq, tau: Joint positions [rad], velocities [rad/s], torques [Nm].
        tcp_pos: TCP position in the arm base frame [m].
        tcp_quat: TCP orientation in the arm base frame (w, x, y, z).
        tcp_twist: TCP twist (v, w) in the base frame [m/s, rad/s].
        ext_wrench: External wrench (f, tau) in the base frame, robot-on-environment [N, Nm].
    """

    t: float = 0.0
    q: np.ndarray = field(default_factory=lambda: np.zeros(0))
    dq: np.ndarray = field(default_factory=lambda: np.zeros(0))
    tau: np.ndarray = field(default_factory=lambda: np.zeros(0))
    tcp_pos: np.ndarray = field(default_factory=lambda: np.zeros(3))
    tcp_quat: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0, 0.0]))
    tcp_twist: np.ndarray = field(default_factory=lambda: np.zeros(6))
    ext_wrench: np.ndarray = field(default_factory=lambda: np.zeros(6))

"""Public and internal data types for the armos simulator."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional

import numpy as np


class ControlMode(Enum):
    """Per-arm control modes.

    Position-actuated arms: JOINT_POS, JOINT_VEL, JOINT_TORQUE, TASK_POSE, TASK_VEL.
    Torque-actuated arms:   JOINT_TORQUE, JOINT_IMPEDANCE, CARTESIAN_IMPEDANCE, CARTESIAN_ADMITTANCE.
    """

    JOINT_POS = auto()  # position target in joint space
    JOINT_VEL = auto()  # velocity target, integrated to position
    JOINT_TORQUE = auto()  # raw torque + gravity compensation
    TASK_POSE = auto()  # 6-D Cartesian target (SE3) — needs Pink + Pinocchio
    TASK_VEL = auto()  # Cartesian twist — needs Pinocchio
    JOINT_IMPEDANCE = auto()  # armos_core JointImpedance (torque actuation)
    CARTESIAN_IMPEDANCE = auto()  # armos_core CartesianImpedance (torque actuation)
    CARTESIAN_ADMITTANCE = auto()  # HybridAdmittance -> virtual pose -> CartesianImpedance (torque actuation)


POSITION_MODES = frozenset({
    ControlMode.JOINT_POS, ControlMode.JOINT_VEL, ControlMode.JOINT_TORQUE,
    ControlMode.TASK_POSE, ControlMode.TASK_VEL,
})
TORQUE_MODES = frozenset({
    ControlMode.JOINT_TORQUE, ControlMode.JOINT_IMPEDANCE,
    ControlMode.CARTESIAN_IMPEDANCE, ControlMode.CARTESIAN_ADMITTANCE,
})


@dataclass
class ToolParams:
    """Tool center point and inertial parameters for one arm."""

    mass: float = 0.0
    com: np.ndarray = field(default_factory=lambda: np.zeros(3))
    inertia: np.ndarray = field(default_factory=lambda: np.zeros(6))
    # [x, y, z, qw, qx, qy, qz] — TCP location in flange frame
    tcp_location: np.ndarray = field(
        default_factory=lambda: np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
    )


@dataclass
class ArmConfig:
    """
    Static description of one arm.

    The fields joint_names, actuator_names, q_lo, q_hi, base_world and
    torque_limits are auto-derived from the MJCF by _RobotLoader. arm_id,
    ee_site, wrench sensor names, q_home, tool and actuation come from the
    YAML file.
    """

    arm_id: str
    start_body: str  # MuJoCo body name for the arm's base (start of kinematic chain)
    joint_names: List[str]
    actuator_names: List[str]
    ee_site: str
    wrench_force_sensor: Optional[str]
    wrench_torque_sensor: Optional[str]
    q_lo: np.ndarray  # lower joint limits [rad]
    q_hi: np.ndarray  # upper joint limits [rad]
    q_home: np.ndarray  # home configuration [rad]
    base_world: np.ndarray  # world-frame origin of start_body [m]
    base_world_rot: np.ndarray  # world-frame orientation of start_body (3×3)
    tool: ToolParams = field(default_factory=ToolParams)
    actuation: str = "position"  # "position" (servo actuators) or "torque" (motors)
    torque_limits: Optional[np.ndarray] = None  # per-joint torque limits [N·m]

    @property
    def ndof(self) -> int:
        """Degrees of freedom for this arm."""
        return len(self.joint_names)


@dataclass
class Telemetry:
    """Snapshot of full robot state at one control tick."""

    timestamp: float
    joint_pos: Dict[str, np.ndarray]  # [rad]
    joint_vel: Dict[str, np.ndarray]  # [rad/s]
    joint_torque: Dict[str, np.ndarray]  # [N·m]
    wrench: Dict[str, np.ndarray]     # [fx, fy, fz, tx, ty, tz] raw, flange frame [N, N·m]
    flange_pose: Dict[str, np.ndarray]  # (4,4) SE3 in arm base frame
    tcp_pose: Dict[str, np.ndarray]     # (4,4) SE3 in arm base frame
    manipulability: Dict[str, float]
    singularity_distance: Dict[str, float]
    control_mode: Dict[str, str]
    flange_twist: Dict[str, np.ndarray] = field(default_factory=dict)  # (6,) [vx,vy,vz,wx,wy,wz] at flange
    tcp_twist: Dict[str, np.ndarray] = field(default_factory=dict)     # (6,) [vx,vy,vz,wx,wy,wz] at TCP
    ext_wrench: Dict[str, np.ndarray] = field(default_factory=dict)    # (6,) gravity-compensated [N, N·m]
    gripper_joints: Dict[str, np.ndarray] = field(default_factory=dict)  # dg3f: 12 joint angles [rad] per arm

    def as_dict(self) -> dict:
        """JSON-serializable representation."""
        payload = {
            "timestamp": self.timestamp,
            "joint_pos": {k: v.tolist() for k, v in self.joint_pos.items()},
            "joint_vel": {k: v.tolist() for k, v in self.joint_vel.items()},
            "joint_torque": {k: v.tolist() for k, v in self.joint_torque.items()},
            "wrench": {k: v.tolist() for k, v in self.wrench.items()},
            "flange_pose": {k: v.tolist() for k, v in self.flange_pose.items()},
            "tcp_pose": {k: v.tolist() for k, v in self.tcp_pose.items()},
            "flange_twist": {k: v.tolist() for k, v in self.flange_twist.items()},
            "tcp_twist": {k: v.tolist() for k, v in self.tcp_twist.items()},
            "ext_wrench": {k: v.tolist() for k, v in self.ext_wrench.items()},
            "manipulability": self.manipulability,
            "singularity_distance": self.singularity_distance,
            "control_mode": self.control_mode,
            "gripper_joints": {k: v.tolist() for k, v in self.gripper_joints.items()},
        }
        return payload


@dataclass
class _JointCmd:
    """Internal: joint-space command."""

    position: Optional[np.ndarray] = None
    velocity: Optional[np.ndarray] = None
    torque: Optional[np.ndarray] = None


@dataclass
class _TaskCmd:
    """Internal: task-space command."""

    position: np.ndarray = field(default_factory=lambda: np.zeros(3))
    orientation: np.ndarray = field(default_factory=lambda: np.eye(3))
    linear_vel: np.ndarray = field(default_factory=lambda: np.zeros(3))
    angular_vel: np.ndarray = field(default_factory=lambda: np.zeros(3))
    use_tcp: bool = False  # True → target is at TCP; False → target is at flange


# ---------------------------------------------------------------------------
# Client-side UDP / JSON parsing (do not add to this module)
# ---------------------------------------------------------------------------
#
# Telemetry datagrams are UTF-8 JSON objects broadcast by sim_cyro.udp_interface.
# Consumers outside the simulator should deserialize in their own package.
#
# In project_lam_dev, use:
#   - lam_dev.robot.types.Telemetry  (application mirror types)
#   - lam_dev.udp.telemetry.telemetry_from_udp_dict
#
# External clients: paste the block below into your client module.
#
# ```python
# from __future__ import annotations
#
# from typing import Any, Dict, Mapping
#
# import numpy as np
#
# from your_app.types import Telemetry  # mirror of sim_cyro.types.Telemetry
#
#
# def _array_dict(values: Mapping[str, Any] | None) -> Dict[str, np.ndarray]:
#     if not values:
#         return {}
#     return {k: np.asarray(v, dtype=float) for k, v in values.items()}
#
#
# def _pose_dict(values: Mapping[str, Any] | None) -> Dict[str, np.ndarray]:
#     if not values:
#         return {}
#     out: Dict[str, np.ndarray] = {}
#     for arm, pose in values.items():
#         arr = np.asarray(pose, dtype=float)
#         if arr.shape == (4, 4):
#             out[arm] = arr
#         elif arr.size == 16:
#             out[arm] = arr.reshape(4, 4)
#         else:
#             out[arm] = arr
#     return out
#
#
# def telemetry_from_udp_dict(msg: Mapping[str, Any]) -> Telemetry:
#     timestamp = float(msg.get("timestamp", msg.get("t", 0.0)))
#     return Telemetry(
#         timestamp=timestamp,
#         joint_pos=_array_dict(msg.get("joint_pos")),
#         joint_vel=_array_dict(msg.get("joint_vel")),
#         joint_torque=_array_dict(msg.get("joint_torque")),
#         flange_pose=_pose_dict(msg.get("flange_pose")),
#         tcp_pose=_pose_dict(msg.get("tcp_pose")),
#         flange_twist=_array_dict(msg.get("flange_twist")),
#         tcp_twist=_array_dict(msg.get("tcp_twist")),
#         wrench=_array_dict(msg.get("wrench")),
#         ext_wrench=_array_dict(msg.get("ext_wrench")),
#         manipulability=dict(msg.get("manipulability") or {}),
#         singularity_distance=dict(msg.get("singularity_distance") or {}),
#         control_mode=dict(msg.get("control_mode") or {}),
#         gripper_joints=_array_dict(msg.get("gripper_joints")),
#     )
# ```

"""Arm kinematics backends.

- `mujoco_kin.MujocoArm`: allocation-free state/kinematics on MjModel/MjData (world frame).
- `pinocchio_arm.PinocchioArm`: Pinocchio model built from the MuJoCo chain (arm base frame).
  Not imported here, so armos_core works without Pinocchio.
"""

from armos_core.kinematics.mujoco_kin import MujocoArm

__all__ = ["MujocoArm"]

"""PinocchioArm: Pinocchio kinematic model of one arm, built from its MuJoCo kinematic chain.

Source: `armos_sim/pinocchio_sync.py::_PinocchioSync` (model construction and per-arm kinematics).
The Pinocchio universe frame is the arm's start body, so poses and Jacobians are in the arm base frame.

Import this module only when Pinocchio is installed (`import pinocchio` happens at import time).
"""

from __future__ import annotations

from typing import Sequence

import mujoco
import numpy as np
import pinocchio as pin

from armos_core.math.so3 import quat_to_rot


class PinocchioArm:
    """One arm: Pinocchio model/data, an end-effector frame and kinematics queries.

    Attributes:
        model: `pin.Model` (revolute joints of the chain, then the EE operational frame).
        data: `pin.Data`.
        ndof: Number of arm joints.
        frame_name: Name of the EE frame.
        frame_id: Id of the EE frame.
    """

    def __init__(self, model, frame_name: str, ndof: int) -> None:
        self.model = model
        self.data = model.createData()
        self.ndof = ndof
        self.frame_name = frame_name
        self.frame_id = model.getFrameId(frame_name)
        self._q_pin = pin.neutral(model)

    # ------------------------------------------------------------------ construction
    @classmethod
    def from_mjcf(
        cls,
        mj_model: mujoco.MjModel,
        start_body: str,
        joint_names: Sequence[str],
        ee_site: str,
        q_lo: np.ndarray,
        q_hi: np.ndarray,
        frame_name: str = "ee",
        model_name: str = "arm",
    ) -> "PinocchioArm":
        """Build the chain start_body -> ... -> ee_site from a compiled MuJoCo model.

        Args:
            mj_model: Compiled model.
            start_body: Body whose frame becomes the Pinocchio universe.
            joint_names: Arm joints in kinematic order (hinge joints).
            ee_site: Site that becomes the EE operational frame.
            q_lo, q_hi: Joint limits [rad].
            frame_name: Name of the EE frame in the Pinocchio model.
            model_name: Pinocchio model name.
        """
        model = pin.Model()
        model.name = model_name
        parent_pin_id = 0

        start_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, start_body)

        for jname in joint_names:
            jid = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_JOINT, jname)
            body_id = int(mj_model.jnt_bodyid[jid])

            if parent_pin_id == 0:
                # First joint: place it relative to the Pinocchio universe (= start_body frame).
                # Walk the fixed body chain from start_body to this joint's body, then
                # apply the joint anchor offset. This handles start_body being an ancestor
                # of the first joint's body (e.g. *_robot_base_link → *_robot_link1).
                jnt_pos = mj_model.jnt_pos[jid].copy()
                T_start_to_body = _se3_from_start_to_body(mj_model, start_id, body_id)
                placement = T_start_to_body * pin.SE3(np.eye(3), jnt_pos)
            else:
                R_body = quat_to_rot(mj_model.body_quat[body_id].copy())
                pos_body = mj_model.body_pos[body_id].copy()
                placement = pin.SE3(R_body, pos_body)
            joint_model = _revolute_model(mj_model.jnt_axis[jid].copy())
            pin_jid = model.addJoint(parent_pin_id, joint_model, placement, jname)

            mass = float(mj_model.body_mass[body_id])
            com = mj_model.body_ipos[body_id].copy()
            inertia = np.diag(mj_model.body_inertia[body_id].copy())
            model.appendBodyToJoint(pin_jid, pin.Inertia(mass, com, inertia), pin.SE3.Identity())
            parent_pin_id = pin_jid

        ee_site_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_SITE, ee_site)
        ee_body_id = int(mj_model.site_bodyid[ee_site_id])

        # Walk any fixed (jointless) bodies between the last joint's body and
        # the ee_site's parent body (e.g. Flexiv's *_robot_flange body).
        last_jid = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_JOINT, joint_names[-1])
        last_body_id = int(mj_model.jnt_bodyid[last_jid])
        T_fixed_chain = _se3_from_start_to_body(mj_model, last_body_id, ee_body_id)

        R_ee = quat_to_rot(mj_model.site_quat[ee_site_id].copy())
        ee_pos = mj_model.site_pos[ee_site_id].copy()
        model.addFrame(pin.Frame(
            frame_name,
            parent_pin_id,
            0,
            T_fixed_chain * pin.SE3(R_ee, ee_pos),
            pin.FrameType.OP_FRAME,
        ))

        model.lowerPositionLimit = np.asarray(q_lo, dtype=float)
        model.upperPositionLimit = np.asarray(q_hi, dtype=float)
        return cls(model, frame_name, len(joint_names))

    # ------------------------------------------------------------------ queries
    def update(self, q: np.ndarray) -> None:
        """Forward kinematics, joint Jacobians and frame placements for joint positions q."""
        m, d = self.model, self.data
        self._q_pin[: self.ndof] = q
        pin.forwardKinematics(m, d, self._q_pin)
        pin.computeJointJacobians(m, d, self._q_pin)
        pin.updateFramePlacements(m, d)

    def fk(self) -> np.ndarray:
        """EE pose in the arm base frame, 4x4 (after `update`)."""
        return self.data.oMf[self.frame_id].homogeneous.copy()

    def jacobian(self) -> np.ndarray:
        """EE Jacobian (6, ndof), LOCAL_WORLD_ALIGNED in the arm base frame (after `update`)."""
        J = pin.getFrameJacobian(
            self.model, self.data, self.frame_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
        )
        return J[:, : self.ndof]

    def manipulability(self) -> float:
        """Yoshikawa manipulability of the translational Jacobian."""
        J = self.jacobian()[:3, :]
        return float(np.sqrt(max(0.0, np.linalg.det(J @ J.T))))

    def singularity_distance(self) -> float:
        """Inverse condition number of the Jacobian (0 = singular)."""
        sv = np.linalg.svd(self.jacobian(), compute_uv=False)
        return 0.0 if sv[0] < 1e-12 else float(sv[-1] / sv[0])


def _se3_from_start_to_body(mj_model, start_id: int, body_id: int):
    """Accumulate SE3 from start_body frame to body_id frame by walking the parent chain."""
    transforms = []
    cur = body_id
    while cur != start_id:
        R = quat_to_rot(mj_model.body_quat[cur].copy())
        pos = mj_model.body_pos[cur].copy()
        transforms.append(pin.SE3(R, pos))
        cur = int(mj_model.body_parentid[cur])
    result = pin.SE3.Identity()
    for t in reversed(transforms):
        result = result * t
    return result


def _revolute_model(axis: np.ndarray):
    a = axis / (np.linalg.norm(axis) + 1e-12)
    if np.allclose(np.abs(a), [0, 0, 1], atol=0.1):
        return pin.JointModelRZ()
    if np.allclose(np.abs(a), [0, 1, 0], atol=0.1):
        return pin.JointModelRY()
    if np.allclose(np.abs(a), [1, 0, 0], atol=0.1):
        return pin.JointModelRX()
    return pin.JointModelRevoluteUnaligned(float(a[0]), float(a[1]), float(a[2]))

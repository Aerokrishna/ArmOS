"""Pinocchio sync: per-arm kinematic models synchronized with MuJoCo."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Dict

import mujoco
import numpy as np

from ..model.mj_model import _MuJoCoModel
from ..types import ArmConfig

if TYPE_CHECKING:
    from armos_core.kinematics.pinocchio_arm import PinocchioArm

logger = logging.getLogger(__name__)


def _import_pinocchio_arm():
    """`PinocchioArm`, or None when Pinocchio is not usable (imported on first use, not with armos_sim)."""
    try:
        from armos_core.kinematics.pinocchio_arm import PinocchioArm
    except Exception as exc:  # ImportError, or a numpy ABI mismatch (ROS pinocchio under numpy 2)
        logger.info("Pinocchio unavailable (%s): MuJoCo kinematics fallback.", exc)
        return None
    return PinocchioArm


class _PinocchioSync:
    """
    Maintains one Pinocchio model per arm (`armos_core.kinematics.PinocchioArm`),
    synchronized with MuJoCo on demand. Falls back to MuJoCo site Jacobians
    when Pinocchio is unavailable.

    `sync()` only reads q / qdot; forward kinematics and Jacobians run on the first
    pose/Jacobian query after it, once per arm.
    """

    def __init__(
        self, arm_configs: Dict[str, ArmConfig], mj: _MuJoCoModel
    ):
        self._mj = mj
        self._cfgs = arm_configs
        self._arms: Dict[str, "PinocchioArm"] = {}
        self._pin_models: Dict[str, object] = {}
        self._pin_data: Dict[str, object] = {}
        self._q: Dict[str, np.ndarray] = {}
        self._qdot: Dict[str, np.ndarray] = {}
        self._stale: set = set()

        PinocchioArm = _import_pinocchio_arm()
        self.available = PinocchioArm is not None
        for arm_id, cfg in arm_configs.items():
            self._q[arm_id] = np.zeros(cfg.ndof)
            self._qdot[arm_id] = np.zeros(cfg.ndof)
            if self.available:
                arm = PinocchioArm.from_mjcf(
                    mj.model, cfg.start_body, cfg.joint_names, cfg.ee_site,
                    cfg.q_lo, cfg.q_hi, frame_name=f"{arm_id}_ee", model_name=arm_id,
                )
                self._arms[arm_id] = arm
                self._pin_models[arm_id] = arm.model
                self._pin_data[arm_id] = arm.data
                logger.info(
                    "Pinocchio model '%s': nq=%d nv=%d", arm_id, arm.model.nq, arm.model.nv
                )
        self._stale.update(self._arms)

    def sync(self) -> None:
        """Read every arm's q / qdot from MuJoCo; kinematics are recomputed on the next query."""
        for arm_id, cfg in self._cfgs.items():
            self._q[arm_id] = self._mj.get_qpos(cfg.joint_names)
            self._qdot[arm_id] = self._mj.get_qvel(cfg.joint_names)
        self._stale.update(self._arms)

    def _kin(self, arm_id: str) -> "PinocchioArm":
        arm = self._arms[arm_id]
        if arm_id in self._stale:
            arm.update(self._q[arm_id])
            self._stale.discard(arm_id)
        return arm

    def get_q(self, arm_id: str) -> np.ndarray:
        return self._q[arm_id].copy()

    def get_qdot(self, arm_id: str) -> np.ndarray:
        return self._qdot[arm_id].copy()

    def get_flange_twist(self, arm_id: str) -> np.ndarray:
        """6D spatial velocity [vx, vy, vz, wx, wy, wz] at the flange.

        Uses the same Jacobian as TASK_VEL control (``get_jacobian @ qdot``).
        """
        return self.get_jacobian(arm_id) @ self._qdot[arm_id]

    def get_jacobian(self, arm_id: str) -> np.ndarray:
        if arm_id in self._arms:
            return self._kin(arm_id).jacobian()
        return self._finite_diff_jacobian(arm_id)

    def _finite_diff_jacobian(self, arm_id: str) -> np.ndarray:
        """Fallback without Pinocchio: MuJoCo site Jacobian, rotated into the arm base frame
        (same convention as the Pinocchio LOCAL_WORLD_ALIGNED Jacobian). Uses the current
        MjData kinematics."""
        cfg = self._cfgs[arm_id]
        m, d = self._mj.model, self._mj.data
        jacp = np.zeros((3, m.nv))
        jacr = np.zeros((3, m.nv))
        mujoco.mj_jacSite(m, d, jacp, jacr, self._mj.site_id(cfg.ee_site))
        dofs = [int(m.jnt_dofadr[self._mj.joint_id(j)]) for j in cfg.joint_names]
        R_bw = cfg.base_world_rot.T
        return np.vstack((R_bw @ jacp[:, dofs], R_bw @ jacr[:, dofs]))

    def get_flange_pose(self, arm_id: str) -> np.ndarray:
        """Returns a (4,4) SE3 transform of the flange in the arm's base frame."""
        if arm_id in self._arms:
            return self._kin(arm_id).fk()
        # Fallback: transform world-frame site pose into arm base frame
        cfg = self._cfgs[arm_id]
        pos_w, R_w = self._mj.get_site_pose(cfg.ee_site)
        T_world_ee = np.eye(4)
        T_world_ee[:3, :3] = R_w
        T_world_ee[:3, 3] = pos_w
        R_b = cfg.base_world_rot
        T_world_base = np.eye(4)
        T_world_base[:3, :3] = R_b
        T_world_base[:3, 3] = cfg.base_world
        T_base_world = np.eye(4)
        T_base_world[:3, :3] = R_b.T
        T_base_world[:3, 3] = -R_b.T @ cfg.base_world
        return T_base_world @ T_world_ee

    def get_manipulability(self, arm_id: str) -> float:
        J = self.get_jacobian(arm_id)[:3, :]
        return float(np.sqrt(max(0.0, np.linalg.det(J @ J.T))))

    def get_singularity_distance(self, arm_id: str) -> float:
        sv = np.linalg.svd(
            self.get_jacobian(arm_id), compute_uv=False
        )
        return 0.0 if sv[0] < 1e-12 else float(sv[-1] / sv[0])

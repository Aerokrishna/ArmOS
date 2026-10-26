"""Resolve an `ArmConfig` into `armos_core.types.ArmHandles` (ids and slices, once at load)."""

from __future__ import annotations

from typing import Optional, Sequence

import mujoco
import numpy as np

from armos_core.types import ArmHandles

from ..types import ArmConfig


def _contiguous_slice(idx: Sequence[int], what: str) -> slice:
    """Indices -> slice; raise if they are not consecutive (views into MjData need slices)."""
    idx = list(idx)
    if not idx or idx != list(range(idx[0], idx[0] + len(idx))):
        raise ValueError(f"{what} indices are not contiguous: {idx}")
    return slice(idx[0], idx[-1] + 1)


def _sensor_slice(model: mujoco.MjModel, name: Optional[str]) -> Optional[slice]:
    if not name:
        return None
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, name)
    if sid < 0:
        raise ValueError(f"Sensor '{name}' not found in MJCF")
    adr = int(model.sensor_adr[sid])
    return slice(adr, adr + int(model.sensor_dim[sid]))


def build_handles(model: mujoco.MjModel, cfg: ArmConfig, gripper_actuator: Optional[str] = None) -> ArmHandles:
    """Resolve one arm's names into ids/slices on the compiled model.

    Args:
        model: Final compiled model.
        cfg: Arm config from `_RobotLoader`.
        gripper_actuator: Name of the arm's gripper actuator, if any.
    """
    obj = mujoco.mjtObj
    jids = [mujoco.mj_name2id(model, obj.mjOBJ_JOINT, n) for n in cfg.joint_names]
    aids = [mujoco.mj_name2id(model, obj.mjOBJ_ACTUATOR, n) for n in cfg.actuator_names]
    tcp = mujoco.mj_name2id(model, obj.mjOBJ_SITE, cfg.ee_site)

    ft_site = -1
    ft_force = _sensor_slice(model, cfg.wrench_force_sensor)
    ft_torque = _sensor_slice(model, cfg.wrench_torque_sensor)
    if ft_force is not None:
        sid = mujoco.mj_name2id(model, obj.mjOBJ_SENSOR, cfg.wrench_force_sensor)
        ft_site = int(model.sensor_objid[sid])

    gid = mujoco.mj_name2id(model, obj.mjOBJ_ACTUATOR, gripper_actuator) if gripper_actuator else -1
    limits = cfg.torque_limits if cfg.torque_limits is not None else np.zeros(cfg.ndof)

    return ArmHandles(
        arm_joint_names=tuple(cfg.joint_names),
        n_arm=cfg.ndof,
        arm_qpos=_contiguous_slice([int(model.jnt_qposadr[j]) for j in jids], f"{cfg.arm_id} qpos"),
        arm_dof=_contiguous_slice([int(model.jnt_dofadr[j]) for j in jids], f"{cfg.arm_id} dof"),
        arm_ctrl=_contiguous_slice(aids, f"{cfg.arm_id} actuator"),
        torque_limits=np.asarray(limits, dtype=np.float64),
        q_home=np.asarray(cfg.q_home, dtype=np.float64).copy(),
        tcp_site_id=tcp,
        ee_body_id=int(model.site_bodyid[tcp]),
        ft_site_id=ft_site,
        ft_force=ft_force,
        ft_torque=ft_torque,
        joint_ids={mujoco.mj_id2name(model, obj.mjOBJ_JOINT, j): j for j in range(model.njnt)
                   if mujoco.mj_id2name(model, obj.mjOBJ_JOINT, j)},
        gripper_act_id=gid,
        gripper_ctrlrange=model.actuator_ctrlrange[gid].copy() if gid >= 0 else np.zeros(2),
    )

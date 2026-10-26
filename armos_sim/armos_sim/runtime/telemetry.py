"""Telemetry collector for robot state snapshots."""

from __future__ import annotations

from typing import Dict

import mujoco
import numpy as np

from ..model.mj_model import _MuJoCoModel
from .pinocchio_sync import _PinocchioSync
from ..types import ArmConfig, Telemetry

from armos_core.math.so3 import quat_to_rot


class _TelemetryCollector:
    """Collects full robot state at each control tick."""

    def __init__(
        self,
        mj: _MuJoCoModel,
        arm_configs: Dict[str, ArmConfig],
        pin_sync: _PinocchioSync,
        controllers: Dict[str, object],
        aux_config: dict | None = None,
    ):
        self._mj = mj
        self._cfgs = arm_configs
        self._sync = pin_sync
        self._ctrls = controllers
        self._dg3f: Dict[str, list] = (aux_config or {}).get("dg3f_grippers", {})

    def _read_gripper_joints(self, actuator_names: list) -> np.ndarray:
        vals = []
        for aname in actuator_names:
            # actuator name is "<joint>_motor"; strip suffix to get joint name
            jname = aname[:-6] if aname.endswith("_motor") else aname
            jid = mujoco.mj_name2id(self._mj.model, mujoco.mjtObj.mjOBJ_JOINT, jname)
            adr = int(self._mj.model.jnt_qposadr[jid])
            vals.append(float(self._mj.data.qpos[adr]))
        return np.array(vals)

    def collect(self, t: float) -> Telemetry:
        jp, jv, jt = {}, {}, {}
        w, fp, tp, ft, tt, ew = {}, {}, {}, {}, {}, {}
        manip, sing, modes = {}, {}, {}

        gravity = self._mj.model.opt.gravity  # world-frame gravity vector [m/s²]

        for arm_id, cfg in self._cfgs.items():
            jp[arm_id] = self._sync.get_q(arm_id)
            jv[arm_id] = self._sync.get_qdot(arm_id)
            jt[arm_id] = self._mj.get_qfrc_actuator(cfg.joint_names)
            fp[arm_id] = self._sync.get_flange_pose(arm_id)
            ft[arm_id] = self._sync.get_flange_twist(arm_id)
            manip[arm_id] = self._sync.get_manipulability(arm_id)
            sing[arm_id] = self._sync.get_singularity_distance(arm_id)
            modes[arm_id] = self._ctrls[arm_id].mode.name

            tool = cfg.tool
            R_flange = fp[arm_id][:3, :3]

            # TCP pose: flange_pose @ T_tcp_in_flange
            T_tcp_in_flange = np.eye(4)
            T_tcp_in_flange[:3, :3] = quat_to_rot(tool.tcp_location[3:])
            T_tcp_in_flange[:3, 3] = tool.tcp_location[:3]
            tp[arm_id] = fp[arm_id] @ T_tcp_in_flange

            # TCP twist (LOCAL_WORLD_ALIGNED frame): rigid-body velocity shift
            omega = ft[arm_id][3:]
            p_tcp_world = R_flange @ tool.tcp_location[:3]
            tt[arm_id] = np.concatenate(
                [ft[arm_id][:3] + np.cross(omega, p_tcp_world), omega]
            )

            # Raw wrench and static gravity compensation
            if cfg.wrench_force_sensor and cfg.wrench_torque_sensor:
                f_raw, t_raw = self._mj.read_wrench(
                    cfg.wrench_force_sensor,
                    cfg.wrench_torque_sensor,
                )
                w[arm_id] = np.concatenate([f_raw, t_raw])
                # Tool gravity contribution in flange frame
                g_fl = R_flange.T @ gravity
                F_tool = tool.mass * g_fl
                tau_tool = np.cross(tool.com, F_tool)
                ew[arm_id] = w[arm_id] - np.concatenate([F_tool, tau_tool])
            else:
                w[arm_id] = np.zeros(6)
                ew[arm_id] = np.zeros(6)

        gj = {
            arm_id: self._read_gripper_joints(act_names)
            for arm_id, act_names in self._dg3f.items()
        }

        return Telemetry(
            timestamp=t,
            joint_pos=jp,
            joint_vel=jv,
            joint_torque=jt,
            wrench=w,
            flange_pose=fp,
            tcp_pose=tp,
            manipulability=manip,
            singularity_distance=sing,
            control_mode=modes,
            flange_twist=ft,
            tcp_twist=tt,
            ext_wrench=ew,
            gripper_joints=gj,
        )

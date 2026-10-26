"""MuJoCo model wrapper."""

from __future__ import annotations

from typing import List, Tuple

import mujoco
import numpy as np

from ..constants import _DEFAULT_DECIMATION


class _MuJoCoModel:
    """Thin wrapper around MjModel / MjData with convenience accessors."""

    def __init__(self, xml_path, preloaded_model=None, n_substeps: int = _DEFAULT_DECIMATION):
        self.model = preloaded_model or mujoco.MjModel.from_xml_path(str(xml_path))
        self.data = mujoco.MjData(self.model)
        self.n_substeps = int(n_substeps)
        self._ids: dict = {}  # (objtype, name) -> id, filled on first lookup

    def _id(self, objtype, name: str) -> int:
        key = (objtype, name)
        i = self._ids.get(key)
        if i is None:
            i = self._ids[key] = mujoco.mj_name2id(self.model, objtype, name)
        return i

    def joint_id(self, name: str) -> int:
        return self._id(mujoco.mjtObj.mjOBJ_JOINT, name)

    def actuator_id(self, name: str) -> int:
        return self._id(mujoco.mjtObj.mjOBJ_ACTUATOR, name)

    def site_id(self, name: str) -> int:
        return self._id(mujoco.mjtObj.mjOBJ_SITE, name)

    def sensor_id(self, name: str) -> int:
        return self._id(mujoco.mjtObj.mjOBJ_SENSOR, name)

    def get_qpos(self, joint_names: List[str]) -> np.ndarray:
        return np.array([
            self.data.qpos[self.model.jnt_qposadr[self.joint_id(n)]]
            for n in joint_names
        ])

    def get_qvel(self, joint_names: List[str]) -> np.ndarray:
        return np.array([
            self.data.qvel[self.model.jnt_dofadr[self.joint_id(n)]]
            for n in joint_names
        ])

    def get_qfrc_bias(self, joint_names: List[str]) -> np.ndarray:
        return np.array([
            self.data.qfrc_bias[self.model.jnt_dofadr[self.joint_id(n)]]
            for n in joint_names
        ])

    def get_qfrc_actuator(self, joint_names: List[str]) -> np.ndarray:
        return np.array([
            self.data.qfrc_actuator[self.model.jnt_dofadr[self.joint_id(n)]]
            for n in joint_names
        ])

    def get_site_pose(self, site_name: str) -> Tuple[np.ndarray, np.ndarray]:
        sid = self.site_id(site_name)
        return (
            self.data.site_xpos[sid].copy(),
            self.data.site_xmat[sid].reshape(3, 3).copy(),
        )

    def read_wrench(
        self, force_sensor: str, torque_sensor: str
    ) -> Tuple[np.ndarray, np.ndarray]:
        f_id = self.sensor_id(force_sensor)
        t_id = self.sensor_id(torque_sensor)
        f_adr = self.model.sensor_adr[f_id]
        t_adr = self.model.sensor_adr[t_id]
        return (
            self.data.sensordata[f_adr : f_adr + 3].copy(),
            self.data.sensordata[t_adr : t_adr + 3].copy(),
        )

    def set_ctrl_pos(
        self, actuator_names: List[str], q_ref: np.ndarray
    ) -> None:
        for name, q in zip(actuator_names, q_ref):
            self.data.ctrl[self.actuator_id(name)] = q

    def set_actuator(self, actuator_name: str, value: float) -> None:
        """Set a single actuator's control value (position or motor)."""
        aid = self.actuator_id(actuator_name)
        self.data.ctrl[aid] = float(value)

    def add_ctrl_torque(
        self, joint_names: List[str], tau: np.ndarray
    ) -> None:
        for name, t in zip(joint_names, tau):
            dof = self.model.jnt_dofadr[self.joint_id(name)]
            self.data.qfrc_applied[dof] += t

    def clear_applied(self) -> None:
        self.data.qfrc_applied[:] = 0.0

    def physics_step(self) -> None:
        for _ in range(self.n_substeps):
            mujoco.mj_step(self.model, self.data)

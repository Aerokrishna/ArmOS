"""ArmPlant: torque-level interface to one arm, the fast core used by torque modes and RL.

Source: `mujoco_rl_bed.sim.plant.FrankaPlant` (actuation, `control_step`, `reset`). The state and
kinematics accessors come from `armos_core.kinematics.MujocoArm`, which this class extends, so the
per-tick path performs no numpy allocation and no name lookups.

Several arms can share one MjModel/MjData: build one `ArmPlant` per arm with the same `data`
(`make_plants`) and step the physics once per tick (see `armos_sim.runtime.simulator`). For a single
arm, `control_step` does the whole tick.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, Optional

import mujoco
import numpy as np

from armos_core.kinematics.mujoco_kin import MujocoArm
from armos_core.types import ArmHandles

from .model.handles import build_handles
from .types import ArmConfig


class ArmPlant(MujocoArm):
    """Torque API and state accessors for one arm (float64 throughout).

    Attributes:
        model, data, h, n: See `MujocoArm`.
    """

    def __init__(self, model: mujoco.MjModel, handles: ArmHandles, data: mujoco.MjData | None = None) -> None:
        super().__init__(model, handles, data)
        h = handles
        self._ctrl_arm = self.data.ctrl[h.arm_ctrl]
        self._gid = h.gripper_act_id
        self._glo, self._ghi = (float(h.gripper_ctrlrange[0]), float(h.gripper_ctrlrange[1]))

    # ------------------------------------------------------------------ construction
    @classmethod
    def from_config(
        cls,
        config_path: str | Path,
        arm_id: str = "left",
        actuation: str = "torque",
        timestep: Optional[float] = None,
    ) -> "ArmPlant":
        """Load a YAML config (see `armos_sim.model.loader`) and return the plant of one arm.

        The plant owns a fresh MjData, reset to the arm's `q_home`.
        """
        from .model.loader import _RobotLoader

        _, arm_configs, aux = _RobotLoader.from_yaml(Path(config_path), actuation=actuation, timestep=timestep)
        model = aux["_preloaded_model"]
        cfg = arm_configs[arm_id]
        plant = cls(model, build_handles(model, cfg, aux.get("grippers", {}).get(arm_id)))
        plant.reset()
        return plant

    # ------------------------------------------------------------------ state
    @property
    def ctrl_arm(self) -> np.ndarray:
        """Arm actuator commands currently applied, shape (n,), view into `data.ctrl`."""
        return self._ctrl_arm

    # ------------------------------------------------------------------ actuation
    def set_joint_torque(self, tau: np.ndarray) -> None:
        """Write arm motor torques [Nm] (clamped by the actuator `ctrlrange` inside MuJoCo)."""
        np.copyto(self._ctrl_arm, tau)

    def set_gripper(self, opening: float) -> None:
        """Gripper opening in [0, 1] (0 = closed, 1 = open), mapped onto its actuator's ctrlrange."""
        if self._gid >= 0:
            o = min(max(opening, 0.0), 1.0)
            self.data.ctrl[self._gid] = self._glo + o * (self._ghi - self._glo)

    def control_step(self, torque_fn: Callable[["ArmPlant"], np.ndarray]) -> None:
        """Advance one physics tick with a torque computed on the current state.

        Sequence: `mj_step1` (kinematics, Jacobian inputs, `qfrc_bias`) -> `ctrl = torque_fn(self)`
        -> `mj_step2`, so the controller sees the state at the beginning of the tick.
        """
        m, d = self.model, self.data
        mujoco.mj_step1(m, d)
        np.copyto(self._ctrl_arm, torque_fn(self))
        mujoco.mj_step2(m, d)

    def step_physics(self) -> None:
        """Advance one physics tick with the currently stored `ctrl` (no controller)."""
        mujoco.mj_step(self.model, self.data)

    # ------------------------------------------------------------------ reset
    def reset(self, q0: np.ndarray | None = None, qd0: np.ndarray | None = None,
              gripper_open: float | None = 1.0) -> None:
        """`mj_resetData`, then set this arm's state and `mj_forward`.

        Resets the whole MjData (all arms); use it for single-arm plants.

        Args:
            q0: Arm joint positions [rad]; None = `h.q_home`.
            qd0: Arm joint velocities [rad/s]; None = zeros.
            gripper_open: Gripper opening in [0, 1]; None = leave ctrl at 0.
        """
        m, d = self.model, self.data
        mujoco.mj_resetData(m, d)
        self._q[:] = self.h.q_home if q0 is None else q0
        if qd0 is not None:
            self._qd[:] = qd0
        if gripper_open is not None:
            self.set_gripper(gripper_open)
        mujoco.mj_forward(m, d)


def make_plants(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    arm_configs: Dict[str, ArmConfig],
    grippers: Optional[Dict[str, str]] = None,
) -> Dict[str, ArmPlant]:
    """One `ArmPlant` per torque-actuated arm, all sharing `data`."""
    grippers = grippers or {}
    return {
        arm_id: ArmPlant(model, build_handles(model, cfg, grippers.get(arm_id)), data)
        for arm_id, cfg in arm_configs.items()
        if cfg.actuation == "torque"
    }

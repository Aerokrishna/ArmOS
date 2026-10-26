"""MujocoArm: allocation-free arm state and kinematics over one MjModel/MjData pair.

Source: the state/kinematics part of `mujoco_rl_bed.sim.plant.FrankaPlant` (views, Jacobian,
EE pose/velocity, wrist wrench). Stepping and actuation live in `armos_sim.plant.ArmPlant`, which
subclasses this.

View semantics: `q`, `qd`, `ee_pos`, `ee_rot` and `qfrc_bias` are *views* into `MjData` memory.
They always reflect the current state; call `.copy()` for a snapshot. `jacobian()`, `ee_quat()`,
`ee_vel()` and `wrist_wrench()` write into owned buffers and return them (overwritten on the next
call). All quantities are in the world frame.
"""

from __future__ import annotations

import mujoco
import numpy as np

from armos_core.types import ArmHandles


class MujocoArm:
    """State accessors for one arm (float64 throughout). Implements `armos_core.types.PlantView`.

    Attributes:
        model: The `MjModel`.
        data: The `MjData`.
        h: Resolved handles (`ArmHandles` or anything with the same fields).
        n: Number of arm joints.
    """

    def __init__(self, model: mujoco.MjModel, handles: ArmHandles, data: mujoco.MjData | None = None) -> None:
        """Cache views into `data` and allocate buffers.

        Args:
            model: Compiled model.
            handles: Arm handles.
            data: Existing `MjData` for `model`; created if None.
        """
        self.model = model
        self.data = data if data is not None else mujoco.MjData(model)
        self.h = handles
        self.n = handles.n_arm
        m, d, h = self.model, self.data, handles

        # --- cached views into MjData (basic slicing -> views; MjData memory is fixed for its lifetime)
        self._q = d.qpos[h.arm_qpos]
        self._qd = d.qvel[h.arm_dof]
        self._bias = d.qfrc_bias[h.arm_dof]
        self._site_xpos = d.site_xpos[h.tcp_site_id]           # (3,)
        self._site_xmat = d.site_xmat[h.tcp_site_id]           # (9,) row-major rotation
        self._has_ft = h.ft_site_id >= 0 and h.ft_force is not None and h.ft_torque is not None
        if self._has_ft:
            self._ft_site_xmat = d.site_xmat[h.ft_site_id]     # (9,)
            self._ft_force = d.sensordata[h.ft_force]          # (3,) in ft_site frame
            self._ft_torque = d.sensordata[h.ft_torque]        # (3,) in ft_site frame
            self._ft_rot = self._ft_site_xmat.reshape(3, 3)    # view, no copy (contiguous row)
        self._tlim = h.torque_limits

        # --- preallocated buffers
        self._jacp = np.zeros((3, m.nv))
        self._jacr = np.zeros((3, m.nv))
        self._jacp_arm = self._jacp[:, h.arm_dof]               # views on the arm columns
        self._jacr_arm = self._jacr[:, h.arm_dof]
        self._J = np.zeros((6, self.n))
        self._J_lin = self._J[:3]
        self._J_ang = self._J[3:]
        self._quat = np.zeros(4)
        self._ee_vel = np.zeros(6)
        self._wrench = np.zeros(6)
        self._wrench_f = self._wrench[:3]
        self._wrench_t = self._wrench[3:]

    # ------------------------------------------------------------------ state (views)
    @property
    def q(self) -> np.ndarray:
        """Arm joint positions [rad], shape (n,), view into `data.qpos`."""
        return self._q

    @property
    def qd(self) -> np.ndarray:
        """Arm joint velocities [rad/s], shape (n,), view into `data.qvel`."""
        return self._qd

    @property
    def ee_pos(self) -> np.ndarray:
        """TCP position in world frame [m], shape (3,), view into `data.site_xpos`."""
        return self._site_xpos

    @property
    def ee_rot(self) -> np.ndarray:
        """TCP rotation matrix (row-major, flattened), shape (9,), view into `data.site_xmat`."""
        return self._site_xmat

    @property
    def qfrc_bias(self) -> np.ndarray:
        """Arm slice of gravity + Coriolis/centrifugal forces [Nm], shape (n,), view."""
        return self._bias

    @property
    def torque_limits(self) -> np.ndarray:
        """Per-joint torque limits [Nm], shape (n,)."""
        return self._tlim

    @property
    def time(self) -> float:
        """Simulation time [s]."""
        return self.data.time

    @property
    def dt(self) -> float:
        """Physics timestep [s]."""
        return self.model.opt.timestep

    @property
    def has_wrench_sensor(self) -> bool:
        """True if the handles name an F/T site and sensors."""
        return self._has_ft

    # ------------------------------------------------------------------ derived quantities (buffers)
    def ee_quat(self) -> np.ndarray:
        """TCP orientation as a unit quaternion (w, x, y, z); owned buffer."""
        mujoco.mju_mat2Quat(self._quat, self._site_xmat)
        return self._quat

    def jacobian(self) -> np.ndarray:
        """Geometric TCP Jacobian on the arm joints, world frame; owned buffer (6, n).

        Rows 0-2 map qd to linear velocity [m/s], rows 3-5 to angular velocity [rad/s].
        Valid after `mj_step1`/`mj_forward` (uses current `site_xpos` and `subtree_com`).
        """
        mujoco.mj_jacSite(self.model, self.data, self._jacp, self._jacr, self.h.tcp_site_id)
        np.copyto(self._J_lin, self._jacp_arm)
        np.copyto(self._J_ang, self._jacr_arm)
        return self._J

    def ee_vel(self, J: np.ndarray | None = None) -> np.ndarray:
        """TCP twist (v, w) in world frame [m/s, rad/s] = J qd; owned buffer (6,).

        Args:
            J: Precomputed Jacobian (6, n); None = call `jacobian()`.
        """
        np.dot(self.jacobian() if J is None else J, self._qd, out=self._ee_vel)
        return self._ee_vel

    def wrist_wrench(self, world_frame: bool = False) -> np.ndarray:
        """F/T sensor reading (force [N], torque [Nm]); zeros without a sensor; owned buffer (6,).

        MuJoCo's force/torque sensors report the wrench exerted on the sensor body's subtree by its
        parent, in the `ft_site` frame, as of the last `mj_step2`/`mj_forward`.

        Args:
            world_frame: If True, rotate both vectors into the world frame.
        """
        if not self._has_ft:
            self._wrench.fill(0.0)
        elif world_frame:
            np.dot(self._ft_rot, self._ft_force, out=self._wrench_f)   # R_site->world @ f_site
            np.dot(self._ft_rot, self._ft_torque, out=self._wrench_t)
        else:
            np.copyto(self._wrench_f, self._ft_force)
            np.copyto(self._wrench_t, self._ft_torque)
        return self._wrench

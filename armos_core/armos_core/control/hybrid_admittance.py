"""Hybrid Cartesian admittance: per-axis pose/force selection, outputs a virtual pose.

One law for the sim and the real FR3. Sources: franka_custom_control/bimanual/admittance_controller.py
(`admittance_control`: TCP axes, rotation-matrix state, force deadband, max deviation) and
armos_control/cartesian_admittance.py (base axes). Both were the same law up to those options.

Poses are a position (3,) and a rotation matrix (3, 3) in the base frame. The virtual velocity is
stored in base coordinates. Wrenches are [f, tau] about the TCP, robot-on-environment (pressing on a
surface with 5 N gives a force of 5 N pointing into it).

Axes (`cfg.axes`): the frame of the mask, the spring and damper, the target and measured wrench:
    "base": base-frame axes ([x, y, z, rx, ry, rz] of the base).
    "tcp":  TCP axes, from the measured TCP orientation passed to `step`; e.g. mask [1, 1, 0, 1, 1, 1]
            is compliant along the tool z axis wherever it points.

Per axis (mask: 1 = pose axis, 0 = force axis), with e = virtual ⊖ target in the chosen axes:

    M dv/dt = force_mode * deadband(F_target - F_measured) - K_pose * e - D v

Pose axes get the virtual spring K and damping D; force axes get no spring and their own damping
D_force (free-space approach speed F_target / D_force). Damping is integrated implicitly (stable for
heavy force-axis damping at the control rate). Acceleration limits apply to pose axes only, so a
force axis can brake immediately on contact. Optionally the virtual pose may lead the measured pose by
at most `max_deviation` (anti-windup).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from armos_core.math.limits import deadband, limit_norm
from armos_core.math.so3 import quat_normalize, quat_to_rot, rot_to_quat, rot_to_rotvec, rotvec_to_rot

AXES = ("base", "tcp")


def _vec6(values) -> np.ndarray:
    return np.asarray(values, dtype=float).reshape(6).copy()


@dataclass
class HybridAdmittanceCfg:
    """Virtual dynamics and limits (defaults: the armos sim admittance; the FR3 values live in
    armos_control/config/admittance.yaml).

    Attributes:
        axes: "base" or "tcp" (see the module docstring).
        mass: Virtual mass (6,) [kg, kg m^2].
        stiffness: Virtual stiffness on pose axes (6,) [N/m, Nm/rad].
        damping: Virtual damping on pose axes (6,) [Ns/m, Nms/rad].
        damping_force: Damping on force axes (6,). Free-space approach speed is F_target / D_force.
        max_lin_vel, max_ang_vel: Virtual velocity caps [m/s, rad/s].
        max_lin_acc, max_ang_acc: Acceleration caps on pose axes [m/s^2, rad/s^2].
        max_force_lin_vel, max_force_ang_vel: Velocity caps on force axes; bound the touch-down force
            spike.
        max_force, max_torque: Norm limits on the target wrench [N, Nm].
        force_deadband: Deadband on the wrench error (force [N], torque [Nm]); 0 disables it. A real
            wrench estimate drifts, and a force axis with a 0 N target would creep at offset / D_force.
        max_deviation: Max lead of the virtual pose over the measured pose (position [m], rotation
            [rad]); None disables it. Needs the measured pose in `step`.
    """

    axes: str = "base"
    mass: np.ndarray = field(default_factory=lambda: np.array([2.5, 2.5, 2.5, 0.25, 0.25, 0.25]))
    stiffness: np.ndarray = field(default_factory=lambda: np.array([100.0, 100.0, 100.0, 10.0, 10.0, 10.0]))
    damping: np.ndarray = field(default_factory=lambda: np.array([30.0, 30.0, 30.0, 3.0, 3.0, 3.0]))
    damping_force: np.ndarray = field(
        default_factory=lambda: np.array([600.0, 600.0, 600.0, 20.0, 20.0, 20.0]))
    max_lin_vel: float = 0.05
    max_ang_vel: float = 0.25
    max_lin_acc: float = 0.25
    max_ang_acc: float = 1.0
    max_force_lin_vel: float = 0.01
    max_force_ang_vel: float = 0.1
    max_force: float = 20.0
    max_torque: float = 3.0
    force_deadband: tuple[float, float] = (0.0, 0.0)
    max_deviation: tuple[float, float] | None = None


class HybridAdmittance:
    """Integrates the virtual pose; the caller sends it to a pose-tracking inner loop.

    Usage:
        law.reset(pos, R)                       # once, from the first valid measured pose
        law.set_target_pose(...); law.set_target_wrench(...); law.set_mask(...)
        pos_v, R_v = law.step(measured_wrench, dt, pos_meas, R_meas)
    """

    def __init__(self, cfg: HybridAdmittanceCfg | None = None) -> None:
        self.cfg = cfg or HybridAdmittanceCfg()
        c = self.cfg
        if c.axes not in AXES:
            raise ValueError(f"axes must be one of {AXES}, got {c.axes!r}")
        self.axes = c.axes
        self.mass = _vec6(c.mass)
        self.K = _vec6(c.stiffness)
        self.D = _vec6(c.damping)
        self.D_force = _vec6(c.damping_force)
        self.force_deadband = tuple(float(v) for v in c.force_deadband)
        self.max_deviation = None if c.max_deviation is None else tuple(float(v) for v in c.max_deviation)

        self.target_pos: np.ndarray | None = None
        self.target_R: np.ndarray | None = None
        self.target_wrench = np.zeros(6)       # in the law's axes
        self.pose_mask = np.ones(6)            # 1 = pose axis, 0 = force axis
        self.virtual_pos: np.ndarray | None = None
        self.virtual_R: np.ndarray | None = None
        self.virtual_vel = np.zeros(6)         # base coordinates
        self.reverted = False                  # True if the last step hit the non-finite guard

    # ------------------------------------------------------------------ setup / targets
    @property
    def ready(self) -> bool:
        """True once `reset` has set a virtual pose and a target exists."""
        return self.virtual_pos is not None and self.target_pos is not None

    def reset(self, pos: np.ndarray, R: np.ndarray) -> None:
        """Start the virtual state at the pose with zero velocity; the target defaults to it."""
        self.virtual_pos = np.asarray(pos, dtype=float).copy()
        self.virtual_R = np.asarray(R, dtype=float).copy()
        self.virtual_vel = np.zeros(6)
        if self.target_pos is None:
            self.set_target_pose(pos, R)

    def clear(self, keep_target: bool = False) -> None:
        """Forget the virtual state (and the target unless keep_target); `reset` starts it again."""
        self.virtual_pos = self.virtual_R = None
        self.virtual_vel = np.zeros(6)
        if not keep_target:
            self.target_pos = self.target_R = None

    def set_target_pose(self, pos: np.ndarray, R: np.ndarray) -> None:
        """Target pose (base frame). The virtual state is not teleported; pose axes move toward it."""
        self.target_pos = np.asarray(pos, dtype=float).copy()
        self.target_R = np.asarray(R, dtype=float).copy()

    def set_target_wrench(self, wrench: np.ndarray) -> np.ndarray:
        """Target wrench [f, tau] in the law's axes (robot-on-environment), norm-limited. Returns it."""
        w = np.asarray(wrench, dtype=float).copy()
        w[:3] = limit_norm(w[:3], self.cfg.max_force)
        w[3:] = limit_norm(w[3:], self.cfg.max_torque)
        self.target_wrench = w
        return w

    def set_mask(self, mask: np.ndarray) -> None:
        """Per-axis selection [x, y, z, rx, ry, rz] in the law's axes: 1 = pose, 0 = force. Keeps the
        virtual state."""
        self.pose_mask = np.asarray(mask, dtype=float).reshape(6).copy()

    def hold(self) -> None:
        """Stop the virtual motion (e.g. on stale measurements); the virtual pose stays."""
        self.virtual_vel[:] = 0.0

    # ------------------------------------------------------------------ law
    def step(
        self,
        wrench: np.ndarray,
        dt: float,
        pos_meas: np.ndarray | None = None,
        R_meas: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Advance the virtual dynamics by dt with the measured wrench; returns the virtual pose.

        Args:
            wrench: Measured wrench [f, tau] in the law's axes, about the TCP, robot-on-environment.
            dt: Time step [s].
            pos_meas, R_meas: Measured TCP pose (base frame). R_meas is required for axes "tcp";
                both are required for `max_deviation`.

        Returns:
            (virtual_pos, virtual_R). On a non-finite result the virtual state keeps its last pose
            with zero velocity (`reverted` is set).
        """
        c = self.cfg
        pose_mode = self.pose_mask
        force_mode = 1.0 - pose_mode

        if self.axes == "tcp":
            if R_meas is None:
                raise ValueError("axes 'tcp' needs the measured TCP orientation (R_meas)")
            R_a = np.asarray(R_meas, dtype=float)          # axes -> base
        else:
            R_a = None

        pos_err = self.virtual_pos - self.target_pos
        rot_err = rot_to_rotvec(self.virtual_R @ self.target_R.T)
        if R_a is not None:
            R_T = R_a.T
            pose_error = np.concatenate((R_T @ pos_err, R_T @ rot_err))
            vel = np.concatenate((R_T @ self.virtual_vel[:3], R_T @ self.virtual_vel[3:]))
        else:
            pose_error = np.concatenate((pos_err, rot_err))
            vel = self.virtual_vel.copy()

        K_active = pose_mode * self.K
        D_active = pose_mode * self.D + force_mode * self.D_force

        wrench_error = self.target_wrench - np.asarray(wrench, dtype=float)
        wrench_error = np.concatenate((deadband(wrench_error[:3], self.force_deadband[0]),
                                       deadband(wrench_error[3:], self.force_deadband[1])))
        generalized_input = force_mode * wrench_error - K_active * pose_error

        # Semi-implicit velocity update: M dv/dt = input - D v.
        new_vel = (self.mass * vel + dt * generalized_input) / (self.mass + dt * D_active)
        dv = new_vel - vel

        # Acceleration limits on pose axes only (translation / rotation separately).
        for sl, max_acc in ((slice(0, 3), c.max_lin_acc), (slice(3, 6), c.max_ang_acc)):
            pose_part = pose_mode[sl] * dv[sl]
            dv[sl] = force_mode[sl] * dv[sl] + limit_norm(pose_part, max_acc * dt)

        vel = vel + dv
        vel[:3] = limit_norm(vel[:3], c.max_lin_vel)
        vel[3:] = limit_norm(vel[3:], c.max_ang_vel)
        force_cap = np.array([c.max_force_lin_vel] * 3 + [c.max_force_ang_vel] * 3)
        vel = np.where(force_mode > 0.5, np.clip(vel, -force_cap, force_cap), vel)
        if R_a is not None:
            self.virtual_vel = np.concatenate((R_a @ vel[:3], R_a @ vel[3:]))
        else:
            self.virtual_vel = vel

        # Integrate in base coordinates; re-orthonormalize the rotation.
        new_pos = self.virtual_pos + self.virtual_vel[:3] * dt
        R_next = rotvec_to_rot(self.virtual_vel[3:] * dt) @ self.virtual_R
        R_next = quat_to_rot(quat_normalize(rot_to_quat(R_next)))

        self.reverted = not (np.all(np.isfinite(new_pos)) and np.all(np.isfinite(R_next)))
        if self.reverted:
            self.hold()
            return self.virtual_pos, self.virtual_R
        self.virtual_pos, self.virtual_R = new_pos, R_next
        if self.max_deviation is not None:
            if pos_meas is None or R_meas is None:
                raise ValueError("max_deviation needs the measured TCP pose (pos_meas, R_meas)")
            self._limit_deviation(np.asarray(pos_meas, dtype=float), np.asarray(R_meas, dtype=float))
        return self.virtual_pos, self.virtual_R

    def _limit_deviation(self, pos_meas: np.ndarray, R_meas: np.ndarray) -> None:
        """Keep the virtual pose within max_deviation of the measured pose (anti-windup)."""
        max_pos, max_rot = self.max_deviation
        d = self.virtual_pos - pos_meas
        n = float(np.linalg.norm(d))
        if n > max_pos:
            u = d / n
            self.virtual_pos = pos_meas + u * max_pos
            if self.virtual_vel[:3] @ u > 0.0:   # drop the velocity pushing further away
                self.virtual_vel[:3] -= (self.virtual_vel[:3] @ u) * u
        r = rot_to_rotvec(self.virtual_R @ R_meas.T)
        a = float(np.linalg.norm(r))
        if a > max_rot:
            u = r / a
            self.virtual_R = rotvec_to_rot(u * max_rot) @ R_meas
            if self.virtual_vel[3:] @ u > 0.0:
                self.virtual_vel[3:] -= (self.virtual_vel[3:] @ u) * u

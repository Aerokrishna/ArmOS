"""Bimanual target generation: moves the arms' Cartesian targets from their start poses to their goals.

Each arm i follows its path by a phase phi_i in [0, 1]:

    p_i = p_i0 + phi_i * dp_i                         (dp_i = p_i,goal - p_i0)
    R_i = exp(phi_i * dtheta_i) R_i0                  (dtheta_i = log(R_i,goal R_i0^T), world frame)

and the phases advance with first-order coupling dynamics (explicit Euler, as specified):

    phi_1^{k+1} = phi_1^k + dt [w_1 + k (phi_2^k - phi_1^k)]
    phi_2^{k+1} = phi_2^k + dt [w_2 + k (phi_1^k - phi_2^k)]

w_i is the arm's phase rate. It ramps up at the
acceleration limit from rest, cruises at the velocity limit and brakes to land on phi = 1 at rest
(w_i <= sqrt(2 a_i (1 - phi_i)), in its discrete form). This gives a trapezoidal profile in phase. Velocity and acceleration
limits in Cartesian space become phase limits per arm: w_max = min(v_max / |dp|, w_max / |dtheta|).

Modes:
    "independent": k = 0, each arm with its own phase limits. Every arm moves as fast as its limits
        allow, and the arms arrive at different times.
    "temporal": both arms share the phase limits of the slower one (the smallest w_max and a_max), so
        their nominal phases are equal and they arrive together. The coupling k pulls the phases
        together whenever they differ (per-arm rates `omega` passed to `set_targets`, or a clamped
        arm).
    "spatial": object-level compliance for an object held rigidly by both arms (two arms only). The
        hybrid admittance law (`HybridAdmittance`, the same law as the per-arm admittance node) runs on
        the OBJECT frame:
            w_obj = G(r_1, r_2) w_ee - w_bias            grasp matrix, r_i = p_i - p_obj (world)
            M dv/dt = mask_force * (w_target - w_obj) - K_pose e - D v     (see HybridAdmittance)
            T_i,cmd = T_obj,virtual T_obj->ee_i                             (rigid grasp)
        `engage_object` captures the end-effector-in-object transforms T_obj->ee_i from the poses on
        entry; the object pose is measured from both arms' poses through them. Not included yet: the
        relative-pose correction and the internal (squeeze) force term (null space of G).
        Wrenches are the ones each end effector applies ON the object (robot-on-environment, as the
        admittance node's), [f, tau] about its TCP, in the world frame. As in HybridAdmittance, pose
        axes ignore the wrench: an axis is compliant only in force mode (mask 0).
        The object's weight shows up in w_obj once both arms carry it (f_z = +m g, plus r_com x m g
        about the object origin); `tare_object` removes it (see its docstring). The squeeze needs no
        tare: G maps internal wrenches to zero.
"""

from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
from armos_core.control.hybrid_admittance import HybridAdmittance, HybridAdmittanceCfg
from armos_core.math.grasp import grasp_matrix
from armos_core.math.se3 import inv_T, make_T
from armos_core.math.so3 import rot_to_rotvec, rotvec_to_rot

MODES = ("independent", "temporal", "spatial")


@dataclass
class BimanualCfg:
    """Target-generation limits and coupling.

    Attributes:
        mode: "independent", "temporal" or "spatial" (see the module docstring).
        coupling: Phase coupling gain k [1/s] (temporal mode). Stability needs n_arms * k * dt < 1.
        max_lin_vel, max_ang_vel: Target velocity limits along the path [m/s, rad/s].
        max_lin_acc, max_ang_acc: Target acceleration limits along the path [m/s^2, rad/s^2].
        object_admittance: Object-frame admittance (spatial mode). axes "base" = world axes, "tcp" =
            object axes.
        object_mask: Initial object axis selection [x, y, z, rx, ry, rz] (1 = pose, 0 = force), applied
            on `engage_object`. All pose by default: the object holds still until axes are switched to
            force mode.
    """

    mode: str = "independent"
    coupling: float = 5.0
    max_lin_vel: float = 0.1
    max_ang_vel: float = 0.5
    max_lin_acc: float = 0.25
    max_ang_acc: float = 1.0
    object_admittance: HybridAdmittanceCfg = field(default_factory=HybridAdmittanceCfg)
    object_mask: tuple = (1, 1, 1, 1, 1, 1)

    def __post_init__(self):
        if self.mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got '{self.mode}'")
        if self.coupling < 0.0:
            raise ValueError("coupling must be >= 0")
        if min(self.max_lin_vel, self.max_ang_vel, self.max_lin_acc, self.max_ang_acc) <= 0.0:
            raise ValueError("velocity and acceleration limits must be positive")


def _phase_limit(lin_limit: float, ang_limit: float, dist: float, angle: float) -> float:
    """Cartesian limit -> phase limit for a path of length `dist` [m] and rotation `angle` [rad]."""
    lim = np.inf
    if dist > 0.0:
        lim = min(lim, lin_limit / dist)
    if angle > 0.0:
        lim = min(lim, ang_limit / angle)
    return lim


class BimanualSync:
    """Phase-based target generator for n arms (n = 2 for the bimanual case).

    Usage per control tick:
        sync.reset([(p_L, R_L), (p_R, R_R)])          # once: hold the measured poses
        sync.set_targets([(g_L, Rg_L), (g_R, Rg_R)])  # new goals (from the commanded poses)
        poses = sync.step(dt)                          # [(p, R), ...] -> each arm's target_pose
    """

    def __init__(self, cfg: BimanualCfg | None = None, n_arms: int = 2):
        if n_arms < 1:
            raise ValueError("n_arms must be >= 1")
        self.cfg = cfg or BimanualCfg()
        self.n = n_arms
        self._p0 = np.zeros((n_arms, 3))
        self._R0 = np.tile(np.eye(3), (n_arms, 1, 1))
        self._dp = np.zeros((n_arms, 3))
        self._dtheta = np.zeros((n_arms, 3))
        self.phase = np.ones(n_arms)        # 1 = at the goal (holding)
        self.rate = np.zeros(n_arms)        # w_i, the arm's own phase rate (without coupling)
        self._w_max = np.zeros(n_arms)
        self._a_max = np.zeros(n_arms)
        self._moving = np.zeros(n_arms, dtype=bool)
        self._initialized = False
        # spatial mode
        self.object_law = HybridAdmittance(self.cfg.object_admittance)
        self._T_obj_ee: list[np.ndarray] | None = None   # end effector in object frame (rigid grasp)
        self._w_bias = np.zeros(6)                         # object wrench bias, world axes
        self._cmd: list[tuple[np.ndarray, np.ndarray]] | None = None
        self.object_wrench = np.zeros(6)                   # last tared object wrench, world axes

    # ------------------------------------------------------------------ state
    @property
    def mode(self) -> str:
        return self.cfg.mode

    @property
    def object_engaged(self) -> bool:
        return self._T_obj_ee is not None

    def set_mode(self, mode: str) -> None:
        """Switch mode. Between independent and temporal it takes effect from the next `set_targets`
        (a motion in progress keeps its limits but uses the new coupling rule). Into spatial: call
        `engage_object` before the next `step`. Out of spatial: the arms hold their last commanded
        poses (bumpless) and the object is released (`disengage_object`)."""
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got '{mode}'")
        if self.cfg.mode == "spatial" and mode != "spatial" and self.object_engaged:
            self.disengage_object()
        self.cfg.mode = mode

    @property
    def done(self) -> bool:
        return bool(np.all(self.phase >= 1.0))

    def reset(self, poses) -> None:
        """Hold the given poses (e.g. the measured ones): no motion until `set_targets`."""
        poses = self._check_poses(poses)
        for i, (p, R) in enumerate(poses):
            self._p0[i] = p
            self._R0[i] = R
        self._dp[:] = 0.0
        self._dtheta[:] = 0.0
        self.phase[:] = 1.0
        self.rate[:] = 0.0
        self._moving[:] = False
        self._initialized = True

    def poses(self) -> list[tuple[np.ndarray, np.ndarray]]:
        """Current commanded poses [(p (3,), R (3, 3)), ...]."""
        if self.object_engaged:
            return [(p.copy(), R.copy()) for p, R in self._cmd]
        return [self._pose(i) for i in range(self.n)]

    # ------------------------------------------------------------------ goals
    def set_targets(self, goals, omega=None) -> None:
        """Start a motion from the current commanded poses to `goals` [(p, R), ...].

        Args:
            goals: One (position (3,), rotation (3, 3)) per arm, in the frame of the commanded poses.
            omega: Optional nominal phase rates w_i [1/s] per arm (temporal mode), e.g. to give the
                arms different speeds and let the coupling reconcile them. Each is still capped by
                that arm's velocity limit. Default: from the limits (see the module docstring).
        """
        if not self._initialized:
            raise RuntimeError("call reset() with the measured poses first")
        if self.cfg.mode == "spatial":
            raise RuntimeError("spatial mode: move the object with set_object_target")
        goals = self._check_poses(goals)
        c = self.cfg
        starts = self.poses()
        dist = np.zeros(self.n)
        angle = np.zeros(self.n)
        for i, ((p0, R0), (pg, Rg)) in enumerate(zip(starts, goals)):
            self._p0[i] = p0
            self._R0[i] = R0
            self._dp[i] = pg - p0
            self._dtheta[i] = rot_to_rotvec(Rg @ R0.T)
            dist[i] = np.linalg.norm(self._dp[i])
            angle[i] = np.linalg.norm(self._dtheta[i])

        w_max = np.array([_phase_limit(c.max_lin_vel, c.max_ang_vel, d, a) for d, a in zip(dist, angle)])
        a_max = np.array([_phase_limit(c.max_lin_acc, c.max_ang_acc, d, a) for d, a in zip(dist, angle)])
        moving = np.isfinite(w_max)        # a zero-length path is already at its goal
        w_max[~moving] = 0.0
        a_max[~moving] = 0.0
        if c.mode == "temporal" and np.any(moving):
            # the slower arm sets the pace for all moving arms
            w_max[moving] = w_max[moving].min()
            a_max[moving] = a_max[moving].min()
        if omega is not None:
            omega = np.asarray(omega, dtype=float).reshape(self.n)
            if np.any(omega <= 0.0):
                raise ValueError("omega must be positive")
            w_max = np.minimum(w_max, omega)
        self._w_max = w_max
        self._a_max = a_max
        self._moving = moving
        self.phase = np.where(moving, 0.0, 1.0)
        self.rate[:] = 0.0

    # ------------------------------------------------------------------ step
    def step(self, dt: float, ee_wrenches=None, ee_poses=None) -> list[tuple[np.ndarray, np.ndarray]]:
        """Advance by dt [s]; returns the commanded poses [(p, R), ...].

        independent / temporal: advance the phases (the arguments after dt are not used).
        spatial: one step of the object admittance. Needs `ee_wrenches` (per end effector [f, tau] it
        applies on the object, about its TCP, world frame) and `ee_poses` (measured, world frame).
        """
        if self.cfg.mode == "spatial":
            return self._step_spatial(dt, ee_wrenches, ee_poses)
        if not self._initialized:
            raise RuntimeError("call reset() with the measured poses first")
        if dt <= 0.0:
            return self.poses()
        phi = self.phase
        active = phi < 1.0

        # own rate: accelerate from rest, cruise, brake to land on phi = 1 at rest. Discrete braking
        # bound: the largest w that still stops within the remaining phase at a_max per step.
        a_half = 0.5 * self._a_max * dt
        brake = np.sqrt(a_half ** 2 + 2.0 * self._a_max * np.maximum(1.0 - phi, 0.0)) - a_half
        rate = np.minimum(np.minimum(self.rate + self._a_max * dt, self._w_max), brake)
        self.rate = np.where(active, rate, 0.0)

        # first-order coupling on the phases at step k (explicit Euler)
        k = self.cfg.coupling if self.cfg.mode == "temporal" else 0.0
        m = self._moving
        coupling = np.where(m, k * (phi[m].sum() - m.sum() * phi), 0.0)   # k * sum_j (phi_j - phi_i)
        phi_dot = np.clip(self.rate + coupling, 0.0, self._w_max)
        phi_dot = np.where(active, phi_dot, 0.0)
        self.phase = np.minimum(phi + dt * phi_dot, 1.0)
        self.rate[self.phase >= 1.0] = 0.0
        return self.poses()

    # ------------------------------------------------------------------ spatial: object compliance
    def engage_object(self, ee_poses, obj_pose=None) -> None:
        """Start object-level compliance for an object held rigidly by both end effectors.

        Captures T_obj->ee_i from `ee_poses` (call it with the grasp closed, object still) and starts
        the object admittance at the object pose with zero velocity, target = that pose, mask =
        `cfg.object_mask`, no wrench bias (see `tare_object`). The commanded poses become `ee_poses`.

        Args:
            ee_poses: [(p, R), (p, R)] end-effector (TCP) poses, world frame.
            obj_pose: Optional (p, R) object frame, world. Default: origin midway between the TCPs
                (near the center of mass for a symmetric grasp on a uniform object), axes of the first
                end effector.
        """
        if self.n != 2:
            raise RuntimeError("spatial mode needs exactly two arms")
        if self.cfg.mode != "spatial":
            raise RuntimeError("set_mode('spatial') before engage_object()")
        poses = self._check_poses(ee_poses)
        if obj_pose is None:
            p_obj, R_obj = 0.5 * (poses[0][0] + poses[1][0]), poses[0][1]
        else:
            (p_obj, R_obj), = self._check_poses_n([obj_pose], 1)
        T_inv = inv_T(make_T(p_obj, R_obj))
        self._T_obj_ee = [T_inv @ make_T(p, R) for p, R in poses]
        law = self.object_law
        law.clear()
        law.reset(p_obj, R_obj)
        law.set_target_wrench(np.zeros(6))
        law.set_mask(self.cfg.object_mask)
        self._w_bias = np.zeros(6)
        self.object_wrench = np.zeros(6)
        self._cmd = poses

    def disengage_object(self) -> None:
        """Leave object compliance: the arms hold their last commanded poses (bumpless); drops the
        grasp transforms and the wrench bias."""
        if not self.object_engaged:
            return
        cmd = self._cmd
        self._T_obj_ee = None
        self._w_bias = np.zeros(6)
        self.reset(cmd)

    def object_pose_from(self, ee_poses) -> tuple[np.ndarray, np.ndarray]:
        """Measured object pose (world) from the end-effector poses through the grasp transforms: the
        mean of the two per-arm estimates (rotation: halfway between them)."""
        self._require_engaged()
        est = [make_T(p, R) @ inv_T(T) for (p, R), T in zip(self._check_poses(ee_poses), self._T_obj_ee)]
        p = 0.5 * (est[0][:3, 3] + est[1][:3, 3])
        R1, R2 = est[0][:3, :3], est[1][:3, :3]
        return p, R1 @ rotvec_to_rot(0.5 * rot_to_rotvec(R1.T @ R2))

    def object_wrench_from(self, ee_poses, ee_wrenches, tared: bool = True) -> np.ndarray:
        """Object wrench (6,) about the measured object origin, world axes: G(r_1, r_2) w_ee, minus
        the tare bias if `tared`. Robot-on-object, like the end-effector wrenches."""
        poses = self._check_poses(ee_poses)
        p_obj, _ = self.object_pose_from(poses)
        w = np.concatenate([np.asarray(w, dtype=float).reshape(6) for w in ee_wrenches])
        if w.shape != (12,) or not np.all(np.isfinite(w)):
            raise ValueError("expected two finite wrenches (6,)")
        w_obj = grasp_matrix(poses[0][0] - p_obj, poses[1][0] - p_obj) @ w
        return w_obj - self._w_bias if tared else w_obj

    def tare_object(self, ee_poses, ee_wrenches) -> np.ndarray:
        """Take the current object wrench as the bias (world axes, about the object origin); returns it.

        Call it with the object lifted (both arms carrying it), still and untouched, before switching
        axes to force mode: it then removes the object's weight and anything else static. The force part
        stays exact for any object orientation (gravity is fixed in world axes); the moment part is
        exact only if the object origin is at the center of mass (else r_com x m g turns with the
        object, and a payload model is needed for large rotations).
        """
        self._w_bias = self.object_wrench_from(ee_poses, ee_wrenches, tared=False)
        return self._w_bias.copy()

    def set_object_target(self, pos, R) -> None:
        """Object target pose (world); pose axes move towards it."""
        self._require_engaged()
        self.object_law.set_target_pose(pos, R)

    def set_object_wrench(self, wrench) -> np.ndarray:
        """Object target wrench [f, tau] in the law's axes (robot-on-object) for force axes."""
        self._require_engaged()
        return self.object_law.set_target_wrench(wrench)

    def set_object_mask(self, mask) -> None:
        """Object axis selection [x, y, z, rx, ry, rz] in the law's axes: 1 = pose, 0 = force."""
        self._require_engaged()
        self.object_law.set_mask(mask)

    def _step_spatial(self, dt, ee_wrenches, ee_poses) -> list[tuple[np.ndarray, np.ndarray]]:
        self._require_engaged()
        if ee_wrenches is None or ee_poses is None:
            raise ValueError("spatial mode: step(dt, ee_wrenches, ee_poses)")
        if dt <= 0.0:
            return self.poses()
        poses = self._check_poses(ee_poses)
        p_m, R_m = self.object_pose_from(poses)
        w = self.object_wrench_from(poses, ee_wrenches)
        self.object_wrench = w
        law = self.object_law
        if law.axes == "tcp":                       # world axes -> object axes
            w = np.concatenate((R_m.T @ w[:3], R_m.T @ w[3:]))
        p_v, R_v = law.step(w, dt, p_m, R_m)
        if law.reverted:                            # non-finite: the law holds its last pose
            return self.poses()
        T_v = make_T(p_v, R_v)
        self._cmd = [((T_v @ T)[:3, 3].copy(), (T_v @ T)[:3, :3].copy()) for T in self._T_obj_ee]
        return self.poses()

    def _require_engaged(self) -> None:
        if not self.object_engaged:
            raise RuntimeError("spatial mode: call engage_object() with the grasp closed first")

    # ------------------------------------------------------------------ helpers
    def _pose(self, i: int) -> tuple[np.ndarray, np.ndarray]:
        phi = self.phase[i]
        p = self._p0[i] + phi * self._dp[i]
        R = rotvec_to_rot(phi * self._dtheta[i]) @ self._R0[i]
        return p, R

    def _check_poses(self, poses) -> list[tuple[np.ndarray, np.ndarray]]:
        return self._check_poses_n(poses, self.n)

    @staticmethod
    def _check_poses_n(poses, n: int) -> list[tuple[np.ndarray, np.ndarray]]:
        poses = list(poses)
        if len(poses) != n:
            raise ValueError(f"expected {n} poses, got {len(poses)}")
        out = []
        for p, R in poses:
            p = np.asarray(p, dtype=float).reshape(3)
            R = np.asarray(R, dtype=float).reshape(3, 3)
            if not (np.all(np.isfinite(p)) and np.all(np.isfinite(R))):
                raise ValueError("non-finite pose")
            out.append((p.copy(), R.copy()))
        return out

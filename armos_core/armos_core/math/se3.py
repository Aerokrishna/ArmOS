"""Rigid transforms (4x4 homogeneous matrices), twists and wrenches.

Source: legacy robot_kinematics/kinematics/spatial_math.py (transformation, adjoint_transform,
transform_vel). Twists are ordered (v, w) and wrenches (f, tau), as in armos telemetry.
"""

from __future__ import annotations

import numpy as np

from armos_core.math.so3 import skew


def make_T(p: np.ndarray, R: np.ndarray | None = None) -> np.ndarray:
    """Position + rotation -> 4x4 transform."""
    T = np.eye(4)
    if R is not None:
        T[:3, :3] = R
    T[:3, 3] = p
    return T


def inv_T(T: np.ndarray) -> np.ndarray:
    """Inverse of a rigid transform."""
    R, p = T[:3, :3], T[:3, 3]
    Ti = np.eye(4)
    Ti[:3, :3] = R.T
    Ti[:3, 3] = -R.T @ p
    return Ti


def transform_point(T: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Point in the child frame -> point in the parent frame."""
    return T[:3, :3] @ p + T[:3, 3]


def adjoint(T: np.ndarray) -> np.ndarray:
    """6x6 adjoint of T for twists ordered (v, w): [v'; w'] = Ad_T [v; w]."""
    R, p = T[:3, :3], T[:3, 3]
    Ad = np.zeros((6, 6))
    Ad[:3, :3] = R
    Ad[:3, 3:] = skew(p) @ R
    Ad[3:, 3:] = R
    return Ad


def transform_twist(T: np.ndarray, twist: np.ndarray) -> np.ndarray:
    """Twist (v, w) given in frame B about B's origin -> frame A about A's origin (T = T_AB)."""
    return adjoint(T) @ twist


def transform_wrench(T: np.ndarray, wrench: np.ndarray) -> np.ndarray:
    """Wrench (f, tau) given in frame B about B's origin -> frame A about A's origin (T = T_AB)."""
    R, p = T[:3, :3], T[:3, 3]
    f = R @ wrench[:3]
    return np.concatenate((f, R @ wrench[3:] + np.cross(p, f)))

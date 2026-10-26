"""Rotations: matrices, quaternions (w, x, y, z), rotation vectors and Euler angles.

Sources: legacy robot_kinematics/kinematics/spatial_math.py, armos_control/utils/kinematics.py,
and the `_wxyz_to_R` copies in armos_sim (pinocchio_sync.py, telemetry.py).

Euler conventions:
- `euler_xyz_to_rot` / `rot_to_euler_xyz`: R = Rx(a) @ Ry(b) @ Rz(c). This is the convention of the
  armos messages (ArmosCartControl a, b, c and ArmosTelemetry tcp_pose).
- `rpy_to_rot` / `rot_to_rpy`: extrinsic X-Y-Z (roll, pitch, yaw), R = Rz(yaw) @ Ry(pitch) @ Rx(roll).
"""

from __future__ import annotations

import math

import mujoco
import numpy as np


def rotX(theta: float) -> np.ndarray:
    """Rotation about x by theta [rad]."""
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def rotY(theta: float) -> np.ndarray:
    """Rotation about y by theta [rad]."""
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def rotZ(theta: float) -> np.ndarray:
    """Rotation about z by theta [rad]."""
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def skew(v: np.ndarray) -> np.ndarray:
    """3-vector -> skew-symmetric matrix, skew(a) @ b = a x b."""
    return np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])


def unskew(M: np.ndarray) -> np.ndarray:
    """Skew-symmetric matrix -> 3-vector."""
    return np.array([M[2, 1], M[0, 2], M[1, 0]])


# --------------------------------------------------------------------------- quaternions (w, x, y, z)
def quat_to_rot(q: np.ndarray) -> np.ndarray:
    """Unit quaternion [w, x, y, z] -> 3x3 rotation matrix."""
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def rot_to_quat(R: np.ndarray) -> np.ndarray:
    """3x3 rotation matrix -> unit quaternion [w, x, y, z] (MuJoCo `mju_mat2Quat`)."""
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.ascontiguousarray(R, dtype=float).reshape(9))
    return q


def quat_normalize(q: np.ndarray) -> np.ndarray:
    """q / |q|."""
    return np.asarray(q, dtype=float) / np.linalg.norm(q)


def quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Hamilton product q1 * q2."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ])


def quat_conj(q: np.ndarray) -> np.ndarray:
    """Conjugate (= inverse for a unit quaternion)."""
    w, x, y, z = q
    return np.array([w, -x, -y, -z])


# --------------------------------------------------------------------------- rotation vectors
def rotvec_to_rot(rotvec: np.ndarray) -> np.ndarray:
    """Rotation vector (axis * angle) -> rotation matrix (Rodrigues)."""
    angle = float(np.linalg.norm(rotvec))
    if angle < 1e-10:
        x, y, z = rotvec
        return np.array([[1.0, -z, y], [z, 1.0, -x], [-y, x, 1.0]])
    K = skew(np.asarray(rotvec, dtype=float) / angle)
    return np.eye(3) + math.sin(angle) * K + (1.0 - math.cos(angle)) * (K @ K)


def rot_to_rotvec(R: np.ndarray) -> np.ndarray:
    """Stable SO(3) logarithm: rotation matrix -> rotation vector (handles angles near 0 and pi)."""
    cos_theta = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    theta = math.acos(float(cos_theta))

    if theta < 1e-8:
        return 0.5 * np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])

    if abs(math.pi - theta) < 1e-5:
        # Near pi, use the symmetric part to estimate the rotation axis.
        A = (R + np.eye(3)) / 2.0
        axis = np.sqrt(np.maximum(np.diag(A), 0.0))
        idx = int(np.argmax(axis))
        if axis[idx] < 1e-8:
            return np.zeros(3)
        if idx == 0:
            axis[1] = math.copysign(axis[1], R[0, 1] + R[1, 0])
            axis[2] = math.copysign(axis[2], R[0, 2] + R[2, 0])
        elif idx == 1:
            axis[0] = math.copysign(axis[0], R[0, 1] + R[1, 0])
            axis[2] = math.copysign(axis[2], R[1, 2] + R[2, 1])
        else:
            axis[0] = math.copysign(axis[0], R[0, 2] + R[2, 0])
            axis[1] = math.copysign(axis[1], R[1, 2] + R[2, 1])
        norm = np.linalg.norm(axis)
        return theta * axis / norm if norm > 1e-8 else np.zeros(3)

    skew_vec = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    return theta / (2.0 * math.sin(theta)) * skew_vec


def orientation_error(R_des: np.ndarray, R_cur: np.ndarray) -> np.ndarray:
    """Rotation vector of R_des @ R_cur^T: the world-frame rotation taking R_cur to R_des."""
    return rot_to_rotvec(R_des @ R_cur.T)


# --------------------------------------------------------------------------- Euler angles
def euler_xyz_to_rot(abc: np.ndarray) -> np.ndarray:
    """(a, b, c) -> R = Rx(a) @ Ry(b) @ Rz(c) (armos message convention)."""
    a, b, c = abc
    return rotX(a) @ rotY(b) @ rotZ(c)


def rot_to_euler_xyz(R: np.ndarray) -> np.ndarray:
    """R -> (a, b, c) with R = Rx(a) @ Ry(b) @ Rz(c); c = 0 at gimbal lock (b = +-pi/2)."""
    b = math.asin(float(np.clip(R[0, 2], -1.0, 1.0)))
    cb = math.cos(b)
    if abs(cb) > 1e-7:
        a = math.atan2(-R[1, 2], R[2, 2])
        c = math.atan2(-R[0, 1], R[0, 0])
    else:
        a = math.atan2(R[2, 1], R[1, 1])
        c = 0.0
    return np.array([a, b, c])


def rpy_to_rot(rpy: np.ndarray) -> np.ndarray:
    """(roll, pitch, yaw) -> R = Rz(yaw) @ Ry(pitch) @ Rx(roll) (extrinsic X-Y-Z)."""
    r, p, y = rpy
    return rotZ(y) @ rotY(p) @ rotX(r)


def rot_to_rpy(R: np.ndarray) -> np.ndarray:
    """R -> (roll, pitch, yaw), inverse of `rpy_to_rot`; roll = 0 at gimbal lock."""
    pitch = math.asin(float(np.clip(-R[2, 0], -1.0, 1.0)))
    if abs(math.cos(pitch)) > 1e-8:
        roll = math.atan2(R[2, 1], R[2, 2])
        yaw = math.atan2(R[1, 0], R[0, 0])
    else:
        roll = 0.0
        yaw = math.atan2(-R[0, 1], R[1, 1])
    return np.array([roll, pitch, yaw])

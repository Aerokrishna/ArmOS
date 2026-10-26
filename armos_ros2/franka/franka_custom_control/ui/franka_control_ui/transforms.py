"""Rotation helpers. Quaternions are numpy arrays in (x, y, z, w) order.

Euler convention used everywhere in the UI: extrinsic XYZ (fixed axes),
i.e. R = Rz(Z) @ Ry(Y) @ Rx(X), angles in radians.
"""

import math

import numpy as np


def euler_xyz_to_quat(rx, ry, rz):
    cr, sr = math.cos(rx / 2), math.sin(rx / 2)
    cp, sp = math.cos(ry / 2), math.sin(ry / 2)
    cy, sy = math.cos(rz / 2), math.sin(rz / 2)
    q = np.array([
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
        cr * cp * cy + sr * sp * sy,
    ])
    return q / np.linalg.norm(q)


def quat_to_euler_xyz(q):
    x, y, z, w = q
    rx = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    ry = math.asin(max(-1.0, min(1.0, 2 * (w * y - z * x))))
    rz = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return np.array([rx, ry, rz])


def unwrap_to(angle, reference):
    """Shift angle by multiples of 2*pi so it lies closest to reference."""
    return angle + 2 * math.pi * round((reference - angle) / (2 * math.pi))


def quat_mul(a, b):
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return np.array([
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    ])


def quat_from_rotvec(v):
    angle = float(np.linalg.norm(v))
    if angle < 1e-12:
        return np.array([0.0, 0.0, 0.0, 1.0])
    axis = v / angle
    s = math.sin(angle / 2)
    return np.array([axis[0] * s, axis[1] * s, axis[2] * s, math.cos(angle / 2)])


def quat_to_matrix(q):
    x, y, z, w = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def quat_angle_between(a, b):
    """Smallest rotation angle (rad) taking a to b."""
    d = abs(float(np.dot(a, b)))
    return 2 * math.acos(min(1.0, d))

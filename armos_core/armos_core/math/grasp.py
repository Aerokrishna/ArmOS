"""Bimanual grasp matrix: maps between end-effector wrenches / twists and the held object's.

Convention: "Mathematical Formulation of the Dual-Arm Admittance Controller" (multi-arm-coordination-control),
section 6 and eqs. (11)-(18); equation numbers below refer to it. Everything is in the WORLD frame.
r_1, r_2 are the end-effector-object lever arms (r_i = p_i - p_obj, world frame). Stacked vectors are
ordered per end effector, linear part first:

    w_ee = [f_1; tau_1; f_2; tau_2] in R^12,     v = [v_1; w_1; v_2; w_2] in R^12

Grasp matrix (11):

    G = [[ I_3,     0,   I_3,     0  ],     in R^(6 x 12)
         [[r_1]x,  I_3, [r_2]x,  I_3 ]]

with [r]x b = r x b, so w_obj = G w_ee gives f_obj = f_1 + f_2, tau_obj = tau_1 + tau_2 + r_1 x f_1 + r_2 x f_2.

    G+  = pinv(G)                 Moore-Penrose pseudo-inverse
    N   = I_12 - G+ G             grasp null-space projector                        (12)
    w_obj = G w_ee                object wrench                                     (13)
    w_ext = G+ w_obj              external (motion-inducing) part of w_ee           (14)
    w_int = N (N+) w_ee = N w_ee  internal (squeeze) part of w_ee (N is a projector) (15)
    v_task = G^T v_obj            object twist -> end-effector twists               (18)

w_ext + w_int = w_ee, and G w_int = 0. Desired squeeze forces act along +-n, n = (p_2 - p_1) / |p_2 - p_1|
(section 7.3).
"""

from __future__ import annotations

import numpy as np

from armos_core.math.so3 import skew


def grasp_matrix(r1: np.ndarray, r2: np.ndarray) -> np.ndarray:
    """Bimanual grasp matrix G (6 x 12), eq. (11).

    Args:
        r1, r2: End-effector-object lever arms r_i = p_i - p_obj (3,), world frame.
    """
    G = np.zeros((6, 12))
    for i, r in enumerate((r1, r2)):
        r = np.asarray(r, dtype=float).reshape(3)
        G[:3, 6 * i:6 * i + 3] = np.eye(3)
        G[3:, 6 * i:6 * i + 3] = skew(r)
        G[3:, 6 * i + 3:6 * i + 6] = np.eye(3)
    return G


def grasp_pinv(G: np.ndarray) -> np.ndarray:
    """G+ = pinv(G) (12 x 6)."""
    return np.linalg.pinv(G)


def nullspace_projector(G: np.ndarray, G_pinv: np.ndarray | None = None) -> np.ndarray:
    """N = I_12 - G+ G (12 x 12), eq. (12)."""
    Gp = grasp_pinv(G) if G_pinv is None else G_pinv
    return np.eye(G.shape[1]) - Gp @ G


def object_wrench(G: np.ndarray, w_ee: np.ndarray) -> np.ndarray:
    """w_obj = G w_ee (6,), eq. (13). w_ee: stacked end-effector wrenches (12,), world frame."""
    return G @ np.asarray(w_ee, dtype=float).reshape(12)


def external_wrench(G: np.ndarray, w_obj: np.ndarray, G_pinv: np.ndarray | None = None) -> np.ndarray:
    """w_ext = G+ w_obj (12,), eq. (14): the least-norm end-effector wrenches producing w_obj."""
    Gp = grasp_pinv(G) if G_pinv is None else G_pinv
    return Gp @ np.asarray(w_obj, dtype=float).reshape(6)


def internal_wrench(G: np.ndarray, w_ee: np.ndarray, N: np.ndarray | None = None) -> np.ndarray:
    """w_int = N (N+) w_ee (12,), eq. (15); N is an orthogonal projector, so N N+ = N."""
    N = nullspace_projector(G) if N is None else N
    return N @ np.asarray(w_ee, dtype=float).reshape(12)


def squeeze_wrench(p1: np.ndarray, p2: np.ndarray, force: float) -> np.ndarray:
    """Stacked end-effector wrenches (12,) squeezing the object with `force` [N] along +-n,
    n = (p2 - p1) / |p2 - p1| (section 7.3): f_1 = +force n, f_2 = -force n, torques zero. Positive force
    pushes the end effectors towards each other; G times it is zero (a pure internal wrench)."""
    d = np.asarray(p2, dtype=float).reshape(3) - np.asarray(p1, dtype=float).reshape(3)
    dist = np.linalg.norm(d)
    if dist < 1e-9:
        raise ValueError("end effectors coincide: squeeze direction undefined")
    n = d / dist
    return np.concatenate((force * n, np.zeros(3), -force * n, np.zeros(3)))


def ee_twists(G: np.ndarray, v_obj: np.ndarray) -> np.ndarray:
    """v_task = G^T v_obj (12,), eq. (18): object twist [v; w] (6,) -> stacked end-effector twists
    [v_1; w_1; v_2; w_2] for a rigid grasp (v_i = v_obj + w_obj x r_i, w_i = w_obj)."""
    return G.T @ np.asarray(v_obj, dtype=float).reshape(6)

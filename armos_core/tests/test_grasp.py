"""Bimanual grasp matrix (convention: dual-arm admittance formulation, eqs. (11)-(18))."""

from __future__ import annotations

import numpy as np
import pytest

from armos_core.math.grasp import (
    ee_twists,
    external_wrench,
    grasp_matrix,
    grasp_pinv,
    internal_wrench,
    nullspace_projector,
    object_wrench,
    squeeze_wrench,
)
from armos_core.math.se3 import make_T, transform_wrench

P_OBJ = np.array([0.45, 0.25, 0.40])           # the handover: hands 39 mm either side of the cylinder center
P1 = P_OBJ + [0.0, -0.039, 0.0]
P2 = P_OBJ + [0.0, 0.039, 0.0]
R1, R2 = P1 - P_OBJ, P2 - P_OBJ                 # lever arms r_i = p_i - p_obj
rng = np.random.default_rng(0)


def test_matrix_is_eq_11():
    G = grasp_matrix(R1, R2)
    rx = lambda r: np.array([[0, -r[2], r[1]], [r[2], 0, -r[0]], [-r[1], r[0], 0]])  # noqa: E731
    I, Z = np.eye(3), np.zeros((3, 3))
    assert np.allclose(G, np.block([[I, Z, I, Z], [rx(R1), I, rx(R2), I]]))


def test_object_wrench_matches_wrench_transport():
    w1, w2 = rng.normal(size=6), rng.normal(size=6)
    expected = transform_wrench(make_T(R1), w1) + transform_wrench(make_T(R2), w2)
    assert np.allclose(object_wrench(grasp_matrix(R1, R2), np.concatenate((w1, w2))), expected)


def test_lifting():
    G = grasp_matrix(R1, R2)
    # both hands lift 0.5 N: 1 N on the object, no torque
    assert np.allclose(object_wrench(G, [0, 0, 0.5, 0, 0, 0, 0, 0, 0.5, 0, 0, 0]), [0, 0, 1, 0, 0, 0])
    # one hand alone: the same force plus the moment r1 x f
    w = object_wrench(G, [0, 0, 1.0, 0, 0, 0] + [0] * 6)
    assert np.allclose(w, [0, 0, 1, *np.cross(R1, [0, 0, 1.0])])


def test_external_and_internal_split():
    G = grasp_matrix(R1, R2)
    w_ee = rng.normal(size=12)
    w_ext = external_wrench(G, object_wrench(G, w_ee))
    w_int = internal_wrench(G, w_ee)
    assert np.allclose(w_ext + w_int, w_ee)                 # (14) + (15) recover w_ee
    assert np.allclose(G @ w_int, 0.0)                       # internal: no object wrench
    assert np.allclose(G @ w_ext, G @ w_ee)                  # external carries all of it
    assert np.isclose(w_ext @ w_int, 0.0)                    # orthogonal split


def test_nullspace_projector_eq_12():
    G = grasp_matrix(R1, R2)
    N = nullspace_projector(G)
    assert np.allclose(N, np.eye(12) - np.linalg.pinv(G) @ G)
    assert np.allclose(N @ N, N) and np.allclose(N, N.T)    # orthogonal projector: N N+ = N
    assert np.allclose(N @ np.linalg.pinv(N), N)
    assert np.linalg.matrix_rank(N) == 6


def test_holding_a_weight_splits_evenly():
    G = grasp_matrix(R1, R2)
    w = external_wrench(G, [0, 0, 0.981, 0, 0, 0])           # hold 0.1 kg
    assert np.allclose(w[:6], w[6:])
    assert np.allclose(w[:3], [0, 0, 0.4905])


def test_squeeze_is_internal():
    G = grasp_matrix(R1, R2)
    w_sq = squeeze_wrench(P1, P2, 10.0)
    assert np.allclose(G @ w_sq, 0.0)
    assert np.allclose(internal_wrench(G, w_sq), w_sq)
    assert w_sq[1] > 0 and w_sq[7] < 0                       # hand 1 pushes +y (towards hand 2), hand 2 -y


def test_twists_eq_18_and_power():
    G = grasp_matrix(R1, R2)
    v_obj = rng.normal(size=6)
    v = ee_twists(G, v_obj)
    for i, r in enumerate((R1, R2)):
        assert np.allclose(v[6 * i:6 * i + 3], v_obj[:3] + np.cross(v_obj[3:], r))   # rigid body
        assert np.allclose(v[6 * i + 3:6 * i + 6], v_obj[3:])
    w_ee = rng.normal(size=12)
    assert np.isclose(object_wrench(G, w_ee) @ v_obj, w_ee @ v)                      # power


def test_coincident_end_effectors():
    G = grasp_matrix(R1, R1)
    assert np.allclose(G @ grasp_pinv(G) @ G, G)             # pinv stays valid when rank-deficient
    with pytest.raises(ValueError):
        squeeze_wrench(P1, P1, 1.0)

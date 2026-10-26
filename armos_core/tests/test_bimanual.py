"""BimanualSync: independent / temporal target generation and spatial (object-level) compliance."""

from __future__ import annotations

import numpy as np
import pytest

from armos_core.control.bimanual import BimanualCfg, BimanualSync
from armos_core.control.hybrid_admittance import HybridAdmittanceCfg
from armos_core.math.so3 import rot_to_rotvec, rotvec_to_rot, rotX

DT = 0.004
# handover-like grasp: TCPs 39 mm either side of the object center, fingers pointing at the object
R_L, R_R = rotX(np.pi), rotX(np.pi) @ rotvec_to_rot([0, 0, np.pi])
P_OBJ = np.array([0.45, 0.25, 0.40])
GRASP = [(P_OBJ + [0, 0.039, 0], R_L), (P_OBJ + [0, -0.039, 0], R_R)]


def run_phase(s, max_t=30.0):
    t, arrive = 0.0, [None, None]
    while not s.done and t < max_t:
        s.step(DT)
        t += DT
        for i in range(2):
            if arrive[i] is None and s.phase[i] >= 1.0:
                arrive[i] = t
    return arrive


@pytest.mark.parametrize("mode", ["independent", "temporal"])
def test_phase_modes(mode):
    s = BimanualSync(BimanualCfg(mode=mode))
    s.reset(GRASP)
    goals = [(GRASP[0][0] + [0.3, 0, 0], GRASP[0][1]), (GRASP[1][0] + [0.1, 0, 0], GRASP[1][1])]
    s.set_targets(goals)
    arrive = run_phase(s)
    for (p, R), (pg, Rg) in zip(s.poses(), goals):
        assert np.allclose(p, pg) and np.allclose(R, Rg)
    if mode == "temporal":
        assert arrive[0] == pytest.approx(arrive[1], abs=DT)
    else:
        assert arrive[1] < arrive[0] - 1.0


def spatial(mask=(1, 1, 1, 1, 1, 1), **adm):
    s = BimanualSync(BimanualCfg(mode="spatial", object_mask=mask,
                                 object_admittance=HybridAdmittanceCfg(**adm)))
    s.reset(GRASP)
    s.engage_object(GRASP)
    return s


def run_spatial(s, wrenches, steps):
    """Closed loop with perfect tracking: measured poses = commanded poses."""
    for _ in range(steps):
        s.step(DT, wrenches, s.poses())
    return s.poses()


def rel(poses):
    (p1, R1), (p2, R2) = poses
    return R1.T @ (p2 - p1), R1.T @ R2


def test_engage_holds_without_jump():
    s = spatial()
    poses = run_spatial(s, [np.zeros(6)] * 2, 100)
    for (p, R), (p0, R0) in zip(poses, GRASP):
        assert np.allclose(p, p0) and np.allclose(R, R0)
    p, R = s.object_pose_from(poses)
    assert np.allclose(p, P_OBJ)


def test_push_on_object_moves_both_arms_rigidly():
    s = spatial(mask=(0, 1, 1, 1, 1, 1))                  # x compliant
    d0 = rel(GRASP)
    # someone pushes the object +x with 6 N: statically the arms hold it back with -3 N each
    push = [np.array([-3.0, 0, 0, 0, 0, 0])] * 2
    poses = run_spatial(s, push, 500)
    moved = [p - p0 for (p, _), (p0, _) in zip(poses, GRASP)]
    assert moved[0][0] > 0.005 and np.allclose(moved[0], moved[1])   # both along +x, equally
    assert np.allclose(moved[0][1:], 0.0, atol=1e-9)                  # pose axes stay
    d = rel(poses)
    assert np.allclose(d[0], d0[0]) and np.allclose(d[1], d0[1])     # grasp kept rigid


def test_pose_axes_ignore_the_push():
    s = spatial()                                           # all pose axes
    poses = run_spatial(s, [np.array([-3.0, 0, 0, 0, 0, 0])] * 2, 500)
    assert np.allclose(poses[0][0], GRASP[0][0])


def test_twist_rotates_about_the_object_origin():
    s = spatial(mask=(1, 1, 1, 1, 1, 0))                  # yaw compliant
    # opposite x forces at the two ends (r = +-39 mm along y): robot-on-object moment about z
    f = 4.0
    w = [np.array([f, 0, 0, 0, 0, 0]), np.array([-f, 0, 0, 0, 0, 0])]
    w_obj = s.object_wrench_from(GRASP, w)
    assert np.allclose(w_obj[:3], 0.0) and w_obj[5] == pytest.approx(-2 * f * 0.039)
    poses = run_spatial(s, w, 500)
    p, R = s.object_pose_from(poses)
    assert np.allclose(p, P_OBJ, atol=1e-9)                 # pivot = object origin
    yaw = rot_to_rotvec(R @ GRASP[0][1].T)[2]
    assert yaw > 0.02                                        # turns with the external moment (+z)
    d0, d = rel(GRASP), rel(poses)
    assert np.allclose(d[0], d0[0]) and np.allclose(d[1], d0[1])


def test_weight_needs_a_tare():
    m_g = 0.1 * 9.81
    carry = [np.array([0, 0, m_g / 2, 0, 0, 0])] * 2      # both arms hold the object up
    s = spatial(mask=(1, 1, 0, 1, 1, 1))                  # z compliant
    poses = run_spatial(s, carry, 500)
    assert poses[0][0][2] < GRASP[0][0][2] - 0.002          # untared: sinks at m g / D_force (1.6 mm/s)
    s = spatial(mask=(1, 1, 0, 1, 1, 1))
    bias = s.tare_object(GRASP, carry)
    assert np.allclose(bias, [0, 0, m_g, 0, 0, 0])
    poses = run_spatial(s, carry, 500)
    assert np.allclose(poses[0][0], GRASP[0][0], atol=1e-9)  # tared: holds


def test_squeeze_has_no_effect():
    s = spatial(mask=(0, 0, 0, 0, 0, 0))                  # everything compliant
    sq = [np.array([0, -10.0, 0, 0, 0, 0]), np.array([0, 10.0, 0, 0, 0, 0])]   # inwards along the axis
    assert np.allclose(s.object_wrench_from(GRASP, sq), 0.0)
    poses = run_spatial(s, sq, 500)
    for (p, _), (p0, _) in zip(poses, GRASP):
        assert np.allclose(p, p0)


def test_mode_rules_and_bumpless_exit():
    s = BimanualSync(BimanualCfg(mode="temporal"))
    s.reset(GRASP)
    with pytest.raises(RuntimeError):
        s.engage_object(GRASP)                              # needs spatial mode first
    s.set_mode("spatial")
    with pytest.raises(RuntimeError):
        s.step(DT, [np.zeros(6)] * 2, GRASP)                # not engaged
    s.engage_object(GRASP)
    s.set_object_mask([0, 1, 1, 1, 1, 1])
    with pytest.raises(RuntimeError):
        s.set_targets(GRASP)
    held = run_spatial(s, [np.array([-3.0, 0, 0, 0, 0, 0])] * 2, 200)
    s.set_mode("temporal")                                  # releases the object, holds the poses
    assert not s.object_engaged
    for (p, R), (ph, Rh) in zip(s.step(DT), held):
        assert np.allclose(p, ph) and np.allclose(R, Rh)

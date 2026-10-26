"""HybridAdmittance (base / TCP axes, deadband, max deviation) and ExternalWrench (tare, sign, transport)."""

from __future__ import annotations

import numpy as np
import pytest

from armos_core.control import ExternalWrench, ExternalWrenchCfg, HybridAdmittance, HybridAdmittanceCfg
from armos_core.control.external_wrench import to_axes
from armos_core.math.so3 import rotX, rotZ

DT = 0.004
POS0 = np.array([0.4, 0.0, 0.3])


def law(**kw) -> HybridAdmittance:
    a = HybridAdmittance(HybridAdmittanceCfg(**kw))
    a.reset(POS0, R_TOOL)
    return a


R_TOOL = rotX(np.pi / 2) @ rotZ(0.3)   # tool z = -base y: base and tool axes differ


def run(a: HybridAdmittance, wrench, steps=250, R_meas=None, pos_meas=None):
    for _ in range(steps):
        a.step(wrench, DT, pos_meas, R_meas)
    return a.virtual_pos - POS0


def test_tcp_equals_base_when_tool_is_aligned() -> None:
    b, t = law(axes="base"), law(axes="tcp")
    rng = np.random.default_rng(0)
    for a in (b, t):
        a.set_mask([1, 0, 0, 1, 0, 1])
        a.set_target_wrench([0.0, 2.0, -1.0, 0.0, 0.1, 0.0])
    for _ in range(300):
        w = rng.normal(0.0, 1.0, 6)
        pb, Rb = b.step(w, DT)
        pt, Rt = t.step(w, DT, R_meas=np.eye(3))
        assert np.allclose(pb, pt, atol=1e-12) and np.allclose(Rb, Rt, atol=1e-12)


@pytest.mark.parametrize("axes", ["base", "tcp"])
def test_force_axis_follows_the_chosen_axes(axes) -> None:
    a = law(axes=axes)
    a.set_mask([1, 1, 0, 1, 1, 1])                 # z is the force axis
    a.set_target_wrench([0.0, 0.0, 3.0, 0.0, 0.0, 0.0])
    d = run(a, np.zeros(6), R_meas=R_TOOL)
    z_axis = R_TOOL[:, 2] if axes == "tcp" else np.array([0.0, 0.0, 1.0])
    along = d @ z_axis
    speed = min(3.0 / a.D_force[2], a.cfg.max_force_lin_vel)   # free-space approach: F / D_force, capped
    assert along > 0.8 * speed * 250 * DT
    assert np.linalg.norm(d - along * z_axis) < 1e-6   # and only along that axis


def test_pose_axes_ignore_force() -> None:
    a = law(axes="tcp")
    d = run(a, np.array([10.0, -5.0, 8.0, 1.0, 1.0, 1.0]), R_meas=R_TOOL)
    assert np.linalg.norm(d) < 1e-9


def test_force_deadband() -> None:
    a = law(axes="base", force_deadband=(0.5, 0.05))
    a.set_mask([0, 1, 1, 1, 1, 1])
    assert np.linalg.norm(run(a, np.array([-0.4, 0, 0, 0, 0, 0]))) < 1e-12   # inside the band
    assert run(a, np.array([-2.0, 0, 0, 0, 0, 0]))[0] > 0.0                   # outside: moves


def test_max_deviation_limits_the_lead() -> None:
    a = law(axes="base", max_deviation=(0.01, 0.3))
    a.set_mask([0, 1, 1, 1, 1, 1])
    a.set_target_wrench([5.0, 0, 0, 0, 0, 0])
    d = run(a, np.zeros(6), steps=500, pos_meas=POS0, R_meas=R_TOOL)   # robot never moves
    assert d[0] == pytest.approx(0.01, abs=1e-9)


def test_tcp_axes_need_the_measured_orientation() -> None:
    with pytest.raises(ValueError):
        law(axes="tcp").step(np.zeros(6), DT)
    with pytest.raises(ValueError):
        HybridAdmittance(HybridAdmittanceCfg(axes="tool"))


def test_external_wrench_tare_sign_and_filter() -> None:
    rng = np.random.default_rng(1)
    bias = np.array([1.0, -2.0, 0.5, 0.1, 0.0, -0.05])
    ew = ExternalWrench(ExternalWrenchCfg(sign=-1.0, tare_samples=100, filter_hz=0.0))
    for k in range(99):
        assert ew.update(bias + rng.normal(0, 0.01, 6), k * DT, DT) is None
    assert ew.update(bias, 99 * DT, DT) is None and ew.tare_event == "done"
    push = np.array([0.0, 0.0, 4.0, 0.0, 0.0, 0.0])
    out = ew.update(bias + push, 100 * DT, DT)
    assert np.allclose(out, -push, atol=0.01)       # bias removed, sign applied


def test_external_wrench_tare_rejects_motion() -> None:
    ew = ExternalWrench(ExternalWrenchCfg(tare_samples=50))
    for k in range(50):
        ew.update(np.array([0.0, 0.0, 5.0 * (k % 2), 0.0, 0.0, 0.0]), k * DT, DT)
    assert ew.tare_event == "rejected" and not ew.tared


def test_to_axes_moves_the_torque_and_rotates() -> None:
    f = np.array([0.0, 0.0, -10.0])
    point, tcp = np.array([0.0, 0.0, 0.1]), np.array([0.05, 0.0, 0.0])
    w = to_axes(np.concatenate((f, np.zeros(3))), tcp, None, point)
    assert np.allclose(w[3:], np.cross(point - tcp, f))
    wt = to_axes(np.concatenate((f, np.zeros(3))), tcp, R_TOOL, point)
    assert np.allclose(R_TOOL @ wt[:3], f) and np.allclose(R_TOOL @ wt[3:], w[3:])

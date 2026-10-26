"""CyRoSimulator checks for Phase 2: actuation defaults, rates, mode validation, torque modes, divergence guard.

The torque-mode checks run on both torque configs (FR3 + hand, dual Panda), headless, with the YAML rates
(physics 500 Hz, control tick 100 Hz, admittance outer loop 250 Hz).
"""

from __future__ import annotations

import subprocess
import sys

import numpy as np
import pytest

from armos_sim import ASSETS_DIR, ControlMode, CyRoSimulator

FR3 = ASSETS_DIR / "franka_fr3" / "fr3.yaml"
DUAL = ASSETS_DIR / "franka_emika_panda" / "dual_franka.yaml"
PANDA = ASSETS_DIR / "franka_emika_panda" / "franka.yaml"
TORQUE_CONFIGS = pytest.mark.parametrize("config", [FR3, DUAL], ids=["fr3", "dual_panda"])


def run(sim: CyRoSimulator, seconds: float, hook=None) -> None:
    for _ in range(round(seconds / sim._dt)):
        if hook is not None:
            hook()
        sim.step()


def tcp(sim: CyRoSimulator, arm: str) -> np.ndarray:
    return sim.get_telemetry().tcp_pose[arm].copy()


def test_import_does_not_load_viewer_pinocchio_or_pink() -> None:
    code = (
        "import sys, armos_sim, armos_sim.plant, armos_sim.runtime.simulator; "
        "print(sorted({'pinocchio', 'pink', 'mujoco.viewer'} & set(sys.modules)))"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "[]"


def test_actuation_defaults_and_rates() -> None:
    sim = CyRoSimulator(FR3, realtime_factor=0)
    assert sim.arm_config("left").actuation == "torque"
    assert sim._mj.model.opt.timestep == pytest.approx(0.002)
    assert sim._dt == pytest.approx(0.010)      # decimation.control = 5
    assert sim._adm_decimation == 2
    dual = CyRoSimulator(DUAL, realtime_factor=0)
    assert [dual.arm_config(a).actuation for a in dual.arm_ids] == ["torque", "torque"]

    sim = CyRoSimulator(PANDA, realtime_factor=0)   # no actuation / physics in the YAML
    assert sim.arm_config("left").actuation == "position"
    assert sim._dt == pytest.approx(2 * sim._mj.model.opt.timestep)

    sim = CyRoSimulator(FR3, actuation="position", control_decimation=3, admittance_decimation=1,
                        realtime_factor=0)
    assert sim.arm_config("left").actuation == "position"
    assert sim._dt == pytest.approx(0.006)
    assert sim._adm_decimation == 1


def test_modes_are_checked_against_actuation() -> None:
    torque = CyRoSimulator(FR3, realtime_factor=0)
    with pytest.raises(ValueError):
        torque.set_mode("left", ControlMode.TASK_POSE)
    position = CyRoSimulator(FR3, actuation="position", realtime_factor=0)
    with pytest.raises(ValueError):
        position.set_mode("left", ControlMode.CARTESIAN_IMPEDANCE)
    with pytest.raises(ValueError):
        position.plant("left")


@TORQUE_CONFIGS
def test_joint_impedance_holds_home_under_gravity(config) -> None:
    sim = CyRoSimulator(config, realtime_factor=0)
    run(sim, 2.0)
    for arm in sim.arm_ids:
        assert sim.get_telemetry().control_mode[arm] == "JOINT_IMPEDANCE"
        assert np.abs(sim.get_telemetry().joint_pos[arm] - sim.arm_config(arm).q_home).max() < 1e-3


@TORQUE_CONFIGS
def test_cartesian_impedance_holds_and_reaches_a_pose(config) -> None:
    sim = CyRoSimulator(config, realtime_factor=0)
    T0 = {arm: tcp(sim, arm) for arm in sim.arm_ids}
    for arm in sim.arm_ids:
        sim.set_tcp_pose(arm, T0[arm][:3, 3], T0[arm][:3, :3])
        sim.set_mode(arm, ControlMode.CARTESIAN_IMPEDANCE)
    run(sim, 2.0)
    for arm in sim.arm_ids:
        assert np.linalg.norm(tcp(sim, arm)[:3, 3] - T0[arm][:3, 3]) < 1e-3

    offset = np.array([0.05, 0.0, -0.05])
    for arm in sim.arm_ids:
        sim.set_tcp_pose(arm, T0[arm][:3, 3] + offset, T0[arm][:3, :3])
    run(sim, 3.0)
    for arm in sim.arm_ids:
        T = tcp(sim, arm)
        # 1 cm: the FR3 model's joint frictionloss leaves a few mm against the default stiffness.
        assert np.linalg.norm(T[:3, 3] - (T0[arm][:3, 3] + offset)) < 1e-2
        assert np.linalg.norm(T[:3, :3] - T0[arm][:3, :3]) < 2e-2


@TORQUE_CONFIGS
def test_cartesian_admittance_complies_to_a_push(config) -> None:
    sim = CyRoSimulator(config, realtime_factor=0)
    m, d = sim._mj.model, sim._mj.data
    for arm in sim.arm_ids:
        # y is a force axis with zero target wrench: the arm gives way to a push along y.
        sim.set_admittance_target(arm, wrench=np.zeros(6), mask=[1, 0, 1, 1, 1, 1])
        sim.set_mode(arm, ControlMode.CARTESIAN_ADMITTANCE)
    run(sim, 0.5)   # F/T tare
    p0 = {arm: tcp(sim, arm)[:3, 3] for arm in sim.arm_ids}
    bodies = {arm: m.site_bodyid[sim.plant(arm).h.tcp_site_id] for arm in sim.arm_ids}

    def push():
        for arm in sim.arm_ids:
            d.xfrc_applied[bodies[arm], :3] = sim.arm_config(arm).base_world_rot @ [0.0, 10.0, 0.0]

    run(sim, 1.5, push)
    for arm in sim.arm_ids:
        dp = tcp(sim, arm)[:3, 3] - p0[arm]   # arm base frame
        assert dp[1] > 0.01                     # complies along the push
        assert abs(dp[0]) < 1e-3 and abs(dp[2]) < 1e-3   # pose axes stay stiff


@TORQUE_CONFIGS
def test_divergence_restores_state_and_holds(config) -> None:
    sim = CyRoSimulator(config, realtime_factor=0)
    d = sim._mj.data
    arm = sim.arm_ids[0]
    run(sim, 0.2)
    sim.set_mode(arm, ControlMode.CARTESIAN_IMPEDANCE)
    q_before = d.qpos[sim.plant(arm).h.arm_qpos].copy()
    for _ in range(2):   # twice: MuJoCo's auto-reset must be caught every time
        d.qpos[sim.plant(arm).h.arm_qpos] = np.nan
        sim.step()
        assert sim.get_telemetry().control_mode[arm] == "JOINT_IMPEDANCE"
        assert np.allclose(d.qpos[sim.plant(arm).h.arm_qpos], q_before)
        sim.set_mode(arm, ControlMode.CARTESIAN_IMPEDANCE)
    run(sim, 0.5)
    assert np.all(np.isfinite(d.qpos))
    assert np.abs(d.qpos[sim.plant(arm).h.arm_qpos] - q_before).max() < 1e-3

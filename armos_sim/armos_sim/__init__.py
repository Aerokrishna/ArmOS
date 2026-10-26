"""
armos_sim — MuJoCo simulation of robot arms for armos.

Quick Start
-----------
    from armos_sim import CyRoSimulator, ControlMode, Telemetry

    sim = CyRoSimulator(realtime_factor=0)      # default: FR3 + hand (fr3.yaml), torque actuation
    sim.set_joint_pos("left", [0.0, -0.3, 0.0, -2.0, 0.0, 1.8, 0.8])   # JOINT_IMPEDANCE target
    sim.set_gripper("left", 0.5)                # 0=closed, 1=open

    # Manual stepping loop
    for tick in range(1000):
        telem: Telemetry = sim.step()

    sim = CyRoSimulator(actuation="position")   # same robot on position servos
    sim.set_mode("left", ControlMode.JOINT_POS)

Control Modes
-------------
  Position-actuated arms (`actuation: position`, the default when a YAML sets none):
    JOINT_POS    – Joint position target           (no extra deps)
    JOINT_VEL    – Joint velocity, integrated      (no extra deps)
    JOINT_TORQUE – Torque + gravity compensation   (no extra deps)
    TASK_POSE    – Cartesian SE3 via Pink IK       (needs pinocchio, pink: armos_sim[ik])
    TASK_VEL     – Cartesian twist via Jacobian    (Pinocchio, else MuJoCo Jacobian)

  Torque-actuated arms (`actuation: torque`; default in fr3.yaml and dual_franka.yaml):
    JOINT_TORQUE         – Torque + gravity/Coriolis compensation
    JOINT_IMPEDANCE      – armos_core JointImpedance (target: set_joint_pos; active after load/reset)
    CARTESIAN_IMPEDANCE  – armos_core CartesianImpedance (target: set_tcp_pose)
    CARTESIAN_ADMITTANCE – HybridAdmittance -> CartesianImpedance (targets: set_admittance_target)

Rates (YAML `physics`): torque laws every physics step (`timestep`); position modes, telemetry and
callbacks every `decimation.control` steps; the admittance outer loop every `decimation.admittance` steps.

Fast core (no viewer, RL use): `armos_sim.plant.ArmPlant`.
"""

from pathlib import Path

from .types import ArmConfig, ControlMode, Telemetry

__version__ = "0.2.0"

ASSETS_DIR: Path = Path(__file__).parent / "assets"

__all__ = [
    "ASSETS_DIR",
    "CyRoSimulator",
    "ControlMode",
    "Telemetry",
    "ArmConfig",
    "get_assets_path",
]


def __getattr__(name):
    # CyRoSimulator is imported on first use, so `import armos_sim.plant` (RL, tests) stays light:
    # no viewer, Pinocchio or Pink imports.
    if name in ("CyRoSimulator", "TriArmSimulator"):
        from .runtime.simulator import CyRoSimulator
        return CyRoSimulator
    raise AttributeError(f"module 'armos_sim' has no attribute {name!r}")


def get_assets_path() -> Path:
    """Return the path to the package's assets directory (configs, XML, meshes)."""
    return ASSETS_DIR

"""Control laws (no ROS). Torque-level laws read a `PlantView`; outer loops are plain numpy.

Torque laws (from mujoco_rl_bed/control, unchanged): `CartesianImpedance`, `JointImpedance`,
`AxisCompliance`. Outer loops: `HybridAdmittance` (one admittance law for sim and real, base or TCP
axes); wrench processing: `WrenchCompensator` (wrist F/T sensor with a payload model, sim) and
`ExternalWrench` (Franka-style external wrench estimate, real robot and the sim bridge).
"""

from armos_core.control.axis_compliance import AxisCompliance, AxisComplianceCfg
from armos_core.control.base import Controller, ControllerCfg
from armos_core.control.cartesian_impedance import CartesianImpedance, CartesianImpedanceCfg
from armos_core.control.external_wrench import ExternalWrench, ExternalWrenchCfg
from armos_core.control.hybrid_admittance import HybridAdmittance, HybridAdmittanceCfg
from armos_core.control.joint_impedance import JointImpedance, JointImpedanceCfg
from armos_core.control.wrench_compensator import WrenchCompensator, WrenchCompensatorCfg

__all__ = [
    "Controller", "ControllerCfg",
    "CartesianImpedance", "CartesianImpedanceCfg",
    "JointImpedance", "JointImpedanceCfg",
    "AxisCompliance", "AxisComplianceCfg",
    "HybridAdmittance", "HybridAdmittanceCfg",
    "WrenchCompensator", "WrenchCompensatorCfg",
    "ExternalWrench", "ExternalWrenchCfg",
]

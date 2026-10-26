# armos_core

Math, kinematics, control laws and types shared by `armos_sim`, the ROS 2 nodes in `armos_ros2` and
`mujoco_rl_bed`. No ROS imports.

| Module | Contents | Source |
|---|---|---|
| `math.so3`, `math.se3` | rotations (quaternions w,x,y,z), Euler conventions, transforms, adjoint | `robot_kinematics/kinematics/spatial_math.py`, `armos_control/utils/kinematics.py` |
| `math.filters`, `math.limits` | `LowPass1`, `limit_norm` | `armos_control/cartesian_admittance.py` |
| `math.mj_math` | allocation-free DLS, quaternion error, gravity torque | `mujoco_rl_bed/control/math_utils.py` |
| `kinematics.mujoco_kin` | `MujocoArm`: views, Jacobian, EE pose/velocity, F/T (world frame) | `mujoco_rl_bed/sim/plant.py` |
| `kinematics.pinocchio_arm` | `PinocchioArm` built from the MuJoCo chain (base frame) | `armos_sim/pinocchio_sync.py` |
| `control` | `CartesianImpedance`, `JointImpedance`, `AxisCompliance` | `mujoco_rl_bed/control` (logic unchanged) |
| `control` | `HybridAdmittance`, `WrenchCompensator` | `armos_control/cartesian_admittance.py` |
| `types` | `ArmHandles`, `PlantView`, `RobotState` | |
| `trajectory`, `control/spiral` | placeholders | |

Install: `pip install -e armos_core -c ../constraints.txt`.

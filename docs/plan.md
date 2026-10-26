Ready for review
Select text to add comments on the plan
Plan: one stack for sim, real FR3, and RL
Context
The work is split across three workspaces that don't share code or interfaces:

Workspace	What it is	Contract it speaks
~/mj_manipulation/mujoco_manipulation (armos_sim + armos_ros2)	Python MuJoCo sim of a dual Panda with position actuators and Pink IK. ArmosBridge runs at 100 Hz. Tk GUI.	Custom ArmosTelemetry/ArmosCartControl msgs, Euler xyzabc, raw F/T wrench
~/franka_ros2_ws/src (franka_ros2 v2.7.1 humble, crisp v2.4.1, franka_custom_control)	Single FR3. crisp torque controllers in ros2_control at 1 kHz. Python outer-loop nodes at 250 Hz. PyQt5 UI.	target_pose/current_pose (PoseStamped), external_wrench_in_base_frame, target_stiffness, controller_manager services
~/mujoco_RL/RL_Mujoco/mujoco_rl_bed	Headless Gymnasium + SB3. Torque-controlled Panda at 500 Hz with its own CartesianImpedance/AxisCompliance. Policies at 15–25 Hz.	No ROS; models saved as SB3 .zip + vecnormalize .pkl
The same code exists several times in parallel:

Two hybrid admittance controllers: armos_control/control/bimanual/cartesian_admittance.py, and franka_custom_control/bimanual/admittance_controller.py, which is a more complete port.
Three sets of rotation/kinematics helpers: robot_kinematics/kinematics/spatial_math.py, control/utils/kinematics.py, and mujoco_rl_bed/control/math_utils.py.
Two UIs.
Logging and plotting scattered across robot_kinematics/manyplot and test scripts.
cartesian_admittance.py is broken right now. It calls self._orientation_error, _limit_vector_norm, _rotvec_to_rot and _rot_to_euler, none of which exist any more. utils/kinematics.py also has an IndentationError at a stray @staticmethod.

Goals

A controller written once runs in sim, then on the real FR3 by flipping one flag.
One UI for both.
Training stays pure Python, with a clean path: policy → sim test → real robot.
Shared math, kinematics and controller laws.
Shared logging and plotting.
Decisions made

The Python sim mimics crisp's ROS contract.
One monorepo with two Python environments.
Single FR3 for now.
armos holds only control + simulation. All RL and policy code (training, obs/action terms, export, the policy-to-ROS bridge) lives in rl/. RL training uses the same low-level impedance law from armos in pure Python. Policy testing runs over ROS against the armos sim and the existing controllers, and then on the real robot.
Core idea: the real robot's contract becomes the standard contract
The real-robot side (franka_ros2 + crisp) is third-party and fixed. The sim is yours. So the crisp/franka topic set becomes the canonical contract, and the armos sim bridge adopts it. Controllers, UI, logger and RL policy bridge only ever see this contract, which makes use_sim:=true|false a pure launch-file switch.

 rl/ (pure Python training)              rl/deploy/rl_policy_bridge (ROS node, lives in rl/)
   TorqueEnv ──uses──┐                      ONNX policy → target_pose / target_stiffness
                     │                                   │
                     ▼                                   ▼
 armos_core.control.CartesianImpedance  ◄── same law ──► existing controllers + UI + recorder
                                                         │   canonical topics
                                            ┌────────────┴────────────┐
                                     use_sim:=true               use_sim:=false
                                  armos_sim_bridge             franka_bringup + crisp
                                  (inner loop = same           (real FR3 / fake hw)
                                   CartesianImpedance)         + wrench relay
Canonical topics (per namespace, default ""):

state: current_pose (PoseStamped), joint_states, ext_wrench_base (WrenchStamped, compensated, fixed sign)
command: target_pose (PoseStamped), target_joint (JointState), target_wrench, target_stiffness
control: controller_manager/{list,switch}_controller, gripper actions
Target layout (monorepo, e.g. ~/robot_stack, git)
robot_stack/
├── armos/                          # CONTROL + SIMULATION ONLY
│   ├── core/armos_core/            # pure Python, NO ROS, NO RL deps; numpy>=1.26,<3; pinocchio optional
│   │   ├── math/       so3.py (quat wxyz internal; Euler only at edges), se3.py, filters.py, limits.py
│   │   ├── kinematics/ model.py (Pinocchio FK/J/manipulability), mujoco_kin.py (mj_jacSite variant)
│   │   ├── control/    cartesian_impedance.py, joint_impedance.py, admittance.py (HybridAdmittance),
│   │   │               axis_compliance.py, wrench.py (payload comp + tare), spiral.py
│   │   └── types.py    RobotState, CartesianTarget (controller I/O dataclasses)
│   ├── sim/armos_sim/              # existing sim lib → FR3, + torque impedance mode
│   └── ros2/                       # colcon packages
│       ├── armos_interfaces/       # extras only (ControllerDebug msg, gripper srv); drop xyzabc msgs
│       ├── armos_control/          # ControllerNode base + admittance, spiral nodes (thin wrappers)
│       ├── armos_sim_bridge/       # replaces armos_ros_sim.py, speaks canonical contract
│       ├── armos_bringup/          # bringup.launch.py use_sim:=…, sim.yaml / real.yaml / controllers.yaml
│       ├── armos_ui/               # PyQt5 UI (from franka_custom_control/ui)
│       └── franka_custom_control/  # franka-specific launch/config/patch + wrench relay
├── tools/datalog/                  # logging + plotting (generic, no ROS in the library part)
│   ├── datalog/   Recorder, schema, plots, compare_runs, CLI `datalog-plot`
│   └── ros/datalog_ros/            # recorder node + rosbag→parquet converter
├── rl/                             # ALL policy work
│   ├── mujoco_rl_bed/              # training; controllers imported from armos_core
│   └── deploy/
│       ├── export_policy.py        # SB3 → ONNX + obs_norm + policy_spec.yaml
│       └── rl_policy_bridge/       # ament_python pkg: policy → canonical topics
└── third_party.repos               # franka_ros2, crisp_controllers, libfranka (vcs import)
colcon: colcon build --base-paths armos/ros2 tools/datalog/ros rl/deploy <third_party dir>.

Environments

~/.venvs/armos (py3.10, numpy 1.26, ROS Humble): pip install -e armos/core armos/sim tools/datalog, plus onnxruntime for the policy bridge.
mjtest conda (numpy 2.x): pip install -e armos/core tools/datalog rl/mujoco_rl_bed.
armos_core must import cleanly in both.
Pin one mujoco version (3.10.0, the RL pin) in both envs, so sim tests match training contact behaviour.
Dependency direction (enforced):

rl → armos_core, rl → datalog, and armos/* → datalog.
armos never imports anything from rl/.
Phase 0 — Housekeeping
Create the monorepo with git init. Move folders into the layout above and add third_party.repos. Fold robot_kinematics code into armos_core; it is currently a broken submodule.
Fix the broken cartesian_admittance.py / utils/kinematics.py refactor. The quick fix is plain functions called without self; the code moves into core in Phase 1.
Switch the sim robot to the Menagerie FR3 with a single-arm config, keeping dual Panda as an extra config. Fix the gripper actuator name bug (*_finger1_joint_motor → *_actuator8).
Make armos_control an ament_python package with a setup.py and entry points. Today control/bimanual and control/utils aren't installed.
Phase 1 — armos_core: shared math and controller laws
math/so3.py: merge robot_kinematics/kinematics/spatial_math.py, control/utils/kinematics.py, and the rotation parts of mujoco_rl_bed/control/math_utils.py (quat_error_world, apply_rotvec, damped_pinv, DampedLeastSquares). Use one convention: quaternion wxyz, rotvec errors.
control/cartesian_impedance.py: move mujoco_rl_bed/control/cartesian_impedance.py. This is the single low-level law used by RL training and by the armos sim's crisp-mimic inner loop.
Keep the zero-alloc design.
Inputs: J, bias torques, pose and twist (backend-agnostic).
Match crisp's CartesianController law and parameters (stiffness, damping, nullspace, torque rate limit) so sim ≈ real.
Same treatment for joint_impedance.py and axis_compliance.py.
control/admittance.py: HybridAdmittance as a pure class (reset, step(state, target, dt)).
Start from the franka version (franka_custom_control/bimanual/admittance_controller.py).
Fold in the armos version's semi-implicit damping and acc limits.
Gains and limits go in a HybridAdmittanceCfg dataclass.
control/wrench.py: payload comp and tare (from cartesian_admittance.py:_compensate_wrench). Used by the sim bridge.
control/spiral.py: from franka_custom_control/insertion/spiral_search.py, which is already a port of compliant_RL/terms.py::SpiralResidual.
Rules:
No rclpy, gymnasium, torch or SB3 imports in armos_core.
Every ROS node is a thin wrapper.
RL update:
mujoco_rl_bed/control/* becomes thin re-exports of, or direct imports from, armos_core.control.
FrankaPlant keeps the MuJoCo plumbing.
Delete the duplicated math.
Add unit tests for each law: admittance vs analytic MSD, quaternion round-trips, wrench comp at random poses, and an impedance step response.
Phase 2 — Canonical contract + one-flag switch
armos_sim_bridge (rewrite of armos_ros_sim.py::ArmosBridge):
Publishes current_pose, joint_states, ext_wrench_base (compensated via armos_core.control.wrench, same sign as the franka wrench) and /clock.
Subscribes target_pose, target_joint, target_wrench and target_stiffness with crisp semantics.
Torque mode in armos_sim: swap actuators to motors (as in mujoco_rl_bed/sim/scene.py::SceneBuilder) and run armos_core CartesianImpedance at ≥500 Hz with gains from franka_custom_control/config/crisp_controllers.yaml. Keep Pink IK as an extra mode.
Mock controller_manager: list_controllers/switch_controller with crisp's names (cartesian_controller, gravity_compensation, joint_impedance_controller), mapped to sim modes.
Add a lock between the ROS thread and the sim thread (it is currently unlocked).
Real side: a wrench relay in franka_custom_control remaps franka_robot_state_broadcaster/external_wrench_in_base_frame to ext_wrench_base. This is the only place sign or frame differences are fixed.
armos_bringup/bringup.launch.py use_sim:=true|false:
true: sim bridge with use_sim_time.
false: include franka_custom_control/launch/cartesian_control.launch.py, plus the relay.
Both: then the same controller nodes, UI and recorder from a shared controllers.yaml. Backend-only params live in sim.yaml / real.yaml.
armos_control/ControllerNode base:
Handles the state subscription, staleness timeout, timer, enable/disable service, ~/debug (an armos_interfaces/ControllerDebug msg with names[] + values[]), and hold-on-shutdown.
admittance_node.py wraps HybridAdmittance.
Delete both old admittance nodes.
Safety gate: with use_sim:=false, the launch loads the real.yaml limits and controllers start disabled; the UI enables them.
Phase 3 — One UI
Move the PyQt5 franka_control_ui into armos_ui; it already speaks the canonical contract.
Detect capabilities: error_recovery and gripper show only if their servers exist. In sim, the bridge provides a no-op error_recovery and a gripper action.
Port the Tk GUI's saved joint poses and jog, then retire armos_gui.py.
Add a controllers panel: enable/disable, live params, admittance targets and control_mode (replaces admittance_test_publisher.py).
Phase 4 — RL connection (everything in rl/, armos untouched)
Principle: the policy only produces targets for a controller that already exists in armos. Training reproduces that controller with the same armos_core law in pure Python. Testing sends the same targets over ROS to the real controller nodes.

Training (pure Python, unchanged workflow):

TorqueEnv uses armos_core.control.CartesianImpedance (and AxisCompliance / HybridAdmittance where a task needs an outer loop) as its inner loop.
Gains come from the same crisp_controllers.yaml/controllers.yaml values (loaded in a task cfg), so the trained controller matches the deployed one.
Deployable action terms ↔ armos controllers: each deployable action term has to map onto an existing controller's command interface. Record the mapping in rl/deploy/README and check it at export time.

RL action term	armos/crisp controller	Topics
delta_ee_pose, absolute_ee_pose, anchor_relative_pos (FORGE)	cartesian_controller	target_pose
delta_ee_pose_with_stiffness	cartesian_controller	target_pose + target_stiffness
axis_compliance_*, spiral_* (compliant_RL)	armos admittance node	admittance/target_pose, target_wrench, control_mode
Shared obs over a plain state:

The deployable obs terms in rl/ (FORGE, compliant_RL) are rewritten to compute from armos_core.types.RobotState (pose, twist, wrench, q, dq, target). This is a controller I/O type, so it fits armos's scope.
In training, FrankaPlant fills it. In deployment, the bridge fills it from canonical topics.
The obs code lives in rl/ and is numpy-1/2 compatible.
Export (rl/deploy/export_policy.py): SB3 .zip + vecnormalize.pkl → policy_dir/, containing:

policy.onnx (actor only; LSTM state as I/O)
obs_norm.npz
policy_spec.yaml: obs terms, action term → target controller, policy rate, controller gains, frames, mujoco version, git hash
rl/deploy/rl_policy_bridge (ament_python, runs in the ROS env with onnxruntime):

Subscribes to canonical state, builds obs, runs ONNX at the spec rate, and publishes targets to the controller named in the spec.
Calls switch_controller to activate it, and only publishes while enabled.
Test ladder (same artifact):

Headless RL eval (existing rl/evaluate.py), plus an --onnx flag to confirm the export matches SB3.
bringup use_sim:=true + rl_policy_bridge → the policy drives the existing controllers in armos sim, which can run faster than real time on sim time. The UI and recorder work as usual.
bringup use_sim:=false + the same bridge command → real FR3.
Share MJCF assets: RL and armos_sim load the same FR3 model and scene objects (peg/hole generated via the MjSpec code in mujoco_rl_bed/sim/assets.py; armos_sim takes a scene XML/spec hook), so step 2 matches step 1.

Phase 5 — Logging & plotting (tools/datalog, used by both armos and rl)
datalog.Recorder: in-process ring buffer → runs/<date>_<tag>/data.parquet + meta.yaml (backend, git hash, params snapshot).
Used by RL eval, the admittance closed-loop test, and the UI.
Replaces robot_kinematics/manyplot/telemetry_logger.py and the UI CSV logging.
datalog_ros/recorder_node: subscribes to canonical topics plus every */debug topic. It writes mcap rosbag2 + the same parquet schema on stop.
Plots:
plot_pose_tracking, plot_wrench, plot_joints, plot_frames_3d (from manyplot/frame_plot.py), plot_controller_debug.
compare_runs([rl_eval, sim_run, real_run]) overlays the runs on one time base.
CLI: datalog-plot runs/<id> [--compare …].
Live: the UI GRAPH tab, and PlotJuggler on canonical and debug topics.
What to reuse
New	Source
armos_core/math	robot_kinematics/kinematics/spatial_math.py, armos_control/control/utils/kinematics.py, mujoco_rl_bed/control/math_utils.py
armos_core/kinematics	armos_sim/sim/pinocchio_sync.py::_PinocchioSync
armos_core/control	mujoco_rl_bed/control/{cartesian_impedance,joint_impedance,axis_compliance}.py, franka_custom_control/bimanual/admittance_controller.py, armos_control/.../cartesian_admittance.py, franka_custom_control/insertion/spiral_search.py
sim torque mode	mujoco_rl_bed/sim/scene.py::SceneBuilder
armos_ui	franka_custom_control/ui/franka_control_ui/*
datalog	robot_kinematics/manyplot/*, plotting in test_cartesian_admittance.py
bringup	franka_custom_control/launch/cartesian_control.launch.py
policy eval / loading	mujoco_rl_bed/rl/evaluate.py (VecNormalize + config.json reload, _check_compatible)
Verification
Phase 1:
pytest armos/core in both envs (numpy 1 and 2).
The rl/mujoco_rl_bed/tests suite passes with the controllers imported from armos_core.
A fixed-seed FORGE eval reproduces the previous success rate. This proves the moved impedance law is unchanged.
Phase 2:
Port test_cartesian_admittance.py to the new bridge + admittance node and run it headless on sim time.
Run bringup use_sim:=false use_fake_hardware:=true to check the wiring.
Run on the real FR3 with low limits.
A check_contract.py script asserts that both backends expose identical canonical topics and types.
Phase 3: the same UI drives sim and real (controller switch, jog, gripper, admittance targets).
Phase 4:
ONNX output matches SB3 to within 1e-5 on recorded obs.
Success rates for ladder steps 1, 2 and 3 are reported side by side on the same seeds.
Going from step 2 to step 3 changes only use_sim.
grep confirms nothing under armos/ imports from rl/.
Phase 5: datalog-plot --compare overlays an RL eval, a sim run and a real run of the same task.
Add Comment
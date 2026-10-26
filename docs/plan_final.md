# Plan: armos_ws — one stack for armos sim, ROS 2 and the real FR3

> **Execution:** step by step, as directed by the user. Nothing runs ahead of the step being worked on.
>
> **Placeholder rule:** only code that already exists is moved, refactored or ported (from armos, `robot_kinematics`,
> `franka_custom_control` or `mujoco_rl_bed`). Anything that does not exist yet gets a **placeholder**: a directory
> holding only a `README.md` that states its purpose, its source material (if any) and `Status: placeholder`. ROS
> placeholder directories also get a `COLCON_IGNORE`. Placeholders are marked `[placeholder]` below.
>
> **Out of scope:** anything learning-related (policies, RL training/eval features, diffusion, classifiers).

## Context

| Source | Today |
|---|---|
| `~/mj_manipulation/mujoco_manipulation` | `armos_sim` (Python MuJoCo, dual Panda, position actuators + Pink IK, module named `sim`); `armos_ros2/pkgs/{armos_interfaces,armos_control}` (custom msgs, Tk GUI, sim bridge, admittance node); `robot_kinematics` (broken submodule). Not a git repo. |
| `~/franka_ros2_ws/src` | franka_ros2 v2.7.1 (local edits), crisp_controllers v2.4.1-3 (local header edits), libfranka 0.17.0, franka_description 2.8.1, `franka_custom_control` (crisp launch wrapper, admittance node, spiral insertion, PyQt5 UI; not a git repo) |
| `~/mujoco_RL/RL_Mujoco/mujoco_rl_bed` | Torque-controlled Panda: `sim/plant.py::FrankaPlant` (`mj_step1` → controller → `mj_step2`), `sim/scene.py::SceneBuilder` (swaps position servos for `motor`s via MjSpec), `control/{base,cartesian_impedance,joint_impedance,axis_compliance,math_utils}.py` (allocation-free hot loop). Runs in conda env `dqn`. Has uncommitted work. |

**Decisions**
- **Workspace:** new git repo + colcon workspace at `~/armos_ws`, created by **copying**. The old workspaces, `~/.venvs/armos` and `mujoco_RL` stay untouched as the fallback.
- **Three parts:** `armos_core` (math, kinematics, control laws, types; no ROS), `armos_sim` (MuJoCo wrapper, usable on its own and by `mujoco_rl_bed`; no ROS), `armos_ros2` (ROS 2 packages + franka third-party).
- **The torque laws come from `mujoco_rl_bed`**, not a fresh port of crisp. They are already tested and RL results depend on them.
- **Torque modes for both FR3 and the dual-Panda config.**
- **Dependencies: `mujoco_rl_bed` dominates** (see *Dependencies*).
- **Control loop split:** torque-level inner loops (joint/Cartesian impedance) run inside the sim process (sim) or in crisp/`ros2_control` C++ (real FR3, 1 kHz). ROS nodes in `armos_control` are outer loops only (admittance → `target_pose`, etc.).
- **Per-arm topics** = one namespace per arm (`/left/…`), using crisp's topic names inside the namespace, so the real FR3 can later run under the same namespace.

`cartesian_admittance.py` is broken right now: it calls `self._orientation_error`, `_limit_vector_norm`, `_rotvec_to_rot` and `_rot_to_euler`, none of which exist any more. `utils/kinematics.py` has a mis-indented `@staticmethod` block and `self`-style functions at module level.

---

## Dependencies

`mujoco_rl_bed/requirements-lock.txt` is the reference: py3.10, **`mujoco==3.10.0`** (pinned: 3.15 changes contact forces 3–4× in the peg tasks), **`numpy==2.2.6`**.

ROS Humble's binary packages (rclpy message typesupport, ROS pinocchio) are built against numpy 1.x and do not load under numpy 2. So there are two envs, and the armos Python packages must work in both:

| Env | Used for | numpy | mujoco | pinocchio / pink |
|---|---|---|---|---|
| conda `dqn` (existing, RL lock) | `mujoco_rl_bed`, standalone `armos_sim` | 2.2.6 | 3.10.0 | pip `pin` + `pin-pink` via `armos_sim[ik]` (only if IK modes are needed) |
| `~/.venvs/armos_ws` (new, system py3.10) | ROS 2 nodes, sim bridge | 1.26.4 (forced by Humble) | **3.10.0** (matches RL) | ROS pinocchio; `pin-pink==4.2.0` + its deps, all `--no-deps` (`requirements-ros-venv.txt`) |

- `~/armos_ws/constraints.txt` holds the shared pin: `mujoco==3.10.0`. Both envs install with `-c constraints.txt`.
- armos packages declare `numpy>=1.24` (RL's floor) with **no upper bound**, and `mujoco>=3.10`. Code must run on numpy 1.26 and 2.x (no removed 1.x aliases, no 2-only APIs).
- `armos_sim` dependencies shrink to `mujoco`, `numpy`, `PyYAML`, `armos-core`. Extras: `ik = [pin, pin-pink, quadprog]`, `convert = [trimesh, pycollada]` (only `urdf2xml.py` uses trimesh), `dev = [pytest, ruff]`. `opencv-contrib-python` and `gdown` are dropped (nothing imports them).
- armos currently runs on mujoco 3.11.0; the move to 3.10.0 is verified in Phase 0.
- `~/.venvs/armos_ws/bin/activate` sets `PYTHONNOUSERSITE=1` (Phase 2). Pink's deps (`qpsolvers`, `loop-rate-limiters`), mujoco's (`glfw`, `PyOpenGL`, `absl-py`, `etils`) and PyYAML 6 had only been found in `~/.local`; they are now pinned in the venv (`requirements-ros-venv.txt`, versions as they were in `~/.local`), plus `scipy==1.11.4` (system scipy 1.8 supports numpy < 1.25 only).
---

## Final layout

```
~/armos_ws/                               (git; colcon workspace root)
├── armos_core/                           # pip "armos-core", module armos_core, no ROS
│   ├── armos_core/
│   │   ├── math/                         # so3, se3, filters, limits, mj_math (from RL math_utils)
│   │   ├── kinematics/                   # pinocchio_arm.py, mujoco_kin.py
│   │   ├── trajectory/                   # [placeholder]
│   │   ├── control/                      # base, cartesian_impedance, joint_impedance, axis_compliance,
│   │   │                                 #   hybrid_admittance, wrench_compensator; spiral [placeholder]
│   │   └── types.py
│   └── tests/                            # control tests moved from mujoco_rl_bed; rest [placeholder]
├── armos_sim/                            # pip "armos-sim", module armos_sim, no ROS
│   ├── armos_sim/
│   │   ├── model/                        # loader, mj_model, actuation swap (from SceneBuilder), urdf2xml
│   │   ├── assets/                       # franka_emika_panda, franka_fr3, duman
│   │   ├── plant.py                      # fast core (from FrankaPlant, per-arm views)
│   │   ├── modes/                        # dispatcher (was controller.py), position modes, torque modes
│   │   ├── runtime/                      # simulator, telemetry, viewer_frames
│   │   └── types.py                      # was sim_types.py
│   └── examples/
├── armos_ros2/                           # colcon base path
│   ├── armos_interfaces/
│   ├── armos_control/                    # ament_python: sim bridge (robot-compatible topics), admittance node
│   │                                     #   (sim + real), launch/admittance.launch.py (use_sim switch)
│   ├── armos_ui/                         # ament_python: Tk GUI (moved from armos_control); PyQt merge [placeholder]
│   └── franka/
│       ├── franka_custom_control/        # tracked, copied as-is
│       ├── franka_ros2/ crisp_controllers/ libfranka/ franka_description/   # gitignored, from third_party.repos
│       └── patches/                      # franka_ros2.patch, crisp_controllers.patch
├── legacy/robot_kinematics/              # absorbed into armos_core, then deleted
├── third_party.repos                     # pinned commits for armos_ros2/franka/*
├── constraints.txt                       # mujoco==3.10.0
├── requirements-ros-venv.txt             # exact --no-deps pins of ~/.venvs/armos_ws
├── docs/                                 # README (setup commands), plan_final.md, baselines/
└── .gitignore                            # build/ install/ log/ armos_ros2/franka/{franka_ros2,crisp_controllers,libfranka,franka_description}/
                                          # __pycache__ *.egg-info MUJOCO_LOG.TXT
```

Build (venv active): `python -m colcon build --base-paths armos_ros2 --symlink-install`. Plain `colcon` runs
under `/usr/bin/python3` and writes system-python shebangs into the entry scripts.

---

## Phase 0 — Workspace and housekeeping

**0.1 Create the repo.** `mkdir ~/armos_ws && git init`, add `.gitignore` and `constraints.txt`.

**0.2 Copy armos.**
- `armos_sim` → `armos_sim/` (without `armos.egg-info`, `MUJOCO_LOG.TXT`, `__pycache__`).
- `armos_ros2/pkgs/*` → `armos_ros2/`.
- `robot_kinematics` → `legacy/robot_kinematics` (without the dangling `.git`).
- `README.md`, `plan.md`, `plan_final.md` → `docs/`.

**0.3 Copy franka.**
- `~/franka_ros2_ws/src/franka_custom_control` → `armos_ros2/franka/franka_custom_control`.
- `cp -a` franka_ros2, crisp_controllers, libfranka, franka_description → `armos_ros2/franka/`.
- For each git repo: `git diff HEAD --binary > armos_ros2/franka/patches/<repo>.patch` (captures franka_ros2's deleted packages, `franka.config.yaml`, crisp's header edits).
- Write `third_party.repos` (exact commit hashes). The restore commands (`vcs import` + `git apply`) go in `docs/README.md`; no setup script.
- `franka_hardware_libfranka_lt_0_18.patch` is already contained in the franka_ros2 diff; record that in the README.

**0.4 Python envs** (see *Dependencies*).
- New `~/.venvs/armos_ws` (system py3.10): the pins from `~/.venvs/armos` (`pip freeze`, old `armos` editable entry removed), with `mujoco==3.10.0` from `constraints.txt`; `pin-pink==4.2.0 --no-deps`.
- `pip install -e armos_sim -c constraints.txt` (and `armos_core` once it exists).
- `dqn` is not touched.

**0.5 Rename the sim module `sim` → `armos_sim`.**
- Move `armos_sim/sim/` → `armos_sim/armos_sim/`. Update `pyproject.toml`: name `armos-sim`, packages include, dependencies/extras per *Dependencies*, drop the "CyRo tri-arm" wording.
- Remove the `sys.path` hack in `__init__.py` and fix its stale `sim_cyro` docstring.
- Rewrite imports `from sim.` → `from armos_sim.` in `armos_ros_sim.py`, `test_cartesian_admittance.py`, the demos and `examples/`. Fix or delete the stale `sim_cyro` import in `examples/dual_arm_task_pose.py`.
- Move `assets/` inside the package so `importlib.resources` finds the models.
- Fix the missing default config in `CyRoSimulator` (`config_cyro_audi.yaml`) to point to `fr3.yaml`.

**0.6 FR3 model.**
- Copy Menagerie `franka_fr3` (Apache-2.0, LICENSE included) from `~/Active-Perception/robot_descriptions/franka_fr3` → `armos_sim/assets/franka_fr3/`.
- Build `fr3_hand.xml` + `scene.xml`: attach the Panda hand at `attachment_site` (mass/COM must match the payload defaults in `cartesian_admittance.py`, otherwise update them); `ft_site` + `ft_force`/`ft_torque` sensors at the flange; `fr3_hand_tcp` site 0.1034 m along flange z; joint names `fr3_joint1..7`.
- Loader config `fr3.yaml`: `start_body: base`, `ee_site: fr3_hand_tcp`, sensors, `q_home = [0, -π/4, 0, -3π/4, 0, π/2, π/4]`, gripper actuator names that exist. It becomes the default; arm key stays `left`.
- Keep `dual_franka.yaml`; fix its gripper actuator names (`*_robot_gripper_finger1_joint_motor` → `*_actuator8`).
- `_RobotLoader` raises when an actuator name doesn't resolve (`mj_name2id == -1`).

**0.7 Convert `armos_control` to `ament_python`.**
- Layout: `armos_control/armos_control/{sim_bridge.py (was armos_ros_sim.py), gui.py, cartesian_admittance.py, utils/kinematics.py}`, `setup.py`, `setup.cfg`, `resource/armos_control`, `test/test_cartesian_admittance.py`.
- `package.xml`: fix the pymoveit2 copy-paste (description, maintainer); deps `rclpy`, `armos_interfaces`, `geometry_msgs`, `std_msgs`.
- Entry points keep the old names: `armos_ros_sim`, `armos_gui`, `cartesian_admittance`. Install `config/*.yaml` via `data_files`.

**0.8 Fix the broken admittance refactor** (minimal, behaviour unchanged).
- `utils/kinematics.py`: `rotvec_to_rot`, `rot_to_euler`, `limit_vector_norm`, `orientation_error(desired, current)` become plain module functions.
- `cartesian_admittance.py` imports them instead of the `self._…` calls, and uses `rot_to_euler` for the duplicated quaternion → Euler block in `_on_target_pose`.

**0.9 Placeholders.** Create `armos_ros2/armos_ui/` (README + `COLCON_IGNORE`; sources to merge later: `franka_custom_control/ui`, `armos_gui.py`).
*Phase 2:* `armos_ui` became an ament_python package holding the Tk GUI (`ros2 run armos_ui armos_gui`); the PyQt merge stays a placeholder.

**0.10 Build and baseline** (in `~/.venvs/armos_ws`, mujoco 3.10.0).
- `source /opt/ros/humble/setup.bash`, `colcon build` (libfranka from source as before).
- `ros2 run armos_control armos_ros_sim` with `fr3.yaml`: viewer up, `/armos/left/telemetry` published.
- `test/test_cartesian_admittance.py --headless` on FR3; save `log.csv` + `plot.png` to `docs/baselines/admittance_phase0/` as the regression reference.

---

## Phase 1 — `armos_core`: math, kinematics, control laws, types

**Package:** `armos_core/pyproject.toml`: name `armos-core`, module `armos_core`; deps `numpy>=1.24`, `mujoco>=3.10` (the RL laws use `mju_*` and `MjData`), `PyYAML`; extras `dev=[pytest]`. Pinocchio is imported lazily. **No rclpy.**

Every item below is moved or merged from existing code; sources in parentheses.

**1.1 `math/`**
- `so3.py` (quaternions **wxyz**): `quat_to_rot`, `rot_to_quat`, `quat_mul`, `quat_conj`, `rotvec_to_rot`, `rot_to_rotvec`, `orientation_error`, `skew`, `euler_xyz_to_rot` / `rot_to_euler_xyz`, `rotX/Y/Z` (from `legacy/robot_kinematics/kinematics/spatial_math.py`, `armos_control/utils/kinematics.py`, the duplicate `_wxyz_to_R` in `pinocchio_sync.py` and `telemetry.py`).
- `se3.py`: `make_T`, `inv_T`, `adjoint`, `transform_twist`, `transform_wrench` (from `spatial_math.py::adjoint_transform`, `transform_vel`).
- `filters.py`: `LowPass1` (from `cartesian_admittance.py`).
- `limits.py`: `limit_norm` (from `utils/kinematics.py`).
- `mj_math.py`: `critical_damping`, `damped_pinv`, `DampedLeastSquares`, `QuatBuffers`, `quat_error_world`, `apply_rotvec`, `GravityTorque` (from `mujoco_rl_bed/control/math_utils.py`, unchanged).

**1.2 `kinematics/`**
- `pinocchio_arm.py`: `PinocchioArm` (from `armos_sim/pinocchio_sync.py::_PinocchioSync`): fk, jacobian (LOCAL_WORLD_ALIGNED), manipulability, singularity distance. `armos_sim/pinocchio_sync.py` becomes a thin adapter over it.
- `mujoco_kin.py`: the same interface on `MjModel`/`MjData` (`mj_jacSite`, `qfrc_bias`, `mj_fullM`), from the accessors in `FrankaPlant` and `armos_sim/mj_model.py`.

**1.3 `types.py`**
- `RobotState` (t, q, dq, tau, tcp pose wxyz, tcp twist, ext wrench).
- `PlantView` Protocol: the attributes the laws read (`q`, `qd`, `jacobian`, `qfrc_bias`, `ee_pos`, `ee_rot`, `ee_quat`, `ee_vel`, `wrist_wrench`, `torque_limits`, `dt`, …), taken from `FrankaPlant`'s public API so `mujoco_rl_bed` keeps working.

**1.4 `control/`**
- From `mujoco_rl_bed/control/`, **logic unchanged** (only imports and the plant type hint change to `PlantView`): `base.py` (`Controller`, `ControllerCfg`), `cartesian_impedance.py`, `joint_impedance.py`, `axis_compliance.py`.
- `hybrid_admittance.py`: `HybridAdmittance` + `Cfg` (from the `cartesian_admittance.py` law: per-axis pose/force mask, virtual M/K/D, velocity/acc limits, force deadband, max deviation). Interface `step(state, target, dt) -> virtual pose`; pure numpy.
- `wrench_compensator.py`: `WrenchCompensator` (from `cartesian_admittance.py::_compensate_wrench` + tare).
- `spiral/` [placeholder] (source: `franka_custom_control/insertion/spiral_search.py`).
- `AxisCompliance` and `HybridAdmittance` stay separate for now: RL results depend on `AxisCompliance` exactly.

**1.5 Switch armos code to `armos_core`.**
- `armos_control/cartesian_admittance.py` becomes a thin node over `HybridAdmittance` + `WrenchCompensator`; hardcoded gains become ROS params with the same defaults; topics unchanged. Delete `utils/kinematics.py`.
- `armos_sim` (`telemetry.py`, `pinocchio_sync.py`, `controller.py`) and `sim_bridge.py` use `armos_core.math`.
- `franka_custom_control` is not changed.

**1.6 Tests.** `mujoco_rl_bed/tests/test_controller.py` moves to `armos_core/tests/` (imports updated). Other armos_core tests: [placeholder].

---

## Phase 2 — `armos_sim` refactor + torque modes

*Status 2026-10-18: done; checks in 2.6 pass (`armos_sim/tests/test_modes.py`, `test_controller.py`; admittance baseline `docs/baselines/admittance_phase2`).*

**2.1 Directory split** (moves only):
- `model/`: `loader.py`, `mj_model.py`, `urdf2xml.py`.
- `modes/`: `controller.py` → `modes/dispatcher.py` (`_ModeLatch`, `_ArmController`), `modes/position.py` (`_PinkSolver`, joint pos/vel, task pose/vel), `modes/torque.py` (2.4).
- `runtime/`: `simulator.py`, `telemetry.py`, `viewer_frames.py`.
- `sim_types.py` → `types.py`.

**2.2 Actuation is a per-arm load option.** `actuation: position | torque` in the loader YAML. `torque` swaps that arm's position servos for `motor`s with the servo's `forcerange` (logic from `mujoco_rl_bed/sim/scene.py::SceneBuilder`, generalized to several arms; the gripper actuator is left as is). Default `torque` for `fr3.yaml` and `dual_franka.yaml`, `position` for the others. Modes are checked against the arm's actuation.
  The ROS bridge's own config copies (`armos_control/config/{fr3,dual_franka}.yaml`) pin `actuation: position`: the bridge maps its messages onto position modes only until the ROS contract step.

**2.3 `plant.py`** (from `FrankaPlant`). One `ArmPlant` per arm over the shared `MjModel`/`MjData`: indices resolved once at load (no `mj_name2id` in the loop), preallocated buffers, implements `PlantView`. The tick is `mj_step1` → each torque-mode arm's `controller.torque(plant)` → `mj_step2`. `mujoco_rl_bed`'s single-arm use is one `ArmPlant`.

**2.4 Modes.**

| Actuation | Mode | Law |
|---|---|---|
| position | `JOINT_POS`, `JOINT_VEL`, `TASK_POSE` (Pink), `TASK_VEL` | existing |
| torque | `JOINT_TORQUE` | raw τ + `qfrc_bias` (existing) |
| torque | `JOINT_IMPEDANCE` | `armos_core` `JointImpedance` |
| torque | `CARTESIAN_IMPEDANCE` | `armos_core` `CartesianImpedance` |
| torque | `CARTESIAN_ADMITTANCE` | `HybridAdmittance` → virtual pose → `CartesianImpedance` |

- Torque laws run every physics tick and use the MuJoCo kinematics (`PlantView`), not Pinocchio.
- Public API additions on the simulator: `set_impedance_target(arm, pos, quat)`, `set_impedance_gains(arm, …)`, `set_admittance_target(arm, pose, wrench, mask)`, `tare(arm)`.

**2.5 Rates and hot path.**
- `_CONTROL_HZ` / `_N_SUBSTEPS` constants become config: `physics.timestep` and per-mode `physics.decimation` (`control`: position modes, telemetry, callbacks, `step()`; `admittance`: outer loop; torque laws every physics step). Without a `physics` block: MJCF timestep, 2 steps per tick (today's behaviour). `fr3.yaml` / `dual_franka.yaml`: 500 Hz physics, 100 Hz control, 250 Hz admittance.
- Pinocchio syncs only when a mode or telemetry needs it (today it runs twice per tick).
- Telemetry is collected on demand / at a configurable stride, not every tick.
- Headless `realtime_factor: 0` disables `time.sleep` pacing. `logging.basicConfig` moves out of the constructor. `mujoco.viewer`, `pinocchio`, `pink` are imported lazily.
- NaN guard in torque modes: restore the last finite state, zero velocity, switch the arm to `JOINT_IMPEDANCE` hold. Also triggered by MuJoCo's bad-qpos/qvel/qacc warnings: MuJoCo resets `MjData` to `qpos0` itself, so the state is never NaN after the step.

**2.6 Checks.**
- Phase 0 admittance baseline still matches (position modes unchanged).
- FR3 and dual Panda: `JOINT_IMPEDANCE` holds `q_home` under gravity; `CARTESIAN_IMPEDANCE` holds a pose; `CARTESIAN_ADMITTANCE` complies to a push in the viewer.
- `armos_sim` imports and runs headless in `dqn` (numpy 2) and in `~/.venvs/armos_ws` (numpy 1.26).

---

## Unified admittance (2026-10-18, done)

One admittance law, one node, shared topics for the sim and the real FR3:
- `armos_core.control.HybridAdmittance` is the superset of the two former copies (the real node in
  `franka_custom_control/bimanual/admittance_controller.py`, the sim node `armos_control/cartesian_admittance.py`):
  rotation-matrix state, `axes: tcp | base`, force deadband, max deviation. `ExternalWrench` holds the
  real node's tare (stillness check), filter and sign; `to_axes` the moment transport to the TCP.
  Replay: the new node reproduces the old real node to 1e-15 (tcp axes); base axes reproduce the old sim law.
- `armos_control/admittance.py` (`admittance_controller`): the real node's topics and parameters, ROS clock
  (sim time in sim), parameter `axes`. Both old nodes are deleted. Parameters: `config/admittance.yaml`
  (FR3 values, used for sim and real).
- Switching: the node does not know sim from real. `ros2 launch armos_control admittance.launch.py
  use_sim:=true|false namespace:=... axes:=...`: sim = armos bridge publishing the robot's topics
  (`current_pose`, `joint_states`, `franka_robot_state_broadcaster/{external_wrench_in_base_frame,
  current_pose}`, `/clock`) and running crisp's Cartesian impedance gains on `target_pose` (torque
  actuation); real = `franka_custom_control/cartesian_control.launch.py`. Namespace: `""` for one arm,
  `<ns>/<arm>` for a multi-arm sim.
- `spiral_search.py` / `spiral_insertion_demo.py` use the ROS clock (unchanged on the robot, work on sim time).
- Sim check `armos_control/test/test_admittance_sim.py --axes tcp|base`: 21/23 with the FR3 parameters
  (`docs/baselines/admittance_unified_{tcp,base}`); the 2 failures are parameter-driven (touch-down
  overshoot at the 50 mm/s force-axis approach, 5.6 mm inner-loop spring-back), to be decided.
  2026-10-20: the bridge had run the inner loop at the law's default 600 N/m (crisp's gains were reset on
  mode entry); with that fixed (crisp's 1500/500): 22/23 on both axes, spring-back passes, touch-down peak
  16.1 N still fails (`result_gainfix.txt`).
- Not emulated yet in the bridge: crisp's `target_wrench`, `target_joint`, `target_stiffness`.

## Later — planned when the user gets there

All [placeholder] until then.
- **ROS contract + sim/real switch:** per-arm namespaces with crisp topic names and the `use_sim` launch flag are done for the admittance path (above); still open: the remaining crisp topics in the bridge and a mode switch that mirrors `controller_manager/switch_controller`.
- **`armos_control` outer loops:** admittance node done (above); spiral node (from `franka_custom_control/insertion`) and the rest of `franka_custom_control/bimanual` still to absorb.
- **`armos_ui`:** merge `franka_custom_control/ui` (PyQt5) and `armos_gui.py`; SIM/REAL badge, safety gate.
- **Logging / plotting:** from `legacy/robot_kinematics/manyplot`; then delete `legacy/`.
- **`armos_core/trajectory`**, **`armos_core/tests`** beyond the moved controller tests.

---

## Verification

**Phase 0**
1. `colcon build` of `~/armos_ws` succeeds (libfranka, franka_ros2, crisp, armos).
2. `ros2 launch franka_custom_control cartesian_control.launch.py use_fake_hardware:=true` comes up from the new workspace.
3. `ros2 run armos_control armos_ros_sim --ros-args -p config_path:=fr3.yaml` shows FR3 + hand on mujoco 3.10.0, publishes telemetry, gripper moves.
4. `test_cartesian_admittance.py --headless` passes; baseline saved.
5. The old `~/mj_manipulation` and `~/franka_ros2_ws` still build and run.

**Phase 1**
1. `pytest armos_core` passes in both `dqn` and `~/.venvs/armos_ws`.
2. The admittance closed-loop test passes on `armos_core` and matches the Phase 0 baseline within tolerance.
3. `grep -r "import rclpy" armos_core armos_sim` finds nothing.

**Phase 2:** the checks in 2.6.

---

## Peg/hole sim scene and bridge gains (2026-10-20)

- Sim: `armos_control/config/peg_hole/fr3_peg_hole.yaml` (FR3 + 8 mm square peg + 9 mm square hole via the loader's new `attach:`).
- Bridge: gains as crisp's parameters (`task.k_*`, `task.d_*`, live), gains kept across mode switches (were reset
  to the law defaults), mode cache follows the sim's own switches, `frictionloss_scale` parameter.

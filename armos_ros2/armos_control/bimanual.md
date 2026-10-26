# Bimanual controller: architecture

Status: design only, nothing implemented. Two open questions at the end decide the rest.

## Goal

One controller for two arms with three modes:

| Mode | Behaviour |
|---|---|
| independent | each arm follows its own target |
| temporal sync | both arms reach their targets at the same time; coupled phase, so a lagging arm slows the other |
| spatial sync | object-level compliance: a force on the held object, or a push on either arm, moves the object (control law supplied by the user) |

It runs unchanged on the dual-Panda sim and on two real FR3s, like the admittance controller.

## Ground rules

**One publisher per `target_pose`.** crisp's `cartesian_controller` ignores a topic with more than one
publisher (the control UI already refuses to enable in that case). So for each arm exactly one node
writes `<arm>/target_pose`: either the per-arm admittance node or the bimanual node, never both.

**Independent mode does not need the admittance node.** The bimanual node can publish straight to each
arm's impedance controller (`<arm>/target_pose`). The impedance alone already makes the arm compliant
(crisp: 1500 / 1500 / 500 N/m, 30 Nm/rad). The admittance layer is only needed for what the impedance
cannot give: force-regulated axes, compliance softer than the impedance stiffness, or its safety limits
(max deviation, force-target timeout). It also adds lag and caps speed at `limits.max_vel` (0.05 m/s),
which would break temporal sync. When per-arm force axes are wanted, the bimanual node runs
`armos_core.HybridAdmittance` per arm internally rather than chaining to the external admittance nodes.

**Same topics for sim and real.** The node only uses the robot-compatible topics, so the sim bridge
(`dual_franka.yaml`, namespaces `/left`, `/right`) and two FR3s in namespaces `left` / `right` are
interchangeable.

## Architecture

```
                    bimanual node  (armos_control/bimanual.py, one process, fixed rate, ROS clock)
  /left/current_pose ─┐   ┌──────────────────────────────────────────────┐
  /left/franka_.../   │   │ inputs: both arms' pose + external wrench     │
   external_wrench    ├──►│   (ExternalWrench tare/filter per arm),       │
  /right/... same     │   │   transformed to one world frame              │
  ~/mode (service)  ──┘   │                                              │
                          │ mode (one active):                           │
                          │   independent ─ per-arm targets              │
                          │   temporal    ─ SyncedTrajectory (shared s)  │
                          │   spatial     ─ ObjectAdmittance (user law)  │
                          │ safety: stale/limits/robot error → hold BOTH │
                          └───────────────┬───────────────┬──────────────┘
                                          ▼               ▼
                                 /left/target_pose   /right/target_pose
                                          ▼               ▼
                       crisp cartesian_controller (real)  or  sim bridge CARTESIAN_IMPEDANCE (sim)
```

Per arm (relative to its namespace):

| Direction | Topic | Type |
|---|---|---|
| in | `current_pose` | PoseStamped, TCP in that arm's `base` |
| in | `franka_robot_state_broadcaster/external_wrench_in_base_frame` | WrenchStamped |
| in | `franka_robot_state_broadcaster/current_pose` | PoseStamped, wrench reference point |
| out | `target_pose` | PoseStamped, to the impedance controller |

## Modes

### Independent
Each arm's target (from the user or a client topic) is rate-limited and streamed to its `target_pose`.
Optional per-arm force axes through an internal `HybridAdmittance` per arm.

### Temporal sync
Both arms follow their paths on one shared phase s ∈ [0, 1]: p_i(t) = path_i(s(t)).
- Duration T = max over the two arms of the time each needs under its velocity / acceleration limits,
  so the slower arm sets the pace.
- Coupled phase: ṡ = (1 / T) · g(e_L, e_R). When either arm's tracking error grows (blocked, pushed),
  the common phase slows, so the arms stay together during the motion, not only at the start.
- Output goes straight to the impedance controllers. Compliance during the motion comes from the
  impedance stiffness, not from the admittance node (its speed cap would distort the timing).

### Spatial sync
The user's object-level law produces a virtual object pose; each arm target follows rigidly:
T_arm_i = T_obj · T_obj→grasp_i.
- Inputs: both arms' wrenches, moved to their grasp points (`armos_core.control.external_wrench.to_axes`
  does the moment transport) and combined through the grasp matrix. A push on the object or on one
  arm shows up in that arm's wrench and moves the object; internal (squeeze) force is part of the law.
- The grasp transforms T_obj→grasp_i are captured from the measured poses on mode entry, after both
  grippers have closed.
- The per-arm admittance must not run underneath: the compliance would be integrated twice.

## Code layout

| Part | Location | Content |
|---|---|---|
| temporal sync | `armos_core/trajectory/synced.py` | `SyncedTrajectory`: shared phase, duration from both arms' limits, phase coupling (fills the `trajectory/` placeholder) |
| spatial sync | `armos_core/control/object_admittance.py` | `ObjectAdmittance`: the user's law, `step(wrenches, dt, poses) -> poses` like `HybridAdmittance` |
| ROS node | `armos_control/armos_control/bimanual.py` | I/O, mode state machine, safety, holding both arms; thin like `admittance.py` |
| parameters | `armos_control/config/bimanual.yaml` | rates, limits, mode gains, arm base poses |
| launch | `armos_control/launch/bimanual.launch.py` | same `use_sim` switch as `admittance.launch.py` |

The laws stay pure numpy (no ROS), testable in `armos_core/tests` and runnable in the sim without ROS.

## Design points

1. **One node with three modes**, not three nodes. All modes need exclusive ownership of both
   `target_pose` topics, and switching must be bumpless (the new mode starts from the measured poses).
   Three separate nodes would need a mux / arbiter in front of `target_pose` to get the same behaviour.
2. **Common world frame.** Each arm's feedback is in its own `base`. The node needs each base's pose in
   the world: from the sim config (`base_world` per arm in the dual-Panda model) in sim, from a
   calibration (parameters or a static TF) on the real robots.
3. **Time alignment and safety.** Both arms' feedback is checked for freshness on one clock; if either
   goes stale, both arms hold. On the real robots a reflex or error on one FR3 must stop the other too
   (in spatial mode they hold the same object), so the node watches both robots' state and their
   controller status.
4. **Rate limiting.** Targets are streamed at the node rate with per-step limits on position and
   rotation change, as the control UI does for its pose targets.

## Open questions

1. **Output of the spatial-sync law.**
   - Motion (virtual object pose or velocity, admittance-style): fits this design, Python at 250 Hz
     feeding crisp's impedance.
   - Force / torque (object-level impedance computing a wrench per arm): belongs in the torque loop. On
     the robot that is a C++ ros2_control controller next to crisp at 1 kHz; in the sim a new torque
     mode in `armos_sim`. A Python node over ROS topics cannot close a torque loop.
2. **One node with three modes, or three nodes behind a mux.** Recommended: one node (point 1 above).

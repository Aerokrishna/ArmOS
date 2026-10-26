"""Per-arm controllers.

- `dispatcher`: `make_controller` (picks by `ArmConfig.actuation`), `_ModeLatch`, `_ArmController` (position arms).
- `position`: position-mode joint references (JOINT_POS/VEL, TASK_POSE via Pink, TASK_VEL via DLS).
- `torque`: `_TorqueArmController`, the armos_core torque laws (motors).

Each controller validates the modes it accepts (`armos_sim.types.POSITION_MODES` / `TORQUE_MODES`).
"""

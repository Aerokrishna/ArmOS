# armos_ui

ROS 2 user interfaces for armos (ament_python).

| Executable | Module | What |
|---|---|---|
| `armos_gui` | `armos_ui/gui.py` (moved from `armos_control`) | Tk control desk for one arm of the sim bridge: joint sliders, Cartesian jog, saved joint poses, telemetry CSV logging. Talks to `/armos/<arm>/{joint_control,cart_control,telemetry}`. |

```bash
ros2 run armos_ui armos_gui
```

## Planned [placeholder]

One UI for the sim and the real FR3, merging:
- `armos_ros2/franka/franka_custom_control/ui` (PyQt5: COMMAND / MONITOR / GRAPH tabs, controller switching, gripper, error recovery)
- `armos_ui/gui.py` (Tk: saved joint poses, jog)

with a SIM/REAL badge and a safety gate.

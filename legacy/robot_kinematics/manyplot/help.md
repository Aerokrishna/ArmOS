contains plotting utils
- plot frames
- plot trajectories
- log and plot robot data
- a user interface to plot frames
- move the frames and run a live plot

Telemetry logging:
- `telemetry_logger.py` contains `TelemetryLogger`.
- Call `start_collection(joint_pose, joint_torque, timestamp)` for joint logs.
- Call `start_task_space_collection(task_pose, task_wrench, timestamp)` for task-space logs.
- Save joint logs with `save_csv(path)` and task logs with `save_task_csv(path)`.

Telemetry plotting:
- Run `telemetry_plot.py <csv_file>` to plot position, velocity, acceleration, jerk, and force/torque.
- Pass `--save output.png` to write the figure to disk.
  
# armos_ws commands

Everything is run from `~/armos_ws` in a ROS terminal (sim, real robot, colcon).

## 1. New terminal

`~/.bashrc` already sources `/opt/ros/humble` (and microros_ws) and auto-activates conda `base`. Leave conda
out of this terminal.

**ROS terminal** (sim, real FR3, colcon build):
```bash
conda deactivate                                # base is auto-activated by .bashrc
source /opt/ros/humble/setup.bash               # already in .bashrc; harmless to repeat
source ~/franka_ros2_ws/install/setup.bash      # crisp_controllers, franka_bringup, libfranka, franka_hardware
source ~/armos_ws/install/setup.bash            # armos_control, armos_ui, armos_interfaces, franka_custom_control
source ~/.venvs/armos_ws/bin/activate           # python 3.10, numpy 1.26, mujoco 3.10, armos_core/armos_sim
cd ~/armos_ws
```
The order matters. `~/armos_ws` must come after `~/franka_ros2_ws`: its `franka_custom_control` (the updated
spiral) replaces the older copy in `~/franka_ros2_ws`. Check with
`ros2 pkg prefix franka_custom_control` (must print `~/armos_ws/install/...`).

Environments:

| Env | Use it for | Don't use it for |
|---|---|---|
| `~/.venvs/armos_ws` | everything ROS (sim bridge, admittance, colcon) | torch / SB3 |
| `~/.venvs/armos` (`armos` alias in .bashrc) | old armos, fallback | — |

## 2. Build (ROS terminal)

```bash
cd ~/armos_ws
python -m colcon build --base-paths armos_ros2 --symlink-install \
    --packages-select armos_interfaces armos_control armos_ui franka_custom_control
source install/setup.bash
```
- **Use `python -m colcon` with the venv active**, never plain `colcon`. Plain colcon writes `/usr/bin/python3`
  shebangs, and those nodes then miss the venv packages.
- **Always pass `--packages-select`.** `armos_ros2/franka/` also holds franka_ros2, crisp, libfranka and
  franka_description sources. They are built and installed in `~/franka_ros2_ws` (unchanged); rebuild there only
  if you change them.
- **When to rebuild:** with `--symlink-install`, edits to existing Python files and configs take effect without
  rebuilding. Rebuild for new files, new entry points or `setup.py` changes.

## 3. Scripted spiral (franka_custom_control, no RL)

Yes, it still exists:
- **Where:** `armos_ros2/franka/franka_custom_control/insertion/` (`spiral_search.py`, `spiral_insertion_demo.py`).
- **How it runs:** on top of **armos_control's admittance node**. The old `bimanual/admittance_controller.py`
  was removed; it is only left in the old `~/franka_ros2_ws` copy, so don't mix the two.
- **Sim and real:** it uses the ROS clock, so the same script runs on both.

**Real FR3:**
```bash
# terminal 1: franka_ros2 + crisp (cartesian_controller active) + admittance node (tares: don't touch for 2 s)
ros2 launch armos_control admittance.launch.py use_sim:=false robot_ip:=<FR3 ip> load_gripper:=true
# terminal 2: calibrate once (hand-guide the peg fully into the hole, Enter) -> ~/.ros/spiral_insertion_hole.yaml
ros2 run franka_custom_control spiral_insertion_demo.py --calibration true
# then runs from a random start error around the calibrated hole
ros2 run franka_custom_control spiral_insertion_demo.py --xy-error 0.003 0.004
```
Close the franka UI first (it also publishes `target_pose`).

**Sim (same peg/hole scene):**
```bash
# terminal 1
ros2 launch armos_control admittance.launch.py use_sim:=true sim_config:=peg_hole/fr3_peg_hole.yaml
# terminal 2
ros2 run franka_custom_control spiral_insertion_demo.py --xy-error 0.002 0.003 \
    --calib-file $(ros2 pkg prefix armos_control)/share/armos_control/config/peg_hole/spiral_demo_sim_hole.yaml \
    --ros-args -p use_sim_time:=true
```
Status (2026-10-20):
- **What works:** the sim run goes through the whole sequence (tare, move, approach, touchdown, 60 s search,
  release).
- **What failed:** it did not insert in the one try. The start pose settled 5.4 mm / 1.2° off because of the
  FR3 joint friction against the admittance gains, and the clearance is only 0.5 mm.
- **Status:** the script's parameters are tuned for the real robot and are not tuned for this sim yet.

## 4. Other runs (ROS terminal)

```bash
ros2 launch armos_control admittance.launch.py                     # admittance node on the sim (FR3, no hole)
ros2 launch armos_control admittance.launch.py use_sim:=false robot_ip:=<ip> load_gripper:=true   # on the robot
ros2 run armos_control armos_ros_sim --ros-args -p config_path:=fr3.yaml      # bridge alone
ros2 run armos_ui armos_gui                                       # Tk GUI (sim)
ros2 launch franka_custom_control cartesian_control.launch.py use_fake_hardware:=true robot_ip:=0.0.0.0 \
    load_gripper:=true rviz:=false start_controller:=cartesian_controller   # crisp without a robot
```

## 5. Tests

```bash
NOROS="-p no:launch_testing -p no:launch_ros -p no:ament_lint -p no:ament_copyright -p no:ament_xmllint -p no:ament_flake8 -p no:ament_pep257"
(cd ~ && python -m pytest -q $NOROS ~/armos_ws/armos_core/tests ~/armos_ws/armos_sim/tests)   # 50 passed
#   (not from ~/armos_ws: there the armos_sim/ folder shadows the installed package)
python armos_ros2/armos_control/test/test_admittance_sim.py --headless --axes tcp           # 22/23
```

## 6. Where things are

| What | Path |
|---|---|
| Sim peg/hole scene (bridge config) | `armos_ros2/armos_control/config/peg_hole/` |
| Admittance node / launch / parameters | `armos_ros2/armos_control/armos_control/admittance.py`, `launch/admittance.launch.py`, `config/admittance.yaml` |
| Sim bridge | `armos_ros2/armos_control/armos_control/sim_bridge.py` |
| crisp launch wrapper, scripted spiral, franka UI | `armos_ros2/franka/franka_custom_control/` |
| crisp gains (real robot) | `armos_ros2/franka/franka_custom_control/config/crisp_controllers.yaml` |
| Plan / status | `docs/plan_final.md` |

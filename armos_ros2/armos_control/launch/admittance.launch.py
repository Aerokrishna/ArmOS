"""Hybrid admittance on the real FR3 or on the armos sim: same node, same parameters, same topics.

    ros2 launch armos_control admittance.launch.py                         # sim (default): FR3 + hand
    ros2 launch armos_control admittance.launch.py sim_config:=dual_franka.yaml arm:=right
    ros2 launch armos_control admittance.launch.py use_sim:=false robot_ip:=192.168.1.17 load_gripper:=true
    ... axes:=base                                                          # base-frame axes (sim or real)

use_sim:=true   armos sim bridge (armos_ros_sim, MuJoCo, torque actuation) publishes the robot's topics
                (current_pose, franka_robot_state_broadcaster/..., joint_states) and /clock, and runs
                crisp's Cartesian impedance on target_pose. The node runs on sim time.
use_sim:=false  franka_custom_control/cartesian_control.launch.py: franka_ros2 + crisp on the FR3 (or
                use_fake_hardware:=true), cartesian_controller active. The node runs on wall time.

Topics are relative to the arm namespace: `namespace` (default "", as in franka_bringup's
franka.config.yaml), plus /<arm> for a multi-arm sim config. On the real robot `namespace` must match
the namespace in robot_config_file. Clients (spiral_search, admittance_test_publisher, ...) use the
admittance/* topics in the same namespace.
"""

import os

import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node


def _sim_config(name, share):
    path = name if os.path.isabs(name) else os.path.join(share, 'config', name)
    with open(path, encoding='utf-8') as fh:
        return yaml.safe_load(fh)


def _setup(context):
    arg = lambda n: context.launch_configurations[n]  # noqa: E731
    share = get_package_share_directory('armos_control')
    params_file = arg('params_file') or os.path.join(share, 'config', 'admittance.yaml')
    namespace = arg('namespace').strip('/')
    node_params = [params_file, {'axes': arg('axes')}]
    actions = []

    if arg('use_sim').lower() == 'true':
        cfg = _sim_config(arg('sim_config'), share)
        arms = list(cfg['arms'])
        arm = arg('arm') or arms[0]
        if arm not in arms:
            raise RuntimeError(f"arm '{arm}' not in {arg('sim_config')} (arms: {arms})")
        node_ns = '/'.join(x for x in (namespace, arm if len(arms) > 1 else '') if x)
        actions.append(Node(
            package='armos_control',
            executable='armos_ros_sim',
            parameters=[{'config_path': arg('sim_config'), 'namespace': namespace,
                         'headless': arg('headless').lower() == 'true', 'use_sim_time': False,
                         'frictionloss_scale': float(arg('joint_friction') or -1.0)}],
            output='screen',
        ))
        node_params.append({'tcp_frame': cfg['arms'][arm]['ee_site'], 'use_sim_time': True})
    else:
        node_ns = namespace
        real = os.path.join(get_package_share_directory('franka_custom_control'), 'launch',
                            'cartesian_control.launch.py')
        passthrough = {k: arg(k) for k in ('robot_config_file', 'robot_ip', 'load_gripper',
                                           'use_fake_hardware', 'start_controller') if arg(k)}
        # An empty robot_config_file declared here would override the included launch's default: pass it.
        passthrough.setdefault('robot_config_file', os.path.join(
            get_package_share_directory('franka_bringup'), 'config', 'franka.config.yaml'))
        actions.append(IncludeLaunchDescription(PythonLaunchDescriptionSource(real),
                                                launch_arguments=passthrough.items()))

    actions.append(Node(
        package='armos_control',
        executable='admittance_controller',
        namespace=node_ns,
        parameters=node_params,
        output='screen',
    ))
    return actions


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('use_sim', default_value='true',
                              description='true: armos MuJoCo sim; false: real FR3 (crisp + franka_ros2).'),
        DeclareLaunchArgument('namespace', default_value='',
                              description='Robot namespace (real: must match robot_config_file).'),
        DeclareLaunchArgument('axes', default_value='tcp',
                              description='Admittance axes: tcp (tool frame) or base.'),
        DeclareLaunchArgument('params_file', default_value='',
                              description='Node parameters (default: armos_control/config/admittance.yaml).'),
        # sim
        DeclareLaunchArgument('sim_config', default_value='fr3.yaml',
                              description='Sim robot config in armos_control/config (or absolute path).'),
        DeclareLaunchArgument('arm', default_value='',
                              description='Arm to control in a multi-arm sim config (default: the first).'),
        DeclareLaunchArgument('headless', default_value='false',
                              description='Sim without the MuJoCo viewer.'),
        DeclareLaunchArgument('joint_friction', default_value='',
                              description='Sim FR3 joint friction scale (0 = none, as the RL training robot; '
                                          '1 = identified FR3); empty = the config\'s.'),
        # real (empty: the value from robot_config_file / cartesian_control.launch.py)
        DeclareLaunchArgument('robot_config_file', default_value=''),
        DeclareLaunchArgument('robot_ip', default_value=''),
        DeclareLaunchArgument('load_gripper', default_value=''),
        DeclareLaunchArgument('use_fake_hardware', default_value=''),
        DeclareLaunchArgument('start_controller', default_value='cartesian_controller',
                              description='crisp controller activated at start (the admittance inner loop).'),
        OpaqueFunction(function=_setup),
    ])

############################################################################
# Brings up the FR3 with the crisp controllers configured, and optionally the control UI.
#
#   ros2 launch franka_custom_control cartesian_control.launch.py
#   ros2 launch franka_custom_control cartesian_control.launch.py ui:=true
#   ros2 launch franka_custom_control cartesian_control.launch.py rviz:=true
#   ros2 launch franka_custom_control cartesian_control.launch.py ui:=true \
#       start_controller:=cartesian_controller robot_ip:=192.168.1.17
#
# What it does:
#   1. Reads the first robot entry of robot_config_file (franka_bringup's franka.config.yaml
#      by default) for robot_ip / namespace / gripper / fake hardware. robot_ip, load_gripper
#      and use_fake_hardware can be overridden from the command line.
#   2. Merges franka_bringup/config/controllers.yaml with config/crisp_controllers.yaml,
#      fills in joint names and the end-effector frame, and hands the result to
#      franka_bringup/franka.launch.py (robot_state_publisher, ros2_control_node,
#      joint_state_broadcaster, franka_robot_state_broadcaster, gripper).
#   3. Spawns pose_broadcaster ACTIVE and the crisp command controllers configured but
#      INACTIVE (or activates start_controller). Activate one later from the UI or with
#      `ros2 control switch_controllers --activate cartesian_controller`.
#   4. ui:=true also starts franka_control_ui in the robot namespace.
#   5. rviz:=true (or use_rviz in robot_config_file) starts RViz with config/
#      franka_custom_control.rviz: robot model, all TF frames with names, the measured TCP,
#      crisp's and the admittance node's target poses, and the admittance (TCP) wrench.
############################################################################

import os
import tempfile

import yaml
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

CRISP_COMMAND_CONTROLLERS = ['cartesian_controller', 'gravity_compensation',
                             'joint_impedance_controller', 'cartesian_admittance_controller']
CRISP_ALL = ['pose_broadcaster'] + CRISP_COMMAND_CONTROLLERS


class _MergingLoader(yaml.SafeLoader):
    """SafeLoader that deep-merges repeated mapping keys.

    franka_bringup's controllers.yaml repeats the top-level '/**' key; plain safe_load
    would keep only the last block.
    """


def _deep_merge(dst, src):
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            _deep_merge(dst[k], v)
        else:
            dst[k] = v
    return dst


def _construct_merged_mapping(loader, node):
    loader.flatten_mapping(node)
    out = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=True)
        value = loader.construct_object(value_node, deep=True)
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


_MergingLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _construct_merged_mapping)


class _NoAliasDumper(yaml.SafeDumper):
    """rcl's params parser rejects YAML anchors/aliases, which PyYAML emits for shared lists."""

    def ignore_aliases(self, data):
        return True


def _load(path):
    with open(path) as f:
        return yaml.load(f, Loader=_MergingLoader) or {}


def launch_setup(context):
    def arg(name):
        return LaunchConfiguration(name).perform(context)

    robots = _load(arg('robot_config_file'))
    robot = next(iter(robots.values()))

    def robot_arg(name):
        override = arg(name) if name in ('robot_ip', 'load_gripper', 'use_fake_hardware') else ''
        return override if override else str(robot.get(name, ''))

    robot_type = robot_arg('robot_type') or 'fr3'
    namespace = robot_arg('namespace')
    load_gripper = robot_arg('load_gripper').lower()
    use_fake_hardware = robot_arg('use_fake_hardware').lower()

    ee_frame = arg('ee_frame')
    if ee_frame in ('', 'auto'):
        ee_frame = f'{robot_type}_hand_tcp' if load_gripper == 'true' else f'{robot_type}_link8'
    joints = [f'{robot_type}_joint{i}' for i in range(1, 8)]

    franka_share = FindPackageShare('franka_bringup').perform(context)
    own_share = FindPackageShare('franka_custom_control').perform(context)
    merged = _load(os.path.join(franka_share, 'config', 'controllers.yaml'))
    _deep_merge(merged, _load(os.path.join(own_share, 'config', 'crisp_controllers.yaml')))
    for name in CRISP_ALL:
        params = merged['/**'][name]['ros__parameters']
        params['joints'] = joints
        params['end_effector_frame'] = ee_frame

    tmp = tempfile.NamedTemporaryFile(
        'w', prefix='franka_custom_controllers_', suffix='.yaml', delete=False)
    yaml.dump(merged, tmp, Dumper=_NoAliasDumper, default_flow_style=False, sort_keys=False)
    tmp.close()
    print(f'[cartesian_control] controllers yaml: {tmp.name}  (end_effector_frame={ee_frame})')

    start = arg('start_controller')
    if start and start not in CRISP_COMMAND_CONTROLLERS:
        raise RuntimeError(f'start_controller must be one of {CRISP_COMMAND_CONTROLLERS}')
    inactive = [c for c in CRISP_COMMAND_CONTROLLERS if c != start]

    def spawner(names, extra=()):
        return Node(
            package='controller_manager',
            executable='spawner',
            namespace=namespace,
            # crisp controllers fetch the URDF from robot_state_publisher with a blocking call
            # inside configure; on the real robot this has taken ~25 s, beyond the spawner's
            # default 10 s service timeout (and each retry queues another configure).
            arguments=list(names) + ['--controller-manager-timeout', '60',
                                     '--service-call-timeout', '60', *extra],
            output='screen',
        )

    nodes = [
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(franka_share, 'launch', 'franka.launch.py')),
            launch_arguments={
                'robot_type': robot_type,
                'arm_prefix': str(robot.get('arm_prefix', '')),
                'namespace': namespace,
                'robot_ip': robot_arg('robot_ip'),
                'load_gripper': load_gripper,
                'use_fake_hardware': use_fake_hardware,
                'fake_sensor_commands': str(robot.get('fake_sensor_commands', 'false')),
                'joint_state_rate': arg('joint_state_rate'),
                'controllers_yaml': tmp.name,
            }.items(),
        ),
        spawner(['pose_broadcaster']),
        spawner(inactive, ['--inactive']),
    ]
    if start:
        nodes.append(spawner([start]))

    if arg('ui').lower() == 'true':
        nodes.append(Node(
            package='franka_custom_control',
            executable='franka_control_ui',
            namespace=namespace,
            parameters=[
                os.path.join(own_share, 'config', 'control_ui.yaml'),
                {'joint_names': joints},
            ],
            output='screen',
        ))

    rviz = arg('rviz') or str(robot.get('use_rviz', 'false'))
    if rviz.lower() == 'true':
        nodes.append(Node(
            package='rviz2',
            executable='rviz2',
            namespace=namespace,   # the config's topic names are relative
            arguments=['--display-config',
                       os.path.join(own_share, 'config', 'franka_custom_control.rviz')],
            output='screen',
        ))
    return nodes


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            'robot_config_file',
            default_value=PathJoinSubstitution(
                [FindPackageShare('franka_bringup'), 'config', 'franka.config.yaml']),
            description='Robot launch config; the first robot entry is used.'),
        DeclareLaunchArgument('robot_ip', default_value='',
                              description='Override robot_ip from robot_config_file.'),
        DeclareLaunchArgument('load_gripper', default_value='',
                              description='Override load_gripper ("true"/"false").'),
        DeclareLaunchArgument('use_fake_hardware', default_value='',
                              description='Override use_fake_hardware ("true"/"false").'),
        DeclareLaunchArgument('joint_state_rate', default_value='100',
                              description='joint_states publish rate (Hz) used by the UI.'),
        DeclareLaunchArgument('ee_frame', default_value='auto',
                              description='End-effector frame for crisp controllers. auto: '
                                          '<robot_type>_hand_tcp with gripper, else _link8.'),
        DeclareLaunchArgument('start_controller', default_value='',
                              description='Command controller to activate at startup '
                                          '(empty: all stay inactive).'),
        DeclareLaunchArgument('ui', default_value='false',
                              description='Launch the franka_control_ui node.'),
        DeclareLaunchArgument('rviz', default_value='',
                              description='Launch RViz showing the TF frames ("true"/"false"; '
                                          'empty: use_rviz from robot_config_file).'),
        OpaqueFunction(function=launch_setup),
    ])

"""ROS side of the control deck: subscriptions, gated command publishers, controller manager."""

import time

import numpy as np
from controller_manager_msgs.srv import (
    ConfigureController,
    ListControllers,
    LoadController,
    SwitchController,
)
from geometry_msgs.msg import PoseStamped, WrenchStamped
from PyQt5.QtCore import QObject, pyqtSignal
from rcl_interfaces.srv import GetParameters, ListParameters
from rclpy.node import Node
from rclpy.qos import QoSProfile, qos_profile_sensor_data
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray
from std_srvs.srv import Trigger

try:
    from franka_msgs.action import ErrorRecovery, Grasp, Homing, Move
    from rclpy.action import ActionClient
except ImportError:  # franka_msgs not built: recovery and gripper buttons are disabled
    ErrorRecovery = Grasp = Homing = Move = None

FR3_JOINTS = [f'fr3_joint{i}' for i in range(1, 8)]
FR3_LOWER = [-2.7437, -1.7837, -2.9007, -3.0421, -2.8065, 0.5445, -3.0159]
FR3_UPPER = [2.7437, 1.7837, 2.9007, -0.1518, 2.8065, 4.5169, 3.0159]

STIFFNESS_PARAMS = ['task.k_pos_x', 'task.k_pos_y', 'task.k_pos_z',
                    'task.k_rot_x', 'task.k_rot_y', 'task.k_rot_z']

DEFAULTS = {
    'topics.target_pose': 'target_pose',
    'topics.target_wrench': 'target_wrench',
    'topics.target_joint': 'target_joint',
    'topics.target_stiffness': 'target_stiffness',
    'topics.current_pose': 'current_pose',
    'topics.joint_states': 'joint_states',
    'topics.external_wrench': 'franka_robot_state_broadcaster/external_wrench_in_base_frame',
    'base_frame': 'base',
    'joint_names': FR3_JOINTS,
    'joint_lower': FR3_LOWER,
    'joint_upper': FR3_UPPER,
    'controller_manager': 'controller_manager',
    'command_controller': 'cartesian_controller',
    'error_recovery_action': 'action_server/error_recovery',
    'gripper.node': 'franka_gripper',
    'gripper.open_width': 0.08,
    'gripper.speed': 0.1,
    'gripper.force': 20.0,
    'ranges.position_min': [0.20, -0.50, 0.05],
    'ranges.position_max': [0.80, 0.50, 0.80],
    'ranges.euler_center_deg': [180.0, 0.0, 0.0],
    'ranges.euler_half_span_deg': [180.0, 90.0, 180.0],
    'limits.stiffness_max': [900.0, 60.0],
    'limits.wrench_max': [20.0, 3.0],
    'limits.wrench_alert': [20.0, 4.0],
    'limits.max_target_offset_pos': 0.25,
    'limits.max_target_offset_rot_deg': 60.0,
    'jog.max_linear': 0.05,
    'jog.max_angular_deg': 15.0,
    'jog.max_lead_pos': 0.03,
    'jog.max_lead_rot_deg': 10.0,
    'jog.rate': 50.0,
    'publish_rate_limit': 30.0,
    'plot.buffer_seconds': 60.0,
    'plot.sample_hz': 100.0,
    'stale_timeout': 0.5,
}

COMMAND_TOPICS = ('target_pose', 'target_wrench', 'target_joint', 'target_stiffness')


class BridgeSignals(QObject):
    controllers = pyqtSignal(list)    # [{name, state, type, claimed}]
    configured = pyqtSignal(list)     # [name]
    switch_done = pyqtSignal(bool, str)
    params = pyqtSignal(str, dict)    # (tag, {param: value})
    log = pyqtSignal(str, str)        # (level, message)


class RosBridge(Node):

    def __init__(self, state):
        super().__init__('franka_control_ui')
        self.state = state
        self.sig = BridgeSignals()
        self.cfg = {}
        for name, default in DEFAULTS.items():
            self.cfg[name] = self.declare_parameter(name, default).value
        self.joint_names = list(self.cfg['joint_names'])
        self.topics = {k: self.cfg[f'topics.{k}'] for k in COMMAND_TOPICS}

        self.create_subscription(PoseStamped, self.cfg['topics.current_pose'],
                                 self._on_pose, qos_profile_sensor_data)
        self.create_subscription(WrenchStamped, self.cfg['topics.external_wrench'],
                                 self._on_wrench, qos_profile_sensor_data)
        self.create_subscription(JointState, self.cfg['topics.joint_states'],
                                 self._on_joints, qos_profile_sensor_data)

        cm = self.cfg['controller_manager']
        self.cli_list = self.create_client(ListControllers, f'{cm}/list_controllers')
        self.cli_switch = self.create_client(SwitchController, f'{cm}/switch_controller')
        self.cli_load = self.create_client(LoadController, f'{cm}/load_controller')
        self.cli_configure = self.create_client(
            ConfigureController, f'{cm}/configure_controller')
        self.cli_cm_params = self.create_client(ListParameters, f'{cm}/list_parameters')
        self.cli_cmd_params = self.create_client(
            GetParameters, f"{self.cfg['command_controller']}/get_parameters")
        self.recovery_client = None
        if ErrorRecovery is not None:
            self.recovery_client = ActionClient(
                self, ErrorRecovery, self.cfg['error_recovery_action'])

        # franka_gripper node (only launched with load_gripper:=true).
        g = self.cfg['gripper.node']
        self.gripper_clients = {}
        if Move is not None:
            self.gripper_clients = {
                'move': ActionClient(self, Move, f'{g}/move'),
                'grasp': ActionClient(self, Grasp, f'{g}/grasp'),
                'homing': ActionClient(self, Homing, f'{g}/homing'),
            }
        self.cli_gripper_stop = self.create_client(Trigger, f'{g}/stop')
        self.gripper_width = None
        self.gripper_time = -float('inf')
        self.create_subscription(JointState, f'{g}/joint_states', self._on_gripper,
                                 qos_profile_sensor_data)

        self._pubs = {}
        self._joint_index = None

    # ------------------------------------------------------------------ feedback
    def _on_pose(self, msg):
        p, o = msg.pose.position, msg.pose.orientation
        self.state.update_pose(np.array([p.x, p.y, p.z]), np.array([o.x, o.y, o.z, o.w]))

    def _on_wrench(self, msg):
        f, t = msg.wrench.force, msg.wrench.torque
        self.state.update_wrench(np.array([f.x, f.y, f.z, t.x, t.y, t.z]))

    def _on_joints(self, msg):
        if self._joint_index is None or len(msg.name) != self._joint_index[1]:
            try:
                idx = [list(msg.name).index(n) for n in self.joint_names]
            except ValueError:
                return
            self._joint_index = (idx, len(msg.name))
        if len(msg.position) < len(msg.name):
            return
        self.state.update_joints(np.array([msg.position[i] for i in self._joint_index[0]]))

    def _on_gripper(self, msg):
        if msg.position:
            self.gripper_width = float(sum(msg.position))   # two fingers, each half the width
            self.gripper_time = time.monotonic()

    # ------------------------------------------------------------------ command gate
    @property
    def enabled(self):
        return bool(self._pubs)

    def foreign_publishers(self):
        """Publisher count on each command topic, excluding this UI's own publisher."""
        counts = {}
        for key, topic in self.topics.items():
            n = self.count_publishers(topic)
            counts[key] = n - (1 if key in self._pubs else 0)
        return counts

    def enable(self):
        if self._pubs:
            return
        qos = QoSProfile(depth=1)
        self._pubs = {
            'target_pose': self.create_publisher(PoseStamped, self.topics['target_pose'], qos),
            'target_wrench': self.create_publisher(
                WrenchStamped, self.topics['target_wrench'], qos),
            'target_joint': self.create_publisher(
                JointState, self.topics['target_joint'], qos),
            'target_stiffness': self.create_publisher(
                Float64MultiArray, self.topics['target_stiffness'], qos),
        }

    def disable(self):
        pubs, self._pubs = self._pubs, {}
        for pub in pubs.values():
            self.destroy_publisher(pub)

    def _stamp(self, msg):
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.cfg['base_frame']

    def publish_pose(self, pos, quat):
        pub = self._pubs.get('target_pose')
        if pub is None:
            return False
        msg = PoseStamped()
        self._stamp(msg)
        msg.pose.position.x, msg.pose.position.y, msg.pose.position.z = map(float, pos)
        o = msg.pose.orientation
        o.x, o.y, o.z, o.w = map(float, quat)
        pub.publish(msg)
        self.state.set_target_pose(pos, quat)
        return True

    def publish_wrench(self, w):
        pub = self._pubs.get('target_wrench')
        if pub is None:
            return False
        msg = WrenchStamped()
        self._stamp(msg)
        f, t = msg.wrench.force, msg.wrench.torque
        f.x, f.y, f.z, t.x, t.y, t.z = map(float, w)
        pub.publish(msg)
        self.state.set_target_wrench(w)
        return True

    def publish_joints(self, q):
        pub = self._pubs.get('target_joint')
        if pub is None:
            return False
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = self.joint_names
        msg.position = [float(v) for v in q]
        pub.publish(msg)
        self.state.set_target_joints(q)
        return True

    def publish_stiffness(self, k):
        pub = self._pubs.get('target_stiffness')
        if pub is None:
            return False
        pub.publish(Float64MultiArray(data=[float(v) for v in k]))
        self.state.set_target_stiffness(k)
        return True

    # ------------------------------------------------------------------ services
    def _call(self, client, request, on_result, what):
        if not client.service_is_ready():
            self.sig.log.emit('warn', f'{what}: service {client.srv_name} not available')
            on_result(None)
            return
        future = client.call_async(request)

        def done(f):
            try:
                result = f.result()
            except Exception as exc:  # noqa: BLE001 - surface any transport error in the log
                self.sig.log.emit('error', f'{what} failed: {exc}')
                result = None
            on_result(result)
        future.add_done_callback(done)

    def refresh_controllers(self, then=None):
        def on_result(res):
            if res is None:
                return
            ctrls = [{'name': c.name, 'state': c.state, 'type': c.type,
                      'claimed': list(c.claimed_interfaces)} for c in res.controller]
            self.sig.controllers.emit(ctrls)
            if then:
                then(ctrls)
        self._call(self.cli_list, ListControllers.Request(), on_result, 'list_controllers')

    def refresh_configured(self):
        def on_result(res):
            if res is None:
                return
            names = sorted({n[:-len('.type')] for n in res.result.names
                            if n.endswith('.type') and n.count('.') == 1})
            self.sig.configured.emit(names)
        req = ListParameters.Request()
        req.depth = 0
        self._call(self.cli_cm_params, req, on_result, 'list configured controllers')

    def _switch(self, activate, deactivate, strict, label):
        req = SwitchController.Request()
        req.activate_controllers = list(activate)
        req.deactivate_controllers = list(deactivate)
        req.strictness = SwitchController.Request.STRICT if strict \
            else SwitchController.Request.BEST_EFFORT
        req.activate_asap = False
        req.timeout.sec = 5

        def on_result(res):
            ok = bool(res and res.ok)
            self.sig.switch_done.emit(ok, label if ok else f'{label} FAILED')
            self.refresh_controllers()
        self._call(self.cli_switch, req, on_result, 'switch_controller')

    def activate_exclusive(self, name):
        """Load/configure `name` if needed, then activate it while deactivating every other
        controller that currently claims command interfaces (broadcasters are untouched)."""

        def switch(ctrls):
            others = [c['name'] for c in ctrls
                      if c['state'] == 'active' and c['claimed'] and c['name'] != name]
            label = f'activated {name}' + (f' (deactivated {", ".join(others)})' if others else '')
            self._switch([name], others, True, label)

        def configure(_=None):
            req = ConfigureController.Request(name=name)
            self._call(self.cli_configure, req,
                       lambda r: self.refresh_controllers(switch) if r and r.ok else
                       self.sig.switch_done.emit(False, f'configure {name} FAILED'),
                       'configure_controller')

        def route(ctrls):
            known = {c['name']: c for c in ctrls}
            if name not in known:
                self._call(self.cli_load, LoadController.Request(name=name),
                           lambda r: configure() if r and r.ok else
                           self.sig.switch_done.emit(False, f'load {name} FAILED'),
                           'load_controller')
            elif known[name]['state'] == 'unconfigured':
                configure()
            elif known[name]['state'] == 'active':
                self.sig.switch_done.emit(True, f'{name} already active')
            else:
                switch(ctrls)

        self.refresh_controllers(route)

    def deactivate(self, names):
        self._switch([], names, False, f'deactivated {", ".join(names)}')

    def deactivate_command_controllers(self):
        """E-stop stage 2: deactivate every active controller that claims command interfaces."""
        def on_list(ctrls):
            claimers = [c['name'] for c in ctrls if c['state'] == 'active' and c['claimed']]
            if claimers:
                self._switch([], claimers, False, f'E-STOP deactivated {", ".join(claimers)}')
            else:
                self.sig.switch_done.emit(True, 'E-STOP: no command controller was active')
        self.refresh_controllers(on_list)

    def fetch_stiffness(self):
        def on_result(res):
            if res is None or len(res.values) != len(STIFFNESS_PARAMS):
                return
            vals = {n: v.double_value for n, v in zip(STIFFNESS_PARAMS, res.values)}
            self.sig.params.emit('stiffness', vals)
        self._call(self.cli_cmd_params, GetParameters.Request(names=STIFFNESS_PARAMS),
                   on_result, f"read {self.cfg['command_controller']} stiffness")

    def error_recovery(self):
        if self.recovery_client is None:
            self.sig.log.emit('error', 'franka_msgs not available: cannot run error recovery')
            return
        if not self.recovery_client.server_is_ready():
            self.sig.log.emit(
                'warn', f"error recovery server {self.cfg['error_recovery_action']} not available")
            return
        self.sig.log.emit('info', 'error recovery requested')

        def on_goal(f):
            handle = f.result()
            if not handle.accepted:
                self.sig.log.emit('error', 'error recovery goal rejected')
                return
            handle.get_result_async().add_done_callback(
                lambda r: self.sig.log.emit('info', 'error recovery finished'))
        self.recovery_client.send_goal_async(ErrorRecovery.Goal()).add_done_callback(on_goal)

    # ------------------------------------------------------------------ gripper
    def _gripper_goal(self, kind, goal, label):
        client = self.gripper_clients.get(kind)
        if client is None:
            self.sig.log.emit('error', 'franka_msgs not available: cannot command the gripper')
            return
        if not client.server_is_ready():
            self.sig.log.emit('warn', f"gripper server {self.cfg['gripper.node']}/{kind} not "
                                      'available (launched with load_gripper:=true?)')
            return
        self.sig.log.emit('info', f'gripper: {label} ...')

        def on_result(f):
            res = f.result().result
            if res.success:
                self.sig.log.emit('ok', f'gripper: {label} done')
            else:
                self.sig.log.emit('error', f'gripper: {label} failed: {res.error}')

        def on_goal(f):
            handle = f.result()
            if not handle.accepted:
                self.sig.log.emit('error', f'gripper: {label} rejected')
                return
            handle.get_result_async().add_done_callback(on_result)
        client.send_goal_async(goal).add_done_callback(on_goal)

    def gripper_move(self, width, speed):
        self._gripper_goal('move', Move.Goal(width=float(width), speed=float(speed)),
                           f'move to {width * 1000:.0f} mm')

    def gripper_grasp(self, speed, force):
        # Close fully; the large outer epsilon makes the grasp succeed at whatever width the
        # fingers stop (any object, or none).
        goal = Grasp.Goal(width=0.0, speed=float(speed), force=float(force))
        goal.epsilon.inner = 0.005
        goal.epsilon.outer = float(self.cfg['gripper.open_width'])
        self._gripper_goal('grasp', goal, f'grasp at {force:.0f} N')

    def gripper_home(self):
        self._gripper_goal('homing', Homing.Goal(), 'homing')

    def gripper_stop(self):
        self._call(self.cli_gripper_stop, Trigger.Request(),
                   lambda r: r is not None and self.sig.log.emit(
                       'ok' if r.success else 'error',
                       'gripper stopped' if r.success else f'gripper stop failed: {r.message}'),
                   'gripper stop')

#!/usr/bin/env python3
"""Hybrid Cartesian admittance node, one for the real FR3 and the armos sim.

The law is `armos_core.control.HybridAdmittance` and the wrench processing is
`armos_core.control.ExternalWrench`; this node moves data between ROS and them. Source:
franka_custom_control/bimanual/admittance_controller.py (real FR3, on crisp) and
armos_control/cartesian_admittance.py (sim), merged.

Sim or real is decided by what runs underneath, not by this node: on the real robot crisp +
franka_ros2 provide the topics below; in sim the armos bridge (armos_control sim_bridge) publishes
the same topics in the same namespace and runs crisp's Cartesian impedance in MuJoCo. Run both with
`ros2 launch armos_control admittance.launch.py use_sim:=true|false`. Time is the ROS clock, so the
node runs on /clock in sim (use_sim_time) and on wall time on the robot.

Topics (relative, so they follow the robot namespace):
  Subscriptions:
    current_pose                    PoseStamped     crisp pose_broadcaster: the TCP
                                                    (crisp's end_effector_frame) in `base`
    franka_robot_state_broadcaster/external_wrench_in_base_frame
                                    WrenchStamped   Franka's O_F_ext_hat_K
    franka_robot_state_broadcaster/current_pose
                                    PoseStamped     Franka's O_T_EE: the point that torque is
                                                    measured about
    admittance/target_pose          PoseStamped     frame `base`
    admittance/target_wrench        WrenchStamped   in the law's axes (frame_id = tcp_frame for
                                                    axes tcp, base_frame for axes base, or empty);
                                                    NOT crisp's `target_wrench`, which is a torque
                                                    feed-forward
    admittance/control_mode         UInt8MultiArray in the law's axes
  Publishers:
    target_pose                     PoseStamped     to crisp cartesian_controller (base frame)
    admittance/wrench               WrenchStamped   tared + filtered wrench used by the law, in the
                                                    law's axes, about the TCP
  Service:
    ~/tare                          std_srvs/Trigger  re-zero the wrench (all axes in pose mode,
                                                      robot untouched)

axes (parameter): "tcp" (default) or "base": the frame of control_mode, target_wrench, the virtual
spring / damper and admittance/wrench. With "tcp", mode [1, 1, 0, 1, 1, 1] is compliant along the
tool's z wherever it points; with "base", along the base z. The target pose is always an absolute
pose in `base`.

control_mode.data is six values in [x, y, z, roll, pitch, yaw] order:
  1 = pose-regulated axis (virtual spring enabled)
  0 = force-regulated axis (virtual spring disabled; force error drives motion)

Wrench:
  - The input wrench has the end-effector load removed already (Franka: the load configured in Desk
    / franka_bringup; sim: the bridge's payload model), so there is no gravity model here.
  - Zero calibration (tare): at start-up the bias is averaged over tare_samples readings with the
    robot holding still and nobody touching it. The node publishes nothing until the tare is done.
    If the wrench std during the tare exceeds wrench.tare_max_std (someone touched the arm, or it
    moved), the tare restarts.
  - The tare and filter run on the base-frame wrench. The torque is then moved from the reference
    point (franka_robot_state_broadcaster/current_pose) to the TCP and the wrench is rotated into the
    law's axes.
  - Convention: the wrench the robot exerts ON THE ENVIRONMENT; target_wrench uses the same (pushing
    along the tool's approach axis with 5 N is force.z = +5 in TCP axes). O_F_ext_hat_K is taken to
    follow it (wrench.sign = +1); check by pushing the fingertips back towards the flange:
    admittance/wrench force.z (axes tcp) must go POSITIVE. If it goes negative, set wrench.sign -1.

Safety:
  - force-error deadband (wrench.force_deadband): a real wrench estimate drifts by ~1 N.
  - force-target timeout (force_target_timeout): with any force axis active and no target_wrench for
    this long, every axis goes back to pose mode holding the measured pose, so a crashed client
    cannot leave the arm pushing. 0 disables it.
  - the virtual pose may lead the measured pose by at most limits.max_deviation (anti-windup).
  - stale pose / wrench: hold the last commanded pose. On shutdown the measured pose is commanded.
"""

import math
import signal

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped, WrenchStamped
from rcl_interfaces.srv import GetParameters
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.signals import SignalHandlerOptions
from std_msgs.msg import UInt8MultiArray
from std_srvs.srv import Trigger

from armos_core.control import ExternalWrench, ExternalWrenchCfg, HybridAdmittance, HybridAdmittanceCfg
from armos_core.control.external_wrench import to_axes
from armos_core.math.so3 import quat_normalize, quat_to_rot, rot_to_quat


def _xyzw_to_rot(o) -> np.ndarray:
    return quat_to_rot(quat_normalize(np.array([o.w, o.x, o.y, o.z])))


class AdmittanceController(Node):

    def __init__(self, **kwargs):
        super().__init__('admittance_controller', **kwargs)
        p = self.declare_parameter

        self.topic_current = p('topics.current_pose', 'current_pose').value
        self.topic_wrench = p('topics.external_wrench',
                              'franka_robot_state_broadcaster/external_wrench_in_base_frame').value
        self.topic_target_pose = p('topics.target_pose', 'admittance/target_pose').value
        self.topic_target_wrench = p('topics.target_wrench', 'admittance/target_wrench').value
        self.topic_mode = p('topics.control_mode', 'admittance/control_mode').value
        self.topic_command = p('topics.command', 'target_pose').value
        self.topic_wrench_out = p('topics.wrench_out', 'admittance/wrench').value
        # Franka's O_T_EE: the point the wrench torque is about. Empty: no moment shift.
        self.topic_wrench_point = p('topics.wrench_point_pose',
                                    'franka_robot_state_broadcaster/current_pose').value
        self.base_frame = p('base_frame', 'base').value
        self.tcp_frame = p('tcp_frame', 'auto').value
        pose_broadcaster = p('pose_broadcaster', 'pose_broadcaster').value
        self.axes = p('axes', 'tcp').value
        if self.axes not in ('tcp', 'base'):
            raise ValueError("axes must be 'tcp' or 'base'")
        rate_hz = float(p('control_rate_hz', 250.0).value)
        if rate_hz <= 0.0:
            raise ValueError('control_rate_hz must be positive')
        self.nominal_dt = 1.0 / rate_hz

        # Wrench: sign, tare, filter.
        self.wrench = ExternalWrench(ExternalWrenchCfg(
            sign=float(p('wrench.sign', 1.0).value),
            tare_samples=int(p('wrench.tare_samples', 500).value),          # 2 s at 250 Hz
            tare_max_std=tuple(p('wrench.tare_max_std', [1.0, 0.15]).value),
            filter_hz=float(p('wrench.filter_hz', 15.0).value),
        ))
        self.wrench_timeout = float(p('wrench.timeout', 0.1).value)

        # Virtual dynamics [x, y, z, rx, ry, rz]; defaults: the FR3 values.
        mass = p('mass', [2.5, 2.5, 2.5, 0.25, 0.25, 0.25]).value
        # Pose axes: K 1000 / D 100 with M 2.5 is critically damped at ~3 Hz, fast enough to
        # follow a few-mm spiral (K 100 tracked at ~1 Hz and smeared it).
        stiffness = p('stiffness', [1000.0, 1000.0, 1000.0, 10.0, 10.0, 10.0]).value
        damping = p('damping', [100.0, 100.0, 100.0, 3.0, 3.0, 3.0]).value
        # Damping on force-regulated axes. Free-space approach speed is F_target / D_force
        # (5 N -> ~8 mm/s at 600). On the FR3 the contact is in series with crisp's impedance
        # (~500 N/m), so the force loop is much softer than a stiff position-controlled contact.
        damping_force = p('damping_force', [100.0, 100.0, 100.0, 5.0, 5.0, 5.0]).value
        max_lin_vel, max_ang_vel = p('limits.max_vel', [0.05, 0.25]).value
        max_lin_acc, max_ang_acc = p('limits.max_acc', [1.0, 1.0]).value
        # Force-axis speed cap: bounds the touch-down force spike for any target force.
        max_force_lin_vel, max_force_ang_vel = p('limits.max_force_vel', [0.05, 0.3]).value
        self.max_force, self.max_torque = p('limits.max_wrench', [20.0, 3.0]).value
        max_dev = p('limits.max_deviation', [0.05, 0.3]).value
        force_deadband = p('wrench.force_deadband', [0.5, 0.05]).value
        self.law = HybridAdmittance(HybridAdmittanceCfg(
            axes=self.axes, mass=mass, stiffness=stiffness, damping=damping,
            damping_force=damping_force, max_lin_vel=max_lin_vel, max_ang_vel=max_ang_vel,
            max_lin_acc=max_lin_acc, max_ang_acc=max_ang_acc, max_force_lin_vel=max_force_lin_vel,
            max_force_ang_vel=max_force_ang_vel, max_force=self.max_force, max_torque=self.max_torque,
            force_deadband=tuple(force_deadband), max_deviation=tuple(max_dev),
        ))
        self.pose_timeout = float(p('pose_timeout', 0.1).value)
        self.force_target_timeout = float(p('force_target_timeout', 1.0).value)

        mode0 = list(p('initial_control_mode', [1, 1, 1, 1, 1, 1]).value)
        if len(mode0) != 6 or any(int(v) not in (0, 1) for v in mode0):
            raise ValueError('initial_control_mode must be six 0/1 values')
        self.law.set_mask(mode0)

        # measured state (times: ROS clock [s])
        self.curr_pos = None
        self.curr_R = None
        self.pose_time = -math.inf
        self.wrench_point = None         # torque reference point, base frame
        self.wrench_point_time = -math.inf
        self._wrench_point_logged = False
        self.curr_wrench = None          # tared, filtered, about the TCP, in the law's axes
        self.wrench_time = -math.inf
        # targets
        self.target_pose_received = False
        self.des_wrench_time = -math.inf
        self.last_update_time = None
        self._stale_logged = False

        self.cmd_pub = self.create_publisher(PoseStamped, self.topic_command, 10)
        self.wrench_pub = self.create_publisher(WrenchStamped, self.topic_wrench_out, 10)
        self.create_subscription(PoseStamped, self.topic_current, self._on_pose,
                                 qos_profile_sensor_data)
        self.create_subscription(WrenchStamped, self.topic_wrench, self._on_wrench,
                                 qos_profile_sensor_data)
        if self.topic_wrench_point:
            self.create_subscription(PoseStamped, self.topic_wrench_point,
                                     self._on_wrench_point, qos_profile_sensor_data)
        self.create_subscription(PoseStamped, self.topic_target_pose, self._on_target_pose, 10)
        self.create_subscription(WrenchStamped, self.topic_target_wrench,
                                 self._on_target_wrench, 10)
        self.create_subscription(UInt8MultiArray, self.topic_mode, self._on_control_mode, 10)
        self.create_service(Trigger, '~/tare', self._on_tare)
        self.timer = self.create_timer(self.nominal_dt, self.admittance_control)

        if self.tcp_frame == 'auto':
            # Name the TCP after crisp's end_effector_frame; until known, headers stay empty.
            self.tcp_frame = ''
            self._ee_client = self.create_client(GetParameters,
                                                 f'{pose_broadcaster}/get_parameters')
            self._ee_pending = False
            self._ee_timer = self.create_timer(1.0, self._query_tcp_frame)

        self.get_logger().info(
            f'axes: {self.axes}. waiting for {self.topic_current} and {self.topic_wrench}. '
            'DO NOT TOUCH THE ROBOT: zeroing the wrench.')

    # ------------------------------------------------------------------ helpers
    def _now(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    @property
    def axes_frame(self) -> str:
        """Frame name of the law's axes (message headers)."""
        return self.tcp_frame if self.axes == 'tcp' else self.base_frame

    @property
    def pose_mode(self) -> np.ndarray:
        return self.law.pose_mask

    # ------------------------------------------------------------------ callbacks
    def _on_pose(self, msg: PoseStamped):
        q, o = msg.pose.position, msg.pose.orientation
        pos = np.array([q.x, q.y, q.z])
        quat = np.array([o.x, o.y, o.z, o.w])
        if not (np.all(np.isfinite(pos)) and np.all(np.isfinite(quat))) or \
                np.linalg.norm(quat) < 1e-8:
            self.get_logger().warning('Ignoring non-finite current pose.',
                                      throttle_duration_sec=1.0)
            return
        self.curr_pos = pos
        self.curr_R = _xyzw_to_rot(o)
        self.pose_time = self._now()

    def _query_tcp_frame(self):
        if self._ee_pending:
            return
        if not self._ee_client.service_is_ready():
            self.get_logger().warning(
                f'{self._ee_client.srv_name} not available: cannot name the TCP frame '
                '(set the tcp_frame parameter)', throttle_duration_sec=10.0)
            return
        self._ee_pending = True

        def done(future):
            self._ee_pending = False
            try:
                values = future.result().values
            except Exception as exc:  # noqa: BLE001 - retried by the timer
                self.get_logger().warning(f'reading end_effector_frame failed: {exc}')
                return
            if values and values[0].string_value:
                self.tcp_frame = values[0].string_value
                self._ee_timer.cancel()
                self.get_logger().info(f"TCP frame: '{self.tcp_frame}' (crisp end_effector_frame)")
        self._ee_client.call_async(
            GetParameters.Request(names=['end_effector_frame'])).add_done_callback(done)

    def _on_wrench_point(self, msg: PoseStamped):
        q = msg.pose.position
        pos = np.array([q.x, q.y, q.z])
        if np.all(np.isfinite(pos)):
            self.wrench_point = pos
            self.wrench_point_time = self._now()

    def _to_axes(self, wrench_base):
        """Base-frame wrench about the reference point -> about the TCP, in the law's axes."""
        point = None
        if self._now() - self.wrench_point_time < self.pose_timeout:
            point = self.wrench_point
            if not self._wrench_point_logged:
                self._wrench_point_logged = True
                r = self.wrench_point - self.curr_pos
                self.get_logger().info(
                    f'wrench reference point is {np.linalg.norm(r) * 1000:.0f} mm from the TCP '
                    f'(base {np.round(r, 4).tolist()}): torque moved to the TCP')
        elif self.topic_wrench_point:
            self.get_logger().warning(
                f'no {self.topic_wrench_point}: torque NOT moved to the TCP',
                throttle_duration_sec=5.0)
        R_axes = self.curr_R if self.axes == 'tcp' else None
        return to_axes(wrench_base, self.curr_pos, R_axes, point)

    def _on_wrench(self, msg: WrenchStamped):
        f, t = msg.wrench.force, msg.wrench.torque
        raw = np.array([f.x, f.y, f.z, t.x, t.y, t.z])
        if not np.all(np.isfinite(raw)):
            self.get_logger().warning('Ignoring non-finite wrench.', throttle_duration_sec=1.0)
            return
        stamp = msg.header.stamp.sec + 1e-9 * msg.header.stamp.nanosec

        was_tared = self.wrench.tared
        filtered = self.wrench.update(raw, stamp, self.nominal_dt)
        if self.wrench.tare_event == 'rejected':
            std = self.wrench.tare_std
            self.get_logger().warning(
                f'wrench varied during the tare (std F {std[:3].max():.2f} N, '
                f'T {std[3:].max():.3f} Nm): robot touched or moving? retrying')
        elif self.wrench.tare_event == 'done':
            self.curr_wrench = None
            bias = self.wrench.bias
            self.get_logger().info(
                f'Wrench tare done: bias F={np.round(bias[:3], 3)} N, '
                f'T={np.round(bias[3:], 4)} Nm (base frame)')
        if filtered is None or not was_tared:
            return
        if self.curr_pos is None:
            return   # no TCP pose yet to express the wrench in
        self.curr_wrench = self._to_axes(filtered)
        self.wrench_time = self._now()

        out = WrenchStamped()
        out.header.stamp = msg.header.stamp
        out.header.frame_id = self.axes_frame
        w = out.wrench
        w.force.x, w.force.y, w.force.z, w.torque.x, w.torque.y, w.torque.z = map(
            float, self.curr_wrench)
        self.wrench_pub.publish(out)

    def _frame_ok(self, msg, what, frame):
        if msg.header.frame_id and msg.header.frame_id != frame:
            self.get_logger().warning(
                f"{what} frame '{msg.header.frame_id}' != '{frame or 'TCP (name unknown yet)'}'"
                ', ignored', throttle_duration_sec=2.0)
            return False
        return True

    def _on_target_pose(self, msg: PoseStamped):
        if not self._frame_ok(msg, 'target pose', self.base_frame):
            return
        q, o = msg.pose.position, msg.pose.orientation
        pos = np.array([q.x, q.y, q.z])
        quat = np.array([o.x, o.y, o.z, o.w])
        if np.linalg.norm(quat) < 1e-8 or not (np.all(np.isfinite(quat)) and
                                                 np.all(np.isfinite(pos))):
            self.get_logger().warning('Ignoring invalid target pose.')
            return
        # The virtual state is not teleported: pose axes move toward the new target.
        self.law.set_target_pose(pos, _xyzw_to_rot(o))
        self.target_pose_received = True

    def _on_target_wrench(self, msg: WrenchStamped):
        if not self._frame_ok(msg, 'target wrench', self.axes_frame):
            return
        f, t = msg.wrench.force, msg.wrench.torque
        wrench = np.array([f.x, f.y, f.z, t.x, t.y, t.z], dtype=float)
        if not np.all(np.isfinite(wrench)):
            self.get_logger().warning('Ignoring non-finite target wrench.')
            return
        self.law.set_target_wrench(wrench)   # norm-limited to limits.max_wrench
        self.des_wrench_time = self._now()

    def _on_control_mode(self, msg: UInt8MultiArray):
        if len(msg.data) != 6 or any(int(v) not in (0, 1) for v in msg.data):
            self.get_logger().warning(
                'Control mode must contain exactly six 0/1 values '
                'in [x,y,z,roll,pitch,yaw] order; 1=pose, 0=force.')
            return
        mode = np.asarray(msg.data, dtype=float)
        if np.array_equal(mode, self.pose_mode):
            return   # clients may republish the mask every cycle
        if np.any(mode == 0) and not np.any(self.pose_mode == 0):
            # Entering force mode starts the force-target timeout from now.
            self.des_wrench_time = self._now()
        self.law.set_mask(mode)
        # Preserve the virtual state; resetting it can create a discontinuity.
        self.get_logger().info(f'Updated pose/force axis mask ({self.axes} axes): {list(msg.data)}')

    def _on_tare(self, request, response):
        if np.any(self.pose_mode == 0):
            response.success = False
            response.message = 'switch every axis to pose mode (1) before taring'
            return response
        self.wrench.reset()
        self.curr_wrench = None
        self.law.hold()
        response.success = True
        response.message = 'taring: do not touch the robot'
        self.get_logger().info('Re-taring the wrench. DO NOT TOUCH THE ROBOT.')
        return response

    # ------------------------------------------------------------------ control loop
    def _hold(self):
        """Stop the virtual motion and keep commanding the virtual pose."""
        self.law.hold()
        self.last_update_time = self._now()
        if self.law.virtual_pos is not None:
            self._publish_pose(self.law.virtual_pos, self.law.virtual_R)

    def admittance_control(self):
        if self.curr_pos is None or not self.wrench.tared or self.curr_wrench is None:
            return   # waiting for data or taring: publish nothing, the inner loop holds
        now = self._now()
        law = self.law

        # Safe default: initialize the virtual and target pose at the measured pose.
        if law.virtual_pos is None:
            if not self.target_pose_received:
                law.set_target_pose(self.curr_pos, self.curr_R)
                self.get_logger().info('Initialized target pose from the measured pose.')
            else:
                self.get_logger().info('Initialized virtual pose; keeping received target pose.')
            law.reset(self.curr_pos, self.curr_R)
            self.get_logger().info(
                f'Hybrid admittance running ({self.axes} axes). Pose: {self.topic_target_pose}; '
                f'wrench: {self.topic_target_wrench}; mode: {self.topic_mode} '
                f'(now {self.pose_mode.astype(int).tolist()}).')
            self._publish_pose(law.virtual_pos, law.virtual_R)
            self.last_update_time = now
            return

        if now - self.pose_time > self.pose_timeout or now - self.wrench_time > self.wrench_timeout:
            if not self._stale_logged:
                self.get_logger().warning('Pose / wrench stale; holding last commanded pose.')
                self._stale_logged = True
            self._hold()
            return
        if self._stale_logged:
            self.get_logger().info('Pose / wrench back.')
            self._stale_logged = False

        if (self.force_target_timeout > 0.0 and np.any(self.pose_mode < 0.5) and
                now - self.des_wrench_time > self.force_target_timeout):
            self.get_logger().warning(
                f'no {self.topic_target_wrench} for {self.force_target_timeout:.1f} s with a '
                'force axis active: all axes back to pose mode, holding the measured pose')
            law.set_mask(np.ones(6))
            law.set_target_wrench(np.zeros(6))
            law.set_target_pose(self.curr_pos, self.curr_R)

        dt = now - self.last_update_time if self.last_update_time is not None else self.nominal_dt
        self.last_update_time = now
        # Guard against timer jitter / long pauses.
        dt = float(np.clip(dt, 0.001, 0.02))

        pos, R = law.step(self.curr_wrench, dt, self.curr_pos, self.curr_R)
        if law.reverted:
            self.get_logger().error('Non-finite command detected; holding last good pose.')
            self._hold()
            return
        self._publish_pose(pos, R)

    def _publish_pose(self, pos, R):
        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.base_frame
        msg.pose.position.x, msg.pose.position.y, msg.pose.position.z = map(float, pos)
        w, x, y, z = rot_to_quat(R)
        o = msg.pose.orientation
        o.x, o.y, o.z, o.w = float(x), float(y), float(z), float(w)
        self.cmd_pub.publish(msg)


def _raise_interrupt(signum, frame):
    raise KeyboardInterrupt


def main():
    # Own signal handling: rclpy's handler would shut the context down first, and the final
    # hold-pose command below could then not be published.
    signal.signal(signal.SIGTERM, _raise_interrupt)
    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
    node = AdmittanceController()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        # Hold the measured pose on shutdown so no push is left behind (the inner loop keeps its
        # last target).
        try:
            if node.curr_pos is not None and node.law.virtual_pos is not None:
                node._publish_pose(node.curr_pos, node.curr_R)
                node.get_logger().info('shutdown: commanded the measured pose')
        except Exception as e:  # noqa: BLE001
            node.get_logger().error(f'shutdown: could not command the measured pose ({e})')
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()

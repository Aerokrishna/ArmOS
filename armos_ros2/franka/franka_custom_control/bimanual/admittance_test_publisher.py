#!/usr/bin/python3
"""Static test publisher for armos_control's admittance_controller.

Latches the first current_pose as the reference, then keeps publishing (topics are relative,
so they follow the robot namespace):
  admittance/control_mode    UInt8MultiArray  `control_mode` parameter (TCP axes)
  admittance/target_pose     PoseStamped      reference pose + `pose_offset` (base frame)
  admittance/target_wrench   WrenchStamped    `target_wrench` parameter (TCP frame)
and prints the tared TCP-frame wrench from admittance/wrench.

pose_offset is [dx, dy, dz, droll, dpitch, dyaw] (m, rad) in the base frame; the rotation is
applied about base axes, at the reference position. Parameters are re-read every cycle, so
they can be changed live:
  ros2 param set /<ns>/admittance_test_publisher pose_offset "[0.0, 0.0, 0.02, 0.0, 0.0, 0.0]"
  ros2 param set /<ns>/admittance_test_publisher control_mode "[1, 1, 0, 1, 1, 1]"
  ros2 param set /<ns>/admittance_test_publisher target_wrench "[0.0, 0.0, 5.0, 0.0, 0.0, 0.0]"
Set `rebase` to true to re-latch the reference from the current pose.

Run:
  ros2 run franka_custom_control admittance_test_publisher.py --ros-args -r __ns:=/<robot_ns> \
      -p pose_offset:="[0.0, 0.0, 0.02, 0.0, 0.0, 0.0]"
"""

import math

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped, WrenchStamped
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import qos_profile_sensor_data
from std_msgs.msg import UInt8MultiArray


def quat_mul(a, b):
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return np.array([aw * bx + ax * bw + ay * bz - az * by,
                     aw * by - ax * bz + ay * bw + az * bx,
                     aw * bz + ax * by - ay * bx + az * bw,
                     aw * bw - ax * bx - ay * by - az * bz])


def quat_from_rpy(r, p, y):
    cr, sr = math.cos(r / 2), math.sin(r / 2)
    cp, sp = math.cos(p / 2), math.sin(p / 2)
    cy, sy = math.cos(y / 2), math.sin(y / 2)
    return np.array([sr * cp * cy - cr * sp * sy,
                     cr * sp * cy + sr * cp * sy,
                     cr * cp * sy - sr * sp * cy,
                     cr * cp * cy + sr * sp * sy])


class AdmittanceTestPublisher(Node):

    def __init__(self):
        super().__init__('admittance_test_publisher')
        p = self.declare_parameter
        p('control_mode', [1, 1, 0, 1, 1, 1])
        p('pose_offset', [0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        p('target_wrench', [0.0, 0.0, 8.0, 0.0, 0.0, 0.0])
        p('rebase', False)
        self.base_frame = p('base_frame', 'base').value
        rate_hz = float(p('rate_hz', 50.0).value)
        print_hz = float(p('print_hz', 2.0).value)

        self.curr_pos = None
        self.curr_quat = None
        self.ref_pos = None
        self.ref_quat = None
        self.wrench = None

        self.mode_pub = self.create_publisher(UInt8MultiArray, 'admittance/control_mode', 10)
        self.pose_pub = self.create_publisher(PoseStamped, 'admittance/target_pose', 10)
        self.wrench_pub = self.create_publisher(WrenchStamped, 'admittance/target_wrench', 10)
        self.create_subscription(PoseStamped, 'current_pose', self._on_pose,
                                 qos_profile_sensor_data)
        self.create_subscription(WrenchStamped, 'admittance/wrench', self._on_wrench, 10)
        self.create_timer(1.0 / rate_hz, self._publish)
        self.create_timer(1.0 / print_hz, self._print)
        self.get_logger().info('waiting for current_pose ...')

    def _on_pose(self, msg: PoseStamped):
        q, o = msg.pose.position, msg.pose.orientation
        self.curr_pos = np.array([q.x, q.y, q.z])
        self.curr_quat = np.array([o.x, o.y, o.z, o.w])
        if self.ref_pos is None:
            self._latch()

    def _on_wrench(self, msg: WrenchStamped):
        f, t = msg.wrench.force, msg.wrench.torque
        self.wrench = np.array([f.x, f.y, f.z, t.x, t.y, t.z])

    def _latch(self):
        self.ref_pos = self.curr_pos.copy()
        self.ref_quat = self.curr_quat.copy()
        self.get_logger().info(
            f'reference pose: pos {np.round(self.ref_pos, 4).tolist()} '
            f'quat {np.round(self.ref_quat, 4).tolist()}')

    def _publish(self):
        if self.ref_pos is None:
            return
        if self.get_parameter('rebase').value:
            self._latch()
            self.set_parameters([Parameter('rebase', value=False)])

        mode = [int(v) for v in self.get_parameter('control_mode').value]
        offset = np.asarray(self.get_parameter('pose_offset').value, dtype=float)
        target_wrench = [float(v) for v in self.get_parameter('target_wrench').value]
        if len(mode) != 6 or len(offset) != 6 or len(target_wrench) != 6:
            self.get_logger().error('control_mode, pose_offset and target_wrench need 6 values',
                                    throttle_duration_sec=2.0)
            return
        stamp = self.get_clock().now().to_msg()

        self.mode_pub.publish(UInt8MultiArray(data=mode))

        pos = self.ref_pos + offset[:3]
        quat = quat_mul(quat_from_rpy(*offset[3:]), self.ref_quat)
        pose = PoseStamped()
        pose.header.stamp = stamp
        pose.header.frame_id = self.base_frame
        pose.pose.position.x, pose.pose.position.y, pose.pose.position.z = map(float, pos)
        o = pose.pose.orientation
        o.x, o.y, o.z, o.w = map(float, quat / np.linalg.norm(quat))
        self.pose_pub.publish(pose)

        wrench = WrenchStamped()
        wrench.header.stamp = stamp
        wrench.header.frame_id = ''   # empty = the controller's TCP frame
        w = wrench.wrench
        w.force.x, w.force.y, w.force.z, w.torque.x, w.torque.y, w.torque.z = target_wrench
        self.wrench_pub.publish(wrench)

    def _print(self):
        if self.wrench is None:
            self.get_logger().info('no admittance/wrench yet (controller still taring?)')
            return
        line = (f'F [{self.wrench[0]:+6.2f} {self.wrench[1]:+6.2f} {self.wrench[2]:+6.2f}] N  '
                f'T [{self.wrench[3]:+5.2f} {self.wrench[4]:+5.2f} {self.wrench[5]:+5.2f}] Nm')
        if self.curr_pos is not None and self.ref_pos is not None:
            d = (self.curr_pos - self.ref_pos) * 1000.0
            line += f'  dpos [{d[0]:+6.1f} {d[1]:+6.1f} {d[2]:+6.1f}] mm'
        self.get_logger().info(line)


def main():
    rclpy.init()
    node = AdmittanceTestPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()

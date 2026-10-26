#!/usr/bin/python3
"""Compliant spiral peg-in-hole search, on top of armos_control's admittance_controller.

Real-robot port of the scripted spiral from the MuJoCo test bed (SpiralResidual in
mujoco_rl_bed/projects/compliant_RL/terms.py), without the catch rule or RL residual.

The admittance node runs the tool z axis as a force axis (control_mode [1, 1, 0, 1, 1, 1],
TCP axes: the node's `axes` parameter must be tcp, the default), like the sim's force-controlled
axial axis, and the rest as pose axes:
    admittance/control_mode    UInt8MultiArray  axis mask, republished every cycle
    admittance/target_wrench   WrenchStamped    TCP frame (empty frame_id): force.z = +f
                                                pushes along the tool axis (robot-on-env)
    admittance/target_pose     PoseStamped      xy spiral + frozen orientation, in `base`
                                                (z is ignored while z is a force axis)
    admittance/wrench          WrenchStamped    the node's tared TCP-frame wrench (contact)
Poses are crisp's end_effector_frame (fr3_hand_tcp) in `base`. The tool must point straight
down (checked at start, max_tilt), so the tool z force axis is the vertical insertion axis.
Everything is relative to the touchdown height, so the peg-tip offset from the TCP does not
need to be known.

Phases:
    APPROACH  push f_approach in free space: the node descends at ~f_approach / D_force (5 mm/s
              at 3 N / 600) until the arm has STALLED (|dz/dt| < v_stall for t_settle) with
              the force above f_contact -> z_touch. Force alone is not enough: Franka's
              estimate reads joint friction as ~2 N of push while the arm moves in free air.
              Forces are relative to the reading while holding still just before the
              approach (removes drift since the node's tare).
    SEARCH    xy follows an Archimedean spiral (r = b*theta, b = pitch / 2pi, constant arc
              speed v, out to r_max and back, repeated) around the start xy;
              z_touch - z > drop_th -> INSERT
    INSERT    hold xy where the drop happened, push f_insert;
              z_touch - z >= success_depth -> inserted
    END       every axis back to pose mode: hold the measured pose (inserted) or lift
              `retract` above the touchdown (failures, Ctrl+C)

The node's force_target_timeout (1 s) also releases the push if this script dies.

Timing uses the node's ROS clock: wall time on the robot, /clock in the armos sim (run the node with
use_sim_time:=true there), so the same script works on both.
"""

import math
from collections import deque
from dataclasses import dataclass

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped, WrenchStamped
from rclpy.executors import ExternalShutdownException
from rclpy.qos import qos_profile_sensor_data
from std_msgs.msg import UInt8MultiArray

POSE_MODE = [1, 1, 1, 1, 1, 1]
Z_FORCE_MODE = [1, 1, 0, 1, 1, 1]


@dataclass
class SpiralParams:
    # spiral (sim: v 0.01, pitch 0.0008, r_max 0.0046)
    v: float = 0.01                # arc speed along the spiral [m/s] (sim 0.01)
    pitch: float = 0.0008          # ring spacing [m]; keep below ~2x the clearance
    r_max: float = 0.005           # search radius [m]
    # forces (pressing down, [N] > 0)
    f_approach: float = 3.0        # push while approaching; must exceed f_contact
    f_search: float = 3.0          # push while spiralling; low = less sliding friction (sim 1.5)
    f_insert: float = 5.0          # push once dropped
    f_contact: float = 2.0         # touchdown threshold
    # touchdown = force above f_contact AND descent stalled below v_stall for t_settle
    v_stall: float = 0.002         # [m/s]
    t_settle: float = 0.2          # [s]
    v_window: float = 0.1          # z-velocity estimation window [s]
    # detection
    max_approach: float = 0.030    # give up after this much descent [m]
    drop_th: float = 0.002         # z below touchdown -> in the hole [m]
    success_depth: float = 0.010   # z below touchdown -> inserted [m]; < hole depth
    # timeouts [s]
    t_approach: float = 20.0
    t_search: float = 60.0
    t_insert: float = 10.0
    # end
    retract: float = 0.020         # lift above touchdown after a failure [m]
    t_end: float = 3.0             # max time for the release / retract [s]
    # misc
    rate: float = 100.0            # publish rate [Hz]
    log_period: float = 0.5        # [s]
    stale_timeout: float = 0.2     # abort when pose / wrench are older than this [s]
    max_tilt: float = 15.0         # tool z must be within this of straight down [deg]
    base_frame: str = 'base'
    topic_pose: str = 'admittance/target_pose'
    topic_wrench: str = 'admittance/target_wrench'
    topic_mode: str = 'admittance/control_mode'
    topic_current_pose: str = 'current_pose'
    topic_current_wrench: str = 'admittance/wrench'


class Spiral:
    """Archimedean spiral r = b * theta at constant arc speed, out to r_max and back to 0."""

    def __init__(self, v, pitch, r_max):
        self.v, self.b, self.r_max = v, pitch / (2.0 * math.pi), r_max
        self.theta = 0.0
        self.out = True

    @property
    def r(self):
        return self.b * self.theta

    def step(self, dt):
        """Advance by dt and return the (dx, dy) offset from the centre [m]."""
        r = self.r
        if self.out and r >= self.r_max:
            self.out = False
        elif not self.out and r <= 0.0:
            self.out = True
        dth = self.v * dt / math.hypot(r, self.b)   # |d(r cos, r sin)/dtheta| = sqrt(r^2 + b^2)
        self.theta = max(0.0, self.theta + (dth if self.out else -dth))
        r = self.r
        return r * math.cos(self.theta), r * math.sin(self.theta)


def _now(node):
    """ROS time [s]: wall time on the robot, sim time with use_sim_time."""
    return node.get_clock().now().nanoseconds * 1e-9


class _Io:
    """Subscriptions and publishers on the caller's node."""

    def __init__(self, node, p):
        self.node, self.p = node, p
        self.pos = self.quat = None
        self.force = np.zeros(3)
        self.pose_stamp = self.wrench_stamp = -math.inf
        self.subs = [
            node.create_subscription(PoseStamped, p.topic_current_pose, self.on_pose,
                                     qos_profile_sensor_data),
            node.create_subscription(WrenchStamped, p.topic_current_wrench, self.on_wrench, 10),
        ]
        self.pose_pub = node.create_publisher(PoseStamped, p.topic_pose, 10)
        self.wrench_pub = node.create_publisher(WrenchStamped, p.topic_wrench, 10)
        self.mode_pub = node.create_publisher(UInt8MultiArray, p.topic_mode, 10)

    def on_pose(self, msg):
        q, o = msg.pose.position, msg.pose.orientation
        self.pos = np.array([q.x, q.y, q.z])
        self.quat = np.array([o.x, o.y, o.z, o.w])
        self.pose_stamp = _now(self.node)

    def on_wrench(self, msg):
        f = msg.wrench.force
        self.force = np.array([f.x, f.y, f.z])   # TCP frame, robot on env: pressing > 0
        self.wrench_stamp = _now(self.node)

    def fresh(self):
        now = _now(self.node)
        return (now - self.pose_stamp < self.p.stale_timeout and
                now - self.wrench_stamp < self.p.stale_timeout)

    def connected(self):
        return all(pub.get_subscription_count() > 0
                   for pub in (self.pose_pub, self.wrench_pub, self.mode_pub))

    def send(self, mode, xyz, quat, f_down=0.0):
        """One command cycle: axis mask, target pose, target wrench (f_down > 0 pushes along the
        tool axis, i.e. down)."""
        stamp = self.node.get_clock().now().to_msg()
        self.mode_pub.publish(UInt8MultiArray(data=list(mode)))
        msg = PoseStamped()
        msg.header.stamp, msg.header.frame_id = stamp, self.p.base_frame
        msg.pose.position.x, msg.pose.position.y, msg.pose.position.z = map(float, xyz)
        o = msg.pose.orientation
        o.x, o.y, o.z, o.w = map(float, quat)
        self.pose_pub.publish(msg)
        w = WrenchStamped()
        w.header.stamp, w.header.frame_id = stamp, ''   # empty = the node's TCP frame
        w.wrench.force.z = float(f_down)
        self.wrench_pub.publish(w)

    def close(self):
        for s in self.subs:
            self.node.destroy_subscription(s)
        for pub in (self.pose_pub, self.wrench_pub, self.mode_pub):
            self.node.destroy_publisher(pub)


def _spin_until(node, t_end):
    """Spin until the node's ROS clock reaches t_end (short waits: sim time may run off wall time)."""
    while True:
        left = t_end - _now(node)
        if left <= 0.0:
            return
        rclpy.spin_once(node, timeout_sec=min(left, 0.01))


def spiral_insert(node, p=None, center_xy=None):
    """Run approach -> spiral search -> insert. Blocks until done.

    Start with the peg vertical, a few mm above the hole estimate, with the admittance controller
    running and tared (and nothing else publishing target_pose or the admittance topics).

    Args:
        node: an rclpy Node (not spun by anyone else while this runs).
        p: SpiralParams.
        center_xy: spiral centre in `base` [m]; default: the TCP xy at start.

    Returns:
        'inserted', 'not_found', 'jammed', 'no_contact' or 'aborted'.
    """
    p = p or SpiralParams()
    log = node.get_logger()
    io = _Io(node, p)
    state = {'z_touch': None, 'quat': None}
    result = 'aborted'
    try:
        result = _run(node, p, io, log, center_xy, state)
    except (KeyboardInterrupt, ExternalShutdownException):
        log.warn('interrupted')
    finally:
        try:
            _end(node, p, io, log, state, lift=(result != 'inserted'))
        except Exception as e:   # context already shut down: the node's timeout releases
            log.error(f'could not send the release ({e}); the admittance node falls back to '
                      'pose mode after its force_target_timeout')
        try:
            io.close()
        except Exception:
            pass
    return result


def _run(node, p, io, log, center_xy, state):
    dt = 1.0 / p.rate

    log.info(f'waiting for {p.topic_current_pose}, {p.topic_current_wrench} (admittance node '
             'running and tared) and for the admittance node to subscribe')
    while not (io.fresh() and io.connected()):
        rclpy.spin_once(node, timeout_sec=0.1)
    pos0, quat0 = io.pos.copy(), io.quat.copy()
    state['quat'] = quat0
    x, y, _, _ = quat0 / np.linalg.norm(quat0)
    tool_z_down = -(1.0 - 2.0 * (x * x + y * y))   # -(base z component of the tool z axis)
    tilt = math.degrees(math.acos(float(np.clip(tool_z_down, -1.0, 1.0))))
    if tilt > p.max_tilt:
        raise ValueError(f'tool z is {tilt:.0f} deg from straight down (max {p.max_tilt}): '
                         'the force axis is the tool z axis, point the peg down first')

    c = np.array(center_xy, dtype=float) if center_xy is not None else pos0[:2].copy()
    if np.linalg.norm(c - pos0[:2]) > 0.02:
        raise ValueError(f'center_xy {c} is more than 2 cm from the start xy {pos0[:2]}')

    # Pin the node's target to the current pose before touching the mode (its target may be
    # an older one), and take the force baseline while holding still.
    t_end = _now(node) + 0.5
    baseline = []
    while _now(node) < t_end:
        io.send(POSE_MODE, pos0, quat0)
        _spin_until(node, _now(node) + dt)
        baseline.append(io.force[2])
    f0 = float(np.mean(baseline[len(baseline) // 2:]))
    log.info(f'force baseline {f0:+.2f} N (subtracted)')
    z_hist = deque()
    t_stalled = 0.0

    spiral = Spiral(p.v, p.pitch, p.r_max)
    phase, t_phase = 'APPROACH', 0.0
    z_drop = None
    hold_xy = c
    t_next = _now(node)
    t_log = 0.0
    log.info(f'APPROACH: pushing {p.f_approach} N on z')
    while True:
        t_next += dt
        _spin_until(node, t_next)
        if not io.fresh():
            log.error('pose / wrench stale: aborting')
            return 'aborted'
        t_phase += dt
        z, f_down = io.pos[2], io.force[2] - f0
        z_touch = state['z_touch']
        now = _now(node)
        z_hist.append((now, z))
        while now - z_hist[0][0] > p.v_window:
            z_hist.popleft()
        span = now - z_hist[0][0]
        vz = (z - z_hist[0][1]) / span if span > 0.5 * p.v_window else math.inf

        if phase == 'APPROACH':
            io.send(Z_FORCE_MODE, np.r_[c, z], quat0, p.f_approach)
            t_stalled = t_stalled + dt if (abs(vz) < p.v_stall and
                                           f_down > p.f_contact) else 0.0
            if t_stalled >= p.t_settle:
                state['z_touch'] = z
                phase, t_phase = 'SEARCH', 0.0
                log.info(f'touchdown at z {z:.4f} (F {f_down:.1f} N, stalled '
                         f'{t_stalled:.2f} s) -> SEARCH')
            elif pos0[2] - z > p.max_approach or t_phase > p.t_approach:
                return 'no_contact'

        elif phase == 'SEARCH':
            dx, dy = spiral.step(dt)
            io.send(Z_FORCE_MODE, np.r_[c[0] + dx, c[1] + dy, z], quat0, p.f_search)
            if z_touch - z > p.drop_th:
                hold_xy, z_drop = io.pos[:2].copy(), z
                phase, t_phase = 'INSERT', 0.0
                log.info(f'drop {(z_touch - z) * 1e3:.1f} mm at r {spiral.r * 1e3:.1f} mm '
                         f'-> INSERT ({p.f_insert} N)')
            elif t_phase > p.t_search:
                return 'not_found'

        else:   # INSERT
            io.send(Z_FORCE_MODE, np.r_[hold_xy, z], quat0, p.f_insert)
            if z_touch - z >= p.success_depth:
                return 'inserted'
            if t_phase > p.t_insert:
                return 'jammed'

        t_log += dt
        if t_log >= p.log_period:
            t_log = 0.0
            travel = 0.0 if z_touch is None else (z_touch - z) * 1e3
            log.info(f'{phase:8s} t {t_phase:5.1f} s  r {spiral.r * 1e3:4.1f} mm  '
                     f'F {f_down:5.1f} N  vz {vz * 1e3:+5.1f} mm/s  travel {travel:5.1f} mm')


def _end(node, p, io, log, state, lift):
    """All axes back to pose mode; hold the measured pose, or lift above the touchdown."""
    if io.pos is None:
        return
    dt = 1.0 / p.rate
    quat = state['quat'] if state['quat'] is not None else io.quat.copy()
    xy, z = io.pos[:2].copy(), io.pos[2]
    if lift:
        z_ref = state['z_touch'] if state['z_touch'] is not None else z
        target = np.r_[xy, max(z, z_ref) + p.retract]
        log.info(f'releasing and lifting to z {target[2]:.4f}')
    else:
        target = np.r_[xy, z]
        log.info('releasing the push, holding the pose')
    t_end = _now(node) + p.t_end
    while _now(node) < t_end:
        io.send(POSE_MODE, target, quat)
        _spin_until(node, _now(node) + dt)
        if lift and io.pos is not None and io.pos[2] > target[2] - 0.002:
            break

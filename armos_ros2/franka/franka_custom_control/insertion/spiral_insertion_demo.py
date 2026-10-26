#!/usr/bin/python3
"""Spiral peg insertion example, with an optional hole calibration.

    ros2 launch armos_control admittance.launch.py use_sim:=false load_gripper:=true
    # (crisp cartesian_controller + armos_control admittance_controller; do not touch: wrench tare)
    # grasp the peg, close the UI (it also publishes target_pose)

Calibrate (once, or whenever the hole moves):
    ros2 run franka_custom_control spiral_insertion_demo.py --calibration true
  Translation becomes hand-guidable (rotation is held, --guide-axes): push the peg fully into
  the hole, let go, press Enter. The TCP pose is saved to --calib-file and the run continues
  like a normal one.

Normal run (uses --calib-file when it exists):
    ros2 run franka_custom_control spiral_insertion_demo.py --xy-error 0.003 0.004 \
        --tilt-error 2
  Lifts straight up to --start-height above the calibrated pose, moves to the calibrated xy
  plus a random --xy-error offset, tilted by --tilt-error about a random horizontal axis, then
  runs the spiral insertion from there. --from-here skips the move and starts where the arm is.
"""

import argparse
import math
import os
import re
import sys
import threading
import time

import numpy as np
import rclpy
import yaml
from rclpy.node import Node
from rclpy.signals import SignalHandlerOptions

from spiral_search import POSE_MODE, SpiralParams, _Io, _now, _spin_until, spiral_insert

DEFAULT_CALIB = os.path.expanduser('~/.ros/spiral_insertion_hole.yaml')


# ---------------------------------------------------------------- quaternions (x, y, z, w)
def quat_mul(a, b):
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return np.array([aw * bx + ax * bw + ay * bz - az * by,
                     aw * by - ax * bz + ay * bw + az * bx,
                     aw * bz + ax * by - ay * bx + az * bw,
                     aw * bw - ax * bx - ay * by - az * bz])


def quat_axis_angle(axis, angle):
    axis = np.asarray(axis, dtype=float) / np.linalg.norm(axis)
    return np.r_[axis * math.sin(angle / 2.0), math.cos(angle / 2.0)]


def quat_angle(a, b):
    return 2.0 * math.acos(min(1.0, abs(float(np.dot(a, b)))))


def slerp(a, b, s):
    a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
    d = float(np.dot(a, b))
    if d < 0.0:
        b, d = -b, -d
    if d > 0.9995:
        q = a + s * (b - a)
        return q / np.linalg.norm(q)
    th = math.acos(d)
    return (math.sin((1 - s) * th) * a + math.sin(s * th) * b) / math.sin(th)


# ---------------------------------------------------------------- calibration file
def save_calibration(path, pos, quat):
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with open(path, 'w') as f:
        yaml.safe_dump({'hole_pose': {
            'frame': 'base',
            'position': [float(v) for v in pos],
            'orientation_xyzw': [float(v) for v in quat],
            'stamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        }}, f, sort_keys=False)


def load_calibration(path):
    with open(path) as f:
        h = yaml.safe_load(f)['hole_pose']
    return np.array(h['position'], dtype=float), np.array(h['orientation_xyzw'], dtype=float)


# ---------------------------------------------------------------- robot helpers
def wait_ready(node, io):
    node.get_logger().info('waiting for current_pose and admittance/wrench (admittance node '
                           'running and tared)')
    while not (io.fresh() and io.connected()):
        rclpy.spin_once(node, timeout_sec=0.1)


def hold(node, io, p, xyz, quat, seconds):
    dt = 1.0 / p.rate
    t_end = _now(node) + seconds
    while _now(node) < t_end:
        io.send(POSE_MODE, xyz, quat)
        _spin_until(node, _now(node) + dt)


def calibrate(node, io, p, guide_axes):
    """Hand-guide on the force axes until Enter; returns the averaged TCP pose."""
    log = node.get_logger()
    pos0, quat0 = io.pos.copy(), io.quat.copy()
    hold(node, io, p, pos0, quat0, 0.3)
    enter = threading.Event()
    threading.Thread(target=lambda: (sys.stdin.readline(), enter.set()), daemon=True).start()
    log.info(f'CALIBRATION: axes {guide_axes} (0 = free) follow your hand. Push the peg fully '
             'into the hole, let go, then press Enter.')
    dt, t_log = 1.0 / p.rate, 0.0
    while not enter.is_set():
        if not io.fresh():
            raise RuntimeError('pose / wrench stale during calibration')
        io.send(guide_axes, pos0, quat0)        # zero target wrench on the free axes
        _spin_until(node, _now(node) + dt)
        t_log += dt
        if t_log >= 1.0:
            t_log = 0.0
            log.info(f'  TCP {np.round(io.pos, 4).tolist()}')
    here, q = io.pos.copy(), io.quat.copy()
    samples = []
    t_end = _now(node) + 0.5
    while _now(node) < t_end:            # hold still and average
        io.send(POSE_MODE, here, q)
        _spin_until(node, _now(node) + dt)
        samples.append(io.pos.copy())
    return np.mean(samples, axis=0), io.quat.copy()


def move_to(node, io, p, xyz, quat, speed, ang_speed, what):
    """Smooth straight-line move of the admittance target (all pose axes), then settle."""
    log = node.get_logger()
    p0, q0 = io.pos.copy(), io.quat.copy()
    dist, ang = float(np.linalg.norm(xyz - p0)), quat_angle(q0, quat)
    T = max(dist / speed, ang / ang_speed, 0.2)
    log.info(f'{what}: {dist * 1e3:.1f} mm, {math.degrees(ang):.1f} deg in {T:.1f} s')
    dt, t0 = 1.0 / p.rate, _now(node)
    while True:
        s = min(1.0, (_now(node) - t0) / T)
        sm = s * s * (3.0 - 2.0 * s)
        io.send(POSE_MODE, p0 + sm * (xyz - p0), slerp(q0, quat, sm))
        _spin_until(node, _now(node) + dt)
        if not io.fresh():
            raise RuntimeError(f'pose / wrench stale during {what}')
        if s >= 1.0:
            break
    t_end = _now(node) + 3.0
    while _now(node) < t_end:
        io.send(POSE_MODE, xyz, quat)
        _spin_until(node, _now(node) + dt)
        if np.linalg.norm(io.pos - xyz) < 0.0015 and quat_angle(io.quat, quat) < math.radians(1):
            return
    log.warning(f'{what}: settled {np.linalg.norm(io.pos - xyz) * 1e3:.1f} mm, '
                f'{math.degrees(quat_angle(io.quat, quat)):.1f} deg from the target; '
                'continuing')


def start_pose(hole_pos, hole_quat, a, rng):
    phi = rng.uniform(0.0, 2.0 * math.pi)
    r = rng.uniform(*a.xy_error)
    pos = hole_pos + [r * math.cos(phi), r * math.sin(phi), a.start_height]
    psi = rng.uniform(0.0, 2.0 * math.pi)       # tilt about a random horizontal base axis
    quat = quat_mul(quat_axis_angle([math.cos(psi), math.sin(psi), 0.0],
                                    math.radians(a.tilt_error)), hole_quat)
    return pos, quat, r, phi


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('1', 'true', 'yes', 'on'):
        return True
    if v.lower() in ('0', 'false', 'no', 'off'):
        return False
    raise argparse.ArgumentTypeError(f'expected true/false, got {v}')


def main():
    # Also accept the launch-style spelling `calibration:=true`.
    argv = []
    for tok in sys.argv[1:]:
        m = re.fullmatch(r'calibration:=(\w+)', tok)
        argv += ['--calibration', m.group(1)] if m else [tok]

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--v', type=float, default=0.01, help='spiral arc speed [m/s]')
    ap.add_argument('--pitch', type=float, default=0.0008, help='ring spacing [m]')
    ap.add_argument('--r-max', type=float, default=0.005, help='search radius [m]')
    ap.add_argument('--f-approach', type=float, default=3.0, help='push while approaching [N]')
    ap.add_argument('--f-search', type=float, default=3.0, help='push while searching [N]')
    ap.add_argument('--f-insert', type=float, default=6.0, help='push once dropped [N]')
    ap.add_argument('--f-contact', type=float, default=2.0, help='touchdown force [N]')
    ap.add_argument('--success-depth', type=float, default=0.008,
                    help='travel below touchdown that counts as inserted [m]')
    ap.add_argument('--t-search', type=float, default=60.0, help='search timeout [s]')
    ap.add_argument('--center', type=float, nargs=2, metavar=('X', 'Y'),
                    help='spiral centre in base [m] (default: TCP xy at start)')
    ap.add_argument('--max-tilt', type=float, default=15.0,
                    help='refuse to start with the tool tilted more than this [deg]')
    # calibration / start pose
    ap.add_argument('--calibration', type=str2bool, nargs='?', const=True, default=False,
                    help='hand-guide the peg into the hole and save that pose first')
    ap.add_argument('--calib-file', default=DEFAULT_CALIB, help='calibrated hole pose (yaml)')
    ap.add_argument('--from-here', action='store_true',
                    help='ignore the calibration file and start where the arm is')
    ap.add_argument('--xy-error', type=float, nargs=2, default=[0.003, 0.004],
                    metavar=('MIN', 'MAX'), help='start offset from the hole, random '
                    'direction, magnitude in [MIN, MAX] [m]')
    ap.add_argument('--tilt-error', type=float, default=0.0,
                    help='start tilt about a random horizontal axis [deg]')
    ap.add_argument('--start-height', type=float, default=0.03,
                    help='start this far above the calibrated (inserted) pose [m]; must clear '
                    'the inserted depth')
    ap.add_argument('--move-speed', type=float, default=0.02, help='transfer speed [m/s]')
    ap.add_argument('--move-ang-speed', type=float, default=0.2,
                    help='transfer rotation speed [rad/s]')
    ap.add_argument('--guide-axes', type=int, nargs=6, default=[0, 0, 0, 1, 1, 1],
                    metavar='M', help='control_mode while calibrating (0 = hand-guidable)')
    ap.add_argument('--seed', type=int, help='random seed for the start error')
    args, ros_args = ap.parse_known_args(argv)

    # No rclpy SIGINT handler: Ctrl+C raises KeyboardInterrupt with the context still alive,
    # so a release / hold can still be sent before shutting down.
    rclpy.init(args=[sys.argv[0]] + ros_args, signal_handler_options=SignalHandlerOptions.NO)
    node = Node('spiral_insertion')
    log = node.get_logger()
    p = SpiralParams(v=args.v, pitch=args.pitch, r_max=args.r_max, f_approach=args.f_approach,
                     f_search=args.f_search,
                     f_insert=args.f_insert, f_contact=args.f_contact,
                     success_depth=args.success_depth, t_search=args.t_search,
                     max_tilt=args.max_tilt)
    io = _Io(node, p)
    hole = None
    result = 'aborted'
    try:
        wait_ready(node, io)
        if args.calibration:
            hole = calibrate(node, io, p, args.guide_axes)
            save_calibration(args.calib_file, *hole)
            log.info(f'saved hole pose {np.round(hole[0], 4).tolist()} to {args.calib_file}. '
                     'Hands off: moving to the start pose in 2 s.')
            hold(node, io, p, hole[0], hole[1], 2.0)
        elif not args.from_here and os.path.exists(args.calib_file):
            hole = load_calibration(args.calib_file)
            log.info(f'hole pose from {args.calib_file}: {np.round(hole[0], 4).tolist()}')
        elif not args.from_here:
            log.warning(f'no {args.calib_file}: starting where the arm is '
                        '(run with --calibration true to make one)')

        if hole is not None:
            if args.tilt_error >= args.max_tilt:
                raise RuntimeError(f'--tilt-error {args.tilt_error} must be below --max-tilt '
                                   f'{args.max_tilt}, or the spiral refuses to start')
            rng = np.random.default_rng(args.seed)
            pos, quat, r, phi = start_pose(hole[0], hole[1], args, rng)
            log.info(f'start error: {r * 1e3:.1f} mm towards {math.degrees(phi):.0f} deg, '
                     f'tilt {args.tilt_error:.1f} deg')
            # Straight up first (out of the hole when calibrating), then across.
            up = np.r_[io.pos[:2], max(io.pos[2], pos[2])]
            move_to(node, io, p, up, io.quat.copy(), args.move_speed, args.move_ang_speed,
                    'lift')
            move_to(node, io, p, pos, quat, args.move_speed, args.move_ang_speed,
                    'to start pose')
            p.max_approach = max(p.max_approach, args.start_height + 0.01)
    except KeyboardInterrupt:
        log.warning('interrupted: holding the measured pose')
        hold(node, io, p, io.pos.copy(), io.quat.copy(), 0.3)
        node.destroy_node()
        rclpy.try_shutdown()
        return
    except RuntimeError as e:
        log.error(str(e))
        node.destroy_node()
        rclpy.try_shutdown()
        return

    # spiral_insert makes its own subscriptions: close ours first, or the single-threaded spin
    # serves these (created first) ahead of its own and its wrench goes stale.
    io.close()
    result = spiral_insert(node, p, center_xy=args.center)
    print(f'spiral insertion: {result}')
    if hole is not None:
        io = _Io(node, p)
        t_end = _now(node) + 1.0
        while io.pos is None and _now(node) < t_end:
            rclpy.spin_once(node, timeout_sec=0.1)
        if io.pos is not None:
            d = io.pos - hole[0]
            print(f'final TCP vs calibrated hole pose: xy {np.linalg.norm(d[:2]) * 1e3:.1f} mm, '
                  f'z {d[2] * 1e3:+.1f} mm')
        io.close()
    node.destroy_node()
    rclpy.try_shutdown()


if __name__ == '__main__':
    main()

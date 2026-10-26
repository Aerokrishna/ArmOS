#!/usr/bin/env python
"""Bimanual handover of a cylinder (5 cm diameter, 10 cm long, 0.1 kg) on the dual Panda, pure MuJoCo (no ROS).

Scene `dual_franka_handover.yaml`: the dual Panda, a table in front of the right arm and the cylinder lying on
it, axis along world y (from the right arm towards the left). The right arm picks it top-down near one end
and lifts it while the left arm moves to a waiting pose. Then both arms MEET at the handover in one motion:
the right arm carries the cylinder to the handover point and the left arm moves to its grasp on the free end.
The left arm closes, the right arm lets go and retreats, and the left arm carries the cylinder away. Every
arm motion goes through `armos_core.control.bimanual.BimanualSync` in the chosen mode and streams to each
arm's CARTESIAN_IMPEDANCE (crisp's gains), as in examples/bimanual_test.py.

The meeting is where the mode matters. In "temporal" mode both arms run on one phase, so they arrive at
the handover at the same time even though the right arm travels about twice as far, and the left hand's
path relative to the cylinder is a straight line from its waiting pose (outside and above the free end)
into the grasp. In "independent" mode each arm moves at its own limits: the left arm arrives first and
waits with an open hand while the cylinder is still on its way. Each move prints both arms' arrival times.
The left grasp at the meeting is planned on where the cylinder will be: its pose in the right hand,
measured before the motion, carried to the right arm's handover goal.

Grasps are planned on the cylinder's actual pose (privileged state from MuJoCo): each hand grasps at
`--grasp-offset` from the cylinder center along its axis, with the fingers across the diameter and the
hand's x axis along the cylinder axis, pointing to the other arm. Each hand is also tilted outward by
`--tilt` about its finger axis (that leaves the pads on the cylinder): with both hands straight down the
two wrists (link5 / link6, ~8 cm behind each hand) run into each other. Approach and retreat go along the
hand's own z axis. At 39 mm the two hands (64 mm thick) clear each other by 14 mm.

The Menagerie Franka Hand squeezes only ~1.25 N per finger on a 5 cm object (100 N/m on the tendon); the
gripper gains are scaled by `--grip-scale` (default 10: ~12.5 N per finger, the real hand gives up to 70 N).
MuJoCo's soft contacts let a grasp creep under a constant load: held 39 mm off-center, the cylinder slowly
turns about the finger axis (torsional friction). `--noslip` iterations (MuJoCo's noslip solver) remove that.

    python examples/bimanual_handover.py --mode temporal
    python examples/bimanual_handover.py --mode independent
    python examples/bimanual_handover.py --headless              # no viewer; prints the result
"""

import argparse

import mujoco
import numpy as np

from armos_core.control.bimanual import MODES, BimanualCfg, BimanualSync
from armos_core.math.se3 import make_T
from armos_sim import ASSETS_DIR, ControlMode, CyRoSimulator, Telemetry

CONFIG = ASSETS_DIR / "franka_emika_panda" / "dual_franka_handover.yaml"
ARMS = ("left", "right")
L, R = 0, 1                                     # indices into ARMS
CRISP_KP = [1500.0, 1500.0, 500.0, 30.0, 30.0, 30.0]
CRISP_KD = [80.0, 80.0, 30.0, 2.0, 2.0, 2.0]
TCP_IN_SITE = np.array([0.0, 0.0, 0.0129])       # finger-pad center below *_attachment_site (hand axes)
HANDOVER = np.array([0.45, 0.25, 0.40])         # cylinder center at the handover [m, world]
PLACE_LEFT = np.array([0.45, 0.62, 0.40])       # where the left arm carries it [m, world, its grasp point]
CYL_HALF_LEN = 0.05
LEFT_WAIT = np.array([0.45, 0.45, 0.50])        # left grasp-TCP waiting position before the meeting [m, world]
APPROACH = 0.12                                 # pre-grasp / retreat distance along the hand z axis [m]


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=MODES, default="temporal")
    ap.add_argument("--coupling", type=float, default=5.0, help="phase coupling k [1/s] (temporal)")
    ap.add_argument("--max-lin-vel", type=float, default=0.1, help="[m/s]")
    ap.add_argument("--max-ang-vel", type=float, default=0.5, help="[rad/s]")
    ap.add_argument("--max-lin-acc", type=float, default=0.25, help="[m/s^2]")
    ap.add_argument("--max-ang-acc", type=float, default=1.0, help="[rad/s^2]")
    ap.add_argument("--grasp-offset", type=float, default=0.039, help="grasp point from the cylinder center [m]")
    ap.add_argument("--tilt", type=float, default=25.0, help="outward hand tilt about the finger axis [deg]")
    ap.add_argument("--grip-scale", type=float, default=10.0, help="gripper gain scale (see the docstring)")
    ap.add_argument("--noslip", type=int, default=5, help="MuJoCo noslip iterations (see the docstring); 0 = off")
    ap.add_argument("--headless", action="store_true", help="no viewer")
    ap.add_argument("--debug", action="store_true", help="print the cylinder tilt and its contacts per stage")
    return ap.parse_args()


def main():
    args = parse_args()
    sim = CyRoSimulator(config_path=CONFIG, realtime_factor=0.0 if args.headless else 1.0)
    m, d = sim._mj.model, sim._mj.data
    cyl = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "cylinder")
    m.opt.noslip_iterations = args.noslip
    for a in ARMS:
        aid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{a}_actuator8")
        m.actuator_gainprm[aid, 0] *= args.grip_scale
        m.actuator_biasprm[aid, 1:3] *= args.grip_scale
        sim.set_impedance_gains(a, kp=CRISP_KP, kd=CRISP_KD, space="cartesian")
    T_wb = {a: make_T(sim.arm_config(a).base_world, sim.arm_config(a).base_world_rot) for a in ARMS}
    sync = BimanualSync(BimanualCfg(
        mode=args.mode, coupling=args.coupling, max_lin_vel=args.max_lin_vel,
        max_ang_vel=args.max_ang_vel, max_lin_acc=args.max_lin_acc, max_ang_acc=args.max_ang_acc))
    s = args.grasp_offset

    # ------------------------------------------------------------------ frames
    # BimanualSync works on grasp-TCP poses (finger-pad center) in the WORLD frame; the controllers take
    # the attachment-site pose in each arm's base frame.
    def site_to_tcp_world(arm, T_base_site):
        T = T_wb[arm] @ T_base_site
        return T[:3, 3] + T[:3, :3] @ TCP_IN_SITE, T[:3, :3]

    def command(arm, p_tcp_w, R_w):
        T_w_site = make_T(p_tcp_w - R_w @ TCP_IN_SITE, R_w)
        T_b = np.linalg.inv(T_wb[arm]) @ T_w_site
        sim.set_tcp_pose(arm, T_b[:3, 3], T_b[:3, :3])

    def cylinder():
        """(center, unit axis pointing to +y, from the right arm to the left) in the world frame."""
        axis = d.xmat[cyl].reshape(3, 3)[:, 2].copy()
        return d.xpos[cyl].copy(), axis if axis[1] >= 0.0 else -axis

    tilt = np.radians(args.tilt)
    Ry_tilt = np.array([[np.cos(tilt), 0.0, np.sin(tilt)], [0.0, 1.0, 0.0], [-np.sin(tilt), 0.0, np.cos(tilt)]])

    def grasp_rot(x_h):
        """Hand rotation with hand x along x_h (towards the other arm), fingers (hand y) across it, hand z
        (approach) as close to straight down as x_h allows, then tilted about hand y so that the wrist
        leans away from x_h (from the other arm)."""
        z = np.array([0.0, 0.0, -1.0])
        z = z - x_h * x_h.dot(z)
        z /= np.linalg.norm(z)
        return np.column_stack((x_h, np.cross(z, x_h), z)) @ Ry_tilt

    def grasp(sign, back=0.0):
        """Grasp pose on the actual cylinder at center + sign * s * axis (right arm: sign -1, left: +1),
        hand x pointing along the axis towards the cylinder center (and the other arm); `back` [m] moves
        it back along the approach axis (pre-grasp)."""
        c, ax = cylinder()
        R_g = grasp_rot(-sign * ax)
        return c + sign * s * ax - back * R_g[:, 2], R_g

    def retreat(i, dist):
        p, R_i = stay(i)
        return p - dist * R_i[:, 2], R_i

    # ------------------------------------------------------------------ task
    st = {"i": -1, "t0": 0.0, "home": None, "log": [], "telem": None, "arr": [None, None],
          "moving": [False, False]}

    def stay(i):
        return sync.poses()[i]

    def move(left=None, right=None, label=None):
        """A motion stage: per arm a callable -> (p, R) goal (world TCP) or None to stay."""
        return ("move", left, right, label)

    Y = np.array([0.0, 1.0, 0.0])

    def handover_right():
        return HANDOVER - s * Y, grasp_rot(Y)

    def meet_left():
        """Left grasp on the cylinder as it will be when the right arm reaches its handover goal: the
        cylinder's current pose in the right hand, carried to that goal."""
        p_r, R_r = site_to_tcp_world("right", st["telem"].tcp_pose["right"])
        c, ax = cylinder()
        c_in, ax_in = R_r.T @ (c - p_r), R_r.T @ ax
        p_g, R_g = handover_right()
        c_pred, ax_pred = p_g + R_g @ c_in, R_g @ ax_in
        ax_pred /= np.linalg.norm(ax_pred)
        return c_pred + s * ax_pred, grasp_rot(-ax_pred)

    stages = [
        ("grip", "left", 1.0, 0.0), ("grip", "right", 1.0, 1.0),                      # open both, settle
        move(left=lambda: (LEFT_WAIT, grasp_rot(-Y)), right=lambda: grasp(-1, APPROACH)),  # wait / pre-grasp
        move(right=lambda: grasp(-1)),
        ("grip", "right", 0.0, 1.0),                                                  # right closes
        move(right=lambda: (stay(R)[0] + [0.0, 0.0, APPROACH], stay(R)[1])),          # lift
        ("wait", 0.5),
        move(left=meet_left, right=handover_right, label="MEET at handover"),          # synchronized meeting
        ("grip", "left", 0.0, 1.0),                                                   # left closes
        ("grip", "right", 1.0, 1.0),                                                  # right lets go
        move(right=lambda: retreat(R, APPROACH)),                                     # right retreats
        move(left=lambda: (PLACE_LEFT, stay(L)[1]), right=lambda: st["home"][R]),      # carry away / home
        ("wait", 1.0),
    ]

    def describe(stage):
        if stage[0] == "move":
            return stage[3] or "move " + "+".join(a for a, g in zip(ARMS, stage[1:3]) if g is not None)
        if stage[0] == "grip":
            return f"{stage[1]} gripper {'open' if stage[2] > 0.5 else 'close'}"
        return "wait"

    def next_stage(t):
        st["i"] += 1
        st["t0"] = t
        if st["i"] >= len(stages):
            return
        stage = stages[st["i"]]
        if stage[0] == "move":
            goals = [g() if g is not None else stay(i) for i, g in enumerate(stage[1:3])]
            sync.set_targets(goals)
            st["moving"] = [bool(ph < 1.0) for ph in sync.phase]
            st["arr"] = [None, None]
            for a, (p, Rg) in zip(ARMS, goals):
                sim.show_custom_frame(f"goal_{a}", make_T(p, Rg))
        elif stage[0] == "grip":
            sim.set_gripper(stage[1], stage[2])

    def stage_done(t):
        stage = stages[st["i"]]
        if stage[0] == "move":
            return sync.done
        return t - st["t0"] >= stage[-1]

    def on_step(tick: int, t: float, telem: Telemetry) -> None:
        if st["i"] == -1:
            st["telem"] = telem
            home = [site_to_tcp_world(a, telem.tcp_pose[a]) for a in ARMS]
            st["home"] = home
            sync.reset(home)
            for a, (p, Rw) in zip(ARMS, home):
                command(a, p, Rw)                                   # target first, mode last
                sim.set_mode(a, ControlMode.CARTESIAN_IMPEDANCE)
            next_stage(t)
            return
        if st["i"] >= len(stages):
            return
        st["telem"] = telem
        for i, (a, (p, Rw)) in enumerate(zip(ARMS, sync.step(sim._dt))):
            command(a, p, Rw)
            sim.show_custom_frame(f"cmd_{a}", make_T(p, Rw))
            if st["moving"][i] and st["arr"][i] is None and sync.phase[i] >= 1.0:
                st["arr"][i] = t - st["t0"]
        if stage_done(t):
            c, _ = cylinder()
            stage = stages[st["i"]]
            arr = ""
            if stage[0] == "move":
                arr = "  arrived " + ", ".join(f"{a} {st['arr'][i]:.2f} s" for i, a in enumerate(ARMS)
                                               if st["moving"][i])
            line = (f"t={t:6.2f}s  {st['i']:2d} {describe(stage):22s} ({t - st['t0']:4.2f} s)  "
                    f"cylinder {np.round(c, 3).tolist()}{arr}")
            st["log"].append(line)
            print(line, flush=True)
            if args.debug:
                debug_contacts()
            next_stage(t)
            if st["i"] >= len(stages):
                result(telem)
                if args.headless:
                    sim.stop()

    def debug_contacts():
        _, ax = cylinder()
        tilt = np.degrees(np.arcsin(np.clip(abs(ax[2]), 0.0, 1.0)))
        fingers = {a: [d.qpos[m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"{a}_finger_joint{k}")]]
                       for k in (1, 2)] for a in ARMS}
        print(f"      tilt {tilt:5.1f} deg | finger opening L {sum(fingers['left']) * 1000:.1f} mm "
              f"R {sum(fingers['right']) * 1000:.1f} mm")
        f6 = np.zeros(6)
        for k in range(d.ncon):
            c = d.contact[k]
            if cyl not in (m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2]):
                continue
            other = c.geom2 if m.geom_bodyid[c.geom1] == cyl else c.geom1
            mujoco.mj_contactForce(m, d, k, f6)
            name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, other) or \
                f"{mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[other])}/geom{other}"
            print(f"      contact {name:28s} N {f6[0]:6.2f}  t {np.linalg.norm(f6[1:3]):5.2f}  "
                  f"torsion {f6[3]:+.4f}  dist {c.dist * 1000:+.2f} mm")

        for k in range(d.ncon):                                   # arm-arm contacts
            c = d.contact[k]
            b1, b2 = (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[g]) or "" for g in (c.geom1, c.geom2))
            if {b1.split("_")[0], b2.split("_")[0]} == {"left", "right"}:
                mujoco.mj_contactForce(m, d, k, f6)
                print(f"      ARM-ARM contact {b1} <-> {b2}  N {f6[0]:6.2f}  dist {c.dist * 1000:+.2f} mm")

    def result(telem):
        c, ax = cylinder()
        p_left, _ = site_to_tcp_world("left", telem.tcp_pose["left"])
        p_right, _ = site_to_tcp_world("right", telem.tcp_pose["right"])
        in_left = np.linalg.norm(c + s * ax - p_left)
        ok = in_left < 0.015 and np.linalg.norm(c + s * ax - PLACE_LEFT) < 0.03
        print(f"\ncylinder center {np.round(c, 3).tolist()}, axis {np.round(ax, 3).tolist()}")
        print(f"left grasp point vs left TCP: {in_left * 1000:.1f} mm; right TCP {np.linalg.norm(c - p_right) * 1000:.0f} mm "
              f"from the cylinder")
        print("HANDOVER OK" if ok else "HANDOVER FAILED", flush=True)

    sim.on_step(on_step)
    try:
        sim.run(headless=args.headless)
    except KeyboardInterrupt:
        pass
    finally:
        sim.close()


if __name__ == "__main__":
    main()

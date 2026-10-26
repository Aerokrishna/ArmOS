#!/usr/bin/env python
"""Coordinated bimanual pick and place of a long box (30 x 5 x 5 cm, 0.3 kg) on the dual Panda, pure MuJoCo.

Scene `dual_franka_pick_place.yaml`: the dual Panda, a table between the arms and the box lying on it, its long
axis pointing from one arm to the other. The green footprint on the table is the place pose (0.14 m further
out and centered between the arms, turned 25 deg). The box starts off-center, nearer the right arm, so the arms
reach different distances: in temporal mode they still arrive together.

    1. reach      both arms to pre-grasps above the box ends, `BimanualSync` temporal mode (arrive together)
    2. descend    both to the grasps (10 cm either side of the box center), temporal
    3. grasp      both grippers close
    4. engage     spatial mode: object-level admittance on the box frame (`engage_object`)
    5. lift       the object target rises 10 cm; both hands follow T_i = T_obj T_obj->ee_i
    6. tare       held still in the air: the object wrench (the box's weight) becomes the bias
    7. carry      the object target moves above the place pose and turns 25 deg
    8. touchdown  z switched to force mode with a 3 N downward target: the box comes down until the table
                  pushes back with 3 N (force-controlled placing, no exact table height needed)
    9. release    z back to pose mode at the measured height, spatial mode left (bumpless), grippers open
    10. retreat   both arms up and home, temporal

The object wrench is measured like the real setup: each arm's wrist wrench (payload-compensated,
robot-on-environment, like Franka's O_F_ext_hat_K), moved to its TCP, in the world frame, low-pass filtered,
summed through the grasp matrix (`armos_core.math.grasp`). Targets stream to each arm's CARTESIAN_IMPEDANCE with
crisp's gains, as in examples/bimanual_test.py. Gripper and contact settings as in examples/bimanual_handover.py
(gripper gains x10, MuJoCo noslip).

    python examples/bimanual_spatial.py
    python examples/bimanual_spatial.py --headless
"""

import argparse

import mujoco
import numpy as np

from armos_core.control.bimanual import BimanualCfg, BimanualSync
from armos_core.control.hybrid_admittance import HybridAdmittanceCfg
from armos_core.math.filters import LowPass1
from armos_core.math.se3 import make_T
from armos_core.math.so3 import rot_to_rotvec, rotvec_to_rot, rotZ
from armos_sim import ASSETS_DIR, ControlMode, CyRoSimulator, Telemetry

CONFIG = ASSETS_DIR / "franka_emika_panda" / "dual_franka_pick_place.yaml"
ARMS = ("left", "right")
L, R = 0, 1
CRISP_KP = [1500.0, 1500.0, 500.0, 30.0, 30.0, 30.0]
CRISP_KD = [80.0, 80.0, 30.0, 2.0, 2.0, 2.0]
TCP_IN_SITE = np.array([0.0, 0.0, 0.0129])     # finger-pad center below *_attachment_site (hand axes)
PLACE = np.array([0.56, 0.25, 0.2255])         # box center at the place pose (resting on the table) [m, world]
PLACE_YAW = np.radians(25.0)
GRASP_OFFSET = 0.10                            # grasp points from the box center along its long axis [m]
PRE_GRASP = 0.10                               # pre-grasp height above the grasp [m]
LIFT = 0.10                                    # lift height [m]
HOVER = 0.02                                   # height above the place pose where touchdown starts [m]

# Object-frame admittance (world axes). Pose axes: K / critical D; force axes (touchdown): z comes down at
# F_target / D_force = 3 N / 100 Ns/m = 3 cm/s until the table pushes back.
OBJECT_ADMITTANCE = HybridAdmittanceCfg(
    axes="base",
    mass=[2.5, 2.5, 2.5, 0.25, 0.25, 0.25],
    stiffness=[400.0, 400.0, 400.0, 30.0, 30.0, 30.0],
    damping=[63.0, 63.0, 63.0, 5.5, 5.5, 5.5],
    damping_force=[100.0, 100.0, 100.0, 5.0, 5.0, 5.0],
    max_lin_vel=0.15, max_ang_vel=0.6, max_lin_acc=0.5, max_ang_acc=2.0,
    max_force_lin_vel=0.03, max_force_ang_vel=0.2,
    force_deadband=(0.3, 0.03), max_deviation=(0.05, 0.3),
)


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reach-mode", choices=("temporal", "independent"), default="temporal",
                    help="BimanualSync mode for the reach / descend / retreat motions")
    ap.add_argument("--press", type=float, default=3.0, help="touchdown contact force [N]")
    ap.add_argument("--obj-vel", type=float, default=0.08, help="object target speed [m/s]")
    ap.add_argument("--obj-ang-vel", type=float, default=0.4, help="object target turn rate [rad/s]")
    ap.add_argument("--grip-scale", type=float, default=10.0, help="gripper gain scale")
    ap.add_argument("--noslip", type=int, default=5, help="MuJoCo noslip iterations")
    ap.add_argument("--wrench-filter", type=float, default=10.0, help="wrist wrench low-pass cutoff [Hz]")
    ap.add_argument("--headless", action="store_true", help="no viewer")
    return ap.parse_args()


def smootherstep(x):
    x = min(max(x, 0.0), 1.0)
    return x ** 3 * (6.0 * x ** 2 - 15.0 * x + 10.0)


def main():
    args = parse_args()
    sim = CyRoSimulator(config_path=CONFIG, realtime_factor=0.0 if args.headless else 1.0)
    m, d = sim._mj.model, sim._mj.data
    box = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "box")
    m.opt.noslip_iterations = args.noslip
    for a in ARMS:
        aid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{a}_actuator8")
        m.actuator_gainprm[aid, 0] *= args.grip_scale
        m.actuator_biasprm[aid, 1:3] *= args.grip_scale
        sim.set_impedance_gains(a, kp=CRISP_KP, kd=CRISP_KD, space="cartesian")
    T_wb = {a: make_T(sim.arm_config(a).base_world, sim.arm_config(a).base_world_rot) for a in ARMS}
    # Both hands turn 90 deg on the reach: at 0.5 rad/s that rotation, the same for both arms, would set both
    # durations; at 1 rad/s the (different) translations do.
    sync = BimanualSync(BimanualCfg(mode=args.reach_mode, max_ang_vel=1.0, max_ang_acc=2.0,
                                    object_admittance=OBJECT_ADMITTANCE))
    filters = {a: LowPass1(args.wrench_filter) for a in ARMS}
    dt = sim._dt

    # ------------------------------------------------------------------ measurements (world frame)
    def tcp_world(arm, telem):
        T = T_wb[arm] @ telem.tcp_pose[arm]
        return T[:3, 3] + T[:3, :3] @ TCP_IN_SITE, T[:3, :3]

    def wrench_world(arm, p_tcp):
        """Wrist wrench the hand applies on the environment, about the TCP, world axes, filtered."""
        w_b, point_b = sim.external_wrench(arm)
        Rw, tw = T_wb[arm][:3, :3], T_wb[arm][:3, 3]
        f, tau = Rw @ w_b[:3], Rw @ w_b[3:]
        tau = tau + np.cross(Rw @ point_b + tw - p_tcp, f)          # moment moved to the TCP
        return filters[arm].update(np.concatenate((f, tau)), dt)

    def command(arm, p_tcp_w, R_w):
        T_b = np.linalg.inv(T_wb[arm]) @ make_T(p_tcp_w - R_w @ TCP_IN_SITE, R_w)
        sim.set_tcp_pose(arm, T_b[:3, 3], T_b[:3, :3])

    def box_pose():
        """Box center and rotation; its y axis (long axis) made to point to +y world."""
        Rb = d.xmat[box].reshape(3, 3).copy()
        if Rb[1, 1] < 0.0:
            Rb = Rb @ rotZ(np.pi)
        return d.xpos[box].copy(), Rb

    def grasp_rot(x_h):
        """Hand x along x_h (along the box axis, towards the other arm), fingers across, approach down."""
        z = np.array([0.0, 0.0, -1.0])
        z = z - x_h * x_h.dot(z)
        z /= np.linalg.norm(z)
        return np.column_stack((x_h, np.cross(z, x_h), z))

    def grasps(up=0.0):
        """[left, right] grasp TCP poses on the actual box, `up` [m] above it."""
        c, Rb = box_pose()
        ax = Rb[:, 1]
        lift = np.array([0.0, 0.0, up])
        return [(c + GRASP_OFFSET * ax + lift, grasp_rot(-ax)), (c - GRASP_OFFSET * ax + lift, grasp_rot(ax))]

    # ------------------------------------------------------------------ stages
    st = {"i": -1, "t0": 0.0, "meas": None, "wr": None, "home": None, "ramp": None, "contact_t": None,
          "arr": [None, None], "moving": [False, False], "goal_obj": None}

    def phase_move(goals_fn):
        def start(t):
            goals = goals_fn()
            sync.set_targets(goals)
            st["moving"] = [bool(ph < 1.0) for ph in sync.phase]
            st["arr"] = [None, None]
            for a, (p, Rg) in zip(ARMS, goals):
                sim.show_custom_frame(f"goal_{a}", make_T(p, Rg))
        return {"start": start, "done": lambda t: sync.done, "phase": True}

    def wait(duration, start=None, end=None):
        return {"start": start or (lambda t: None), "done": lambda t: t - st["t0"] >= duration, "end": end}

    def grippers(opening):
        def start(t):
            for a in ARMS:
                sim.set_gripper(a, opening)
        return wait(1.0, start)

    def engage(t):
        sync.set_mode("spatial")
        poses = sync.poses()                                  # commanded: bumpless
        c, Rb = box_pose()
        sync.engage_object(poses, obj_pose=(0.5 * (poses[0][0] + poses[1][0]), Rb))
        st["goal_obj"] = (sync.object_law.target_pos.copy(), sync.object_law.target_R.copy())

    def object_move(goal_fn):
        """Ramp the object target to goal_fn() with a smootherstep at --obj-vel / --obj-ang-vel."""
        def start(t):
            p0, R0 = sync.object_law.target_pos.copy(), sync.object_law.target_R.copy()
            pg, Rg = goal_fn(p0, R0)
            dth = rot_to_rotvec(Rg @ R0.T)
            T = 1.875 * max(np.linalg.norm(pg - p0) / args.obj_vel, np.linalg.norm(dth) / args.obj_ang_vel, 0.1)
            st["ramp"] = (t, T, p0, pg - p0, R0, dth)
            st["goal_obj"] = (pg, Rg)
            sim.show_custom_frame("object_goal", make_T(pg, Rg))

        def tick(t):
            t0, T, p0, dp, R0, dth = st["ramp"]
            s = smootherstep((t - t0) / T)
            sync.set_object_target(p0 + s * dp, rotvec_to_rot(s * dth) @ R0)

        def done(t):
            t0, T = st["ramp"][:2]
            return t - t0 >= T + 0.3                           # + settle
        return {"start": start, "tick": tick, "done": done}

    def tare(t):
        bias = sync.tare_object(st["meas"], st["wr"])
        print(f"      tare: object wrench bias {np.round(bias, 3).tolist()}  (box weight {0.3 * 9.81:.2f} N)",
              flush=True)

    def touchdown():
        def start(t):
            st["contact_t"] = None
            sync.set_object_wrench([0.0, 0.0, -args.press, 0.0, 0.0, 0.0])   # push down (robot-on-object)
            sync.set_object_mask([1, 1, 0, 1, 1, 1])

        def done(t):
            fz = sync.object_wrench[2]
            if fz < -0.7 * args.press:
                st["contact_t"] = st["contact_t"] or t
            else:
                st["contact_t"] = None
            return (st["contact_t"] is not None and t - st["contact_t"] > 0.3) or t - st["t0"] > 8.0
        return {"start": start, "done": done}

    def settle_on_table(t):
        p_obj, R_obj = sync.object_pose_from(st["meas"])
        sync.set_object_target(p_obj, R_obj)                 # stop pressing: hold where it rests
        sync.set_object_wrench(np.zeros(6))
        sync.set_object_mask([1, 1, 1, 1, 1, 1])

    def release(t):
        sync.set_mode(args.reach_mode)                        # leaves spatial: holds the commanded poses
        for a in ARMS:
            sim.set_gripper(a, 1.0)

    stages = [
        ("open grippers", grippers(1.0)),
        ("reach pre-grasps", phase_move(lambda: grasps(PRE_GRASP))),
        ("descend to grasps", phase_move(lambda: grasps())),
        ("close grippers", grippers(0.0)),
        ("engage spatial", wait(0.0, engage)),
        ("lift", object_move(lambda p, Rm: (p + [0.0, 0.0, LIFT], Rm))),
        ("hold + tare", wait(0.5, end=tare)),
        ("carry", object_move(lambda p, Rm: (PLACE + [0.0, 0.0, HOVER], rotZ(PLACE_YAW) @ Rm))),
        ("touchdown (force)", touchdown()),
        ("settle", wait(0.8, settle_on_table)),
        ("release", wait(1.0, release)),
        ("retreat", phase_move(lambda: [(p + [0.0, 0.0, 0.12], Rp) for p, Rp in sync.poses()])),
        ("home", phase_move(lambda: st["home"])),
    ]

    def next_stage(t):
        st["i"] += 1
        st["t0"] = t
        if st["i"] < len(stages):
            stages[st["i"]][1]["start"](t)

    def report_stage(t):
        name, stage = stages[st["i"]]
        c, Rb = box_pose()
        yaw = np.degrees(np.arctan2(Rb[1, 1], Rb[0, 1]) - np.pi / 2)
        extra = ""
        if stage.get("phase"):
            extra = "  arrived " + ", ".join(f"{a} {st['arr'][i]:.2f} s" for i, a in enumerate(ARMS) if st["moving"][i])
        if sync.object_engaged:
            extra += f"  F_obj {np.round(sync.object_wrench[:3], 2).tolist()} N"
        print(f"t={t:6.2f}s  {st['i']:2d} {name:20s} ({t - st['t0']:5.2f} s)  box {np.round(c, 3).tolist()} "
              f"yaw {yaw:5.1f} deg{extra}", flush=True)

    def result():
        c, Rb = box_pose()
        yaw = np.arctan2(Rb[1, 1], Rb[0, 1]) - np.pi / 2
        e_xy = np.linalg.norm(c[:2] - PLACE[:2])
        e_z = c[2] - PLACE[2]
        e_yaw = np.degrees(yaw - PLACE_YAW)
        tilt = np.degrees(np.arccos(np.clip(Rb[2, 2], -1.0, 1.0)))
        ok = e_xy < 0.01 and abs(e_z) < 0.003 and abs(e_yaw) < 3.0 and tilt < 3.0
        print(f"\nbox at {np.round(c, 4).tolist()}: xy error {e_xy * 1000:.1f} mm, height error {e_z * 1000:.1f} mm, "
              f"yaw error {e_yaw:.2f} deg, tilt {tilt:.2f} deg")
        print("PLACE OK" if ok else "PLACE FAILED", flush=True)

    def on_step(tick: int, t: float, telem: Telemetry) -> None:
        meas = [tcp_world(a, telem) for a in ARMS]
        st["meas"] = meas
        st["wr"] = [wrench_world(a, p) for a, (p, _) in zip(ARMS, meas)]
        if st["i"] == -1:
            st["home"] = meas
            sync.reset(meas)
            for a, (p, Rw) in zip(ARMS, meas):
                command(a, p, Rw)                           # target first, mode last
                sim.set_mode(a, ControlMode.CARTESIAN_IMPEDANCE)
            next_stage(t)
            return
        if st["i"] >= len(stages):
            return
        name, stage = stages[st["i"]]
        if "tick" in stage:
            stage["tick"](t)
        if sync.object_engaged:
            poses = sync.step(dt, st["wr"], meas)
            p_v, R_v = sync.object_law.virtual_pos, sync.object_law.virtual_R
            sim.show_custom_frame("object_virtual", make_T(p_v, R_v))
        else:
            poses = sync.step(dt)
        for i, (a, (p, Rw)) in enumerate(zip(ARMS, poses)):
            command(a, p, Rw)
            sim.show_custom_frame(f"cmd_{a}", make_T(p, Rw))
            if st["moving"][i] and st["arr"][i] is None and sync.phase[i] >= 1.0:
                st["arr"][i] = t - st["t0"]
        if stage["done"](t):
            if stage.get("end"):
                stage["end"](t)
            report_stage(t)
            next_stage(t)
            if st["i"] >= len(stages):
                result()
                if args.headless:
                    sim.stop()

    sim.on_step(on_step)
    try:
        sim.run(headless=args.headless)
    except KeyboardInterrupt:
        pass
    finally:
        sim.close()


if __name__ == "__main__":
    main()

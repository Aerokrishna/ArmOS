#!/usr/bin/env python
"""Bimanual target generation (armos_core.control.bimanual) on the dual Panda, pure MuJoCo (no ROS).

Both arms start at their home TCP poses. Each cycle draws a random goal per arm (home + random offset and
rotation), moves both arms there with `BimanualSync` in the chosen mode, holds, and comes back home. The
targets stream to each arm's CARTESIAN_IMPEDANCE with crisp's gains (the same as the sim bridge), as the
ROS node will stream them to `<arm>/target_pose`.

Viewer overlays (world frame): `goal_<arm>` is the segment's goal; `cmd_<arm>` is the moving target.
Per segment it prints each arm's target arrival time and its tracking error after the hold (settled).

    python examples/bimanual_test.py --mode independent
    python examples/bimanual_test.py --mode temporal
    python examples/bimanual_test.py --mode temporal --omega 0.5 0.2       # unequal rates, coupled
    python examples/bimanual_test.py --mode temporal --omega 0.5 0.2 --coupling 0
"""

import argparse

import numpy as np

from armos_core.control.bimanual import MODES, BimanualCfg, BimanualSync
from armos_core.math.se3 import make_T
from armos_core.math.so3 import rot_to_rotvec, rotvec_to_rot
from armos_sim import ASSETS_DIR, ControlMode, CyRoSimulator, Telemetry

CONFIG = ASSETS_DIR / "franka_emika_panda" / "dual_franka.yaml"
ARMS = ("left", "right")
# crisp_controllers.yaml cartesian_controller gains (as armos_control's sim bridge)
CRISP_KP = [1500.0, 1500.0, 500.0, 30.0, 30.0, 30.0]
CRISP_KD = [80.0, 80.0, 30.0, 2.0, 2.0, 2.0]


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=MODES, default="independent")
    ap.add_argument("--coupling", type=float, default=5.0, help="phase coupling k [1/s] (temporal)")
    ap.add_argument("--omega", type=float, nargs=2, metavar=("LEFT", "RIGHT"), default=None,
                    help="nominal phase rates [1/s] per arm (temporal); default: from the limits")
    ap.add_argument("--max-lin-vel", type=float, default=0.1, help="[m/s]")
    ap.add_argument("--max-ang-vel", type=float, default=0.5, help="[rad/s]")
    ap.add_argument("--max-lin-acc", type=float, default=0.25, help="[m/s^2]")
    ap.add_argument("--max-ang-acc", type=float, default=1.0, help="[rad/s^2]")
    ap.add_argument("--pos-range", type=float, default=0.15, help="random goal offset per axis, +- [m]")
    ap.add_argument("--rot-range", type=float, default=0.4, help="random goal rotation per axis, +- [rad]")
    ap.add_argument("--hold", type=float, default=1.0, help="pause at each end [s]")
    ap.add_argument("--cycles", type=int, default=0, help="out-and-back cycles (0 = until the viewer closes)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--headless", action="store_true", help="no viewer (needs --cycles)")
    return ap.parse_args()


def main():
    args = parse_args()
    if args.headless and args.cycles <= 0:
        raise SystemExit("--headless needs --cycles > 0")
    rng = np.random.default_rng(args.seed)
    sim = CyRoSimulator(config_path=CONFIG, realtime_factor=0.0 if args.headless else 1.0)
    sync = BimanualSync(BimanualCfg(
        mode=args.mode, coupling=args.coupling, max_lin_vel=args.max_lin_vel,
        max_ang_vel=args.max_ang_vel, max_lin_acc=args.max_lin_acc, max_ang_acc=args.max_ang_acc))
    T_wb = {a: make_T(sim.arm_config(a).base_world, sim.arm_config(a).base_world_rot) for a in ARMS}
    for a in ARMS:
        sim.set_impedance_gains(a, kp=CRISP_KP, kd=CRISP_KD, space="cartesian")

    st = {"seg": "init", "t0": 0.0, "home": None, "goals": None, "arrive": [None, None], "cycle": 0}

    def show(name, arm, p, R):
        sim.show_custom_frame(f"{name}_{arm}", T_wb[arm] @ make_T(p, R))

    def start(seg, goals, t):
        st.update(seg=seg, t0=t, goals=goals, arrive=[None, None])
        sync.set_targets(goals, omega=args.omega if args.mode == "temporal" else None)
        for a, (p, R) in zip(ARMS, goals):
            show("goal", a, p, R)

    def random_goals():
        goals = []
        for p, R in st["home"]:
            dp = rng.uniform(-args.pos_range, args.pos_range, 3)
            dr = rng.uniform(-args.rot_range, args.rot_range, 3)
            goals.append((p + dp, rotvec_to_rot(dr) @ R))
        return goals

    def report(telem):
        parts = []
        for i, a in enumerate(ARMS):
            T = telem.tcp_pose[a]
            pg, Rg = st["goals"][i]
            e_p = np.linalg.norm(T[:3, 3] - pg) * 1000
            e_r = np.degrees(np.linalg.norm(rot_to_rotvec(T[:3, :3] @ Rg.T)))
            arr = st["arrive"][i]
            parts.append(f"{a}: target arrived {arr:5.2f} s, settled err {e_p:4.1f} mm {e_r:4.2f} deg")
        print(f"[cycle {st['cycle']}] {st['last']:4s}  " + " | ".join(parts), flush=True)

    def on_step(tick: int, t: float, telem: Telemetry) -> None:
        dt = sim._dt
        if st["seg"] == "init":
            home = [(telem.tcp_pose[a][:3, 3].copy(), telem.tcp_pose[a][:3, :3].copy()) for a in ARMS]
            st["home"] = home
            sync.reset(home)
            for a, (p, R) in zip(ARMS, home):
                sim.set_tcp_pose(a, p, R)                       # target first, mode last
                sim.set_mode(a, ControlMode.CARTESIAN_IMPEDANCE)
            st.update(seg="wait", t0=t, next="out")
            print(f"mode={args.mode} coupling={args.coupling} omega={args.omega} | home TCP (base): "
                  + ", ".join(f"{a} {np.round(p, 3).tolist()}" for a, (p, _) in zip(ARMS, home)), flush=True)
            return

        if st["seg"] == "wait":
            if t - st["t0"] >= args.hold:
                if st["goals"] is not None:
                    report(telem)
                if st["next"] == "out":
                    if args.cycles and st["cycle"] >= args.cycles:
                        sim.stop()
                        return
                    st["cycle"] += 1
                    start("out", random_goals(), t)
                else:
                    start("back", st["home"], t)
            return

        poses = sync.step(dt)
        for i, (a, (p, R)) in enumerate(zip(ARMS, poses)):
            sim.set_tcp_pose(a, p, R)
            show("cmd", a, p, R)
            if st["arrive"][i] is None and sync.phase[i] >= 1.0:
                st["arrive"][i] = t - st["t0"]
        if sync.done:
            st.update(seg="wait", t0=t, last=st["seg"], next="back" if st["seg"] == "out" else "out")

    sim.on_step(on_step)
    try:
        sim.run(headless=args.headless)
    except KeyboardInterrupt:
        pass
    finally:
        sim.close()


if __name__ == "__main__":
    main()

#!/usr/bin/env python

import time
import numpy as np
from armos_sim import ASSETS_DIR, CyRoSimulator
from armos_sim import ControlMode, Telemetry
from pathlib import Path


dual_arm = False

assets = ASSETS_DIR / "duman"
if dual_arm:
    default = assets / "dual_franka.yaml"
else:
    default = assets / "duman.yaml"

def main_demo() -> None:
    time.sleep(1)

    sim = CyRoSimulator(config_path=default)

    def on_step(tick: int, t: float, telem: Telemetry) -> None:
        if tick == 2:
            sim.set_mode("left", ControlMode.JOINT_POS)

        if tick == 400:
            print("ROBOT SHUD MOVE")
            sim.set_mode("left", ControlMode.JOINT_POS)
            sim.set_joint_pos("left",q=[0.4,0.,0.5,0,1,0.6])
    sim.on_step(on_step)
    sim.set_gripper()

    try:
        sim.run(headless=False)
    except KeyboardInterrupt:
        print("\nStopping simulation...")
    finally:
        print("Cleaning up...")
        try:
            # sim.stop()
            sim.close()
        except AttributeError:
            pass

if __name__ == "__main__":
    main_demo()
    
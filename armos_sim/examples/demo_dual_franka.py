#!/usr/bin/env python
import time
import numpy as np
from armos_sim import ASSETS_DIR, CyRoSimulator
from armos_sim import ControlMode, Telemetry
from pathlib import Path


dual_arm = False
assets = ASSETS_DIR / "franka_emika_panda"
default = assets / "dual_franka.yaml"

def main_demo() -> None:
    time.sleep(1)

    sim = CyRoSimulator(config_path=default, actuation="position")

    def on_step(tick: int, t: float, telem: Telemetry) -> None:
        if tick == 2:
            sim.set_mode("left", ControlMode.JOINT_POS)
            sim.set_mode("right", ControlMode.JOINT_POS)

            sim.set_joint_pos("left",[0.0, -0.78539816, 0.0, -2.35619449, 0.0, 1.57079633,0.78539816])
            sim.set_joint_pos("right",[0.0, -0.78539816, 0.0, -2.35619449, 0.0, 1.57079633,0.78539816])


        if tick == 200:
            sim.set_joint_pos("left",[0.0, 0.78539816, 0.0, -2.35619449, 0.0, -1.57079633,0.78539816])
            sim.set_joint_pos("right",[0.0, 0.78539816, 0.0, -1.35619449, 0.0, 0,0.78539816])
            
        if tick == 600:
            sim.set_mode("left", ControlMode.JOINT_VEL)
            sim.set_mode("right", ControlMode.JOINT_VEL)

            sim.set_joint_vel("left",qdot=[0.1,0, 0.1,0,0.1,0.1,0.])
            sim.set_joint_vel("right",qdot=[0.1,0.,-0.2,0,-0.1,0.1,0.])

        if tick == 2000:
            sim.set_mode("left", ControlMode.JOINT_VEL)
            sim.set_mode("right", ControlMode.JOINT_VEL)

            sim.set_joint_vel("left",qdot=[0,0,0,0,0,0,0])
            sim.set_joint_vel("right",qdot=[0,0,0,0,0,0,0])

            sim.set_mode("left", ControlMode.JOINT_POS)
            sim.set_mode("right", ControlMode.JOINT_POS)

            sim.set_joint_pos("left",[0.0, -0.78539816, 0.0, -2.35619449, 0.0, 1.57079633,0.78539816])
            sim.set_joint_pos("right",[0.0, -0.78539816, 0.0, -2.35619449, 0.0, 1.57079633,0.78539816])

            sim.set_mode("left", ControlMode.TASK_VEL)
            sim.set_tcp_vel("left", linear=[0,0,0.1], angular=[0.1,0,0])

            sim.set_mode("right", ControlMode.TASK_VEL)
            sim.set_tcp_pose("right", linear=[0,0,-0.1], angular=[-0.1,0,0])

        if tick == 3001:
            sim.set_mode("left", ControlMode.JOINT_VEL)
            sim.set_mode("right", ControlMode.JOINT_VEL)

            sim.set_joint_vel("left",qdot=[0,0,0,0,0,0,0])
            sim.set_joint_vel("right",qdot=[0,0,0,0,0,0,0])


        
    sim.on_step(on_step)
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
    
#!/usr/bin/env python
"""Run the CyRo simulation demo."""

import time
import numpy as np
from armos_sim import ASSETS_DIR, CyRoSimulator
from armos_sim import ControlMode, Telemetry
from pathlib import Path

dual_arm = True  # the demo drives both arms

assets = ASSETS_DIR / "franka_emika_panda"
if dual_arm:
    default = assets / "dual_franka.yaml"
else:
    default = assets / "franka.yaml"

def main_demo() -> None:
    """Run the built-in demo (interactive MuJoCo viewer)."""
    time.sleep(1)  # let OS scheduler settle before launching viewer

    sim = CyRoSimulator(config_path=default, actuation="position")

    def on_step(tick: int, t: float, telem: Telemetry) -> None:
        # print(tick)
        if tick==2:
            sim.set_mode("left", ControlMode.JOINT_POS)
            sim.set_joint_pos("left", q = [0.0, -0.78539816, 0.0, -2.35619449, 0.0, 1.57079633, 0.78539816])
            
            sim.set_mode("right", ControlMode.JOINT_POS)
            sim.set_joint_pos("right", q = [0.0, -0.78539816, 0.0, -2.35619449, 0.0, 1.57079633, 0.78539816])

        if tick ==100:
            sim.set_mode("left", ControlMode.TASK_POSE)
            sim.set_mode("right", ControlMode.TASK_POSE)

            sim.show_object_frame("left_attachment_site")
            sim.show_object_frame("right_attachment_site")

            sim.show_object_frame("panda")

        if tick==101:
            # sim.set_tcp_vel("left", linear=[-0.05, 0., -0.06], angular=[0.,0.0,0.])
            sim.set_tcp_pose("left", pos=[0.3, 0, 0.2], rot = [[ 9.99995667e-01,  2.00273503e-03, -2.15762363e-03],
                                                                [ 2.00269853e-03, -9.99997994e-01, -1.90771176e-05 ],
                                                                [-2.15765751e-03,  1.47559653e-05, -9.99997672e-01 ]])
        
        elif tick==500:
            sim.set_tcp_pose("right", pos=[0.3, 0.2, 0.6], rot = [[ 9.99995667e-01,  2.00273503e-03, -2.15762363e-03],
                                                                [ 2.00269853e-03, -9.99997994e-01, -1.90771176e-05 ],
                                                                [-2.15765751e-03,  1.47559653e-05, -9.99997672e-01 ]])
            
    sim.on_step(on_step)
    sim.run(headless=False)

if __name__ == "__main__":
    main_demo()
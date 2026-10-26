"""Torque actuation: replace an arm's position servos with motors.

Source: `mujoco_rl_bed.sim.scene.SceneBuilder` (`_discover_arm` torque limits, servo -> motor swap),
generalized to any set of arm joints so several arms in one scene can be swapped.
"""

from __future__ import annotations

from typing import Sequence

import mujoco
import numpy as np


def joint_torque_limits(model: mujoco.MjModel, joint_names: Sequence[str]) -> np.ndarray:
    """Per-joint torque limits [Nm] from the joint actuators' forcerange, else the joints' actuatorfrcrange.

    Args:
        model: Compiled model (with the original actuators).
        joint_names: Arm joints.

    Returns:
        Limits, shape (n,); 0 where neither range is set.
    """
    trn_joint = int(mujoco.mjtTrn.mjTRN_JOINT)
    act_of_joint = {
        int(model.actuator_trnid[a, 0]): a
        for a in range(model.nu)
        if int(model.actuator_trntype[a]) == trn_joint
    }
    limits = np.zeros(len(joint_names))
    for k, jn in enumerate(joint_names):
        j = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jn)
        a = act_of_joint.get(j, -1)
        if a >= 0 and model.actuator_forcelimited[a]:
            limits[k] = max(abs(model.actuator_forcerange[a, 0]), abs(model.actuator_forcerange[a, 1]))
        elif model.jnt_actfrclimited[j]:
            limits[k] = max(abs(model.jnt_actfrcrange[j, 0]), abs(model.jnt_actfrcrange[j, 1]))
    return limits


def swap_to_motors(
    spec: mujoco.MjSpec,
    model: mujoco.MjModel,
    joint_names: Sequence[str],
    actuator_names: Sequence[str],
) -> np.ndarray:
    """Replace the given position actuators with pure torque motors (in place on `spec`).

    Each motor is named `<joint>_motor`, has gear 1 and `ctrlrange = +-limit`, so MuJoCo clamps ctrl
    (torque saturation for free). Keyframes are dropped: their `ctrl` vectors no longer match the
    actuator layout.

    Args:
        spec: Spec of the scene; modified in place.
        model: Model compiled from the unmodified spec (to read resolved defaults such as class forceranges).
        joint_names: Arm joints, in kinematic order.
        actuator_names: Their current actuators, same order.

    Returns:
        Torque limits [Nm], shape (n,).
    """
    limits = joint_torque_limits(model, joint_names)
    if np.any(limits <= 0):
        raise ValueError(f"Missing torque limit for arm joints {list(joint_names)}: {limits}")

    for name in actuator_names:
        spec.delete(spec.actuator(name))
    for jn, lim in zip(joint_names, limits):
        act = spec.add_actuator(name=f"{jn}_motor", target=jn, trntype=mujoco.mjtTrn.mjTRN_JOINT)
        act.set_to_motor()  # gainprm=[1,0..], biasprm=0, dyntype none -> force = ctrl
        act.gear = [1.0, 0, 0, 0, 0, 0]
        act.ctrllimited = mujoco.mjtLimited.mjLIMITED_TRUE
        act.ctrlrange = [-float(lim), float(lim)]

    for key in list(spec.keys):
        spec.delete(key)
    return limits

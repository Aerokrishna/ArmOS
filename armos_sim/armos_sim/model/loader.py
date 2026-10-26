"""Robot loader: parses YAML config + MJCF and produces ArmConfig objects."""

from __future__ import annotations

import logging
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import mujoco
import numpy as np

from ..types import ArmConfig, ToolParams
from .actuation import joint_torque_limits, swap_to_motors

try:
    import yaml as _yaml
    _YAML_OK = True
except ImportError:
    _YAML_OK = False

logger = logging.getLogger(__name__)


class _RobotLoader:
    """Parses a YAML config + referenced MJCF and produces ArmConfig objects."""

    _DEFAULT_AUX = {
        "grippers": {"left": "left_robot_gripper_finger1_joint_motor", "right": "right_robot_gripper_finger1_joint_motor"},
        "gripper_open": -0.65,
        "gripper_closed": 0.0,
        "stereo_left_yaw_motor": "stereo_left_yaw_motor",
        "stereo_right_yaw_motor": "stereo_right_yaw_motor",
    }

    _ACTUATIONS = ("position", "torque")
    _DECIMATIONS = ("control", "admittance")

    @staticmethod
    def from_yaml(
        config_path: Path,
        actuation: Optional[str] = None,
        timestep: Optional[float] = None,
        decimation: Optional[Dict[str, int]] = None,
    ) -> Tuple[Path, Dict[str, ArmConfig], dict]:
        """Load a YAML config and its MJCF.

        Args:
            config_path: YAML config.
            actuation: Override for every arm: "position" or "torque". Default: per arm from the YAML
                (`arms.<id>.actuation`, else top-level `actuation`, else "position").
            timestep: Override of the physics timestep [s] (YAML: `physics.timestep`).
            decimation: Overrides of the per-mode decimation, physics steps per update
                (YAML: `physics.decimation`): "control" (control tick: position modes, telemetry,
                callbacks) and "admittance" (CARTESIAN_ADMITTANCE outer loop).

        Returns:
            (scene_path, arm_configs, aux). aux["_preloaded_model"] is the compiled model (with motors
            for torque arms) and aux["physics"] holds {"decimation": {"control": int | None,
            "admittance": int | None}}.

        Optional YAML `attach`: MJCF bodies added to the scene, e.g. a tool held by the hand or a fixture:

            attach:
              - file: peg.xml            # MJCF whose worldbody holds one top-level body (relative to the YAML)
                site: fr3_hand_tcp       # attach at this site (body frame = site frame), or
              - file: hole.xml
                pos: [0.6, 0.0, 0.08]    # a fixed world pose (quat w x y z, default identity)
                quat: [1, 0, 0, 0]
                prefix: ""               # optional name prefix of the attached elements

        Optional YAML `physics.frictionloss_scale`: multiplies every joint's frictionloss (0 = no joint friction,
        e.g. to match a model identified without it).
        """
        if not _YAML_OK:
            raise ImportError("pyyaml required: pip install pyyaml")

        with open(config_path, encoding="utf-8") as fh:
            cfg = _yaml.safe_load(fh)

        scene_path = config_path.parent / cfg["scene_xml_path"]
        robot_xml_name = cfg["robot_xml_path"]

        tree = ET.parse(scene_path)
        for include in tree.findall(".//include"):
            if Path(include.get("file", "")).stem.startswith("CyRo"):
                include.set("file", robot_xml_name)
                break

        with tempfile.NamedTemporaryFile(
            suffix=".xml", dir=scene_path.parent, delete=False
        ) as tmp:
            patched_path = Path(tmp.name)
        tree.write(str(patched_path), encoding="unicode")

        physics = cfg.get("physics") or {}
        timestep = timestep if timestep is not None else physics.get("timestep")
        dec_yaml = physics.get("decimation") or {}
        unknown = set(dec_yaml) - set(_RobotLoader._DECIMATIONS)
        if unknown:
            raise ValueError(f"physics.decimation: unknown keys {sorted(unknown)}, valid: {_RobotLoader._DECIMATIONS}")
        dec = {k: (decimation or {}).get(k, dec_yaml.get(k)) for k in _RobotLoader._DECIMATIONS}
        for k, v in dec.items():
            if v is not None and int(v) < 1:
                raise ValueError(f"physics.decimation.{k} must be >= 1, got {v}")
        actuations = {}
        for arm_id, arm_yaml in cfg["arms"].items():
            act = actuation or arm_yaml.get("actuation") or cfg.get("actuation") or "position"
            if act not in _RobotLoader._ACTUATIONS:
                raise ValueError(f"Arm '{arm_id}': actuation must be one of {_RobotLoader._ACTUATIONS}, got {act!r}")
            actuations[arm_id] = act

        try:
            tmp_model = mujoco.MjModel.from_xml_path(str(patched_path))
            tmp_data = mujoco.MjData(tmp_model)
            mujoco.mj_forward(tmp_model, tmp_data)
            arm_configs = {
                arm_id: _RobotLoader._build(arm_id, arm_yaml, tmp_model, tmp_data)
                for arm_id, arm_yaml in cfg["arms"].items()
            }
            attach = cfg.get("attach") or []
            if timestep is not None or "torque" in actuations.values() or attach:
                # Rebuild through MjSpec: attach bodies, swap torque arms' servos for motors, apply physics options.
                spec = mujoco.MjSpec.from_file(str(patched_path))
                _RobotLoader._attach(spec, attach, config_path.parent)
                for arm_id, arm_cfg in arm_configs.items():
                    if actuations[arm_id] == "torque":
                        swap_to_motors(spec, tmp_model, arm_cfg.joint_names, arm_cfg.actuator_names)
                if timestep is not None:
                    spec.option.timestep = float(timestep)
                tmp_model = spec.compile()
                tmp_data = mujoco.MjData(tmp_model)
                mujoco.mj_forward(tmp_model, tmp_data)
                arm_configs = {
                    arm_id: _RobotLoader._build(arm_id, arm_yaml, tmp_model, tmp_data)
                    for arm_id, arm_yaml in cfg["arms"].items()
                }
        finally:
            patched_path.unlink(missing_ok=True)

        for arm_id, arm_cfg in arm_configs.items():
            arm_cfg.actuation = actuations[arm_id]
            if arm_cfg.actuation == "torque":
                # Motors carry the limits in their ctrlrange.
                aids = [mujoco.mj_name2id(tmp_model, mujoco.mjtObj.mjOBJ_ACTUATOR, n) for n in arm_cfg.actuator_names]
                arm_cfg.torque_limits = np.abs(tmp_model.actuator_ctrlrange[aids, 1]).astype(float)
            else:
                arm_cfg.torque_limits = joint_torque_limits(tmp_model, arm_cfg.joint_names)
            logger.info(
                "Loaded arm '%s': %d DOF  %s actuation  joints=%s  base_world=%s",
                arm_id,
                arm_cfg.ndof,
                arm_cfg.actuation,
                arm_cfg.joint_names,
                arm_cfg.base_world.round(4),
            )

        aux = dict(_RobotLoader._DEFAULT_AUX)
        if "grippers" in cfg:
            # Replace entirely so `grippers: {}` in YAML clears stale defaults.
            aux["grippers"] = cfg["grippers"]
        for key in ["gripper_open", "gripper_closed", "stereo_left_yaw_motor", "stereo_right_yaw_motor"]:
            if key in cfg:
                aux[key] = cfg[key]
        if "dg3f_grippers" in cfg:
            aux["dg3f_grippers"] = cfg["dg3f_grippers"]
        if "robot_xml_path" in cfg:
            aux["robot_xml_path"] = config_path.parent / cfg["robot_xml_path"]
        _RobotLoader._check_aux_actuators(tmp_model, aux)
        scale = physics.get("frictionloss_scale")
        if scale is not None:
            tmp_model.dof_frictionloss[:] *= float(scale)
        aux["_preloaded_model"] = tmp_model
        aux["physics"] = {"decimation": {k: None if v is None else int(v) for k, v in dec.items()}}
        return scene_path, arm_configs, aux

    @staticmethod
    def _attach(spec: mujoco.MjSpec, items: List[dict], base_dir: Path) -> None:
        """Attach the YAML `attach` bodies to the scene spec (see `from_yaml`)."""
        for item in items:
            path = Path(item["file"]).expanduser()
            path = path if path.is_absolute() else base_dir / path
            child = mujoco.MjSpec.from_file(str(path))
            body = child.worldbody.first_body()
            if body is None:
                raise ValueError(f"attach: {path} has no body in its worldbody")
            prefix = str(item.get("prefix", ""))
            if "site" in item:
                site = spec.site(item["site"])
                if site is None:
                    raise ValueError(f"attach: site {item['site']!r} not found for {path.name}")
                site.attach_body(body, prefix, "")
            else:
                frame = spec.worldbody.add_frame(pos=list(item.get("pos", (0.0, 0.0, 0.0))),
                                                 quat=list(item.get("quat", (1.0, 0.0, 0.0, 0.0))))
                frame.attach_body(body, prefix, "")
            logger.info("Attached %s at %s", path.name, item.get("site", item.get("pos")))

    @staticmethod
    def _check_aux_actuators(mj_model, aux: dict) -> None:
        """Raise if a gripper actuator named in the config is not in the model."""
        names = list(aux.get("grippers", {}).values())
        for act_names in aux.get("dg3f_grippers", {}).values():
            names.extend(act_names)
        missing = [
            n for n in names
            if n and mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_ACTUATOR, n) < 0
        ]
        if missing:
            raise ValueError(f"Gripper actuators not found in MJCF: {missing}")

    @staticmethod
    def _build(arm_id: str, arm_yaml: dict, mj_model, mj_data) -> ArmConfig:
        start_name = arm_yaml["start_body"]
        ee_site_name = arm_yaml["ee_site"]

        start_id = mujoco.mj_name2id(
            mj_model, mujoco.mjtObj.mjOBJ_BODY, start_name
        )
        ee_site_id = mujoco.mj_name2id(
            mj_model, mujoco.mjtObj.mjOBJ_SITE, ee_site_name
        )
        if start_id < 0:
            raise ValueError(f"start_body '{start_name}' not found in MJCF")
        if ee_site_id < 0:
            raise ValueError(f"ee_site '{ee_site_name}' not found in MJCF")

        ee_body_id = int(mj_model.site_bodyid[ee_site_id])
        chain = _RobotLoader._trace_chain(mj_model, start_id, ee_body_id)
        joint_names = _RobotLoader._joints_on_chain(mj_model, chain)
        if not joint_names:
            raise ValueError(
                f"No actuated joints on chain '{start_name}' → '{ee_site_name}'"
            )

        actuator_names = _RobotLoader._actuators_for_joints(
            mj_model, joint_names
        )
        if len(actuator_names) != len(joint_names):
            raise ValueError(
                f"Arm '{arm_id}': {len(joint_names)} joints {joint_names} but only "
                f"{len(actuator_names)} joint actuators {actuator_names}"
            )
        q_lo, q_hi = _RobotLoader._joint_limits(mj_model, joint_names)
        base_world = mj_data.xpos[start_id].copy()
        base_world_rot = mj_data.xmat[start_id].reshape(3, 3).copy()
        ndof = len(joint_names)
        q_home = np.array(
            arm_yaml.get("q_home", [0.0] * ndof), dtype=float
        )
        if len(q_home) != ndof:
            raise ValueError(
                f"Arm '{arm_id}': q_home has {len(q_home)} values, "
                f"but {ndof} joints found"
            )
        tool_yaml = arm_yaml.get("tool") or {}
        tool = ToolParams(
            mass=float(tool_yaml.get("mass", 0.0)),
            com=np.array(tool_yaml.get("com", [0.0, 0.0, 0.0]), dtype=float),
            inertia=np.array(tool_yaml.get("inertia", [0.0] * 6), dtype=float),
            tcp_location=np.array(
                tool_yaml.get("tcp_location", [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]),
                dtype=float,
            ),
        )
        return ArmConfig(
            arm_id=arm_id,
            start_body=start_name,
            joint_names=joint_names,
            actuator_names=actuator_names,
            ee_site=ee_site_name,
            wrench_force_sensor=arm_yaml.get("wrench_force_sensor") or None,
            wrench_torque_sensor=arm_yaml.get("wrench_torque_sensor") or None,
            q_lo=q_lo,
            q_hi=q_hi,
            q_home=q_home,
            base_world=base_world,
            base_world_rot=base_world_rot,
            tool=tool,
        )

    @staticmethod
    def _trace_chain(mj_model, start_id: int, ee_id: int) -> List[int]:
        chain, cur, visited = [], ee_id, set()
        while True:
            if cur in visited:
                raise ValueError("Cycle in MuJoCo body tree")
            visited.add(cur)
            chain.append(cur)
            if cur == start_id:
                break
            parent = int(mj_model.body_parentid[cur])
            if parent == 0 and start_id != 0:
                raise ValueError(
                    f"start_body id={start_id} is not ancestor of ee id={ee_id}"
                )
            cur = parent
        chain.reverse()
        return chain

    @staticmethod
    def _joints_on_chain(mj_model, body_ids: List[int]) -> List[str]:
        allowed = {
            int(mujoco.mjtJoint.mjJNT_HINGE),
            int(mujoco.mjtJoint.mjJNT_SLIDE),
        }
        names: List[str] = []
        for bid in body_ids:
            adr = int(mj_model.body_jntadr[bid])
            for j in range(adr, adr + int(mj_model.body_jntnum[bid])):
                if int(mj_model.jnt_type[j]) in allowed:
                    n = mujoco.mj_id2name(
                        mj_model, mujoco.mjtObj.mjOBJ_JOINT, j
                    )
                    if n:
                        names.append(n)
        return names

    @staticmethod
    def _actuators_for_joints(
        mj_model, joint_names: List[str]
    ) -> List[str]:
        trn_joint = int(mujoco.mjtTrn.mjTRN_JOINT)
        jid_to_act: Dict[int, str] = {}
        for aid in range(mj_model.nu):
            if int(mj_model.actuator_trntype[aid]) == trn_joint:
                jid = int(mj_model.actuator_trnid[aid, 0])
                name = mujoco.mj_id2name(
                    mj_model, mujoco.mjtObj.mjOBJ_ACTUATOR, aid
                )
                if name:
                    jid_to_act[jid] = name
        acts: List[str] = []
        for jn in joint_names:
            jid = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_JOINT, jn)
            if jid in jid_to_act:
                acts.append(jid_to_act[jid])
        return acts

    @staticmethod
    def _joint_limits(
        mj_model, joint_names: List[str]
    ) -> Tuple[np.ndarray, np.ndarray]:
        lo, hi = [], []
        for jn in joint_names:
            jid = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_JOINT, jn)
            if mj_model.jnt_limited[jid]:
                lo.append(float(mj_model.jnt_range[jid, 0]))
                hi.append(float(mj_model.jnt_range[jid, 1]))
            else:
                lo.append(-np.pi)
                hi.append(np.pi)
        return np.array(lo), np.array(hi)

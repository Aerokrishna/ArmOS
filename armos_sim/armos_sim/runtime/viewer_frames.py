"""RGB axis overlays for the MuJoCo passive viewer (X=red, Y=green, Z=blue)."""

from __future__ import annotations

import mujoco
import numpy as np

AXIS_LENGTH = 0.06
AXIS_RADIUS = 0.003
FRAME_ALPHA = 0.9

ObjectRef = tuple[int, int]

_AXIS_RGBA = (
    np.array([1.0, 0.15, 0.15, FRAME_ALPHA], dtype=np.float32),
    np.array([0.15, 1.0, 0.15, FRAME_ALPHA], dtype=np.float32),
    np.array([0.15, 0.15, 1.0, FRAME_ALPHA], dtype=np.float32),
)


def resolve_object_ref(model, object_name: str) -> ObjectRef | None:
    """Resolve *object_name* to ``(mjtObj, id)`` or None if not in the model."""
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, object_name)
    if body_id >= 0:
        ref = (int(mujoco.mjtObj.mjOBJ_BODY), body_id)
        return ref

    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, object_name)
    if site_id >= 0:
        ref = (int(mujoco.mjtObj.mjOBJ_SITE), site_id)
        return ref

    geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, object_name)
    if geom_id >= 0:
        ref = (int(mujoco.mjtObj.mjOBJ_GEOM), geom_id)
        return ref

    return None


def read_object_pose(
    data,
    obj_type: int,
    obj_id: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return world-frame ``(pos, rmat)`` for a resolved object reference."""
    if obj_type == int(mujoco.mjtObj.mjOBJ_BODY):
        pos = np.asarray(data.xpos[obj_id], dtype=float)
        rmat = np.asarray(data.xmat[obj_id], dtype=float).reshape(3, 3)
    elif obj_type == int(mujoco.mjtObj.mjOBJ_SITE):
        pos = np.asarray(data.site_xpos[obj_id], dtype=float)
        rmat = np.asarray(data.site_xmat[obj_id], dtype=float).reshape(3, 3)
    else:
        pos = np.asarray(data.geom_xpos[obj_id], dtype=float)
        rmat = np.asarray(data.geom_xmat[obj_id], dtype=float).reshape(3, 3)

    pose = (pos, rmat)
    return pose


def poses_equal(
    pos_a: np.ndarray,
    rmat_a: np.ndarray,
    pos_b: np.ndarray,
    rmat_b: np.ndarray,
    *,
    atol: float = 1e-9,
) -> bool:
    """Return True when two object poses match within *atol*."""
    pos_match = np.allclose(pos_a, pos_b, atol=atol)
    rmat_match = np.allclose(rmat_a, rmat_b, atol=atol)
    equal = pos_match and rmat_match
    return equal


def draw_custom_frame(
    viewer,
    pos,
    rmat,
    *,
    length: float = AXIS_LENGTH,
    radius: float = AXIS_RADIUS,
    alpha: float = FRAME_ALPHA,
) -> None:
    """Draw a frame at custom position and rotation matrix."""
    for axis_idx, base_rgba in enumerate(_AXIS_RGBA):
        rgba = base_rgba.copy()
        rgba[3] = alpha
        direction = rmat[:, axis_idx]
        end_pt = pos + direction * length

        mujoco.mjv_initGeom(
            viewer.user_scn.geoms[viewer.user_scn.ngeom],
            type=mujoco.mjtGeom.mjGEOM_CAPSULE,
            size=np.array([radius, length / 2, 0]),
            pos=(pos + end_pt) / 2,
            mat=np.eye(3).flatten(),
            rgba=rgba,
        )

        geom = viewer.user_scn.geoms[viewer.user_scn.ngeom]
        z_axis = direction / (np.linalg.norm(direction) + 1e-12)
        up = np.array([0.0, 0.0, 1.0])
        if abs(np.dot(z_axis, up)) > 0.99:
            up = np.array([1.0, 0.0, 0.0])
        x_axis = np.cross(up, z_axis)
        x_axis /= np.linalg.norm(x_axis) + 1e-12
        y_axis = np.cross(z_axis, x_axis)
        rot = np.column_stack([x_axis, y_axis, z_axis])
        geom.mat[:] = rot

        viewer.user_scn.ngeom += 1

        mujoco.mjv_initGeom(
            viewer.user_scn.geoms[viewer.user_scn.ngeom],
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=np.array([radius * 2, 0, 0]),
            pos=end_pt,
            mat=np.eye(3).flatten(),
            rgba=rgba,
        )
        viewer.user_scn.ngeom += 1


def draw_object_frame(
    viewer,
    model,
    data,
    object_name: str,
    *,
    ref: ObjectRef | None = None,
    length: float = AXIS_LENGTH,
    radius: float = AXIS_RADIUS,
    alpha: float = FRAME_ALPHA,
) -> bool:
    """Draw a frame for a named MuJoCo body, site, or geom.

    Returns True when *object_name* was found in the model.
    """
    if ref is None:
        ref = resolve_object_ref(model, object_name)
    if ref is None:
        found = False
        return found

    obj_type, obj_id = ref
    pos, rmat = read_object_pose(data, obj_type, obj_id)
    draw_custom_frame(viewer, pos, rmat, length=length, radius=radius, alpha=alpha)
    found = True
    return found


def draw_all_frames(
    viewer,
    model,
    data,
    object_names,
    custom_frames,
    object_ref_cache: dict[str, ObjectRef],
    *,
    length: float = AXIS_LENGTH,
    radius: float = AXIS_RADIUS,
    alpha: float = FRAME_ALPHA,
) -> None:
    """Draw scene-object frames and client-supplied 4x4 pose overlays."""
    viewer.user_scn.ngeom = 0
    for name in object_names:
        ref = object_ref_cache.get(name)
        if ref is not None:
            draw_object_frame(
                viewer, model, data, name, ref=ref,
                length=length, radius=radius, alpha=alpha,
            )
    for T in custom_frames.values():
        draw_custom_frame(
            viewer, T[:3, 3], T[:3, :3],
            length=length, radius=radius, alpha=alpha,
        )

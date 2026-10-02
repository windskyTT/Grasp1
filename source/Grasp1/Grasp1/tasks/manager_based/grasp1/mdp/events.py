"""Teacher object placement, pre-grasp reset and initial collision fallback.

This module migrates the RobustDexGrasp Teacher reset pipeline while keeping it
compatible with IsaacLab's Manager-Based lifecycle.

Main responsibilities:
- cache per-object affordance points / mesh triangles / lowest point metadata;
- load optional evaluation stable states;
- sample source object XY / yaw reset poses;
- ray-cast visible affordance points;
- generate and rank analytic UR5 pre-grasp IK candidates;
- write robot/object reset state and actuator targets;
- check initial arm collision after the first physics step and apply fallback;
- reproduce the optional one-shot object-position bias curriculum.

The original code checked the initial candidate after a zero-action physics
step. IsaacLab's reset event runs before physics; ``sim.forward()`` updates
kinematics but does not solve contacts. The collision check therefore runs in
the interval event after the first control step.
"""

from __future__ import annotations

import math
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Final

import numpy as np
import torch

from isaaclab.utils.math import matrix_from_quat, quat_apply, quat_mul

from Grasp1.robots.robot_profile import CONTROLLED_JOINT_NAMES
from Grasp1.utils.paths import object_asset_dir

from . import geometry
from .keypoints import hand_keypoints_w


# -----------------------------------------------------------------------------
# Source constants
# -----------------------------------------------------------------------------

SOURCE_STABLE_SUPPORT_HEIGHT: Final[float] = 0.771
STABLE_STATE_HEIGHT_OFFSET: Final[float] = 0.005
SOURCE_TRAINING_DATASET: Final[str] = "new_training_set"

XY_ANGLE_MIN: Final[float] = -0.7 * math.pi
XY_ANGLE_MAX: Final[float] = -0.3 * math.pi
XY_DISTANCE_MIN: Final[float] = 0.45
XY_DISTANCE_MAX: Final[float] = 0.75
XY_ABS_X_LIMIT: Final[float] = 0.25

# 源 URDF 的 platform2ur5_joint 使用 yaw=-1.57；移除平台后，源工作
# 坐标系需先 Rz(+1.57) 转入 base_link，再经真实 robot root 转入世界系。
_SOURCE_TO_BASE_QUAT: Final[tuple[float, float, float, float]] = (
    math.cos(1.57 / 2.0), 0.0, 0.0, math.sin(1.57 / 2.0),
)

BIAS_DISTANCE_THRESHOLD: Final[float] = 0.07
BIAS_RANGE: Final[float] = 0.05

# Source collision fallback.
_FALLBACK_ARM_TAIL: Final[tuple[float, ...]] = (-1.57, 1.57, 0.0, 1.57, -1.57)
_FALLBACK_OBJECT_XY: Final[tuple[float, float]] = (0.1, -0.5)

# global_state[:,124:128] == contacts_arm_all[1:5].
_COLLISION_ARM_SENSOR_INDICES: Final[tuple[int, ...]] = (1, 2, 3, 4)


# -----------------------------------------------------------------------------
# Small helpers
# -----------------------------------------------------------------------------

def _env_ids(
    env,
    env_ids: Sequence[int] | torch.Tensor | slice | None,
) -> torch.Tensor:
    """Convert IsaacLab event indices to a device ``long`` tensor."""
    if env_ids is None:
        return torch.arange(env.num_envs, device=env.device, dtype=torch.long)

    if isinstance(env_ids, slice):
        return torch.arange(env.num_envs, device=env.device, dtype=torch.long)[env_ids]

    if isinstance(env_ids, torch.Tensor):
        return env_ids.to(device=env.device, dtype=torch.long)

    return torch.as_tensor(env_ids, device=env.device, dtype=torch.long)


def _runtime_controlled_joint_names() -> tuple[str, ...]:
    return tuple(name.replace(".", "_") for name in CONTROLLED_JOINT_NAMES)


def _resolve_controlled_joint_ids(robot) -> tuple[int, ...]:
    runtime_names = _runtime_controlled_joint_names()
    joint_ids, resolved = robot.find_joints(
        [re.escape(name) for name in runtime_names],
        preserve_order=True,
    )

    if tuple(resolved) != runtime_names:
        raise RuntimeError(
            "Teacher controlled-joint order mismatch: "
            f"expected={runtime_names}, resolved={tuple(resolved)}."
        )

    if len(joint_ids) != 22:
        raise RuntimeError(
            f"Teacher requires 22 controlled joints, resolved {len(joint_ids)}."
        )

    return tuple(int(index) for index in joint_ids)


def _stable_state_path(dataset_name: str, object_name: str) -> Path:
    return object_asset_dir(dataset_name, object_name) / f"{object_name}.npy"


def _load_stable_state(path: Path) -> torch.Tensor:
    """Load the last ``[x,y,z,qw,qx,qy,qz]`` state from ``<object>.npy``."""
    array = np.load(path)

    if array.ndim != 2 or array.shape[0] == 0 or array.shape[1] < 7:
        raise ValueError(
            f"Stable-state file must have shape [K,>=7], got {array.shape}: {path}"
        )

    state = torch.as_tensor(array[-1, :7], dtype=torch.float32, device="cpu").clone()
    quaternion = state[3:7]
    norm = torch.linalg.vector_norm(quaternion)

    if not torch.isfinite(norm) or norm <= 1.0e-8:
        raise ValueError(f"Invalid stable quaternion in {path}: {quaternion.tolist()}")

    state[3:7] = quaternion / norm
    return state


# -----------------------------------------------------------------------------
# Startup cache
# -----------------------------------------------------------------------------

def initialize_teacher_data(
    env,
    env_ids,
    dataset_name: str,
    object_names: tuple[str, ...],
) -> None:
    """Cache Teacher geometry, stable states and runtime indices once.

    ``ObservationManager`` probes the policy observation before startup events
    in IsaacLab 2.3.2.  The observation term therefore normally creates
    ``env._teacher_object_metadata`` and ``env._teacher_affordance_points_o``
    first. This function reuses that exact point sample and lowest-point
    metadata, then uploads reset mesh triangles and stable states once.
    """
    del env_ids

    names = tuple(object_names)
    if len(names) != env.num_envs:
        raise ValueError(
            "Teacher requires one object name per environment: "
            f"num_envs={env.num_envs}, len(object_names)={len(names)}."
        )

    robot = env.scene["robot"]
    joint_ids = _resolve_controlled_joint_ids(robot)

    unique_names = tuple(dict.fromkeys(names))
    object_index = {name: index for index, name in enumerate(unique_names)}
    object_ids = torch.tensor(
        [object_index[name] for name in names],
        device=env.device,
        dtype=torch.long,
    )

    # ObservationManager is constructed before startup events in IsaacLab
    # 2.3.2, so observations.py should already have sampled and published the
    # shared affordance cloud.  Keep task31 independent of object_set naming.
    cached_points = getattr(env, "_teacher_affordance_points_o", None)
    if not isinstance(cached_points, torch.Tensor):
        raise RuntimeError(
            "Teacher startup requires env._teacher_affordance_points_o. "
            "The TeacherObservation term must initialize the shared point "
            "cache before startup events run."
        )

    cached_points = cached_points.to(
        device=env.device,
        dtype=robot.data.joint_pos.dtype,
    )
    env._teacher_affordance_points_o = cached_points

    if cached_points.ndim != 3 or cached_points.shape[0] != env.num_envs or cached_points.shape[-1] != 3:
        raise RuntimeError(
            "Teacher affordance point cache must have shape [N,P,3], got "
            f"{tuple(cached_points.shape)}."
        )

    # Observation 初始化已读取最低点；重复物体无需再次访问文件。
    metadata = env._teacher_object_metadata
    lowest_values = [metadata[name].lowest_point for name in names]

    lowest = torch.tensor(
        lowest_values,
        device=env.device,
        dtype=robot.data.joint_pos.dtype,
    )

    # 全部 float32 triangles：训练集约 0.87 MiB，ShapeNet 约 6.17 MiB。
    # 启动时一次搬到 GPU，partial reset 仅索引对应物体，保留 process=False 语义。
    try:
        import trimesh
    except ImportError as exc:
        raise ImportError(
            "Teacher reset geometry requires trimesh during startup."
        ) from exc

    triangles: dict[str, torch.Tensor] = {}
    stable_states_cpu: dict[str, torch.Tensor] = {}

    for name in unique_names:
        object_dir = object_asset_dir(dataset_name, name)
        mesh_path = object_dir / "top_watertight_tiny.obj"
        if not mesh_path.is_file():
            raise FileNotFoundError(f"Missing top affordance mesh: {mesh_path}")

        mesh = trimesh.load_mesh(mesh_path, process=False)
        if isinstance(mesh, trimesh.Scene):
            geometries = tuple(mesh.geometry.values())
            if not geometries:
                raise ValueError(f"Mesh scene contains no geometry: {mesh_path}")
            mesh = trimesh.util.concatenate(geometries)

        if not isinstance(mesh, trimesh.Trimesh) or mesh.faces.shape[0] == 0:
            raise ValueError(f"Invalid triangular affordance mesh: {mesh_path}")

        triangles[name] = torch.as_tensor(
            np.asarray(mesh.triangles),
            device=env.device,
            dtype=robot.data.joint_pos.dtype,
        )

        stable_path = _stable_state_path(dataset_name, name)
        if stable_path.is_file():
            stable_states_cpu[name] = _load_stable_state(stable_path)

    # stable state 按环境排好序并一次搬到 GPU；缺失状态只在确实请求评估时报告。
    missing_stable_state = torch.zeros(7)
    stable_states = torch.stack(
        [stable_states_cpu.get(name, missing_stable_state) for name in names]
    ).to(device=env.device, dtype=robot.data.joint_pos.dtype)
    stable_state_available = torch.tensor(
        [name in stable_states_cpu for name in names], device=env.device, dtype=torch.bool
    )

    env._teacher_reset_data = {
        "dataset_name": dataset_name,
        "object_names": names,
        "object_ids": object_ids,
        "unique_names": unique_names,
        "lowest": lowest,
        "points": cached_points,
        "triangles": triangles,
        "stable_states": stable_states,
        "stable_state_available": stable_state_available,
        "joint_ids": joint_ids,
        "joint_pos": robot.data.default_joint_pos.clone(),
        "object_pose": torch.zeros(
            (env.num_envs, 7),
            device=env.device,
            dtype=robot.data.joint_pos.dtype,
        ),
        "angle": torch.zeros(
            env.num_envs,
            device=env.device,
            dtype=robot.data.joint_pos.dtype,
        ),
        "bias": torch.zeros(
            (env.num_envs, 3),
            device=env.device,
            dtype=robot.data.joint_pos.dtype,
        ),
        "bias_pending": torch.zeros(
            env.num_envs,
            device=env.device,
            dtype=torch.bool,
        ),
        "collision_pending": torch.zeros(
            env.num_envs,
            device=env.device,
            dtype=torch.bool,
        ),
    }

    env._teacher_last_ik_feasible = torch.zeros(
        env.num_envs,
        device=env.device,
        dtype=torch.bool,
    )


def _ensure_initialized(
    env,
    dataset_name: str,
    object_names: tuple[str, ...],
) -> None:
    data = getattr(env, "_teacher_reset_data", None)

    if (
        not isinstance(data, dict)
        or data.get("dataset_name") != dataset_name
        or tuple(data.get("object_names", ())) != tuple(object_names)
    ):
        initialize_teacher_data(
            env,
            None,
            dataset_name=dataset_name,
            object_names=tuple(object_names),
        )


# -----------------------------------------------------------------------------
# Source object sampling
# -----------------------------------------------------------------------------

def _source_workspace_quat_w(robot, ids: torch.Tensor) -> torch.Tensor:
    """返回源物体采样/相机工作坐标系相对世界系的旋转。"""
    root_quat = robot.data.root_quat_w[ids]
    return quat_mul(root_quat, root_quat.new_tensor(_SOURCE_TO_BASE_QUAT).expand(len(ids), -1))


def _sample_object_xy(
    count: int,
    device: str,
    dtype: torch.dtype,
    non_uniform: bool,
    *,
    max_attempts: int = 4096,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample the source annular sector, rejecting positions outside ``|x|<0.25``.

    With non-uniform training enabled, each environment chooses the source
    uniform branch or Beta(0.5,0.5) edge-biased branch with 50% probability.
    """
    xy = torch.empty((count, 2), device=device, dtype=dtype)
    angles = torch.empty(count, device=device, dtype=dtype)
    pending = torch.arange(count, device=device)

    beta = torch.distributions.Beta(
        torch.tensor(0.5, device=device, dtype=dtype),
        torch.tensor(0.5, device=device, dtype=dtype),
    )

    for _ in range(max_attempts):
        if pending.numel() == 0:
            break

        n = pending.numel()
        angle_u = torch.rand(n, device=device, dtype=dtype)
        distance_u = torch.rand(n, device=device, dtype=dtype)

        if non_uniform:
            edge = torch.rand(n, device=device) < 0.5
            angle_u = torch.where(edge, beta.sample((n,)), angle_u)
            distance_u = torch.where(edge, beta.sample((n,)), distance_u)

        angle = XY_ANGLE_MIN + (XY_ANGLE_MAX - XY_ANGLE_MIN) * angle_u
        distance = XY_DISTANCE_MIN + (XY_DISTANCE_MAX - XY_DISTANCE_MIN) * distance_u
        x = distance * angle.cos()
        y = distance * angle.sin()

        accepted = x.abs() < XY_ABS_X_LIMIT
        accepted_ids = pending[accepted]

        xy[accepted_ids, 0] = x[accepted]
        xy[accepted_ids, 1] = y[accepted]
        angles[accepted_ids] = angle[accepted]

        pending = pending[~accepted]

    if pending.numel() > 0:
        raise RuntimeError(
            "Teacher XY rejection sampling exceeded "
            f"{max_attempts} attempts for {pending.numel()} environments."
        )

    return xy, angles


def _sample_object_pose(
    env,
    ids: torch.Tensor,
    *,
    non_uniform_sampling: bool,
    support_height: float,
    use_stable_states: bool,
    stable_state_height_offset: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return world-frame object root poses and sampled polar angles."""
    data = env._teacher_reset_data
    robot = env.scene["robot"]
    dtype = robot.data.joint_pos.dtype

    # Source evaluation always uses the uniform branch.
    use_non_uniform = (
        non_uniform_sampling
        and data["dataset_name"] == SOURCE_TRAINING_DATASET
        and not use_stable_states
    )

    xy, angles = _sample_object_xy(
        len(ids),
        env.device,
        dtype,
        use_non_uniform,
    )

    object_pose = torch.zeros((len(ids), 7), device=env.device, dtype=dtype)
    # 保留源距离/角度分布，将源 -Y 工作区旋转到当前底座前方。
    workspace_quat = _source_workspace_quat_w(robot, ids)
    object_pose[:, :2] = xy
    object_pose[:, :3] = (
        quat_apply(workspace_quat, object_pose[:, :3]) + robot.data.root_pos_w[ids]
    )

    if use_stable_states:
        missing_ids = ids[~data["stable_state_available"][ids]]
        if missing_ids.numel() > 0:
            missing = tuple(dict.fromkeys(data["object_names"][index] for index in missing_ids.cpu().tolist()))
            raise FileNotFoundError(
                "Stable-state reset requested, but no valid <object>.npy was "
                f"loaded for: {missing}."
            )
        stable = data["stable_states"][ids]

        # quantitative_eval.py resamples X/Y and copies only stable Z + quat.
        object_pose[:, 2] = (
            stable[:, 2] - SOURCE_STABLE_SUPPORT_HEIGHT
            + support_height + stable_state_height_offset
        )
        object_pose[:, 3:7] = stable[:, 3:7]
    else:
        object_pose[:, 2] = support_height - data["lowest"][ids]

        yaw = (
            torch.rand(len(ids), device=env.device, dtype=dtype) * 2.0 - 1.0
        ) * math.pi
        object_pose[:, 3] = (yaw * 0.5).cos()
        object_pose[:, 6] = (yaw * 0.5).sin()

    object_pose[:, 3:7] = quat_mul(workspace_quat, object_pose[:, 3:7])
    object_pose[:, 2] += env.scene.env_origins[ids, 2]

    quaternion_norm = torch.linalg.vector_norm(
        object_pose[:, 3:7],
        dim=-1,
        keepdim=True,
    ).clamp_min(1.0e-8)
    object_pose[:, 3:7] /= quaternion_norm

    return object_pose, angles


# -----------------------------------------------------------------------------
# Pre-grasp geometry / IK
# -----------------------------------------------------------------------------

def _visible_points(
    env,
    ids: torch.Tensor,
    object_pose: torch.Tensor,
    camera_position: tuple[float, float, float],
) -> torch.Tensor:
    """Ray-cast the source 200 visible affordance points in world coordinates."""
    data = env._teacher_reset_data
    dtype = object_pose.dtype

    robot = env.scene["robot"]
    camera_w = quat_apply(
        _source_workspace_quat_w(robot, ids),
        object_pose.new_tensor(camera_position).expand(len(ids), -1),
    ) + robot.data.root_pos_w[ids]
    camera_o = geometry.world_points_to_object(
        camera_w[:, None, :],
        object_pose[:, :3],
        object_pose[:, 3:7],
    )[:, 0]

    visible_o = torch.empty_like(data["points"][ids])
    object_ids = data["object_ids"][ids]

    for object_index, name in enumerate(data["unique_names"]):
        group = (object_ids == object_index).nonzero(as_tuple=True)[0]
        if group.numel() == 0:
            continue

        triangles = data["triangles"][name]
        visible_o[group] = geometry.visible_points(
            data["points"][ids[group]],
            triangles,
            camera_o[group],
        )

    return geometry.object_points_to_world(
        visible_o,
        object_pose[:, :3],
        object_pose[:, 3:7],
    )


def _solver_base_pose(
    robot,
    ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """以新 USD 的 base_link root 位姿构造解析 UR5 的 base 坐标系。

    移除旧平台后 root 就是 base_link，不再添加平台的 0.771 m 高度。
    UR5 DH/base 与 ROS base_link 相差 Rz(pi)，对应 URDF 中保留的
    base_link-base_fixed_joint；DH 参数及 IK 算法保持不变。
    """
    root_pose = robot.data.root_link_pose_w[ids]
    base_pos = root_pose[:, :3]
    half_yaw = math.pi / 2.0
    base_quat_local = root_pose.new_tensor(
        (math.cos(half_yaw), 0.0, 0.0, math.sin(half_yaw))
    ).expand(len(ids), -1)
    base_quat = quat_mul(root_pose[:, 3:7], base_quat_local)

    return base_pos, matrix_from_quat(base_quat)


def _solve_pregrasp(
    env,
    ids: torch.Tensor,
    visible_w: torch.Tensor,
    *,
    top: bool,
    camera_position: tuple[float, float, float],
    sample_num: int,
    length_score_coeff: float,
    angle_score_coeff: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate, solve and rank batched UR5 pre-grasp candidates."""
    dtype = visible_w.dtype

    robot = env.scene["robot"]
    camera_w = quat_apply(
        _source_workspace_quat_w(robot, ids),
        torch.tensor(camera_position, device=env.device, dtype=dtype).expand(len(ids), -1),
    ) + robot.data.root_pos_w[ids]
    direction = geometry.approach_direction(
        visible_w,
        camera_w,
        top=top,
    )
    rotations, widths = geometry.sample_rot_mats(
        direction,
        sample_num,
        visible_w,
    )

    base_pos, base_rotation = _solver_base_pose(robot, ids)

    targets = geometry.wrist_poses_for_ur5_ik(
        visible_w,
        direction,
        rotations,
        base_pos,
        base_rotation,
    )
    ik_joints, feasible = geometry.ur5_inverse_kinematics(targets)

    arm_joints, selected_feasible = geometry.select_pregrasp_candidate(
        widths,
        ik_joints,
        feasible,
        length_score_coeff,
        angle_score_coeff,
    )

    return arm_joints, selected_feasible


def _source_fallback_arm(
    angles: torch.Tensor,
) -> torch.Tensor:
    """Return the source six-joint safe fallback pose."""
    return torch.stack(
        (
            angles + math.pi / 2.0 - 0.3,
            torch.full_like(angles, _FALLBACK_ARM_TAIL[0]),
            torch.full_like(angles, _FALLBACK_ARM_TAIL[1]),
            torch.full_like(angles, _FALLBACK_ARM_TAIL[2]),
            torch.full_like(angles, _FALLBACK_ARM_TAIL[3]),
            torch.full_like(angles, _FALLBACK_ARM_TAIL[4]),
        ),
        dim=-1,
    )


# -----------------------------------------------------------------------------
# State writing / collision fallback
# -----------------------------------------------------------------------------

def _write_reset_state(
    env,
    env_ids: torch.Tensor,
    joint_pos: torch.Tensor,
    object_pose: torch.Tensor,
) -> None:
    """Write robot/object state and hold targets at the reset pose."""
    env._teacher_runtime_features.invalidate(env_ids)
    robot = env.scene["robot"]
    obj = env.scene["object"]

    zero_joint_vel = torch.zeros_like(joint_pos)
    robot.write_joint_state_to_sim(
        joint_pos,
        zero_joint_vel,
        env_ids=env_ids,
    )
    robot.set_joint_position_target(
        joint_pos,
        env_ids=env_ids,
    )
    robot.set_joint_velocity_target(
        zero_joint_vel,
        env_ids=env_ids,
    )

    obj.write_root_pose_to_sim(
        object_pose,
        env_ids=env_ids,
    )
    obj.write_root_velocity_to_sim(
        torch.zeros(
            (len(env_ids), 6),
            device=env.device,
            dtype=object_pose.dtype,
        ),
        env_ids=env_ids,
    )

    if obj.data.joint_pos.shape[1] > 0:
        # reset_state() receives a zero-initialized articulated-object tail.
        object_joint_pos = torch.zeros_like(obj.data.joint_pos[env_ids])
        object_joint_vel = torch.zeros_like(obj.data.joint_vel[env_ids])
        obj.write_joint_state_to_sim(
            object_joint_pos,
            object_joint_vel,
            env_ids=env_ids,
        )


def _current_arm_collision(
    env,
    ids: torch.Tensor,
) -> torch.Tensor:
    """Read source arm-collision slots after a completed physics step."""
    collision = torch.zeros(len(ids), device=env.device, dtype=torch.bool)

    for sensor_index in _COLLISION_ARM_SENSOR_INDICES:
        sensor = env.scene[f"teacher_arm_contact_{sensor_index}"]
        force = sensor.data.net_forces_w[ids, 0, :]
        collision |= torch.linalg.vector_norm(force, dim=-1) > sensor.cfg.force_threshold

    return collision


def _apply_collision_fallback(
    env,
    ids: torch.Tensor,
    names: tuple[str, ...],
    collision: torch.Tensor,
    joint_pos: torch.Tensor,
    object_pose: torch.Tensor,
    angles: torch.Tensor,
    ur5_joint_ids: tuple[int, ...],
) -> None:
    """Copy a collision-free same-object peer or use the source safe pose."""
    if not bool(collision.any()):
        return

    safe = ~collision
    robot = env.scene["robot"]
    base_positions = robot.data.root_pos_w[ids]

    for local_index in (
        torch.nonzero(collision, as_tuple=False)
        .flatten()
        .detach()
        .cpu()
        .tolist()
    ):
        same_object_safe = [
            candidate
            for candidate, candidate_name in enumerate(names)
            if candidate_name == names[local_index] and bool(safe[candidate])
        ]

        if same_object_safe:
            chosen = same_object_safe[
                int(
                    torch.randint(
                        len(same_object_safe),
                        (1,),
                        device=env.device,
                    ).item()
                )
            ]

            joint_pos[local_index] = joint_pos[chosen]

            # 将同物体安全姿态按真实 robot root 平移到目标环境。
            chosen_local_position = object_pose[chosen, :3] - base_positions[chosen]
            object_pose[local_index, :3] = (
                chosen_local_position + base_positions[local_index]
            )
            object_pose[local_index, 3:7] = object_pose[chosen, 3:7]
            continue

        fallback = _source_fallback_arm(
            angles[local_index : local_index + 1]
        )[0]
        for column, joint_id in enumerate(ur5_joint_ids):
            joint_pos[local_index, joint_id] = fallback[column]

        fallback_offset = object_pose.new_tensor((*_FALLBACK_OBJECT_XY, 0.0)).unsqueeze(0)
        fallback_xy = quat_apply(
            _source_workspace_quat_w(robot, ids[local_index : local_index + 1]),
            fallback_offset,
        )[0, :2]
        object_pose[local_index, :2] = base_positions[local_index, :2] + fallback_xy


# -----------------------------------------------------------------------------
# Main reset event
# -----------------------------------------------------------------------------

def reset_teacher(
    env,
    env_ids: Sequence[int] | torch.Tensor | None,
    dataset_name: str,
    object_names: tuple[str, ...],
    biased: bool,
    top: bool,
    non_uniform_sampling: bool,
    support_height: float,
    camera_position: tuple[float, float, float],
    sample_num: int,
    length_score_coeff: float,
    angle_score_coeff: float,
    use_stable_states: bool = False,
    stable_state_height_offset: float = STABLE_STATE_HEIGHT_OFFSET,
    collision_check: bool = True,
) -> None:
    """Place selected objects and solve/reset batched UR5 pre-grasp candidates.

    ``use_stable_states`` is part of the public event contract because the
    quantitative ShapeNet evaluator injects it dynamically.
    """
    ids = _env_ids(env, env_ids)
    if ids.numel() == 0:
        return

    names_all = tuple(object_names)
    _ensure_initialized(env, dataset_name, names_all)
    data = env._teacher_reset_data

    object_pose, angles = _sample_object_pose(
        env,
        ids,
        non_uniform_sampling=non_uniform_sampling,
        support_height=support_height,
        use_stable_states=use_stable_states,
        stable_state_height_offset=stable_state_height_offset,
    )

    visible_w = _visible_points(
        env,
        ids,
        object_pose,
        camera_position,
    )

    arm_joints, ik_feasible = _solve_pregrasp(
        env,
        ids,
        visible_w,
        top=top,
        camera_position=camera_position,
        sample_num=sample_num,
        length_score_coeff=length_score_coeff,
        angle_score_coeff=angle_score_coeff,
    )

    robot = env.scene["robot"]
    joint_pos = robot.data.default_joint_pos[ids].clone()
    ur5_joint_ids = tuple(data["joint_ids"][:6])

    # Source train.py keeps a zero-filled IK row when the chosen candidate is
    # infeasible; only the collision branch uses the fixed fallback pose.
    joint_pos[:, list(ur5_joint_ids)] = arm_joints

    _write_reset_state(
        env,
        ids,
        joint_pos,
        object_pose,
    )

    data["joint_pos"][ids] = joint_pos
    data["object_pose"][ids] = object_pose
    data["angle"][ids] = angles
    data["collision_pending"][ids] = collision_check

    env._teacher_last_ik_feasible[ids] = ik_feasible

    # Source biased curriculum samples one vector per reset and applies X/Y
    # once when the hand first approaches the affordance.
    if biased:
        data["bias"][ids] = (
            torch.rand(
                (len(ids), 3),
                device=env.device,
                dtype=joint_pos.dtype,
            )
            * (2.0 * BIAS_RANGE)
            - BIAS_RANGE
        )
        data["bias_pending"][ids] = True
    else:
        data["bias"][ids] = 0.0
        data["bias_pending"][ids] = False


def check_initial_collision(
    env,
    env_ids: Sequence[int] | torch.Tensor | slice | None,
) -> None:
    """Apply collision fallback after each selected environment's first physics step."""
    data = env._teacher_reset_data
    ids = _env_ids(env, env_ids)
    ids = ids[data["collision_pending"][ids] & (env.episode_length_buf[ids] == 1)]
    if ids.numel() == 0:
        return

    collision = _current_arm_collision(env, ids)
    if bool(collision.any()):
        joint_pos = data["joint_pos"][ids].clone()
        object_pose = data["object_pose"][ids].clone()
        names = tuple(data["object_names"][index] for index in ids.cpu().tolist())
        _apply_collision_fallback(
            env,
            ids,
            names,
            collision,
            joint_pos,
            object_pose,
            data["angle"][ids],
            tuple(data["joint_ids"][:6]),
        )
        bad_ids = ids[collision]
        _write_reset_state(env, bad_ids, joint_pos[collision], object_pose[collision])
        env.sim.forward()
        env.action_manager.reset(bad_ids)
        env.observation_manager.reset(bad_ids)
        env.reward_manager.get_term_cfg("obj_displacement_reward").func.reset(bad_ids)
        data["joint_pos"][bad_ids] = joint_pos[collision]
        data["object_pose"][bad_ids] = object_pose[collision]

    data["collision_pending"][ids] = False


# -----------------------------------------------------------------------------
# Optional rollout-time object bias
# -----------------------------------------------------------------------------

def apply_object_position_bias(
    env,
    env_ids: Sequence[int] | torch.Tensor | slice | None,
    distance_threshold: float = BIAS_DISTANCE_THRESHOLD,
) -> None:
    """Shift object X/Y once when the hand first reaches the affordance.

    The source checks ``min(dis_info[0:17]) < 0.07`` after each policy step and
    calls ``switch_obj_pos`` once for that episode.
    """
    data = getattr(env, "_teacher_reset_data", None)
    if not isinstance(data, dict):
        return

    ids = _env_ids(env, env_ids)
    ids = ids[data["bias_pending"][ids]]

    if ids.numel() == 0:
        return

    obj = env.scene["object"]
    top_pose = obj.data.body_link_pose_w[ids, env._teacher_body_indices.object_top]
    hand_points_w = hand_keypoints_w(env)[ids]

    # Compare in object/top frame, matching observe_vision_new's dis_info.
    hand_points_o = geometry.world_points_to_object(
        hand_points_w,
        top_pose[:, :3],
        top_pose[:, 3:7],
    )
    affordance_points_o = env._teacher_affordance_points_o[ids]

    distances = torch.cdist(
        hand_points_o,
        affordance_points_o,
    ).amin(dim=-1)
    selected = ids[distances.amin(dim=-1) < distance_threshold]

    if selected.numel() == 0:
        return

    pose = torch.cat(
        (
            obj.data.root_pos_w[selected],
            obj.data.root_quat_w[selected],
        ),
        dim=-1,
    ).clone()

    # Environment.hpp::switch_obj_pos changes only X/Y.
    pose[:, :2] += data["bias"][selected, :2]
    obj.write_root_pose_to_sim(pose, env_ids=selected)
    env._teacher_runtime_features.invalidate(selected)

    data["bias_pending"][selected] = False


__all__ = [
    "SOURCE_STABLE_SUPPORT_HEIGHT",
    "STABLE_STATE_HEIGHT_OFFSET",
    "SOURCE_TRAINING_DATASET",
    "BIAS_DISTANCE_THRESHOLD",
    "initialize_teacher_data",
    "reset_teacher",
    "check_initial_collision",
    "apply_object_position_bias",
]

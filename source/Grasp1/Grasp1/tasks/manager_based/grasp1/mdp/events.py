"""Teacher训练集预抓取Reset；状态写入仅发生在合法Reset事件。"""
from __future__ import annotations
import math
import re
from collections.abc import Sequence
import numpy as np
import torch
from isaaclab.utils.math import matrix_from_quat, quat_apply, quat_mul
from Grasp1.robots.robot_profile import CONTROLLED_JOINT_NAMES
from Grasp1.utils.paths import object_asset_dir
from . import geometry

SOURCE_TRAINING_DATASET = "new_training_set"
XY_ANGLE_MIN, XY_ANGLE_MAX = -0.7*math.pi, -0.3*math.pi
XY_DISTANCE_MIN, XY_DISTANCE_MAX, XY_ABS_X_LIMIT = 0.45, 0.75, 0.25
_SOURCE_TO_BASE_QUAT = (math.cos(1.57/2), 0., 0., math.sin(1.57/2))


def configure_object_joint(env, env_ids, armature):
    env.scene["object"].write_joint_armature_to_sim(armature, env_ids=env_ids)


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



def initialize_teacher_data(env, env_ids, dataset_name, object_names):
    if dataset_name != SOURCE_TRAINING_DATASET or len(object_names) != env.num_envs:
        raise ValueError("Teacher requires one new_training_set object per environment.")
    import trimesh
    robot = env.scene["robot"]
    unique = tuple(dict.fromkeys(object_names))
    indices = {name: i for i, name in enumerate(unique)}
    triangles = {}
    for name in unique:
        mesh = trimesh.load_mesh(object_asset_dir(dataset_name, name)/"top_watertight_tiny.obj", process=False)
        triangles[name] = torch.as_tensor(np.array(mesh.triangles, copy=True), dtype=torch.float32, device=env.device)
    env._teacher_reset_data = dict(
        dataset_name=dataset_name, object_names=tuple(object_names), unique_names=unique,
        object_ids=torch.tensor([indices[n] for n in object_names], device=env.device),
        lowest=torch.tensor([env._teacher_object_metadata[n].lowest_point for n in object_names], device=env.device),
        points=env._teacher_affordance_points_o, triangles=triangles,
        joint_ids=_resolve_controlled_joint_ids(robot),
    )
    env._teacher_last_ik_feasible = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)


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
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return world-frame object root poses and sampled polar angles."""
    data = env._teacher_reset_data
    robot = env.scene["robot"]
    dtype = robot.data.joint_pos.dtype

    # Source evaluation always uses the uniform branch.
    use_non_uniform = (
        non_uniform_sampling
        and data["dataset_name"] == SOURCE_TRAINING_DATASET
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

    object_pose[:, 2] = support_height - data["lowest"][ids]
    yaw = (torch.rand(len(ids), device=env.device, dtype=dtype)*2.0-1.0)*math.pi
    object_pose[:, 3] = (yaw*0.5).cos()
    object_pose[:, 6] = (yaw*0.5).sin()

    object_pose[:, 3:7] = quat_mul(workspace_quat, object_pose[:, 3:7])
    object_pose[:, 2] += env.scene.env_origins[ids, 2]

    quaternion_norm = torch.linalg.vector_norm(
        object_pose[:, 3:7],
        dim=-1,
        keepdim=True,
    ).clamp_min(1.0e-8)
    object_pose[:, 3:7] /= quaternion_norm

    return object_pose, angles


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


def _write_reset_state(
    env,
    env_ids: torch.Tensor,
    joint_pos: torch.Tensor,
    object_pose: torch.Tensor,
) -> None:
    """Write robot/object state and hold targets at the reset pose."""
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

    # 只在Reset同步动作目标，rollout不写机器人/物体状态。
    env.action_manager.get_term("teacher").reset(env_ids)

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



def reset_teacher(env, env_ids, dataset_name, object_names, top, non_uniform_sampling,
                  support_height, camera_position, sample_num, length_score_coeff, angle_score_coeff):
    """只重置选定行；不可行IK必须重新采样，不能静默使用零关节姿态。"""
    ids = _env_ids(env, env_ids)
    data = env._teacher_reset_data
    robot = env.scene["robot"]
    pending = ids
    for _ in range(32):
        if not len(pending):
            break
        pose, _ = _sample_object_pose(env, pending, non_uniform_sampling=non_uniform_sampling,
                                      support_height=support_height)
        visible = _visible_points(env, pending, pose, camera_position)
        arm, feasible = _solve_pregrasp(env, pending, visible, top=top, camera_position=camera_position,
                                       sample_num=sample_num, length_score_coeff=length_score_coeff,
                                       angle_score_coeff=angle_score_coeff)
        q = robot.data.default_joint_pos[pending].clone()
        q[:, list(data["joint_ids"][:6])] = arm
        limits = robot.data.joint_pos_limits[pending]
        valid = feasible & torch.isfinite(q).all(1) & ((q >= limits[..., 0]) & (q <= limits[..., 1])).all(1)
        accepted = pending[valid]
        if len(accepted):
            _write_reset_state(env, accepted, q[valid], pose[valid])
            env._teacher_last_ik_feasible[accepted] = True
        pending = pending[~valid]
    if len(pending):
        raise RuntimeError(f"Teacher Reset has no valid IK: env_ids={pending.cpu().tolist()}")
    env.sim.forward()
    env._teacher_grasp.reset(ids)

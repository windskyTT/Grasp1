"""Teacher 预抓取使用的可见点、坐标变换、姿态采样和 UR5 解析 IK。

本模块只处理几何与 Torch 张量，不读取或写入仿真状态。

主要职责：

- object/world 点坐标变换；
- reset-time 可见表面点 ray casting；
- Teacher approach direction 与 palm rotation sampling；
- pre-grasp wrist pose 构造；
- 旧 UR5 DH 模型的批量解析 IK；
- pre-grasp candidate 评分与选择。

四元数采用 Isaac Lab 的 ``(w, x, y, z)`` 顺序。旋转矩阵最后一维表示
局部坐标轴在父坐标系中的方向。

ray-triangle intersection 按 triangle chunk 计算，避免一次构造完整的
``[N, P, F, 3]`` 张量。
"""

from __future__ import annotations

import math
from typing import Final

import torch

from isaaclab.utils.math import matrix_from_quat

from Grasp1.robots.robot_profile import HAND_CENTER


# -----------------------------------------------------------------------------
# 从 RobustDexGrasp Teacher 迁移的几何和候选筛选常量。
# -----------------------------------------------------------------------------

# 可见表面点云中的点数。
VISIBLE_POINT_COUNT: Final[int] = 200
# 手腕目标沿接近方向离开物体中心的预抓取距离，单位为米。
PREGRASP_DISTANCE: Final[float] = 0.25
# 用于按投影宽度筛选抓取方向的阈值，单位为米。
PROJECTION_THRESHOLD: Final[float] = 0.18
# 候选无效或没有可行 IK 解时使用的高惩罚分数。
INVALID_SCORE: Final[float] = 10000.0

# 旧 Teacher 在选择 UR5 解析解时使用的默认参考关节角，单位为弧度。
DEFAULT_UR5_REFERENCE_JOINTS: Final[tuple[float, ...]] = (
    0.0,
    -1.57,
    1.57,
    0.0,
    1.57,
    -1.57,
)

# 旧 UR5 解析 IK 使用的 DH 几何参数和统一关节角限位，长度单位为米、角度单位为弧度。
_UR5_D1: Final[float] = 0.089159
_UR5_D4: Final[float] = 0.10915
_UR5_D5: Final[float] = 0.09465
_UR5_D6: Final[float] = 0.0823
_UR5_A2: Final[float] = -0.425
_UR5_A3: Final[float] = -0.39225
_UR5_JOINT_LIMIT: Final[float] = 3.14


# -----------------------------------------------------------------------------
# 点云的刚体正变换和逆变换。
# -----------------------------------------------------------------------------

def transform_points(
    points: torch.Tensor,
    rotation: torch.Tensor,
    translation: torch.Tensor,
) -> torch.Tensor:
    """将局部点 ``[..., P, 3]`` 变换到父坐标系。

    ``rotation[..., 3, 3]`` 的列向量表示局部坐标轴在父坐标系中的方向。
    """
    if points.shape[-1] != 3:
        raise ValueError(
            f"points must end with dimension 3, got {tuple(points.shape)}."
        )
    if rotation.shape[-2:] != (3, 3):
        raise ValueError(
            f"rotation must end with [3,3], got {tuple(rotation.shape)}."
        )
    if translation.shape[-1] != 3:
        raise ValueError(
            f"translation must end with dimension 3, got {tuple(translation.shape)}."
        )

    return points @ rotation.transpose(-1, -2) + translation.unsqueeze(-2)


def inverse_transform_points(
    points: torch.Tensor,
    rotation: torch.Tensor,
    translation: torch.Tensor,
) -> torch.Tensor:
    """将父坐标系点 ``[..., P, 3]`` 变换到局部坐标系。"""
    if points.shape[-1] != 3:
        raise ValueError(
            f"points must end with dimension 3, got {tuple(points.shape)}."
        )
    if rotation.shape[-2:] != (3, 3):
        raise ValueError(
            f"rotation must end with [3,3], got {tuple(rotation.shape)}."
        )
    if translation.shape[-1] != 3:
        raise ValueError(
            f"translation must end with dimension 3, got {tuple(translation.shape)}."
        )

    return (points - translation.unsqueeze(-2)) @ rotation


def object_points_to_world(
    points_o: torch.Tensor,
    object_pos_w: torch.Tensor,
    object_quat_w: torch.Tensor,
) -> torch.Tensor:
    """把物体局部点云 ``[N, P, 3]`` 转换到世界系。"""
    return transform_points(
        points_o,
        matrix_from_quat(object_quat_w),
        object_pos_w,
    )


def world_points_to_object(
    points_w: torch.Tensor,
    object_pos_w: torch.Tensor,
    object_quat_w: torch.Tensor,
) -> torch.Tensor:
    """把世界系点云 ``[N, P, 3]`` 转换到物体系。"""
    return inverse_transform_points(
        points_w,
        matrix_from_quat(object_quat_w),
        object_pos_w,
    )


# -----------------------------------------------------------------------------
# 从相机位置向采样点发射射线，取得网格可见表面点。
# -----------------------------------------------------------------------------

def visible_points(
    sampled_points_o: torch.Tensor,
    triangles_o: torch.Tensor,
    camera_o: torch.Tensor,
    *,
    face_chunk_size: int = 2048,
    eps: float = 1.0e-8,
) -> torch.Tensor:
    """沿相机到目标采样点的射线求网格首个交点，返回可见点云。

    参数 ``sampled_points_o`` 是形状 ``[N,P,3]`` 的 mesh 采样目标点；
    ``triangles_o`` 是共享三角面 ``[F,3,3]``；``camera_o`` 是物体系下的相机位置
    ``[N,3]``。不同环境使用不同 mesh 时，应由 reset/event 分别调用本函数。

    ``face_chunk_size`` 控制每次处理的三角面数量，以限制峰值显存；``eps`` 用于
    判断退化或近平行的射线/三角形情况。返回首个命中点 ``[N,P,3]``。

    保持旧实现 ``multiple_hits=False`` 的首个交点语义和固定射线顺序；没有命中的
    射线使用该环境第一个有效命中点补齐。

    """
    if sampled_points_o.ndim != 3 or sampled_points_o.shape[-1] != 3:
        raise ValueError(
            "sampled_points_o must have shape [N,P,3], got "
            f"{tuple(sampled_points_o.shape)}."
        )
    if triangles_o.ndim != 3 or triangles_o.shape[-2:] != (3, 3):
        raise ValueError(
            "triangles_o must have shape [F,3,3], got "
            f"{tuple(triangles_o.shape)}."
        )
    if camera_o.shape != (sampled_points_o.shape[0], 3):
        raise ValueError(
            "camera_o must have shape [N,3], got "
            f"{tuple(camera_o.shape)}."
        )
    if triangles_o.shape[0] == 0:
        raise ValueError("triangles_o must contain at least one triangle.")
    if face_chunk_size <= 0:
        raise ValueError(
            f"face_chunk_size must be positive, got {face_chunk_size}."
        )

    # 从相机指向每个目标采样点，并归一化为射线方向。
    ray_vectors = sampled_points_o - camera_o[:, None, :]
    ray_norms = torch.linalg.vector_norm(
        ray_vectors,
        dim=-1,
        keepdim=True,
    )

    directions = ray_vectors / ray_norms

    num_envs, num_rays, _ = sampled_points_o.shape

    nearest_distance = torch.full(
        (num_envs, num_rays),
        torch.inf,
        dtype=sampled_points_o.dtype,
        device=sampled_points_o.device,
    )

    # 分块遍历三角面，避免为完整网格一次性构造 [N,P,F,3] 中间张量。
    for start in range(0, triangles_o.shape[0], face_chunk_size):
        triangles = triangles_o[
            start : start + face_chunk_size
        ]

        vertex0 = triangles[:, 0]
        edge1 = triangles[:, 1] - vertex0
        edge2 = triangles[:, 2] - vertex0

        # 用 Möller-Trumbore 算法计算当前射线与三角面的交点参数。
        ray_cross = torch.cross(
            directions[:, :, None, :],
            edge2[None, None, :, :],
            dim=-1,
        )
        determinant = (
            edge1[None, None, :, :]
            * ray_cross
        ).sum(dim=-1)

        determinant_valid = determinant.abs() > eps

        # 对将被判为平行/退化的交点令倒数为零，避免除法产生 inf 或 NaN。
        inverse_det = torch.where(
            determinant_valid,
            determinant.reciprocal(),
            torch.zeros_like(determinant),
        )

        origin_edge = (camera_o[:, None, None, :] - vertex0[None, None, :, :])
        u = (origin_edge * ray_cross).sum(dim=-1) * inverse_det
        origin_cross = torch.cross(origin_edge,edge1[None, None, :, :],dim=-1)
        v = (directions[:, :, None, :]* origin_cross).sum(dim=-1) * inverse_det
        distance = (edge2[None, None, :, :]* origin_cross).sum(dim=-1) * inverse_det

        # 保留三角形内部且位于射线正向的有效交点。
        hit = (
            determinant_valid
            & (u >= 0.0)
            & (v >= 0.0)
            & (u + v <= 1.0)
            & (distance > eps)
        )

        # 每条射线只记录当前三角面块中的最近有效交点。
        chunk_nearest = distance.masked_fill(
            ~hit,
            torch.inf,
        ).amin(dim=-1)

        nearest_distance = torch.minimum(
            nearest_distance,
            chunk_nearest,
        )

    # 汇总所有三角面块后，最近距离有限即表示该射线命中网格。
    hit_mask = torch.isfinite(nearest_distance)
    hit_points = (
        camera_o[:, None, :]
        + nearest_distance[..., None] * directions
    )

    # 找出每个环境第一个有效命中点，供未命中射线维持固定点数时复用。
    first_hit_index = (
        hit_mask.to(torch.int64)
        .argmax(dim=1)
    )
    first_hit = hit_points[
        torch.arange(num_envs, device=sampled_points_o.device),
        first_hit_index,
    ]

    return torch.where(
        hit_mask[..., None],
        hit_points,
        first_hit[:, None, :],
    )


# -----------------------------------------------------------------------------
# 预抓取接近方向、手腕朝向候选及目标位姿。
# -----------------------------------------------------------------------------

def approach_direction(
    visible_points_w: torch.Tensor,
    camera_w: torch.Tensor,
    *,
    top: bool = False,
) -> torch.Tensor:
    """计算每个环境的手部接近方向 ``[N,3]``。

    ``top=True`` 时使用世界系竖直向上方向 ``[0,0,1]``；否则使用从可见点云中心
    指向相机的单位向量。
    """
    if visible_points_w.ndim != 3 or visible_points_w.shape[-1] != 3:
        raise ValueError(
            "visible_points_w must have shape [N,P,3], got "
            f"{tuple(visible_points_w.shape)}."
        )

    num_envs = visible_points_w.shape[0]

    if camera_w.shape == (3,):
        camera_w = camera_w.unsqueeze(0).expand(num_envs, -1)

    if camera_w.shape != (num_envs, 3):
        raise ValueError(
            f"camera_w must have shape [3] or [{num_envs},3], "
            f"got {tuple(camera_w.shape)}."
        )

    if top:
        direction = torch.zeros_like(camera_w)
        direction[:, 2] = 1.0
        return direction

    # 以点云中心到相机的方向作为手部接近方向。
    direction = (
        camera_w
        - visible_points_w.mean(dim=1)
    )

    norm = torch.linalg.vector_norm(
        direction,
        dim=-1,
        keepdim=True,
    )
    return direction / norm


def sample_rot_mats(
    hand_dir_x_w: torch.Tensor,
    num_samples: int,
    visible_points_w: torch.Tensor,
    *,
    eps: float = 1.0e-8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """围绕给定接近方向等角采样手掌朝向，并计算各朝向下的点云投影宽度。

    输入 ``hand_dir_x_w`` 为世界系接近方向 ``[N,3]``，``num_samples`` 是每个
    环境采样的候选数量，``visible_points_w`` 是可见点云 ``[N,P,3]``。

    返回 ``(rotations, projection_widths)``：旋转矩阵形状为 ``[N,S,3,3]``，
    投影宽度形状为 ``[N,S]``。

    保留源实现约定：当 ``abs(hand_x[0]) < 0.9`` 时以世界 X 轴构造临时基向量，
    否则使用世界 Y 轴；若采样垂直方向的世界 Y 分量小于零则反向。旋转矩阵按
    ``-stack((hand_x, hand_y, perpendicular), dim=-1)`` 构造。
    """
    if num_samples <= 0:
        raise ValueError(
            f"num_samples must be positive, got {num_samples}."
        )
    if hand_dir_x_w.ndim != 2 or hand_dir_x_w.shape[-1] != 3:
        raise ValueError(
            "hand_dir_x_w must have shape [N,3], got "
            f"{tuple(hand_dir_x_w.shape)}."
        )
    if (
        visible_points_w.ndim != 3
        or visible_points_w.shape[0] != hand_dir_x_w.shape[0]
        or visible_points_w.shape[-1] != 3
    ):
        raise ValueError(
            "visible_points_w must have shape [N,P,3] with the same N "
            "as hand_dir_x_w."
        )

    # 将接近方向单位化，并选取不与其近似平行的世界轴作为构造基准。
    hand_norm = torch.linalg.vector_norm(
        hand_dir_x_w,
        dim=-1,
        keepdim=True,
    )
    hand_dir_x_w = hand_dir_x_w / hand_norm

    world_x = hand_dir_x_w.new_tensor((1.0, 0.0, 0.0)).expand_as(hand_dir_x_w)
    world_y = hand_dir_x_w.new_tensor((0.0, 1.0, 0.0)).expand_as(hand_dir_x_w)

    reference = torch.where((hand_dir_x_w[:, :1].abs() < 0.9),world_x,world_y)

    # 构造与接近方向垂直的平面基底，用于绕接近轴生成候选方向。
    first = torch.cross(hand_dir_x_w,reference,dim=-1)
    first = torch.nn.functional.normalize(first,dim=-1,eps=eps)
    second = torch.cross(hand_dir_x_w,first,dim=-1)
    second = torch.nn.functional.normalize(second,dim=-1,eps=eps)

    # 在一整圈上等间隔采样旋转角度，并据此生成垂直接近方向的候选向量。
    angles = torch.arange(num_samples,device=hand_dir_x_w.device,dtype=hand_dir_x_w.dtype)
    angles = angles * (2.0 * math.pi / num_samples)

    perpendicular = (
        first[:, None, :]
        * angles.cos()[None, :, None]
        + second[:, None, :]
        * angles.sin()[None, :, None]
    )
    perpendicular = torch.nn.functional.normalize(perpendicular,dim=-1,eps=eps)

    # 遵循源约定：让采样的 z/垂直方向保持世界 Y 分量非负。
    perpendicular = torch.where(
        perpendicular[..., 1:2] < 0.0,
        -perpendicular,
        perpendicular,
    )

    # 将可见点云移到中心原点，再投影到每个候选垂直方向上计算宽度。
    centered = (
        visible_points_w
        - visible_points_w.mean(dim=1, keepdim=True)
    )

    projections = torch.einsum(
        "npc,nsc->nsp",
        centered,
        perpendicular,
    )

    widths = (
        projections.amax(dim=-1)
        - projections.amin(dim=-1)
    )

    hand_x = hand_dir_x_w[:, None, :].expand_as(
        perpendicular
    )
    # 由接近方向和垂直方向构造手掌局部 y 轴，并按源约定组成旋转矩阵。
    y_direction = torch.cross(
        hand_x,
        perpendicular,
        dim=-1,
    )
    y_direction = torch.nn.functional.normalize(
        y_direction,
        dim=-1,
        eps=eps,
    )

    rotations = -torch.stack(
        (
            hand_x,
            y_direction,
            perpendicular,
        ),
        dim=-1,
    )

    return rotations, widths


def wrist_poses_for_ur5_ik(
    visible_points_w: torch.Tensor,
    hand_dir_x_w: torch.Tensor,
    wrist_rotations_w: torch.Tensor,
    ur5_base_pos_w: torch.Tensor,
    ur5_base_rotation_w: torch.Tensor,
    *,
    pregrasp_distance: float = PREGRASP_DISTANCE,
) -> torch.Tensor:
    """构造候选手腕位姿，并转换为 UR5 base 坐标系下的 IK 目标 ``[N,S,4,4]``。

    先以 ``可见点云中心 + pregrasp_distance × hand_dir_x_w`` 得到预抓取手部中心，
    再将 ``HAND_CENTER`` 通过候选手腕旋转变换后加到该位置，得到手腕位置。
    随后使用仿真场景中 UR5 base 的实际世界位姿，将候选手腕位姿变换到 base 坐标系，
    因而可以适配机器人整体平移或旋转。

    输入依次为世界系可见点 ``[N,P,3]``、接近方向 ``[N,3]``、候选旋转
    ``[N,S,3,3]``、UR5 base 世界位置 ``[N,3]`` 和旋转矩阵 ``[N,3,3]``；
    旋转矩阵的列向量表示 base 轴在世界系中的方向。距离默认沿用源值 0.25 m。
    返回 base 坐标系下的齐次变换 ``[N,S,4,4]``。
    """
    if visible_points_w.ndim != 3 or visible_points_w.shape[-1] != 3:
        raise ValueError(
            "visible_points_w must have shape [N,P,3]."
        )

    num_envs = visible_points_w.shape[0]

    if hand_dir_x_w.shape != (num_envs, 3):
        raise ValueError(
            f"hand_dir_x_w must have shape [{num_envs},3]."
        )
    if (
        wrist_rotations_w.ndim != 4
        or wrist_rotations_w.shape[0] != num_envs
        or wrist_rotations_w.shape[-2:] != (3, 3)
    ):
        raise ValueError(
            "wrist_rotations_w must have shape [N,S,3,3]."
        )
    if ur5_base_pos_w.shape != (num_envs, 3):
        raise ValueError(
            f"ur5_base_pos_w must have shape [{num_envs},3]."
        )
    if ur5_base_rotation_w.shape != (num_envs, 3, 3):
        raise ValueError(
            f"ur5_base_rotation_w must have shape [{num_envs},3,3]."
        )

    # 用可见点云的均值作为物体抓取区域中心，并沿接近方向移动到预抓取距离。
    center_w = visible_points_w.mean(dim=1)
    approach_pos_w = (
        center_w
        + pregrasp_distance * hand_dir_x_w
    )

    hand_center = wrist_rotations_w.new_tensor(
        HAND_CENTER
    )
    # 按源定义先将末端执行器坐标系中的 HAND_CENTER 旋转到世界系。
    wrist_pos_w = (
        approach_pos_w[:, None, :]
        + wrist_rotations_w @ hand_center
    )

    # ur5_base_rotation_w 的列表示 base 轴在世界系中的方向；列向量从世界转 base 时
    # 使用其转置。
    base_rotation_t = (
        ur5_base_rotation_w.transpose(-1, -2)
    )

    # 对行向量位置使用等价形式：p_b = (p_w - t_wb) @ R_wb。
    position_b = (
        wrist_pos_w
        - ur5_base_pos_w[:, None, :]
    ) @ ur5_base_rotation_w

    rotation_b = (
        base_rotation_t[:, None, :, :]
        @ wrist_rotations_w
    )

    target = torch.eye(
        4,
        device=wrist_rotations_w.device,
        dtype=wrist_rotations_w.dtype,
    ).expand(
        *wrist_rotations_w.shape[:2],
        4,
        4,
    ).clone()

    target[..., :3, :3] = rotation_b
    target[..., :3, 3] = position_b

    return target


# -----------------------------------------------------------------------------
# UR5 解析逆运动学计算。
# -----------------------------------------------------------------------------

def _dh_transform(
    theta: torch.Tensor,
    a: float,
    d: float,
    alpha: float,
) -> torch.Tensor:
    """按 UR5 DH 参数批量构造一个关节的齐次变换矩阵 ``[...,4,4]``。

    ``theta`` 可包含任意批次维度；``a``、``d`` 和 ``alpha`` 是该关节的 DH 连杆参数。
    """
    c = theta.cos()
    s = theta.sin()
    ca = math.cos(alpha)
    sa = math.sin(alpha)

    result = torch.zeros(
        (*theta.shape, 4, 4),
        device=theta.device,
        dtype=theta.dtype,
    )

    result[..., 0, 0] = c
    result[..., 0, 1] = -s * ca
    result[..., 0, 2] = s * sa
    result[..., 0, 3] = a * c

    result[..., 1, 0] = s
    result[..., 1, 1] = c * ca
    result[..., 1, 2] = -c * sa
    result[..., 1, 3] = a * s

    result[..., 2, 1] = sa
    result[..., 2, 2] = ca
    result[..., 2, 3] = d
    result[..., 3, 3] = 1.0

    return result


def _inverse_rigid(
    transform: torch.Tensor,
) -> torch.Tensor:
    """批量求刚体齐次变换 ``[...,4,4]`` 的逆变换。"""
    rotation_t = (
        transform[..., :3, :3]
        .transpose(-1, -2)
    )

    result = torch.zeros_like(transform)
    result[..., :3, :3] = rotation_t
    result[..., :3, 3] = -(
        rotation_t
        @ transform[..., :3, 3, None]
    ).squeeze(-1)
    result[..., 3, 3] = 1.0

    return result


def _broadcast_reference_joints(
    reference_joints: torch.Tensor,
    batch_shape: torch.Size,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """将参考关节角 ``[6]`` 或 ``[...,6]`` 扩展到 IK 目标的批次形状。

    参数指定目标批次形状、设备和数据类型；返回形状为 ``[*batch_shape,6]`` 的张量。
    """
    reference = torch.as_tensor(
        reference_joints,
        device=device,
        dtype=dtype,
    )

    if reference.shape[-1:] != (6,):
        raise ValueError(
            "reference_joints must end with dimension 6, got "
            f"{tuple(reference.shape)}."
        )

    # 示例：targets 为 [N,S,4,4]、reference 为 [N,6] 时，扩展成 [N,1,6]，
    # 再广播成 [N,S,6]，使同一环境参考姿态可用于其全部候选目标。
    while reference.ndim < len(batch_shape) + 1:
        reference = reference.unsqueeze(-2)

    try:
        reference = torch.broadcast_to(
            reference,
            (*batch_shape, 6),
        )
    except RuntimeError as exc:
        raise ValueError(
            "reference_joints is not broadcastable to IK target batch "
            f"{tuple(batch_shape)}; got {tuple(reference_joints.shape)}."
        ) from exc

    return reference


def ur5_inverse_kinematics(
    targets_b: torch.Tensor,
    reference_joints: torch.Tensor | None = None,
    *,
    joint_limit: float = _UR5_JOINT_LIMIT,
    eps: float = 1.0e-7,
) -> tuple[torch.Tensor, torch.Tensor]:
    """用旧 UR5 DH 几何模型为一批目标求最近的解析逆运动学解。

    ``targets_b`` 是 UR5 base 坐标系下的目标齐次变换 ``[...,4,4]``。
    ``reference_joints`` 是最近解选择所用的参考关节角，可为 ``[6]``、``[N,6]``
    或可广播到目标批次的 ``[...,6]``；省略时使用旧 Teacher 默认姿态。
    ``joint_limit`` 是统一关节角绝对限位（默认 3.14 弧度），``eps`` 是奇异位形和
    零向量判断使用的阈值。

    返回 ``(joint_solution, feasible)``，形状分别为 ``[...,6]`` 和 ``[...]``。
    不可行目标的关节结果置零，调用方应依据 ``feasible`` 判断是否能使用该结果。
    """
    if targets_b.ndim < 2 or targets_b.shape[-2:] != (4, 4):
        raise ValueError(
            "targets_b must end with [4,4], got "
            f"{tuple(targets_b.shape)}."
        )

    # 展平任意目标批次维，统一执行分支计算；返回前再恢复原批次形状。
    batch_shape = targets_b.shape[:-2]
    target = targets_b.reshape(-1, 4, 4)
    count = target.shape[0]

    # 求肩部关节 theta1 的两种解析分支，并判断目标是否落在可达肩部范围内。

    p05 = (target[:, :3, 3]- _UR5_D6 * target[:, :3, 2])
    radius = torch.linalg.vector_norm(p05[:, :2],dim=-1)

    # 与源 getFlags() 一致，允许可达性比例在 1 附近有 1% 的数值容差。
    shoulder_valid = (_UR5_D4 / radius.clamp_min(eps)).abs() < 1.01

    safe_radius = radius.clamp_min(_UR5_D4)
    phi = torch.acos((_UR5_D4 / safe_radius).clamp(-1.0, 1.0))
    psi = torch.atan2(p05[:, 1],p05[:, 0],)

    theta1 = torch.stack((psi + phi + math.pi / 2.0,psi - phi + math.pi / 2.0,),dim=-1)

    # 对每种肩部分支求腕部关节 theta5 的正负两种分支。
    p16z = (target[:, 0, 3, None] * theta1.sin()- target[:, 1, 3, None] * theta1.cos())

    wrist_ratio = (p16z - _UR5_D4) / _UR5_D6
    wrist_valid = (wrist_ratio.abs() < 1.01)
    theta5_abs = torch.acos(wrist_ratio.clamp(-1.0, 1.0))
    theta5 = torch.stack((theta5_abs,-theta5_abs,),dim=-1)

    # 求 theta6；在 theta5 接近奇异时按源求解器取 theta6=0。
    t1 = _dh_transform(theta1,0.0,_UR5_D1,math.pi / 2.0)
    t61 = _inverse_rigid(_inverse_rigid(t1) @ target[:, None, :, :])

    sin5 = theta5.sin()
    singular5 = sin5.abs() < eps

    safe_sin5 = torch.where(singular5,torch.ones_like(sin5),sin5)

    theta6 = torch.atan2(-t61[:, :, None, 1, 2] / safe_sin5,t61[:, :, None, 0, 2] / safe_sin5)

    # 奇异情况下源求解器将 theta6 设为 0。
    theta6 = torch.where(singular5,torch.zeros_like(theta6),theta6)

    # 根据腕部目标位置求肘部关节 theta2、theta3 的两种弯曲分支。
    t45 = _dh_transform(theta5,0.0,_UR5_D5,-math.pi / 2.0)
    t56 = _dh_transform(theta6,0.0,_UR5_D6,0.0)
    t14 = (_inverse_rigid(t1)@ target[:, None, :, :])[:, :, None, :, :]
    t14 = (t14 @ _inverse_rigid(t45 @ t56))
    p13_h = (t14@ target.new_tensor((0.0, -_UR5_D4, 0.0, 1.0)))
    p13 = p13_h[..., :3]

    elbow_ratio = (
        p13.square().sum(dim=-1)
        - _UR5_A2 * _UR5_A2 - _UR5_A3 * _UR5_A3
    ) / (2.0 * _UR5_A2 * _UR5_A3)

    elbow_valid = (elbow_ratio.abs() < 1.01)
    theta3_abs = torch.acos(elbow_ratio.clamp(-1.0, 1.0))
    theta3 = torch.stack((theta3_abs,-theta3_abs,),dim=-1)

    p13_norm = torch.linalg.vector_norm(p13,dim=-1).clamp_min(eps)

    theta2 = -torch.atan2(p13[..., 1, None],-p13[..., 0, None])
    theta2 = theta2 + torch.asin((_UR5_A3* theta3.sin()/ p13_norm[..., None]).clamp(-1.0, 1.0))

    # 利用目标旋转和已求出的前序关节角回算最后一个关节 theta4。
    t12 = _dh_transform(theta2, _UR5_A2,0.0,0.0)
    t23 = _dh_transform(theta3,_UR5_A3, 0.0,0.0)
    t34 = (_inverse_rigid(t12 @ t23) @ t14[..., None, :, :])
    theta4 = torch.atan2(t34[..., 1, 0],t34[..., 0, 0])

    # 组合肩、腕和肘的分支，得到最多 2×2×2=8 组完整关节解。
    joints = torch.stack(
        (
            theta1[
                :, :, None, None
            ].expand(count, 2, 2, 2),
            theta2,
            theta3,
            theta4,
            theta5[
                :, :, :, None
            ].expand(count, 2, 2, 2),
            theta6[
                :, :, :, None
            ].expand(count, 2, 2, 2),
        ),
        dim=-1,
    )

    # 将各关节角归一化到主值范围，与源 normalize() 的非奇异解行为一致。
    joints = torch.atan2(joints.sin(),joints.cos(),).reshape(count, 8, 6)
    valid = (shoulder_valid[:, None, None, None] & wrist_valid[:, :, None, None] & elbow_valid[..., None])
    valid = valid.expand(count,2,2,2).reshape(count, 8)

    valid = (valid & joints.isfinite().all(dim=-1)
        & (joints.abs() <= joint_limit).all(dim=-1)
    )

    # 按源 findClosestIK() 的加权 L1 距离选择最接近参考姿态的可行解。
    if reference_joints is None:
        reference = target.new_tensor(
            DEFAULT_UR5_REFERENCE_JOINTS
        ).reshape(1, 6).expand(count, -1)
    else:
        reference = _broadcast_reference_joints(
            reference_joints,
            batch_shape,
            device=target.device,
            dtype=target.dtype,
        ).reshape(count, 6)

    costs = (joints - reference[:, None, :]).abs().sum(dim=-1)
    costs = costs.masked_fill(~valid,torch.inf)
    best = costs.argmin(dim=-1)
    rows = torch.arange(count,device=target.device)
    result = joints[rows,best]
    feasible = valid.any(dim=-1)

    # 与原 train.py 一致：IK 失败时对应结果行保持为零，调用方同时收到 feasible=False。
    result = torch.where(feasible[:, None],result,torch.zeros_like(result))

    return (result.reshape(*batch_shape, 6),feasible.reshape(*batch_shape))


# -----------------------------------------------------------------------------
# 按源 Teacher 评分规则选择预抓取候选。
# -----------------------------------------------------------------------------

def select_pregrasp_candidate(
    projection_widths: torch.Tensor,
    ik_joints: torch.Tensor,
    feasible: torch.Tensor,
    length_score_coeff: float,
    angle_score_coeff: float,
    *,
    projection_threshold: float = PROJECTION_THRESHOLD,
) -> tuple[torch.Tensor, torch.Tensor]:
    """按旧 Teacher 规则从投影宽度和 UR5 IK 解中选择一个预抓取候选。

    输入投影宽度和可行标志为 ``[N,S]``，IK 解为 ``[N,S,6]``。短投影候选的分数为
    ``projection * length_score_coeff + abs(q[4] - 1.57) * angle_score_coeff
    + (abs(q[4]) - 3.2) * angle_score_coeff * 0.5``。若存在宽度小于阈值的候选，
    只在其中选择可行 IK 分数最低者；若不存在短投影候选，则源实现直接选最窄投影，
    即使其 IK 不可行。若短投影候选均不可行，分数均为 10000 并选索引 0。

    返回选中的关节角 ``[N,6]`` 和对应可行标志 ``[N]``。
    """
    if projection_widths.ndim != 2:
        raise ValueError(
            "projection_widths must have shape [N,S]."
        )
    if (
        ik_joints.ndim != 3
        or ik_joints.shape[:2] != projection_widths.shape
        or ik_joints.shape[-1] != 6
    ):
        raise ValueError(
            "ik_joints must have shape [N,S,6] matching projection_widths."
        )
    if feasible.shape != projection_widths.shape:
        raise ValueError(
            "feasible must have shape [N,S] matching projection_widths."
        )

    feasible = feasible.to(dtype=torch.bool,device=projection_widths.device)

    # 源评分使用第五个 UR5 关节角 theta5 作为角度代价。
    wrist_angle = ik_joints[..., 4]

    weighted_score = (
        projection_widths * length_score_coeff
        + (
            wrist_angle - 1.57
        ).abs() * angle_score_coeff
        + (
            wrist_angle.abs() - 3.2
        ) * angle_score_coeff * 0.5
    )

    # 判断每个环境是否存在投影宽度小于阈值的候选。
    any_short =projection_widths.amin(dim=-1,keepdim=True) < projection_threshold


    # 短投影候选使用加权分数；不可行或宽度超阈值的候选标为高惩罚。
    short_scores = torch.where(
        feasible & (projection_widths < projection_threshold),
        weighted_score,
        INVALID_SCORE,
    )

    # 若无短投影候选，按源规则直接选最窄投影，不附加 IK 可行性筛选。
    scores = torch.where(any_short,short_scores,projection_widths)
    best = scores.argmin(dim=-1)
    rows = torch.arange(ik_joints.shape[0],device=ik_joints.device)
    selected = ik_joints[rows,best]

    selected_feasible = feasible[rows, best]

    return selected, selected_feasible


# 对外提供的几何常量、点云坐标变换、可见点计算、预抓取位姿和 UR5 IK 接口。
__all__ = [
    "VISIBLE_POINT_COUNT",
    "PREGRASP_DISTANCE",
    "PROJECTION_THRESHOLD",
    "DEFAULT_UR5_REFERENCE_JOINTS",
    "transform_points",
    "inverse_transform_points",
    "object_points_to_world",
    "world_points_to_object",
    "visible_points",
    "approach_direction",
    "sample_rot_mats",
    "wrist_poses_for_ur5_ik",
    "ur5_inverse_kinematics",
    "select_pregrasp_candidate",
]

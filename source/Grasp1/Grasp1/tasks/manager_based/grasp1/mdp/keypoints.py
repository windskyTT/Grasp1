"""批量计算 Teacher 的手部与 UR5 关键点，并在世界系和物体系间转换。

USD 导入可能合并源 URDF 中由固定关节连接的连杆。本模块通过读取保留的 parent
刚体位姿并应用固定偏移，重建源 Teacher 使用的关键点位置和顺序。
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

import torch

from isaaclab.utils.math import quat_apply, quat_apply_inverse

from Grasp1.robots.robot_profile import (
    ARM_KEYPOINT_LINK_NAMES,
    HAND_FINGERTIP_KEYPOINT_INDICES,
    HAND_KEYPOINT_LINK_NAMES,
    UR5_LINK_NAMES,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# 源 URDF 的 hand base 和四个 fingertip 由固定关节连接。USD 导入时这些连杆
# 可能与 parent 合并，因此从仍存在的 parent 刚体位姿加固定偏移恢复关键点。
# 第 0 个手部点位于 wrist_3_link，其余关键点对应各手指关节的 parent link。
_HAND_PARENT_LINK_NAMES = (UR5_LINK_NAMES[-1],) + tuple(
    name.removesuffix("_tip") for name in HAND_KEYPOINT_LINK_NAMES[1:]
)
# 用 USD 中的关节/刚体命名规则规范名称，并去重后供 find_bodies 一次查询。
_HAND_BODY_NAMES = tuple(dict.fromkeys(name.replace(".", "_") for name in _HAND_PARENT_LINK_NAMES))
# 对每个关键点记录其 parent 刚体在去重 body 列表中的位置，以恢复 Teacher 原顺序。
_HAND_BODY_SELECT = tuple(_HAND_BODY_NAMES.index(name.replace(".", "_")) for name in _HAND_PARENT_LINK_NAMES)
# 四个指尖 frame 相对各自 parent link 沿局部 z 轴的固定平移距离，单位为米。
_TIP_OFFSETS_Z = (0.0267, 0.0267, 0.0267, 0.0423)


def hand_keypoints_w(env: ManagerBasedRLEnv) -> torch.Tensor:
    """重建并返回世界坐标系下的 17 个手部关键点，形状为 ``[N, 17, 3]``。

    ``N`` 是并行环境数。先解析去重后的 parent 刚体，再按原 Teacher 顺序取位姿；
    wrist/tool 和指尖关键点的局部固定偏移经刚体朝向旋转后加到 parent 位置上。
    """
    robot = env.scene["robot"]
    # 按 USD 刚体名称查询 parent link，并要求查询结果保持传入名称的顺序。
    body_ids, _ = robot.find_bodies([re.escape(name) for name in _HAND_BODY_NAMES], preserve_order=True)
    # 将去重后的 body 索引重新排列成 17 个关键点各自对应的 parent 索引。
    ordered_ids = [body_ids[index] for index in _HAND_BODY_SELECT]
    body_poses = robot.data.body_link_pose_w[:, ordered_ids, :]
    positions = body_poses[..., :3]
    orientations = body_poses[..., 3:7]

    # 为每个 parent link 构造局部坐标系中的固定 frame 偏移。
    offsets = torch.zeros_like(positions)
    # wrist_3_link 到 tool0/手腕基座 frame 的局部 y 轴平移；中间固定关节无平移。
    offsets[:, 0, 1] = 0.0823
    for index, offset_z in zip(HAND_FINGERTIP_KEYPOINT_INDICES, _TIP_OFFSETS_Z):
        offsets[:, index, 2] = offset_z

    # 将局部偏移旋转到世界系，再与 parent link 的世界位置相加。
    rotated_offsets = quat_apply(orientations.reshape(-1, 4), offsets.reshape(-1, 3))
    return positions + rotated_offsets.reshape_as(positions)


def arm_keypoints_w(env: ManagerBasedRLEnv) -> torch.Tensor:
    """按 robot_profile 顺序返回世界系下六个 UR5 关键点位置 ``[N, 6, 3]``。"""
    robot = env.scene["robot"]
    # 使用 URDF 对应连杆名查找刚体，并保持六个手臂关键点的语义顺序。
    body_ids, _ = robot.find_bodies(
        [re.escape(name) for name in ARM_KEYPOINT_LINK_NAMES], preserve_order=True
    )
    return robot.data.body_link_pose_w[:, body_ids, :3]


def keypoints_w_to_object(
    keypoints_w: torch.Tensor, object_pos_w: torch.Tensor, object_quat_w: torch.Tensor
) -> torch.Tensor:
    """把世界系关键点变换到物体系，输入 ``[N, K, 3]``，四元数采用 wxyz 顺序。

    ``object_pos_w`` 和 ``object_quat_w`` 分别是物体参考坐标系在世界系中的位置和
    朝向；返回张量形状与 ``keypoints_w`` 相同。
    """
    # 先平移到以物体原点为原点的向量，再用物体朝向的逆旋转转入物体系。
    relative = keypoints_w - object_pos_w[:, None, :]
    quaternions = object_quat_w[:, None, :].expand(-1, keypoints_w.shape[1], -1)
    return quat_apply_inverse(quaternions.reshape(-1, 4), relative.reshape(-1, 3)).reshape_as(relative)


def _object_top_pose_w(env: ManagerBasedRLEnv) -> tuple[torch.Tensor, torch.Tensor]:
    """取得原 Teacher 使用的 ``top`` 刚体世界位姿，返回位置和 wxyz 四元数。"""
    obj = env.scene["object"]
    # 获取 top 刚体索引；即使物体 articulation 包含多个刚体，也明确使用 top frame。
    body_ids, _ = obj.find_bodies(["top"], preserve_order=True)
    pose = obj.data.body_link_pose_w[:, body_ids[0], :]
    return pose[:, :3], pose[:, 3:7]


def hand_keypoints_o(env: ManagerBasedRLEnv) -> torch.Tensor:
    """返回物体 ``top`` 坐标系下的 17 个手部关键点 ``[N, 17, 3]``。"""
    # 取物体参考位姿，并将世界系手部关键点转换到物体系。
    object_pos_w, object_quat_w = _object_top_pose_w(env)
    return keypoints_w_to_object(hand_keypoints_w(env), object_pos_w, object_quat_w)


def arm_keypoints_o(env: ManagerBasedRLEnv) -> torch.Tensor:
    """返回物体 ``top`` 坐标系下的六个 UR5 关键点 ``[N, 6, 3]``。"""
    # 取物体参考位姿，并将世界系 UR5 关键点转换到物体系。
    object_pos_w, object_quat_w = _object_top_pose_w(env)
    return keypoints_w_to_object(arm_keypoints_w(env), object_pos_w, object_quat_w)


# 对外提供的世界系关键点、物体系关键点和坐标转换函数。
__all__ = [
    "hand_keypoints_w",
    "arm_keypoints_w",
    "keypoints_w_to_object",
    "hand_keypoints_o",
    "arm_keypoints_o",
]

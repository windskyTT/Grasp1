"""定义 UR5 + Allegro Teacher 环境使用的回合终止条件函数。

这些函数按环境逐项返回布尔值，供 Isaac Lab 的 termination manager 判断
哪些并行环境需要结束当前回合并重置。
"""

from __future__ import annotations

from collections.abc import Callable # 用于表示“可调用对象”的类型
from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _has_non_finite_rows(value: torch.Tensor) -> torch.Tensor:
    """检查张量中每个环境的数据行是否含 NaN 或无穷值。

    假定第 0 维是环境维；将其余维度展平后，只要某个环境的数据包含非有限值，
    对应位置就返回 ``True``。
    """
    return ~torch.isfinite(value).flatten(start_dim=1).all(dim=1)


def invalid_hand_height(
    env: ManagerBasedRLEnv,
    hand_keypoints_w_fn: Callable[[ManagerBasedRLEnv], torch.Tensor],
    support_height: float,
) -> torch.Tensor:
    """当任一手部关键点低于桌面高度时终止对应环境的回合。

    ``hand_keypoints_w_fn`` 从环境取得世界坐标系下的手部关键点；检查每个关键点
    的 z 坐标是否小于 ``support_height``，并为每个并行环境返回一个终止标志。
    """
    hand_keypoints_w = hand_keypoints_w_fn(env)
    return torch.any(hand_keypoints_w[:, :, 2] < support_height, dim=1)


def invalid_observation(env: ManagerBasedRLEnv, group_names: tuple[str, ...]) -> torch.Tensor:
    """当指定观测组包含 NaN 或无穷值时终止对应环境的回合。

    通过 observation manager 逐组计算观测，并合并各组逐环境的非有限值标志。
    ``group_names`` 通常包含策略和 critic 所使用的观测组。
    """
    invalid = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    for group_name in group_names:
        observation = env.observation_manager.compute_group(group_name)
        invalid |= _has_non_finite_rows(observation)
    return invalid


def invalid_asset_state(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """当场景资产的根状态或 articulation 状态包含非有限值时终止对应环境。

    所有资产都会检查世界坐标系根状态；若资产是 articulation，还会检查各刚体
    状态、关节位置和关节速度。``asset_cfg`` 指定场景资产，默认检查机器人。
    """
    asset = env.scene[asset_cfg.name]
    invalid = _has_non_finite_rows(asset.data.root_state_w)
    if isinstance(asset, Articulation):
        invalid |= _has_non_finite_rows(asset.data.body_state_w)
        invalid |= _has_non_finite_rows(asset.data.joint_pos)
        invalid |= _has_non_finite_rows(asset.data.joint_vel)
    return invalid


# 对外提供的手部高度、观测值和资产状态终止条件。
__all__ = ["invalid_hand_height", "invalid_observation", "invalid_asset_state"]

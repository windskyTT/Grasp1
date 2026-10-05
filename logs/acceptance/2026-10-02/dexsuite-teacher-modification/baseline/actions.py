"""定义 RobustDexGrasp Teacher 的 22 维残差关节位置动作及其配置类。

动作项在 UR5 与 Allegro 关节上应用缩放后的相对位置目标，并按配置复现旧 Teacher
在物理仿真子步中随机延迟一个子步更新 PD 目标的行为。
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.envs.mdp.actions.actions_cfg import JointActionCfg
from isaaclab.envs.mdp.actions.joint_actions import JointAction
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


class TeacherJointPositionAction(JointAction):
    """Teacher 的残差关节位置动作，并复现旧实现的一物理子步随机延迟。

    每个控制周期的目标为::

        q_target = q_current + action * scale

    ``scale`` 由配置按关节给出；目标随后裁剪到从 USD/URDF 载入的关节位置限位。

    Isaac Lab 每个环境步调用一次 :meth:`process_actions`，每个物理子步调用一次
    :meth:`apply_actions`。因此该类复现旧 Teacher 的以下行为：

    - 每个控制周期、每个环境独立抽样是否延迟；
    - 抽中延迟时，前 ``delay_sim_steps`` 个物理子步使用上一控制周期的位置目标；
    - 随后的物理子步使用当前控制周期的位置目标；
    - 当前控制周期最后一个物理子步结束后，将当前目标保存为下一周期的历史目标。
    """

    cfg: TeacherJointPositionActionCfg

    def __init__(
        self,
        cfg: TeacherJointPositionActionCfg,
        env: ManagerBasedEnv,
    ) -> None:
        """初始化关节动作缓存、PhysX 关节限位及延迟控制所需状态。

        ``cfg`` 指定关节、动作缩放和延迟参数；``env`` 提供并行环境、机器人
        articulation 和每个控制步对应的物理子步数。
        """
        super().__init__(cfg, env)
        # 每个控制步包含的物理仿真子步数，用来判断何时开始下一控制周期。
        self._decimation = int(env.cfg.decimation)

        # 读取 PhysX 实际使用的关节位置硬限位，形状为 [环境数, 动作维度, 2]；
        # 最后一维分别存储每个关节的下限和上限。
        self._joint_pos_limits = self._asset.data.joint_pos_limits[:, self._joint_ids, :].clone()

        current_target = self._asset.data.joint_pos[:, self._joint_ids].clone()

        # 上一个控制周期的 PD 位置目标，对应 RobustDexGrasp 的 pTarget_prev_r。
        self._previous_target = current_target.clone()

        # 首次策略动作到来前，将处理后动作初始化为当前关节位置。
        self._processed_actions.copy_(current_target)

        # 原 Teacher 始终使用零关节速度目标。
        self._zero_velocity_target = torch.zeros_like(current_target)

        # 为每个环境保存当前控制周期是否抽中延迟。
        self._delay_mask = torch.zeros(
            (self.num_envs, 1),
            dtype=torch.bool,
            device=self.device,
        )

        # 当前控制周期已经执行的物理子步数。并行环境同步推进，因此使用一个计数器。
        self._substep = 0

    def process_actions(self, actions: torch.Tensor) -> None:
        """在每个控制周期处理策略动作并生成新的关节位置目标。

        先由父类保存原始动作、应用关节级缩放和可选动作裁剪，再将结果作为相对
        当前测量关节位置的残差，最后裁剪到 PhysX 关节位置限位。
        """
        # 父类负责保存原始动作、应用每关节缩放和执行可选的动作范围裁剪。
        super().process_actions(actions)

        # 旧实现会在上一控制周期后用实测关节位置更新 actionMean_r_；因此当前残差
        # 以本控制周期开始时的实测关节位置为基准。
        target = (
            self._asset.data.joint_pos[:, self._joint_ids]
            + self._processed_actions
        )

        self._processed_actions = torch.clamp(
            target,
            min=self._joint_pos_limits[..., 0],
            max=self._joint_pos_limits[..., 1],
        )

        # 每个控制周期、每个环境独立抽样一次是否延迟新目标。
        self._delay_mask = (
            torch.rand((self.num_envs, 1), device=self.device)
            < self.cfg.delay_probability
        )

        self._substep = 0

    def apply_actions(self) -> None:
        """在当前物理子步向 articulation 写入位置目标和零速度目标。

        对抽中延迟的环境，在本控制周期的前 ``delay_sim_steps`` 个子步继续使用
        上一控制周期目标；之后所有环境都改用当前目标。
        """
        if self._substep < self.cfg.delay_sim_steps:
            target = torch.where(
                self._delay_mask,
                self._previous_target,
                self._processed_actions,
            )
        else:
            target = self._processed_actions

        # 向所选关节写入本子步生效的位置目标及零速度目标。
        self._asset.set_joint_position_target(
            target,
            joint_ids=self._joint_ids,
        )
        self._asset.set_joint_velocity_target(
            self._zero_velocity_target,
            joint_ids=self._joint_ids,
        )

        self._substep += 1

        # 到达本控制周期最后一个物理子步后，将当前目标保存为下一周期的历史目标。
        if self._substep >= self._decimation:
            self._previous_target.copy_(self._processed_actions)

    def reset(
        self,
        env_ids: Sequence[int] | torch.Tensor | None = None,
    ) -> None:
        """将指定环境的动作缓存和延迟状态同步到 reset 后的关节状态。

        ``env_ids`` 为 ``None`` 时重置所有环境；也可仅重置给定环境索引。
        """
        if env_ids is None:
            env_ids = slice(None)

        super().reset(env_ids)

        # 以 reset 后实测关节位置重新初始化当前目标和历史目标，避免沿用旧回合数据。
        current_target = self._asset.data.joint_pos[env_ids][:, self._joint_ids]

        self._processed_actions[env_ids] = current_target
        self._previous_target[env_ids] = current_target
        self._zero_velocity_target[env_ids] = 0.0
        self._delay_mask[env_ids] = False

        self._substep = 0


@configclass
class TeacherJointPositionActionCfg(JointActionCfg):
    """配置 Teacher 关节动作项；关节名、缩放和裁剪参数继承自 ``JointActionCfg``。"""

    # Isaac Lab action manager 根据此类型创建自定义动作项实例。
    class_type: type[TeacherJointPositionAction] = TeacherJointPositionAction

    delay_probability: float = 0.5
    """每个控制周期中，每个环境独立延迟新位置目标的概率。"""

    delay_sim_steps: int = 1
    """抽中延迟后继续使用上一控制周期目标的物理子步数。"""


# 对外公开的 Teacher 动作类和动作配置类。
__all__ = [
    "TeacherJointPositionAction",
    "TeacherJointPositionActionCfg",
]

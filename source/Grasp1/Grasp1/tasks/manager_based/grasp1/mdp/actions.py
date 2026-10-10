"""Teacher 相对位置动作：每个策略步生成一次限位目标，物理子步保持。"""

from isaaclab.envs.mdp.actions import RelativeJointPositionAction
import torch


class TeacherRelativeJointPositionAction(RelativeJointPositionAction):
    """按源Teacher控制周期缓存目标，供物理子步和真实误差观测共用。"""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._target = self._asset.data.joint_pos[:, self._joint_ids].clone()

    @property
    def target(self) -> torch.Tensor:
        """按22维策略顺序返回最近实际发送的关节目标。"""
        return self._target

    def reset(self, env_ids=None) -> None:
        super().reset(env_ids)
        if env_ids is None:
            env_ids = slice(None)
        self._processed_actions[env_ids] = 0.0
        self._target[env_ids] = self._asset.data.joint_pos[:, self._joint_ids][env_ids]

    def process_actions(self, actions: torch.Tensor) -> None:
        super().process_actions(actions)
        target = self._asset.data.joint_pos[:, self._joint_ids] + self.processed_actions
        limits = self._asset.data.joint_pos_limits[:, self._joint_ids]
        target.clamp_(min=limits[..., 0], max=limits[..., 1])
        self._target[:] = target

    def apply_actions(self) -> None:
        self._asset.set_joint_position_target(self._target, joint_ids=self._joint_ids)
        # 源vTarget始终为零；位置PD保持同一策略步目标，速度PD不引入运动目标。
        self._asset.set_joint_velocity_target(0.0, joint_ids=self._joint_ids)

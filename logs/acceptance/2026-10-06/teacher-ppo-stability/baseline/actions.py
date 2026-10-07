"""Teacher 相对位置动作：保留官方逐物理步语义并恢复源目标限位。"""

from isaaclab.envs.mdp.actions import RelativeJointPositionAction
import torch


class TeacherRelativeJointPositionAction(RelativeJointPositionAction):
    """逐物理步限制目标，并保存实际发送的目标供观测读取。"""

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

    def apply_actions(self) -> None:
        target = self._asset.data.joint_pos[:, self._joint_ids] + self.processed_actions
        limits = self._asset.data.joint_pos_limits[:, self._joint_ids]
        target.clamp_(min=limits[..., 0], max=limits[..., 1])
        self._target[:] = target
        self._asset.set_joint_position_target(self._target, joint_ids=self._joint_ids)

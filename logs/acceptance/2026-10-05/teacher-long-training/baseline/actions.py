"""Teacher 相对位置动作：保留官方逐物理步语义并恢复源目标限位。"""

from isaaclab.envs.mdp.actions import RelativeJointPositionAction


class TeacherRelativeJointPositionAction(RelativeJointPositionAction):
    """仅限制最终目标，不增加动作延迟或上一目标状态。"""

    def apply_actions(self) -> None:
        target = self._asset.data.joint_pos[:, self._joint_ids] + self.processed_actions
        limits = self._asset.data.joint_pos_limits[:, self._joint_ids]
        target.clamp_(min=limits[..., 0], max=limits[..., 1])
        self._asset.set_joint_position_target(target, joint_ids=self._joint_ids)

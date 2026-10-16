"""Teacher 22维相对关节目标：实测 q + 缩放动作，硬限位后在子步保持。"""

import torch
from isaaclab.envs.mdp.actions import RelativeJointPositionAction


class TeacherRelativeJointPositionAction(RelativeJointPositionAction):
    """复用官方动作处理；每个控制步只计算一次相对实测 q 的目标。"""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._target = self._asset.data.joint_pos[:, self._joint_ids].clone()

    @property
    def target(self):
        return self._target

    def reset(self, env_ids=None):
        super().reset(env_ids)
        ids = slice(None) if env_ids is None else env_ids
        self._processed_actions[ids] = 0.0
        self._target[ids] = self._asset.data.joint_pos[ids][:, self._joint_ids]

    def process_actions(self, actions):
        if actions.shape != self._target.shape or not torch.isfinite(actions).all():
            raise RuntimeError(f"Invalid Teacher policy actions: shape={tuple(actions.shape)}")
        super().process_actions(actions)
        limits = self._asset.data.joint_pos_limits[:, self._joint_ids]
        joint_pos = self._asset.data.joint_pos[:, self._joint_ids]
        self._target[:] = torch.clamp(joint_pos + self.processed_actions, limits[..., 0], limits[..., 1])

    def apply_actions(self):
        # 上一物理子步已完成并更新 scene；末子步由 termination 读取。
        self._env._teacher_grasp.sample(self._env._sim_step_counter - 1)
        self._asset.set_joint_position_target(self._target, joint_ids=self._joint_ids)
        self._asset.set_joint_velocity_target(0.0, joint_ids=self._joint_ids)

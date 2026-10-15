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


class FullGraspJointDeltaAction(TeacherRelativeJointPositionAction):
    """仅v2：策略增量累计到上一目标，零动作保持；22D无辅助控制。

    复用官方raw/scale处理及Teacher的reset/子步缓存；旧Teacher仍以当前q为基准。
    观测中的target-q使累计目标对actor可观测，目标只在控制步更新一次。
    """

    def process_actions(self, actions: torch.Tensor) -> None:
        RelativeJointPositionAction.process_actions(self, actions)
        target = self._target + self.processed_actions
        limits = self._asset.data.joint_pos_limits[:, self._joint_ids]
        target.clamp_(min=limits[..., 0], max=limits[..., 1])
        self._target[:] = target


class TableClearanceJointDeltaAction(FullGraspJointDeltaAction):
    """v3候选：以真实碰撞顶点和刚体Jacobian约束目标下降，保留22D策略控制。

    对下一关节目标的一阶预测按桌面间距统一缩放；不写实际q/qd，
    不删除碰撞、不控制物体、不生成抓取/抬升动作。真实物理仍独立验收。
    """

    def process_actions(self, actions):
        from .full_grasp_metrics import world_vertices
        from ..grasp1_env_cfg import SUPPORT_HEIGHT, TABLE_XY_BOUNDS
        super().process_actions(actions)
        q = self._asset.data.joint_pos[:, self._joint_ids]
        difference = self._target-q
        jacobians = self._asset.root_physx_view.get_jacobians()
        fraction = torch.ones(self._env.num_envs, device=self._env.device)
        for body, points in self._env._full_grasp_state.hand_geometry:
            pose = self._asset.data.body_link_pose_w[:, body]
            vertices = world_vertices(pose, points)
            # PhysX dense Jacobian的线速度参考点为COM，顶点杠杆臂必须也从COM起算。
            relative = vertices-self._asset.data.body_com_pos_w[:, body, None]
            jacobian = jacobians[:, body-1, :, self._joint_ids]
            motion = (jacobian @ difference[..., None]).squeeze(-1)
            dz = motion[:, None, 2]+motion[:, None, 3]*relative[..., 1]-motion[:, None, 4]*relative[..., 0]
            local = vertices-self._env.scene.env_origins[:, None]
            inside = ((local[..., 0] >= TABLE_XY_BOUNDS[0]) & (local[..., 0] <= TABLE_XY_BOUNDS[1])
                & (local[..., 1] >= TABLE_XY_BOUNDS[2]) & (local[..., 1] <= TABLE_XY_BOUNDS[3]))
            clearance = local[..., 2]-SUPPORT_HEIGHT
            descending = inside & (dz < 0)
            allowed = (clearance-self._env.cfg.action_table_clearance_m)/(-dz).clamp_min(1e-12)
            fraction = torch.minimum(fraction, torch.where(descending, allowed, torch.ones_like(allowed)).amin(1))
        fraction.clamp_(0, 1)
        projected = q+fraction[:, None]*difference
        limits = self._asset.data.joint_pos_limits[:, self._joint_ids]
        projected.clamp_(min=limits[..., 0], max=limits[..., 1])
        self._target[:] = torch.where(fraction[:, None] < 1, projected, self._target)
        self.projection_fraction = fraction

"""优化8独立诊断候选；通过物理和可学性Gate后才能用于正式训练。"""
from isaaclab.utils import configclass

from Grasp1.utils.paths import object_asset_dir
from ...mdp.rewards_grasp_v1 import apply_reward_profile
from ...grasp1_env_cfg import SOURCE_CONTROL_DT
from .teacher_env_cfg import UR5AllegroTeacherEnvCfg


@configclass
class UR5AllegroGraspTrainEnvCfg(UR5AllegroTeacherEnvCfg):
    """复用标准Manager和153D/22D语义，保留bottom/top Articulation。

    当前尚无Gate A通过的物理修复，保留旧物理参数作为独立候选。
    C1姿态范围/课程扩容在物理通过后实施，不能称当前初始化已可抓。
    """
    decimation = 4
    episode_length_s = 4.0
    reward_profile: str = 'source_physics_aligned'
    training_objects: tuple[str, ...] = ('wood_block_oriented',)
    curriculum_stage: str = 'C1_pending_physics'

    def __post_init__(self):
        super().__post_init__()
        self.object_names = self.training_objects
        self.set_num_envs(1)

    def synchronize_control_timing(self):
        """入口在Hydra/CLI覆盖后同步实际控制频率及Reward选择。"""
        super().synchronize_control_timing()
        names = self.actions.teacher.joint_names
        arm_scale = 0.005 * self.control_dt() / SOURCE_CONTROL_DT
        hand_scale = 0.015 * self.control_dt() / SOURCE_CONTROL_DT
        self.actions.teacher.scale = {name: arm_scale if i < 6 else hand_scale for i, name in enumerate(names)}
        apply_reward_profile(self, self.reward_profile)

    def set_num_envs(self, num_envs):
        """对象候选池循环分配；1对象可复制到16/128个环境。"""
        names = tuple(self.training_objects[i % len(self.training_objects)] for i in range(num_envs))
        self.scene.num_envs = num_envs
        self.object_names = names
        self.scene.object.spawn.usd_path = [str(object_asset_dir(self.object_dataset, name) / f'{name}.usd') for name in names]
        self.events.initialize_teacher_data.params['object_names'] = names
        self.events.reset_teacher.params['object_names'] = names

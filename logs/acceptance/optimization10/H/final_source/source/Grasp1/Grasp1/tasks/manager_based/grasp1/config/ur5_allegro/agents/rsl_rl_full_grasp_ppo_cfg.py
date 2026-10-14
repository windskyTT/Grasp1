"""完整自主抓取v2的独立标准RSL-RL配置；禁止跨旧MDP续训。"""
from isaaclab.utils import configclass
from .rsl_rl_grasp_train_ppo_cfg import UR5AllegroGraspTrainPPORunnerCfg


@configclass
class UR5AllegroFullGraspPPORunnerCfg(UR5AllegroGraspTrainPPORunnerCfg):
    experiment_name = 'grasp1_ur5_allegro_full_grasp_v2'
    save_interval = 50
    clip_actions = 1.0  # 将v2每秒目标增量限制在物理筛查的范围。

    def __post_init__(self):
        super().__post_init__()
        self.policy.actor_obs_normalization = True
        self.policy.critic_obs_normalization = True

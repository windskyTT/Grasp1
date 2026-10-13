"""优化8独立PPO候选，未通过物理Gate前不进行PPO性能比较。"""
from isaaclab.utils import configclass

from .rsl_rl_ppo_cfg import UR5AllegroTeacherPPORunnerCfg


@configclass
class UR5AllegroGraspTrainPPORunnerCfg(UR5AllegroTeacherPPORunnerCfg):
    experiment_name = 'grasp1_ur5_allegro_grasp_train_v1'
    save_interval = 100

    def __post_init__(self):
        super().__post_init__()
        self.algorithm.learning_rate = 3e-4
        self.synchronize_control_timing(4 / 120)

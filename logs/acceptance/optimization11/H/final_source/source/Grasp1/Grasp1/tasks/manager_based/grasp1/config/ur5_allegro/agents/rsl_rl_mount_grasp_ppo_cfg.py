"""v3独立标准PPO目录；每50个update保存，现有噪声和奖励保持不变。"""
from isaaclab.utils import configclass
from .rsl_rl_full_grasp_ppo_cfg import UR5AllegroFullGraspPPORunnerCfg


@configclass
class UR5AllegroMountGraspPPORunnerCfg(UR5AllegroFullGraspPPORunnerCfg):
    experiment_name = 'grasp1_ur5_allegro_mount_grasp_v3'

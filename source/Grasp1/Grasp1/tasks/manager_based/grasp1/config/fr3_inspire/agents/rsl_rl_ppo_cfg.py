"""沿用现有Teacher PPO及Gamma/Lambda时间换算，仅隔离实验名称。"""
from isaaclab.utils import configclass
from ...ur5_allegro.agents.rsl_rl_ppo_cfg import UR5AllegroTeacherPPORunnerCfg


@configclass
class FR3InspireTeacherPPORunnerCfg(UR5AllegroTeacherPPORunnerCfg):
    experiment_name = "grasp1_fr3_inspire_teacher"

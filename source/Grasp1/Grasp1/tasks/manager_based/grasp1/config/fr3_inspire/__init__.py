"""Register the FR3-Inspire Teacher task independently of UR5-Allegro."""
import gymnasium as gym
from . import agents

gym.register(
    id="Grasp1-FR3-Inspire-Teacher-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.teacher_env_cfg:FR3InspireTeacherEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:FR3InspireTeacherPPORunnerCfg",
    },
)

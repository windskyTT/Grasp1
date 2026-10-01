"""Register the UR5-Allegro Teacher task with Gymnasium."""

import gymnasium as gym

from . import agents


gym.register(
    id="Grasp1-UR5-Allegro-Teacher-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.teacher_env_cfg:UR5AllegroTeacherEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UR5AllegroTeacherPPORunnerCfg",
    },
    disable_env_checker=True,
)

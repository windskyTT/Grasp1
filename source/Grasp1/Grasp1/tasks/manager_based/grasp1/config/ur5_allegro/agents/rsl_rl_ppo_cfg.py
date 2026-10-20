"""Isaac Lab 2.3.2 / RSL-RL 3.1.2 的 Teacher PPO配置。"""
from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class UR5AllegroTeacherPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    class_name = "OnPolicyRunner"
    num_steps_per_env = 32
    max_iterations = 100
    save_interval = 10
    experiment_name = "grasp1_ur5_allegro_teacher"
    clip_actions = None
    obs_groups = {"policy": ["policy"], "critic": ["policy"]}
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.5, noise_std_type="log",
        actor_obs_normalization=True, critic_obs_normalization=True,
        actor_hidden_dims=[256, 256, 128], critic_hidden_dims=[256, 256, 128], activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0, use_clipped_value_loss=True, clip_param=0.2,
        entropy_coef=0.005, num_learning_epochs=5, num_mini_batches=4,
        learning_rate=3e-4, schedule="adaptive", gamma=0.99, lam=0.95,
        desired_kl=0.01, max_grad_norm=1.0,
    )

    def synchronize_control_timing(self, control_dt):
        # 以原始RobustDexGrasp的0.2s控制周期、gamma=.996/lambda=.95为参考。
        self.algorithm.gamma = 0.996 ** (control_dt/0.2)
        self.algorithm.lam = 0.95 ** (control_dt/0.2)

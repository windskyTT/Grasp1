"""配置 UR5 + Allegro Teacher 使用的 RSL-RL PPO 训练器、策略网络和算法参数。

这里将 RobustDexGrasp Teacher 的网络结构和 PPO 超参数映射到 Isaac Lab
提供的 RSL-RL 配置接口；具体训练循环由 RSL-RL 实现。
"""

from __future__ import annotations

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)
from ....grasp1_env_cfg import CONTROL_DT, SOURCE_CONTROL_DT


@configclass
class UR5AllegroTeacherPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """Grasp1 UR5 + Allegro Teacher 的 PPO 运行器配置。"""

    # 随机数种子，用于初始化训练中的随机过程。
    seed = 1

    # 每次 PPO 更新采集 32 个策略步；与 240 步的 episode 分开配置。
    num_steps_per_env = 32

    # Dexsuite v2.3.2 reference = 15000；Grasp1 selected = 20001。
    max_iterations = 20001

    # 检查点保存间隔，单位为训练迭代次数；原训练配置每 500 次评估/记录一次。
    save_interval = 500

    # 此运行配置在日志和输出目录中使用的实验名称。
    experiment_name = "grasp1_ur5_allegro_teacher"

    # 将环境观测组映射给策略网络：actor 和 critic 都读取 policy 观测。
    obs_groups = {
        "policy": ["policy"],
        "critic": ["policy"],
    }

    # Actor-Critic 策略网络结构与输入归一化配置。
    policy = RslRlPpoActorCriticCfg(
        # 高斯策略动作分布的初始标准差。
        # 原始实现更新后的最小标准差为 0.2；RSL-RL 配置接口没有对应字段。
        init_noise_std=1.0,

        # 分别控制 actor 和 critic 是否对输入观测做归一化；原实现未归一化观测。
        actor_obs_normalization=False,
        critic_obs_normalization=False,

        # Actor 策略网络和 critic 价值网络的隐藏层宽度，均为两层、每层 128 个单元。
        actor_hidden_dims=[128, 128],
        critic_hidden_dims=[128, 128],

        # 隐藏层使用 LeakyReLU 激活函数，对应原始网络实现。
        activation="lrelu",
    )

    # PPO 更新规则及优势估计、优化器等超参数。
    algorithm = RslRlPpoAlgorithmCfg(
        # 价值函数损失权重、价值裁剪开关、策略裁剪范围和熵正则权重。
        # RSL-RL 会打乱 mini-batch；原实现使用 shuffle_batch=False，配置接口无此选项。
        value_loss_coef=0.5,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.0,

        # 每次更新的学习轮数，以及每轮将 rollout 划分成的 mini-batch 数。
        num_learning_epochs=4,
        num_mini_batches=4,

        # 优化器学习率及自适应学习率调整策略。
        learning_rate=5.0e-4,
        schedule="adaptive",

        # 5 Hz → 60 Hz 保持源gamma的物理时间折扣；不硬编码近似值。
        # lambda的时间等效值只作为候选，本轮隔离gamma，仍保留0.95。
        gamma=0.996 ** (CONTROL_DT / SOURCE_CONTROL_DT),
        lam=0.95,

        # 自适应学习率使用的目标 KL 散度，以及梯度范数裁剪上限。
        desired_kl=0.01,
        max_grad_norm=0.5,
    )


# 本模块公开的 UR5 + Allegro Teacher PPO 运行器配置类。
__all__ = [
    "UR5AllegroTeacherPPORunnerCfg",
]

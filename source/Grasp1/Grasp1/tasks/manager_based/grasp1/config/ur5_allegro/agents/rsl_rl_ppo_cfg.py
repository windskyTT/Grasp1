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


@configclass
class UR5AllegroTeacherPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """Grasp1 UR5 + Allegro Teacher 的 PPO 运行器配置。"""

    # 随机数种子，用于初始化训练中的随机过程。
    seed = 1

    # 每个环境在一次 PPO 更新前采集的步数；对应原 Teacher 的 grasp_steps=70，
    # 表示 rollout 长度，不等同于环境 episode 长度。
    num_steps_per_env = 70

    # 最大训练迭代次数，对应原 train.py 的默认 --num_iterations。
    max_iterations = 50001

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

        # 折扣因子 gamma 和 GAE 优势估计参数 lambda。
        gamma=0.996,
        lam=0.95,

        # 自适应学习率使用的目标 KL 散度，以及梯度范数裁剪上限。
        desired_kl=0.01,
        max_grad_norm=0.5,
    )


# 本模块公开的 UR5 + Allegro Teacher PPO 运行器配置类。
__all__ = [
    "UR5AllegroTeacherPPORunnerCfg",
]

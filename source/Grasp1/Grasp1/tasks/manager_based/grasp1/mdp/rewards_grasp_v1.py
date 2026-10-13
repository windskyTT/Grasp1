"""优化8源方向对齐Reward；旧Teacher继续使用rewards.py。

ContactSensor已按刚体聚合，无法精确恢复源逐接触<0.001 N·s过滤。
这里只对齐世界XY及有符号Z定义，冲量只乘一个真实PhysX积分步。
"""
import torch

from . import rewards


def world_contact_impulses(env, prefix):
    sensors = env._teacher_runtime_features.contact_sensors[prefix]
    normal = torch.stack([s.data.force_matrix_w[:, 0].sum(1) for s in sensors], 1)
    friction = torch.stack([s.data.friction_forces_w[:, 0].sum(1) for s in sensors], 1)
    return (normal + friction) * env.physics_dt


def affordance_impulse_reward(env, contact_weights):
    """对应源世界XY总冲量模，保留原clip、权重及采样时刻。"""
    impulse = world_contact_impulses(env, rewards.AFFORDANCE_CONTACT_SENSOR_PREFIX)
    return rewards._weighted(rewards._clip_hand_impulses(impulse[..., :2].norm(dim=-1)), contact_weights)


def push_reward(env):
    """对应源有符号世界Z冲量；源系数仍为0，不加入训练附加奖励。"""
    z = world_contact_impulses(env, rewards.AFFORDANCE_CONTACT_SENSOR_PREFIX)[..., 2]
    return ((z[:, 0] - 1).clamp_min(0) + (z[:, 1:] - 2).clamp_min(0).sum(-1)).clamp_max(10)


def apply_reward_profile(cfg, profile):
    """由独立GraspTrain配置消费；旧Teacher默认值不受影响。"""
    if profile == 'source_legacy':
        cfg.rewards.affordance_impulse_reward.func = rewards.affordance_impulse_reward
        cfg.rewards.push_reward.func = rewards.push_reward
    elif profile == 'source_physics_aligned':
        cfg.rewards.affordance_impulse_reward.func = affordance_impulse_reward
        cfg.rewards.push_reward.func = push_reward
    else:
        raise ValueError(f'Unvalidated reward profile: {profile}')

"""原始Teacher桌面高度/NaN终止及Isaac Lab非有限数值适配。"""

import torch
from .keypoints import hand_keypoints_w


def invalid_state(env):
    s = env._teacher_grasp
    s.sample(env._sim_step_counter)
    # 源isTerminalState检查当步obDouble/global_state；物理状态检查覆盖
    # global_state的位姿来源，另检查实际153维观测，不能读取上一控制步obs_buf。
    observation = env.observation_manager.compute_group("policy")
    s.invalid |= ~torch.isfinite(observation).all(1)
    # Reset前快照，独立评估可据此排除中途重置，训练不统计抬升成功。
    env.extras["teacher_episode"] = dict(
        lift_height=s.lift.clone(), invalid=s.invalid.clone(),
        hand_height=env.termination_manager.get_term("hand_height").clone(),
        timeout=env.episode_length_buf >= env.max_episode_length,
    )
    env.extras["teacher_grip_rate"] = s.grip.float().mean()
    env.extras["teacher_good_contact_rate"] = s.good_contact.float().mean()
    env.extras["log"] = dict(
        good_contact_rate=s.good_contact.float().mean(),
        opposing_grip_rate=s.grip.float().mean(), lift_height=s.lift.mean(),
        invalid_rate=s.invalid.float().mean(),
    )
    return s.invalid


def hand_height(env):
    """源Teacher：17个手部关键点中任一点相对桌面高度严格小于0。"""
    heights = hand_keypoints_w(env)[..., 2] - env.scene.env_origins[:, None, 2] - env.cfg.support_height
    return (heights < 0.0).any(1)

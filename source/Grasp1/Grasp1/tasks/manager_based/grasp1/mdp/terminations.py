"""Teacher 终止项：先采样末物理子步，再读取同一状态。"""

from .keypoints import hand_keypoints_w


def unsafe(env):
    s = env._teacher_grasp
    s.sample(env._sim_step_counter)
    # 所有PhysX子步结束后仅更新一次成功，原始判据没有保持时间或接触门控。
    s.update_success()
    # Reset前发布当前高度与同一成功状态，play和train不读取已重置的数据。
    env.extras["teacher_episode"] = dict(
        success=s.success.clone(), lift_height=s.lift.clone(), unsafe=s.unsafe.clone(),
        timeout=env.episode_length_buf >= env.max_episode_length,
    )
    env.extras["teacher_grip_rate"] = s.grip.float().mean()
    env.extras["teacher_good_contact_rate"] = s.good_contact.float().mean()
    env.extras["log"] = dict(
        good_contact_rate=s.good_contact.float().mean(),
        opposing_grip_rate=s.grip.float().mean(), lift_height=s.lift.mean(),
        success_rate=s.success.float().mean(), unsafe_rate=s.unsafe.float().mean(),
    )
    return s.unsafe


def invalid_state(env):
    return env._teacher_grasp.invalid


def hand_height(env):
    """源Teacher：17个手部关键点中任一点相对桌面高度严格小于0。"""
    heights = hand_keypoints_w(env)[..., 2] - env.scene.env_origins[:, None, 2] - env.cfg.support_height
    return (heights < 0.0).any(1)


def success(env):
    return env._teacher_grasp.success

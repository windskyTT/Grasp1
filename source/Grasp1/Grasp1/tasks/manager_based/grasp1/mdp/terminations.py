"""Teacher 终止项：先采样末物理子步，再读取同一状态。"""


def unsafe(env):
    s = env._teacher_grasp
    s.sample(env._sim_step_counter)
    # Reset 前发布回合末状态，play和train不读取已重置的lift/hold。
    env.extras["teacher_episode"] = dict(
        success=s.success.clone(), lift_height=s.max_lift.clone(),
        hold_duration=s.max_hold.clone(), unsafe=s.unsafe.clone(),
        timeout=env.episode_length_buf >= env.max_episode_length,
    )
    env.extras["teacher_grip_rate"] = s.grip.float().mean()
    env.extras["teacher_good_contact_rate"] = s.good_contact.float().mean()
    env.extras["log"] = dict(
        good_contact_rate=s.good_contact.float().mean(),
        opposing_grip_rate=s.grip.float().mean(), lift_height=s.lift.mean(),
        hold_duration=s.hold.mean(), success_rate=s.success.float().mean(), unsafe_rate=s.unsafe.float().mean(),
    )
    return s.unsafe


def invalid_state(env):
    return env._teacher_grasp.invalid


def dropped(env):
    return env._teacher_grasp.dropped


def success(env):
    s = env._teacher_grasp
    return s.success & ~s.unsafe & ~s.invalid

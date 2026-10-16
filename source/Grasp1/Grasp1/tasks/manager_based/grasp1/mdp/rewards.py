"""Teacher 完整抓取奖励；观测、奖励和终止共用真实物理步状态。"""
import torch
from isaaclab.utils.math import quat_apply
from .observations import nearest_affordance_vectors_w


def approach(env):
    distances = nearest_affordance_vectors_w(env)[:, (4, 8, 12, 16)].norm(dim=-1)
    return torch.exp(-distances.mean(1)/0.08)


def direction(env):
    s = env._teacher_grasp
    palm = s.robot.data.body_link_quat_w[:, s.palm]
    axis = quat_apply(palm, palm.new_tensor((1., 0., 0.)).expand(env.num_envs, -1))
    offset = s.obj.data.body_link_pos_w[:, s.top]-s.robot.data.body_link_pos_w[:, s.palm]
    # Allegro 掌的+X法向朝向物体，接近时才贡献方向奖励。
    align = (axis*torch.nn.functional.normalize(offset, dim=-1)).sum(-1).clamp_min(0)
    return align*torch.exp(-offset.norm(dim=-1)/0.20)


def contact(env):
    s = env._teacher_grasp
    return (s.tip_forces.norm(dim=-1) > env.cfg.tip_contact_threshold_N).float().mean(1)


def opposing(env):
    return env._teacher_grasp.grip.float()


def stable_grasp(env):
    return env._teacher_grasp.stable.float()


def lift(env):
    s = env._teacher_grasp
    return s.grip.float()*(s.lift/env.cfg.success_lift_m).clamp(0, 1)


def hold(env):
    return (env._teacher_grasp.hold/env.cfg.success_hold_s).clamp(0, 1)


def success_bonus(env):
    # success 当步即终止；除以step_dt抵消RewardManager缩放，只发一次。
    return env._teacher_grasp.success.float()/env.step_dt


def unsafe_penalty(env):
    return env._teacher_grasp.unsafe.float()/env.step_dt


def dropped_penalty(env):
    return env._teacher_grasp.dropped.float()/env.step_dt


def joint_speed(env):
    s = env._teacher_grasp
    return s.robot.data.joint_vel[:, s.joints].square().mean(1)


def target_error(env):
    s = env._teacher_grasp
    return (env.action_manager.get_term("teacher").target-s.robot.data.joint_pos[:, s.joints]).square().mean(1)

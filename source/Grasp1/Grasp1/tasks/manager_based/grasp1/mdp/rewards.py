"""Teacher的17项抓取奖励及终止奖励、DexSuite Lift两项扩展与Grasp1成功奖金。

公式：Environment.hpp:642–734，train.py:216–225/609–625。
cfg_reg.yaml原系数由TeacherRewardsCfg配置；这里返回未加权项。
Lift扩展来自Isaac Lab 2.3.2 DexSuite，不属于原始Teacher奖励。
"""
import torch
from isaaclab.utils.math import quat_apply
from .keypoints import arm_keypoints_w, hand_keypoints_w
from .observations import dexsuite_good_finger_contact, nearest_affordance_vectors_w


# 源train.py先用包含掌点的33归一化，再将掌点置零，最后乘16。
_POINT_WEIGHTS = (0., 1., 1., 1., 4., 1., 1., 1., 4., 1., 1., 1., 4., 1., 1., 1., 8.)
# 源Environment.hpp:196–205，拇指末槽为12；归一化后乘13。
_CONTACT_WEIGHTS = (0., 1., 1., 3., 1., 1., 3., 1., 1., 3., 2., 2., 12.)


def _contact_state(env):
    state = env._teacher_reward_state
    state.update()
    return state


def _weighted_impulse(impulses):
    high = impulses.new_tensor((0.1,)*10 + (0.2,)*3)
    clipped = torch.minimum(impulses.clamp_min(0), high)
    return (clipped * impulses.new_tensor(_CONTACT_WEIGHTS) * (13./31.)).sum(1)


def affordance_reward(env):
    distances = nearest_affordance_vectors_w(env).norm(dim=-1)
    return -(distances[:, 1:17] * distances.new_tensor(_POINT_WEIGHTS[1:]) * (16./33.)).sum(1)


def affordance_contact_reward(env):
    impulse = _contact_state(env).affordance_impulses.norm(dim=-1)
    return ((impulse > 0.01).to(impulse.dtype) * impulse.new_tensor(_CONTACT_WEIGHTS)).sum(1)/31.


def affordance_impulse_reward(env):
    # 世界XY的总冲量，不能改成摩擦冲量或三维模长。
    return _weighted_impulse(_contact_state(env).affordance_impulses[..., :2].norm(dim=-1))


def table_reward(env):
    heights = hand_keypoints_w(env)[..., 2] - env.scene.env_origins[:, None, 2] - env.cfg.support_height
    return -(torch.log(50*heights.clamp(0.002, 0.02)) * heights.new_tensor(_POINT_WEIGHTS) * (16./33.)).sum(1)


def table_contact_reward(env):
    impulse = _contact_state(env).table_impulses.norm(dim=-1)
    return ((impulse > 0.01).to(impulse.dtype) * impulse.new_tensor(_CONTACT_WEIGHTS)).sum(1)/31.


def table_impulse_reward(env):
    return _weighted_impulse(_contact_state(env).table_impulses.norm(dim=-1))


def arm_height_reward(env):
    # 源obs[:,89:93]对应最后四个机械臂关键点。
    heights = arm_keypoints_w(env)[:, 2:6, 2] - env.scene.env_origins[:, None, 2] - env.cfg.support_height
    return -torch.log(50*heights.clamp(0.002, 0.02)).sum(1)


def arm_contact_reward(env):
    impulse = _contact_state(env).arm_impulses.norm(dim=-1)
    return (impulse > 0.01).to(impulse.dtype).norm(dim=1)


def arm_impulse_reward(env):
    return _contact_state(env).arm_impulses.flatten(1).norm(dim=1)


def arm_collision_reward(env):
    return _contact_state(env).arm_contacts_all.sum(1).float()


def push_reward(env):
    impulse_z = _contact_state(env).affordance_impulses[..., 2]
    thresholds = impulse_z.new_tensor((1.,) + (2.,)*12)
    return (impulse_z-thresholds).clamp_min(0).sum(1).clamp_max(10.)


def wrist_vel_reward_(env):
    robot = env.scene["robot"]
    wrist = env._teacher_body_indices.wrist
    offset = quat_apply(robot.data.body_link_quat_w[:, wrist],
                        robot.data.joint_pos.new_tensor((0., 0.0823, 0.)).expand(env.num_envs, -1))
    velocity = robot.data.body_link_lin_vel_w[:, wrist] + torch.cross(
        robot.data.body_link_ang_vel_w[:, wrist], offset, dim=-1)
    # 世界系与腕系范数相同；固定关节frame速度需包含omega×offset。
    squared = velocity.square().sum(1)
    return torch.where(velocity.norm(dim=1) > 0.25, squared*10., squared)


def wrist_qvel_reward_(env):
    return env.scene["robot"].data.body_link_ang_vel_w[:, env._teacher_body_indices.wrist].square().sum(1)


def obj_vel_reward_(env):
    return env.scene["object"].data.body_link_lin_vel_w[:, env._teacher_body_indices.object_top].square().sum(1)


def obj_qvel_reward_(env):
    return env.scene["object"].data.body_link_ang_vel_w[:, env._teacher_body_indices.object_top].square().sum(1)


def obj_displacement_reward(env):
    position = env.scene["object"].data.body_link_pos_w[:, env._teacher_body_indices.object_top]
    # 源obj_pos_init_.head(3)来自Reset时的物体root，不是上一帧位置。
    return (position-env._teacher_reward_state.initial_object_pos).norm(dim=1)


def arm_joint_vel_reward_(env):
    velocity = env.scene["robot"].data.joint_vel[:, env._teacher_body_indices.arm_joints]
    return torch.where(velocity.abs() > 0.5, velocity*4., velocity).square().sum(1)


def good_finger_contact(env, threshold):
    """DexSuite contacts()：拇指与至少另一指的法向接触力均严格大于threshold。"""
    return dexsuite_good_finger_contact(env, threshold)


def position_tracking(env, std):
    """DexSuite原tanh及接触门控；命令位置适配为Reset位置正上方0.10m。"""
    initial = env._teacher_reward_state.initial_object_pos
    target = initial + initial.new_tensor((0., 0., env.cfg.success_lift_m))
    distance = (env.scene["object"].data.root_pos_w-target).norm(dim=1)
    return (1-torch.tanh(distance/std))*good_finger_contact(env, 1.0).float()


def success_bonus(env):
    """复用源高度>0.10m成功状态；成功终止为训练适配，抵消step_dt后发10分。"""
    return env._teacher_grasp.success.to(torch.float64)/env.step_dt


def terminal_reward(env):
    """源高度异常terminalReward=-10；由配置加权并抵消RewardManager的step_dt。"""
    return env.termination_manager.get_term("hand_height").to(torch.float64)/env.step_dt

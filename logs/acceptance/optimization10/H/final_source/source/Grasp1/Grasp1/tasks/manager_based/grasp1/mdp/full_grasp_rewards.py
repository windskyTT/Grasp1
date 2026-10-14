"""自主完整抓取奖励，连续项按step_dt积分，终止成功只奖励一次。"""
import torch
from isaaclab.utils.math import quat_apply
from .full_grasp_metrics import state
from .observations import nearest_affordance_vectors_w


def approach(env):
    distance = nearest_affordance_vectors_w(env)[:, (4, 8, 12, 16)].norm(dim=-1).mean(1)
    return torch.exp(-distance/0.08)


def alignment(env):
    s = state(env)
    # C1为俯视抓取：手掌+x朝下，同时减小物体相对手的横向偏差。
    return torch.exp(-s['relpos'][:, 1:].norm(dim=-1)/0.06)*top_down_alignment(s['palm'][:,3:7])


def top_down_alignment(quaternion):
    axis = quaternion.new_tensor((1.,0.,0.)).expand(len(quaternion),-1)
    return (-quat_apply(quaternion,axis)[:,2]).clamp(0,1)


def stable_grasp(env):
    s = state(env)
    return s['grasp'].float()*(0.5+0.5*s['opposing'].float())*torch.exp(-s['relvel'][:, :3].norm(dim=-1)/0.2)


def lift_progress(lift, grasp, safe, target):
    return (lift/target).clamp(0, 1)*grasp.float()*safe.float()


def lift(env):
    s = state(env)
    return lift_progress(s['lift'], s['grasp'] & s['opposing'], ~s['unsafe'], env.cfg.lift_height)


def hold(env):
    return state(env)['holding'].float()


def success_event(env):
    return env.termination_manager.get_term('success').float()/env.step_dt


def safety_event(env):
    manager = env.termination_manager
    return (manager.get_term('unsafe') | manager.get_term('reset_failed') | manager.get_term('dropped')).float()/env.step_dt


def table_contact(env):
    s = state(env)
    return (s['table']*env.physics_dt).sum(1) + (s['arm_force']*env.physics_dt) + (-s['clearance']).clamp_min(0)*100


def speed(env):
    s = state(env)
    # 允许慢速真实抬升，不沿用Teacher的物体Z位移/全速度负项。
    return (s['qd'][:, :6].square().mean(1) + s['qd'][:, 6:].square().mean(1)*0.05
            + (s['relvel'][:, :3].norm(dim=-1)-0.35).clamp_min(0).square())


def action_rate(env):
    return (env.action_manager.action-env.action_manager.prev_action).square().mean(1)


def limit(env):
    return state(env)['violation'].sum(1)

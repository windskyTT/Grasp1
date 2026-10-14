"""成功必须连续保持；失败原因独立，不再以任一关键点擦桌终止。"""
from .full_grasp_metrics import state


def unsafe(env):
    return state(env)['unsafe']


def reset_failed(env):
    return ~env._full_grasp_state.valid_reset


def dropped(env):
    s = state(env)
    return (s['lift'] < -0.05) | (s['object_clearance'] < -0.04)


def success(env):
    return state(env)['success']

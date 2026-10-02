"""同一 PhysX 状态上比较优化前快照与当前 MDP；不改变任务配置。"""
import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument('--steps', type=int, default=5)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import copy
import importlib
import importlib.util
import json
from pathlib import Path
import sys
import types
import torch
import gymnasium as gym
import Grasp1.tasks
from isaaclab_tasks.utils import parse_env_cfg

ROOT = Path(__file__).parent
PACKAGE = 'Grasp1.tasks.manager_based.grasp1.mdp'
legacy_name = PACKAGE + '._perf_before'
legacy = types.ModuleType(legacy_name)
legacy.__path__ = [str(ROOT / 'before/mdp')]
sys.modules[legacy_name] = legacy
old_obs = importlib.import_module(legacy_name + '.observations')
old_rw = importlib.import_module(legacy_name + '.rewards')
old_kp = importlib.import_module(legacy_name + '.keypoints')
old_events = importlib.import_module(legacy_name + '.events')
old_done = importlib.import_module(legacy_name + '.terminations')
new_rw = importlib.import_module(PACKAGE + '.rewards')
new_kp = importlib.import_module(PACKAGE + '.keypoints')
new_events = importlib.import_module(PACKAGE + '.events')
new_done = importlib.import_module(PACKAGE + '.terminations')

errors = {}
def compare(label, a, b, finite=True):
    assert a.shape == b.shape and a.dtype == b.dtype and a.device == b.device, label
    if a.dtype == torch.bool:
        assert torch.equal(a, b), label
        errors[label] = 0.0
        return
    if finite:
        assert torch.isfinite(a).all() and torch.isfinite(b).all(), label
    assert torch.allclose(a, b, atol=1e-6, rtol=1e-5, equal_nan=not finite), (label, (a-b).abs().max())
    error = float((a-b).abs().max()) if a.numel() else 0.0
    errors[label] = max(errors.get(label, 0.0), error)

def clone_data(data):
    return {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in data.items()}

try:
    cfg = parse_env_cfg('Grasp1-UR5-Allegro-Teacher-v0', device='cuda:0', num_envs=16)
    cfg.set_num_envs(16)
    cfg.seed = 42
    env = gym.make('Grasp1-UR5-Allegro-Teacher-v0', cfg=cfg)
    e = env.unwrapped
    obs_cfg = e.observation_manager._group_obs_term_cfgs['policy'][0]
    current = obs_cfg.func
    previous = old_obs.TeacherObservation(obs_cfg, e)
    class Proxy:
        def __getattr__(self, name):
            return getattr(e, name)
    proxy = Proxy()
    proxy.observation_manager = types.SimpleNamespace(compute_group=lambda group: previous(e, **obs_cfg.params))
    old_displacement = old_rw.ObjectDisplacementReward(e.reward_manager.get_term_cfg('obj_displacement_reward'), e)
    wrist_fields = ('_initial_wrist_quat_w', '_previous_wrist_euler', '_needs_wrist_init')

    def check_outputs():
        for name in wrist_fields:
            getattr(previous, name).copy_(getattr(current, name))
        a = previous(e, **obs_cfg.params)
        b = current(e, **obs_cfg.params)
        assert b.shape == (16, 153)
        compare('observation', a, b)
        for name in wrist_fields:
            compare('wrist/' + name, getattr(previous, name), getattr(current, name))
        for name in ('hand_keypoints_w', 'arm_keypoints_w', 'hand_keypoints_o', 'arm_keypoints_o'):
            compare(name, getattr(old_kp, name)(e), getattr(new_kp, name)(e))
        old_displacement._initial_base_position_w.copy_(e.reward_manager.get_term_cfg('obj_displacement_reward').func._initial_base_position_w)
        total_a = torch.zeros(16, device=e.device)
        total_b = torch.zeros_like(total_a)
        for name in e.reward_manager.active_terms:
            term = e.reward_manager.get_term_cfg(name)
            previous_func = old_displacement if name == 'obj_displacement_reward' else getattr(old_rw, name)
            a = previous_func(proxy, **term.params)
            b = term.func(e, **term.params)
            compare('reward/' + name, a, b)
            if term.weight != 0:
                total_a += a * term.weight * e.step_dt
                total_b += b * term.weight * e.step_dt
        compare('total_reward', total_a, total_b)
        for name in e.termination_manager.active_terms:
            term = e.termination_manager.get_term_cfg(name)
            if name == 'invalid_observation':
                a = old_done.invalid_observation(proxy, **term.params)
            elif name == 'invalid_hand_height':
                params = dict(term.params, hand_keypoints_w_fn=old_kp.hand_keypoints_w)
                a = old_done.invalid_hand_height(proxy, **params)
            elif term.func.__module__.endswith('terminations') and hasattr(old_done, term.func.__name__):
                a = getattr(old_done, term.func.__name__)(proxy, **term.params)
            else:
                a = term.func(e, **term.params)
            b = term.func(e, **term.params)
            assert torch.equal(a, b), name

    with torch.inference_mode():
        env.reset(seed=42)
        check_outputs()
        for step in range(args.steps):
            actions = torch.empty((16,22), device=e.device).uniform_(-1, 1)
            before = e.scene['robot'].data.joint_pos[:, e.action_manager.get_term('teacher')._joint_ids].clone()
            observation, reward, terminated, truncated, _ = env.step(actions)
            assert torch.isfinite(observation['policy']).all() and torch.isfinite(reward).all()
            term = e.action_manager.get_term('teacher')
            # 非 reset 行上的实际处理结果；reset 行已由动作 manager 重新初始化。
            keep = ~(terminated | truncated)
            expected = (before + actions * term._scale).clamp(term._joint_pos_limits[...,0], term._joint_pos_limits[...,1])
            compare('processed_actions', expected[keep], term.processed_actions[keep])
            check_outputs()

        ids = torch.tensor([3,7,11], device=e.device)
        keep = torch.ones(16, dtype=torch.bool, device=e.device)
        keep[ids] = False
        reset_data = clone_data(e._teacher_reset_data)
        rng_cpu, rng_cuda = torch.get_rng_state(), torch.cuda.get_rng_state()
        params = e.cfg.events.reset_teacher.params
        if "triangles" in reset_data:
            e._teacher_reset_data["triangles_cpu"] = {name: value.cpu() for name, value in reset_data["triangles"].items()}
            e._teacher_reset_data["stable_states_cpu"] = {
                name: reset_data["stable_states"][index].cpu()
                for index, name in enumerate(reset_data["object_names"])
                if bool(reset_data["stable_state_available"][index])
            }
        old_events.reset_teacher(e, ids, **params)
        old_reset = clone_data(e._teacher_reset_data)
        old_ik = e._teacher_last_ik_feasible.clone()
        old_states = {(asset, field): getattr(e.scene[asset].data, field).clone()
                      for asset in ('robot','object') for field in ('root_state_w','joint_pos','joint_vel')}
        e._teacher_reset_data = clone_data(reset_data)
        torch.set_rng_state(rng_cpu)
        torch.cuda.set_rng_state(rng_cuda)
        new_events.reset_teacher(e, ids, **params)
        for key, value in old_reset.items():
            if isinstance(value, torch.Tensor):
                compare('reset/' + key, value, e._teacher_reset_data[key])
        assert torch.equal(old_ik, e._teacher_last_ik_feasible)
        for (asset, field), value in old_states.items():
            compare('reset/' + asset + '/' + field, value, getattr(e.scene[asset].data, field))
        # 正式 partial reset 同时验证 manager 状态和缓存，不只验证 reset 函数。
        states = {(asset, field): getattr(e.scene[asset].data, field).clone()
                  for asset in ('robot','object') for field in ('root_state_w','joint_pos','joint_vel')}
        buffers = clone_data(e._teacher_reset_data)
        wrist_before = {name: getattr(current, name).clone() for name in wrist_fields}
        cache = e._teacher_runtime_features
        cache_before = {name: tuple(x.clone() for x in value) if isinstance(value, tuple) else value.clone()
                        for name, value in cache._values.items()}
        e._reset_idx(ids)
        e.scene.write_data_to_sim()
        e.sim.forward()
        for (asset, field), value in states.items():
            assert torch.equal(value[keep], getattr(e.scene[asset].data, field)[keep]), (asset,field)
        for key, value in buffers.items():
            if isinstance(value, torch.Tensor) and value.ndim and value.shape[0] == 16:
                assert torch.equal(value[keep], e._teacher_reset_data[key][keep]), key
        for name, value in wrist_before.items():
            assert torch.equal(value[keep], getattr(current, name)[keep]), name
        assert current._needs_wrist_init[ids].all()
        assert (current._previous_wrist_euler[ids] == 0).all()
        for asset in ('robot','object'):
            assert (e.scene[asset].data.joint_vel[ids] == 0).all()
        assert (e.scene['object'].data.root_vel_w[ids] == 0).all()
        partial_computes = []
        original_get = cache.get
        def counted_get(name, compute):
            def counted(rows):
                partial_computes.append((name, 16 if isinstance(rows, slice) else rows.numel()))
                return compute(rows)
            return original_get(name, counted)
        cache.get = counted_get
        check_outputs()
        cache.get = original_get
        assert partial_computes and all(count == 3 for _, count in partial_computes), partial_computes
        for name, value in cache_before.items():
            after = cache._values[name]
            if isinstance(value, tuple):
                assert all(torch.equal(a[keep], b[keep]) for a,b in zip(value,after)), name
            else:
                assert torch.equal(value[keep], after[keep]), name

        # 同一步 reuse、下一个真实控制步失效；运行路径不应再解析 body/joint 名称。
        cached_hand = new_kp.hand_keypoints_w(e)
        assert cached_hand is new_kp.hand_keypoints_w(e)
        saved_lookups = []
        def no_lookup(*a, **kw):
            raise AssertionError('runtime body/joint name lookup')
        for asset in ('robot','object'):
            for method in ('find_bodies','find_joints'):
                saved_lookups.append((e.scene[asset], method, getattr(e.scene[asset], method)))
                setattr(e.scene[asset], method, no_lookup)
        env.step(torch.zeros((16,22), device=e.device))
        for asset, method, saved in saved_lookups:
            setattr(asset, method, saved)
        assert new_kp.hand_keypoints_w(e) is not cached_hand
        check_outputs()

        # 非有限力：原奖励保留 NaN/Inf，原 observation 分量清理；两条语义分别比对。
        sensor = e.scene['teacher_af_contact_0']
        normal, friction = sensor.data.force_matrix_w, sensor.data.friction_forces_w
        normal_before, friction_before = normal.clone(), friction.clone()
        normal[0,0,0,0] = float('nan')
        friction[1,0,0,1] = float('inf')
        cache.invalidate(torch.tensor([0,1], device=e.device))
        legacy_impulses = old_rw._filtered_impulses(e, 'teacher_af_contact_', 13)
        current_impulses = new_rw._filtered_impulses(e, 'teacher_af_contact_')
        for a,b in zip(legacy_impulses, current_impulses):
            compare('nonfinite_contact_reward', a, b, finite=False)
        for a,b in zip(previous._contact_observation(e), current._contact_observation(e)):
            compare('nonfinite_contact_observation', a, b)
        normal.copy_(normal_before)
        friction.copy_(friction_before)
        cache.invalidate(torch.tensor([0,1], device=e.device))
        # 当前 RewardManager 必须跳过零权重项，同时保留 17 项和其日志。
        push = e.reward_manager.get_term_cfg('push_reward')
        saved = push.func
        def must_skip(*a, **kw):
            raise AssertionError('zero-weight push_reward executed')
        push.func = must_skip
        e.reward_manager.compute(e.step_dt)
        push.func = saved
        assert len(e.reward_manager.active_terms) == 17
        print('VALIDATION_RESULT', json.dumps(dict(max_abs_errors=errors, steps=args.steps,
              partial_reset=[3,7,11], partial_feature_recomputes=partial_computes,
              done_masks_equal=True, zero_weight_skipped=True, runtime_name_lookups=0,
              cache_shapes={name: [list(x.shape) for x in value] if isinstance(value,tuple) else list(value.shape)
                            for name,value in cache._values.items()})), flush=True)
    env.close()
finally:
    app.close()

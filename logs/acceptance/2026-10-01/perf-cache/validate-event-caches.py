"""覆盖 stable-state、bias 写入、碰撞回退及 reset 无文件访问的缓存边界。"""
import argparse
from isaaclab.app import AppLauncher
parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import importlib
import json
from pathlib import Path
import sys
import types
from unittest.mock import patch
import numpy as np
import trimesh
import torch
import gymnasium as gym
import Grasp1.tasks
from isaaclab_tasks.utils import parse_env_cfg

package = 'Grasp1.tasks.manager_based.grasp1.mdp'
legacy_name = package + '._event_before'
legacy = types.ModuleType(legacy_name)
legacy.__path__ = [str(Path(__file__).parent / 'before/mdp')]
sys.modules[legacy_name] = legacy
old_events = importlib.import_module(legacy_name + '.events')
old_kp = importlib.import_module(legacy_name + '.keypoints')
events = importlib.import_module(package + '.events')
kp = importlib.import_module(package + '.keypoints')
features = importlib.import_module(package + '.observations')

def same(a,b):
    assert torch.allclose(a,b,atol=1e-6,rtol=1e-5), (a-b).abs().max()

try:
    cfg = parse_env_cfg('Grasp1-UR5-Allegro-Teacher-v0', device='cuda:0', num_envs=16)
    cfg.set_num_envs(16)
    env = gym.make('Grasp1-UR5-Allegro-Teacher-v0', cfg=cfg)
    e = env.unwrapped
    ids = torch.tensor([3,7,11], device=e.device)
    keep = torch.ones(16,dtype=torch.bool,device=e.device)
    keep[ids] = False
    with torch.inference_mode():
        env.reset(seed=42)
        data = e._teacher_reset_data
        stable_cpu = {name: torch.tensor([0.,0.,0.8+index*0.001,1.,0.,0.,0.])
                      for index,name in enumerate(data['unique_names'])}
        data['stable_states'] = torch.stack([stable_cpu[name] for name in data['object_names']]).to(e.device)
        data['stable_state_available'].fill_(True)
        class Proxy:
            def __getattr__(self,name):
                return getattr(e,name)
        proxy = Proxy()
        proxy._teacher_reset_data = dict(data, stable_states_cpu=stable_cpu)
        params = {name: cfg.events.reset_teacher.params[name] for name in
                  ('non_uniform_sampling','support_height','stable_state_height_offset')
                  if name in cfg.events.reset_teacher.params}
        params.update(use_stable_states=True,stable_state_height_offset=events.STABLE_STATE_HEIGHT_OFFSET)
        state = torch.cuda.get_rng_state()
        expected = old_events._sample_object_pose(proxy, ids,
                    tuple(data['object_names'][index] for index in ids.cpu().tolist()), **params)
        torch.cuda.set_rng_state(state)
        actual = events._sample_object_pose(e,ids,**params)
        for a,b in zip(expected,actual): same(a,b)

        cache = e._teacher_runtime_features
        old_hand = kp.hand_keypoints_w(e).clone()
        old_vectors = features.nearest_affordance_vectors_w(e).clone()
        old_root = e.scene['object'].data.root_pos_w.clone()
        data['bias_pending'][ids] = True
        data['bias'][ids,:2] = 0.005
        events.apply_object_position_bias(e,ids,distance_threshold=1000000.0)
        assert 'affordance_vectors_w' in cache._pending
        same(e.scene['object'].data.root_pos_w[ids,:2], old_root[ids,:2]+0.005)
        assert torch.equal(e.scene['object'].data.root_pos_w[keep],old_root[keep])
        e.scene.write_data_to_sim()
        e.sim.forward()
        same(kp.hand_keypoints_w(e),old_kp.hand_keypoints_w(e))
        same(kp.hand_keypoints_o(e),old_kp.hand_keypoints_o(e))
        assert torch.equal(kp.hand_keypoints_w(e)[keep],old_hand[keep])
        updated_vectors = features.nearest_affordance_vectors_w(e)
        assert torch.equal(updated_vectors[keep],old_vectors[keep])
        assert not torch.equal(updated_vectors[ids],old_vectors[ids])

        # 强制首次碰撞全为 True，使三个环境进入原始固定安全回退分支。
        data['collision_pending'][ids] = True
        e.episode_length_buf[ids] = 1
        old_robot = e.scene['robot'].data.joint_pos.clone()
        original_collision = events._current_arm_collision
        events._current_arm_collision = lambda env, selected: torch.ones(len(selected),device=e.device,dtype=torch.bool)
        events.check_initial_collision(e,ids)
        events._current_arm_collision = original_collision
        assert not data['collision_pending'][ids].any()
        assert torch.equal(e.scene['robot'].data.joint_pos[keep],old_robot[keep])
        same(kp.hand_keypoints_w(e),old_kp.hand_keypoints_w(e))
        same(kp.hand_keypoints_o(e),old_kp.hand_keypoints_o(e))
        current = e.observation_manager._group_obs_term_cfgs['policy'][0].func
        assert current._needs_wrist_init[ids].all()
        assert (current._previous_wrist_euler[ids] == 0).all()

        def no_files(*a,**kw): raise AssertionError('reset file IO')
        with patch.object(Path,'read_text',no_files), patch.object(np,'load',no_files), patch.object(trimesh,'load_mesh',no_files):
            e._reset_idx([3,7,11])
            kp.hand_keypoints_w(e)
            features.nearest_affordance_vectors_w(e)
        print('EVENT_CACHE_RESULT',json.dumps(dict(stable_state_parity=True,bias_invalidation=True,
              collision_fallback_invalidation=True,reset_file_reads=0)),flush=True)
    env.close()
finally:
    app.close()

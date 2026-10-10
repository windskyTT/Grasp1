"""优化6逐项正确性与等模拟时长物理A/B；使用真实Isaac环境。"""
import argparse
import json
from pathlib import Path
import sys
import time
from isaaclab.app import AppLauncher
parser = argparse.ArgumentParser()
parser.add_argument('--decimation', type=int, choices=(2, 4), required=True)
parser.add_argument('--episode', type=float, choices=(4., 10.), required=True)
parser.add_argument('--duration', type=float, default=10.)
parser.add_argument('--output_dir',type=Path,default=Path('logs/acceptance/optimization6'))
parser.add_argument('--trace_mode',choices=('zero','random'),default=None)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app
import gymnasium as gym
import torch
import Grasp1.tasks
from Grasp1.tasks.manager_based.grasp1.config.ur5_allegro.teacher_env_cfg import UR5AllegroTeacherEnvCfg
from Grasp1.tasks.manager_based.grasp1.config.ur5_allegro.agents.rsl_rl_ppo_cfg import UR5AllegroTeacherPPORunnerCfg
from Grasp1.tasks.manager_based.grasp1.mdp import events, keypoints, runtime
sys.path.insert(0, str(Path('scripts/rsl_rl').resolve()))
from grasp_diagnostics import PhysicsDiagnostics
out = args.output_dir
out.mkdir(parents=True,exist_ok=True)
result = dict(decimation=args.decimation, episode_s=args.episode, seed=1, num_envs=16)
cfg = UR5AllegroTeacherEnvCfg(decimation=args.decimation, episode_length_s=args.episode)
cfg.set_num_envs(16)
cfg.synchronize_control_timing()
agent = UR5AllegroTeacherPPORunnerCfg()
agent.synchronize_control_timing(cfg.control_dt())
result['timing'] = dict(physics_dt=cfg.sim.dt, control_dt=cfg.control_dt(), gamma=agent.algorithm.gamma,
                        lam=agent.algorithm.lam, seconds_to_steps={str(s): round(s/cfg.control_dt()) for s in (4,10,20)},
                        rollout_s=32*cfg.control_dt(), event_intervals=[cfg.events.check_initial_collision.interval_range_s,
                                                                   cfg.events.apply_object_position_bias.interval_range_s])
import ast
from types import SimpleNamespace
from Grasp1.data.object_set import EVALUATION_DATASET,evaluation_object_names
from Grasp1.utils.paths import object_asset_dir
module=ast.parse(Path('scripts/rsl_rl/evaluate.py').read_text())
configure_node=next(n for n in module.body if isinstance(n,ast.FunctionDef) and n.name=='_configure_evaluation')
namespace=dict(args_cli=SimpleNamespace(grasp_duration_s=20.,lift_duration_s=20.),EVALUATION_DATASET=EVALUATION_DATASET,
               object_asset_dir=object_asset_dir,ManagerBasedRLEnvCfg=object)
exec(compile(ast.Module(body=[configure_node],type_ignores=[]),'scripts/rsl_rl/evaluate.py','exec'),namespace)
eval_cfg=cfg.copy()
namespace['_configure_evaluation'](eval_cfg,tuple(evaluation_object_names(1)))
assert round(eval_cfg.episode_length_s/cfg.control_dt())==round(40/cfg.control_dt())+2
result['evaluation_timing']=dict(grasp_steps=round(20/cfg.control_dt()),lift_steps=round(20/cfg.control_dt()),
                                ramp_steps=round(16/cfg.control_dt()),episode_s=eval_cfg.episode_length_s,
                                paper_ramp_steps=round(14/cfg.control_dt()))

env = gym.make('Grasp1-UR5-Allegro-Teacher-v0', cfg=cfg).unwrapped
try:
    print('CHECK environment initialized',flush=True)
    env.reset()
    print('CHECK full reset',flush=True)
    robot, obj = env.scene['robot'], env.scene['object']
    term = env.action_manager.get_term('teacher')
    ids = torch.tensor([0,2], device=env.device)
    keep = torch.tensor([i for i in range(16) if i not in (0,2)], device=env.device)
    zeros = torch.zeros((16,22), device=env.device)
    assert env.step_dt == cfg.control_dt()
    assert env.max_episode_length == round(args.episode / env.step_dt)
    result['registered_observation_shape'] = list(env.observation_manager.compute_group('policy').shape)
    assert result['registered_observation_shape'] == [16,153]
    # 零/随机/极值和上下限：实际apply并人为改变实测q，检查不在子步重新生成target。
    q_original = robot.data.joint_pos.clone()
    limits = robot.data.joint_pos_limits[:,term._joint_ids]
    cases = 0
    max_obs_error = 0.
    for baseline in ('current','lower','upper'):
        for value in ('zero','random','positive','negative'):
            q = q_original.clone()
            if baseline != 'current':
                q[:,term._joint_ids] = limits[...,0 if baseline == 'lower' else 1]
            robot.write_joint_state_to_sim(q, torch.zeros_like(q))
            action = zeros if value == 'zero' else torch.randn_like(zeros) if value == 'random' else torch.full_like(zeros, 1e4 if value == 'positive' else -1e4)
            term.process_actions(action)
            expected = (q[:,term._joint_ids] + term.processed_actions).clamp(limits[...,0],limits[...,1])
            assert torch.equal(term.target, expected)
            for substep in range(args.decimation):
                changed = q.clone()
                changed[:,term._joint_ids] += 0.00001*(substep+1)
                robot.write_joint_state_to_sim(changed, torch.zeros_like(q))
                term.apply_actions()
                assert torch.equal(term.target, expected)
                assert torch.equal(robot.data.joint_pos_target[:,term._joint_ids], expected)
                assert (robot.data.joint_vel_target[:,term._joint_ids] == 0).all()
            env._teacher_runtime_features.invalidate()
            obs = env.observation_manager.compute_group('policy')
            error = (obs[:,22:44] - (expected-robot.data.joint_pos[:,term._joint_ids])).abs().max()
            max_obs_error = max(max_obs_error,float(error))
            assert float(error) == 0.
            cases += 16*22
    result['action'] = dict(cases=cases, target_substeps_equal=True, target_outside_fraction=0., observation_error=max_obs_error)
    env.reset()
    assert torch.equal(term.target, robot.data.joint_pos[:,term._joint_ids])
    term.process_actions(torch.ones_like(zeros))
    before = term.target.clone()
    cache_before = keypoints.hand_keypoints_o(env).clone()
    env._reset_idx(ids)
    assert torch.equal(term.target[ids], robot.data.joint_pos[:,term._joint_ids][ids])
    assert torch.equal(term.target[keep], before[keep])
    after = keypoints.hand_keypoints_o(env).clone()
    assert torch.equal(after[keep],cache_before[keep])
    env._teacher_runtime_features.invalidate()
    assert torch.allclose(after,keypoints.hand_keypoints_o(env),atol=1e-7,rtol=0)
    result['partial_reset'] = dict(target_synced=True, other_targets_unchanged=True, cache_matches_uncached=True)
    # 强制真实collision fallback分支，仅测试选择的两个槽位。
    env.reset()
    term.process_actions(torch.ones_like(zeros))
    before = term.target.clone()
    env._teacher_reset_data['collision_pending'][ids] = True
    env.episode_length_buf[ids] = 1
    original_collision = events._current_arm_collision
    events._current_arm_collision = lambda e, selected: torch.ones(selected.numel(),device=e.device,dtype=torch.bool)
    events.check_initial_collision(env,ids)
    events._current_arm_collision = original_collision
    assert torch.equal(term.target[ids],robot.data.joint_pos[:,term._joint_ids][ids])
    assert torch.equal(term.target[keep],before[keep])
    cached=keypoints.hand_keypoints_o(env).clone()
    env._teacher_runtime_features.invalidate()
    assert torch.allclose(cached,keypoints.hand_keypoints_o(env),atol=1e-7,rtol=0)
    result['collision_fallback'] = dict(target_synced=True, other_targets_unchanged=True, cache_matches_uncached=True)
    # 物体偏移消费者的实际写状态与部分缓存失效。
    env.reset()
    cached = keypoints.hand_keypoints_o(env).clone()
    env._teacher_reset_data['bias_pending'][ids] = True
    env._teacher_reset_data['bias'][ids,:2] = 0.01
    events.apply_object_position_bias(env, ids, distance_threshold=100.)
    biased = keypoints.hand_keypoints_o(env).clone()
    assert torch.equal(biased[keep],cached[keep])
    assert not torch.equal(biased[ids],cached[ids])
    env._teacher_runtime_features.invalidate()
    assert torch.allclose(biased,keypoints.hand_keypoints_o(env),atol=1e-7,rtol=0)
    result['object_bias'] = dict(cache_matches_uncached=True, other_rows_unchanged=True)
    # 只更换高度termination测试输入，仍使用真实manager顺序和奖励累加。
    env.reset()
    height_cfg=env.termination_manager.get_term_cfg('invalid_hand_height')
    original_height=height_cfg.func
    height_cfg.func=lambda e,**kw: torch.arange(e.num_envs,device=e.device)==0
    env.termination_manager.set_term_cfg('invalid_hand_height',height_cfg)
    original_reward=env.reward_manager.compute
    weights=[term_cfg.weight for term_cfg in env.reward_manager._term_cfgs]
    for name,term_cfg in zip(env.reward_manager.active_terms,env.reward_manager._term_cfgs):
        if name != 'invalid_hand_height_terminal':
            term_cfg.weight=0.
    penalties=[]
    penalty_idx=env.reward_manager.active_terms.index('invalid_hand_height_terminal')
    def reward(dt):
        value=original_reward(dt)
        penalties.append(value.clone())
        return value
    env.reward_manager.compute=reward
    _,_,terminated,_,_=env.step(zeros)
    assert bool(terminated[0])
    assert float(penalties[-1][0]) == -10.
    assert (penalties[-1][1:] == 0).all()
    height_cfg.func=original_height
    env.termination_manager.set_term_cfg('invalid_hand_height',height_cfg)
    env.step(zeros)
    assert (penalties[-1] == 0).all()
    env.reward_manager.compute=original_reward
    for term_cfg,weight in zip(env.reward_manager._term_cfgs,weights):
        term_cfg.weight=weight
    result['terminal_penalty'] = dict(height=-10., other_envs=0., next_step=0.)
    env.reset()
    env.episode_length_buf[:]=env.max_episode_length-1
    _,_,_,truncated,_=env.step(zeros)
    assert truncated.all()
    assert (env.reward_manager._step_reward[:,penalty_idx] == 0).all()
    result['timeout'] = dict(max_steps=env.max_episode_length, boundary_pass=True, penalty=0.)
    # 非法观测终止不套用高度惩罚，不把NaN写入物理状态。
    obs_cfg=env.termination_manager.get_term_cfg('invalid_observation')
    original_obs=obs_cfg.func
    obs_cfg.func=lambda e,**kw: torch.arange(e.num_envs,device=e.device)==1
    env.termination_manager.set_term_cfg('invalid_observation',obs_cfg)
    env.step(zeros)
    assert (env.reward_manager._step_reward[:,penalty_idx] == 0).all()
    obs_cfg.func=original_obs
    env.termination_manager.set_term_cfg('invalid_observation',obs_cfg)
    result['invalid_observation_penalty'] = 0.
    # 提前终止必须冻结首次终止前物体状态，后续自动reset的高度不能进入同一trial。
    from grasp_diagnostics import EvaluationTrajectories
    env.reset()
    trial_args=SimpleNamespace(diagnostics=False,paper_stability=False,success_height=0.1,
                               max_relative_translation_drift_m=0.02,max_relative_rotation_drift_rad=0.2)
    trials=EvaluationTrajectories(env,cfg.object_names,trial_args)
    trials.start_round(0)
    pose=obj.data.root_pose_w[1:2].clone()
    pose[:,2]+=0.2
    obj.write_root_pose_to_sim(pose,env_ids=torch.tensor([1],device=env.device))
    obs_cfg.func=lambda e,**kw: torch.arange(e.num_envs,device=e.device)==1
    env.termination_manager.set_term_cfg('invalid_observation',obs_cfg)
    env.step(zeros)
    trials.sample('grasp')
    frozen=trials.final_object_pose[1].clone()
    assert not bool(trials.active[1]) and float(frozen[2]-trials.initial_z[1])>0.1
    obs_cfg.func=original_obs
    env.termination_manager.set_term_cfg('invalid_observation',obs_cfg)
    env.step(zeros)
    trials.sample('lift')
    assert torch.equal(trials.final_object_pose[1],frozen)
    source=(trials.final_object_pose[:,2]-trials.initial_z)>0.1
    trials.finish_round(source,trials.active & source)
    assert trials.rows[1]['failure_reason']=='invalid_observation'
    assert not trials.rows[1]['active_height_success']
    trial_path=out/f'early_termination_D{args.decimation}_T{int(args.episode)}'
    trial_path.mkdir()
    trials.save(trial_path)
    env._reset_idx=trials.original_reset
    result['evaluation_early_termination']=dict(frozen=True,new_episode_excluded=True,failure='invalid_observation')
    # 注入真实NaN观测，保持物理状态有限，走原invalid_observation检测和reset链。
    env.reset()
    observation_term=env.observation_manager._group_obs_term_cfgs['policy'][0].func
    observation_type=type(observation_term)
    original_call=observation_type.__call__
    def observation_with_nan(self,e,**params):
        value=original_call(self,e,**params).clone()
        if int(e.episode_length_buf[1])>0:
            value[1,0]=torch.nan
        return value
    observation_type.__call__=observation_with_nan
    invalid_obs,_,invalid_done,_,_=env.step(zeros)
    observation_type.__call__=original_call
    assert bool(invalid_done[1])
    assert torch.isfinite(invalid_obs['policy']).all()
    assert float(env.reward_manager._step_reward[1,penalty_idx])==0.
    assert torch.equal(term.target[1],robot.data.joint_pos[:,term._joint_ids][1])
    result['actual_nan_observation_reset']=dict(terminated=True,reset_observation_finite=True,target_synced=True,penalty=0.)


    result['object'] = dict(body_names=obj.body_names, joint_names=obj.joint_names,
                            stiffness=obj.data.joint_stiffness.tolist(), damping=obj.data.joint_damping.tolist(),
                            armature=obj.root_physx_view.get_dof_armatures().tolist(), limits=obj.data.joint_pos_limits.tolist())
    assert obj.num_bodies == 2 and obj.num_joints == 1
    assert (obj.data.joint_stiffness == 0).all() and (obj.data.joint_damping == 0).all()
    # 每物理步测量，零动作和固定seed高斯动作均执行同样10秒模拟时间。
    result['sensors']=[dict(name=name, bodies=sensor.body_names, filters=sensor.cfg.filter_prim_paths_expr,
                            physics_dt=sensor._sim_physics_dt, normal_shape=list(sensor.data.force_matrix_w.shape),
                            friction_shape=list(sensor.data.friction_forces_w.shape)) for name,sensor in env.scene.sensors.items()]
    physics=[]
    for mode in ('zero','random'):
        env.reset(seed=1)
        stats=PhysicsDiagnostics(env)
        torch.manual_seed(1)
        trace=[]
        original_update=env.scene.update
        if args.trace_mode == mode:
            def traced_update(dt):
                original_update(dt)
                if dt==env.physics_dt:
                    trace.append(dict(q=robot.data.joint_pos[:,term._joint_ids].clone(),
                                      v=robot.data.joint_vel[:,term._joint_ids].clone(),target=term.target.clone(),
                                      torque=robot.data.applied_torque[:,term._joint_ids].clone(),
                                      object_pose=obj.data.root_pose_w.clone(),object_q=obj.data.joint_pos.clone(),
                                      contacts=torch.stack([sensor.data.net_forces_w[:,0].norm(dim=-1) for sensor in env.scene.sensors.values()],1)))
            env.scene.update=traced_update
        torch.cuda.synchronize()
        started=time.perf_counter()
        counts={name:torch.zeros((),device=env.device,dtype=torch.long) for name in env.termination_manager.active_terms}
        finite=torch.tensor(True,device=env.device)
        obs_equal=torch.tensor(True,device=env.device)
        substeps_equal=torch.tensor(True,device=env.device)
        original_process=term.process_actions
        original_apply=term.apply_actions
        held={}
        def process(action):
            original_process(action)
            held['target']=term.target.clone()
        def apply():
            original_apply()
            substeps_equal.logical_and_((term.target==held['target']).all())
        term.process_actions=process
        term.apply_actions=apply
        steps=round(args.duration/env.step_dt)
        for i in range(steps):
            action=zeros if mode=='zero' else torch.randn_like(zeros)
            obs,rewards,term_done,time_done,_=env.step(action)
            finite &= torch.isfinite(obs['policy']).all() & torch.isfinite(rewards).all()
            for name in counts:
                counts[name]+=env.termination_manager.get_term(name).sum()
            expected=term.target-robot.data.joint_pos[:,term._joint_ids]
            obs_equal &= (obs['policy'][:,22:44] == expected).all()
        torch.cuda.synchronize()
        wall=time.perf_counter()-started
        assert bool(finite) and bool(obs_equal) and bool(substeps_equal)
        term.process_actions=original_process
        term.apply_actions=original_apply
        counts={name:int(value) for name,value in counts.items()}
        row=stats.summary()
        if trace:
            data={name:torch.stack([entry[name] for entry in trace]).cpu() for name in trace[0]}
            torch.save(dict(physics_dt=env.physics_dt,names=cfg.object_names,joints=list(term._joint_names),
                            sensors=list(env.scene.sensors),**data),out/f'{mode}_physics_trace.pt')
        env.scene.update=original_update
        stats.close()
        row.update(mode=mode,simulated_s=args.duration,wall_s=wall,transitions=steps*16,
                   wall_s_per_simulated_s=wall/args.duration,wall_s_per_million_transitions=wall/(steps*16)*1e6,
                   environment_simulated_s=args.duration*16, env_simulated_s_per_gpu_hour=args.duration*16/wall*3600,
                   termination_counts=counts, fps=steps*16/wall, cuda_peak_allocated_mib=torch.cuda.max_memory_allocated()/1024**2)
        # 传感器按physics_dt把冲量转成N；reward路径只读控制周期末帧，再乘physics_dt。
        for prefix in env._teacher_runtime_features.contact_sensors:
            impulses=runtime.filtered_contact_impulses(env,prefix)
            sensors=env._teacher_runtime_features.contact_sensors[prefix]
            n=torch.stack([sensor.data.force_matrix_w[:,0].sum(1) for sensor in sensors],1)
            f=torch.stack([sensor.data.friction_forces_w[:,0].sum(1) for sensor in sensors],1)
            assert torch.allclose(impulses[0],(n+f).norm(dim=-1)*env.physics_dt,atol=1e-7,rtol=1e-6)
        row['impulse_unit_check']=True
        physics.append(row)
    result['physics_ab']=physics
    result['gate_pass']=all(p['finite'] and p['hand_velocity_peak'] < 100. and p['contact_force_peak_N'] < 10000.
                            and p['termination_counts']['invalid_observation']==0 for p in physics)
    result['gate_threshold_basis']='Stop before repeats of prior 406rad/s and 87kN anomalies: screening limits 100rad/s, 10kN; not physical safety guarantees.'
    (out/f'D{args.decimation}_T{int(args.episode)}.json').write_text(json.dumps(result,indent=2,allow_nan=False))
    print('OPT6_RESULT',json.dumps(dict(gate_pass=result['gate_pass'],timing=result['timing'],physics=[{k:r[k] for k in ('mode','hand_velocity_peak','contact_force_peak_N','wall_s')} for r in physics])))
except Exception:
    import traceback
    error=traceback.format_exc()
    (out/f'D{args.decimation}_T{int(args.episode)}.error.txt').write_text(error)
    print(error,flush=True)
    raise
finally:
    env.close()
    app.close()

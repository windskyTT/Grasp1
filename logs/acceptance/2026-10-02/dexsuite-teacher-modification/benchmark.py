"""优化2分阶段的实际运行、奖励尺度、缓存及 CPU/CUDA 测量。"""
import argparse
from pathlib import Path
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument('--num_envs', type=int, default=16)
parser.add_argument('--steps', type=int, nargs='+', default=[300])
parser.add_argument('--profile', action='store_true')
parser.add_argument('--validate', action='store_true')
parser.add_argument('--inspect_assets', action='store_true')
parser.add_argument('--self_collision', action='store_true')
parser.add_argument('--scale', type=float)
parser.add_argument('--output', required=True)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import json
import time
import torch
import gymnasium as gym
import Grasp1.tasks
from isaaclab_tasks.utils import parse_env_cfg
from Grasp1.tasks.manager_based.grasp1.mdp.keypoints import hand_keypoints_w

out = Path(args.output)
cfg = parse_env_cfg('Grasp1-UR5-Allegro-Teacher-v0', device='cuda:0', num_envs=args.num_envs)
cfg.set_num_envs(args.num_envs)
cfg.seed = 42
if args.self_collision:
    cfg.scene.robot.spawn.articulation_props.enabled_self_collisions = True
if args.scale is not None:
    cfg.actions.teacher.scale = args.scale
env = gym.make('Grasp1-UR5-Allegro-Teacher-v0', cfg=cfg)
e = env.unwrapped
robot, obj = e.scene['robot'], e.scene['object']
term = e.action_manager.get_term('teacher')
from Grasp1.robots.robot_profile import UR5_JOINT_NAMES, ALLEGRO_JOINT_NAMES
assert tuple(robot.joint_names[i] for i in term._joint_ids) == UR5_JOINT_NAMES + tuple(name.replace('.', '_') for name in ALLEGRO_JOINT_NAMES)
result = dict(num_envs=e.num_envs, physics_dt=e.physics_dt, step_dt=e.step_dt,
              decimation=cfg.decimation, physics_hz=1/e.physics_dt, policy_hz=1/e.step_dt,
              episode_seconds=cfg.episode_length_s, episode_steps=e.max_episode_length,
              action_class=type(term).__name__, joint_names=[robot.joint_names[i] for i in term._joint_ids],
              robot_solver=cfg.scene.robot.spawn.articulation_props.to_dict(),
              object_solver=cfg.scene.object.spawn.articulation_props.to_dict(),
              solver_type=cfg.sim.physx.solver_type, measurements=[],
              object_body_names=obj.body_names, object_joint_names=obj.joint_names,
              object_joint_limits=obj.data.joint_pos_limits[0].tolist())
from pxr import Usd, UsdPhysics, PhysxSchema
result['loaded_articulation_properties'] = {}
for asset_name in ['Robot', 'Object']:
    prim = e.sim.stage.GetPrimAtPath('/World/envs/env_0/'+asset_name)
    props = []
    for node in Usd.PrimRange(prim):
        if node.HasAPI(UsdPhysics.ArticulationRootAPI):
            api = PhysxSchema.PhysxArticulationAPI(node)
            props.append(dict(path=str(node.GetPath()), position=api.GetSolverPositionIterationCountAttr().Get(),
                              velocity=api.GetSolverVelocityIterationCountAttr().Get(), self_collision=api.GetEnabledSelfCollisionsAttr().Get()))
    result['loaded_articulation_properties'][asset_name] = props
with torch.inference_mode():
    obs, _ = env.reset(seed=42)
    assert obs['policy'].shape == (e.num_envs, 153)
    assert e.action_manager.total_action_dim == 22
    actions = torch.zeros((e.num_envs, 22), device=e.device)
    # 首个真实控制步之后检查 arm contact 数据与 collision pending。
    env.step(actions)
    result['first_step_contacts'] = {str(i): dict(
        finite=bool(torch.isfinite(e.scene[f'teacher_arm_contact_{i}'].data.net_forces_w).all()),
        max_force=float(e.scene[f'teacher_arm_contact_{i}'].data.net_forces_w.abs().max()),
        sensor_timestamp=float(e.scene[f'teacher_arm_contact_{i}']._timestamp.max())) for i in range(6)}
    result['collision_pending_after_first_step'] = int(e._teacher_reset_data['collision_pending'].sum())
    object_joint_min = obj.data.joint_pos.clone()
    object_joint_max = object_joint_min.clone()
    for _ in range(9):
        env.step(actions)
    for steps in args.steps:
        env.reset(seed=42)
        done_counts = {name: torch.zeros((), device=e.device, dtype=torch.long) for name in e.termination_manager.active_terms}
        finite = torch.ones((), device=e.device, dtype=torch.bool)
        reward_sum = torch.zeros((e.num_envs, len(e.reward_manager.active_terms)), device=e.device)
        reward_squares = torch.zeros_like(reward_sum)
        episode_term_running = torch.zeros_like(reward_sum)
        completed_terms = torch.zeros(reward_sum.shape[1], device=e.device)
        completed_count = torch.zeros((), device=e.device, dtype=torch.long)
        totals = torch.zeros(e.num_envs, device=e.device)
        episode_rewards = []
        max_velocity = max_effort = max_contact = target_delta_max = 0.0
        clipped = total_targets = 0
        torch.cuda.synchronize()
        start = time.perf_counter()
        for index in range(steps):
            actions.uniform_(-1, 1)
            obs, reward, terminated, truncated, _ = env.step(actions)
            finite &= torch.isfinite(obs['policy']).all() & torch.isfinite(reward).all()
            finite &= torch.isfinite(robot.data.joint_vel).all() & torch.isfinite(obj.data.root_state_w).all()
            for name in done_counts:
                done_counts[name] += e.termination_manager.get_term(name).sum()
            values = e.reward_manager._step_reward * e.step_dt
            reward_sum += values
            reward_squares += values.square()
            totals += reward
            done = terminated | truncated
            episode_term_running += values
            completed_terms += episode_term_running[done].sum(0)
            completed_count += done.sum()
            episode_term_running[done] = 0
            if args.validate and bool(done.any()):
                episode_rewards.extend(totals[done].tolist())
                totals[done] = 0
            object_joint_min = torch.minimum(object_joint_min, obj.data.joint_pos)
            object_joint_max = torch.maximum(object_joint_max, obj.data.joint_pos)
            if args.validate or args.scale is not None or args.self_collision:
                max_velocity = max(max_velocity, float(robot.data.joint_vel.abs().max()))
                max_effort = max(max_effort, float(robot.data.applied_torque.abs().max()))
                contact_vectors = torch.cat([e.scene[f'teacher_af_contact_{i}'].data.net_forces_w for i in range(13)] + [e.scene[f'teacher_arm_contact_{i}'].data.net_forces_w for i in range(6)], dim=1)
                max_contact = max(max_contact, float(torch.linalg.vector_norm(contact_vectors, dim=-1).max()))
                target_delta_max = max(target_delta_max, float((robot.data.joint_pos_target[:, term._joint_ids] - robot.data.joint_pos[:, term._joint_ids]).abs().max()))
                limits = robot.data.joint_pos_limits[:, term._joint_ids]
                targets = robot.data.joint_pos_target[:, term._joint_ids]
                clipped += int(((targets < limits[..., 0]) | (targets > limits[..., 1])).sum())
                total_targets += targets.numel()
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        assert bool(finite)
        mean = reward_sum.mean(0)/steps
        completed = int(completed_count)
        entry = dict(steps=steps, wall_seconds=elapsed, simulated_seconds=steps*e.step_dt,
                     env_steps_per_second=e.num_envs*steps/elapsed,
                     physics_substeps_per_second=steps*cfg.decimation/elapsed,
                     physics_env_substeps_per_second=e.num_envs*steps*cfg.decimation/elapsed,
                     simulated_seconds_per_wall_second=steps*e.step_dt/elapsed,
                     wall_seconds_per_simulated_second=elapsed/(steps*e.step_dt),
                     done_counts={name:int(value) for name,value in done_counts.items()}, observed_episode_reward_mean=(sum(episode_rewards)/len(episode_rewards) if episode_rewards else None),
                     reward_terms={name: dict(mean_per_step=float(mean[i]), mean_per_second=float(mean[i]/e.step_dt),
                                            horizon_equivalent=float(mean[i]*e.max_episode_length),
                                            completed_episode_mean=float(completed_terms[i]/completed) if completed else None,
                                            rms_per_step=float((reward_squares[:,i].mean()/steps).sqrt()))
                                   for i,name in enumerate(e.reward_manager.active_terms)},
                     max_joint_velocity=max_velocity, max_joint_effort=max_effort, max_contact_force=max_contact,
                     max_target_delta=target_delta_max, max_contact_impulse=max_contact*e.physics_dt, target_outside_limits_fraction=clipped/max(1,total_targets))
        result['measurements'].append(entry)
        print('BENCHMARK', json.dumps({key: value for key,value in entry.items() if key != 'reward_terms'}), flush=True)
    result['object_joint_min'] = object_joint_min.min(0).values.tolist()
    result['object_joint_max'] = object_joint_max.max(0).values.tolist()
    if args.validate:
        ids = torch.tensor([3,7,11], device=e.device)
        keep = torch.ones(e.num_envs, dtype=torch.bool, device=e.device)
        keep[ids] = False
        cached = hand_keypoints_w(e)
        assert cached is hand_keypoints_w(e)
        cache_before = cached.clone()
        states = robot.data.joint_pos.clone()
        object_states = obj.data.root_state_w.clone()
        processed_before = term.processed_actions.clone()
        e._reset_idx(ids)
        e.scene.write_data_to_sim()
        e.sim.forward()
        assert torch.equal(states[keep], robot.data.joint_pos[keep])
        assert torch.equal(object_states[keep], obj.data.root_state_w[keep])
        assert torch.equal(cache_before[keep], hand_keypoints_w(e)[keep])
        assert (obj.data.root_vel_w[ids] == 0).all()
        if type(term).__name__ == 'RelativeJointPositionAction':
            assert (term.processed_actions[ids] == 0).all()
            assert torch.equal(processed_before[keep], term.processed_actions[keep])
            result['partial_reset_relative_delta_zero'] = True
        e.observation_manager.compute()
        actions.zero_()
        env.step(actions)
        assert hand_keypoints_w(e) is not cached
        saved_lookups = []
        def no_lookup(*a, **kw):
            raise AssertionError('runtime body/joint name lookup')
        for asset in [robot, obj]:
            for method in ['find_bodies','find_joints']:
                saved_lookups.append((asset, method, getattr(asset,method)))
                setattr(asset, method, no_lookup)
        env.step(actions)
        for asset, method, original in saved_lookups:
            setattr(asset, method, original)
        result['runtime_name_lookups'] = 0
        result['partial_reset'] = [3,7,11]
        result['cache_reuse_and_step_invalidation'] = True
        if type(term).__name__ == 'RelativeJointPositionAction':
            # 官方 RelativeJointPositionAction 在每个物理子步基于当前 q 生成目标。
            for joint in range(22):
                actions.zero_()
                actions[:, joint] = 1
                term.process_actions(actions)
                before = robot.data.joint_pos[:, term._joint_ids].clone()
                term.apply_actions()
                expected = before + term.processed_actions
                torch.testing.assert_close(robot.data.joint_pos_target[:, term._joint_ids], expected)
                assert torch.count_nonzero(term.processed_actions[0]).item() == 1
                obs = e.observation_manager.compute()['policy']
                torch.testing.assert_close(obs[:,22:44], term.processed_actions)
            actions.zero_()
            term.process_actions(actions)
            term.apply_actions()
            torch.testing.assert_close(robot.data.joint_pos_target[:,term._joint_ids], robot.data.joint_pos[:,term._joint_ids])
            result['relative_action_all_22_joints_and_zero'] = True
    if args.profile:
        from torch.profiler import profile, ProfilerActivity, record_function
        wall_profile = {}
        for owner, method, label in [(e.sim,'step','physics'), (e.scene,'update','scene_update'),
                                      (e.reward_manager,'compute','reward'), (e.observation_manager,'compute','observation')]:
            original = getattr(owner, method)
            def measured(*a, _original=original, _label=label, **kw):
                begin = time.perf_counter_ns()
                with record_function('phase/'+_label):
                    value = _original(*a, **kw)
                duration = (time.perf_counter_ns() - begin)/1000
                stats = wall_profile.setdefault(_label, dict(total_us=0, calls=0))
                stats['total_us'] += duration
                stats['calls'] += 1
                return value
            setattr(owner, method, measured)
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            for _ in range(10):
                env.step(actions)
        prof.export_chrome_trace(str(out.with_suffix('.trace.json')))
        result['cpu_wall_profile'] = wall_profile
        result['cpu_profile'] = {ev.key: dict(total_us=ev.cpu_time_total, self_us=ev.self_cpu_time_total, calls=ev.count)
                                 for ev in prof.key_averages() if ev.key.startswith('phase/')}
        trace = json.loads(out.with_suffix('.trace.json').read_text())
        result['cpu_trace_profile'] = {}
        for label in ['physics','scene_update','reward','observation']:
            rows = [ev for ev in trace['traceEvents'] if ev.get('name') == 'phase/'+label and ev.get('ph') == 'X' and ev.get('cat') == 'user_annotation']
            result['cpu_trace_profile'][label] = dict(total_us=sum(ev['dur'] for ev in rows), calls=len(rows))
        kernels = [ev for ev in trace['traceEvents'] if ev.get('cat') == 'kernel']
        total = sum(ev.get('dur',0) for ev in kernels)
        result['gpu_kernel_total_us'] = total
        result['gpu_profile'] = {}
        for name in ['artiSolveInternalConstraintsTGS1T','stepArticulation1TTGS','artiPropagateRigidImpulsesAndSolveSelfConstraintsTGS1T']:
            matches = [ev for ev in kernels if name in ev['name']]
            duration = sum(ev['dur'] for ev in matches)
            result['gpu_profile'][name] = dict(duration_us=duration, count=len(matches), percent=duration/total*100 if total else 0)
if args.inspect_assets:
    from pxr import Usd, UsdPhysics
    import xml.etree.ElementTree as ET
    asset_rows = []
    for urdf_path in sorted(Path('assets/objects').glob('*/*/*.urdf')):
        if urdf_path.stem != urdf_path.parent.name:
            continue
        source = ET.parse(urdf_path).getroot()
        stage = Usd.Stage.Open(str(urdf_path.with_suffix('.usd')))
        rigid_bodies, joints, articulation_roots = [], [], []
        for prim in stage.Traverse():
            if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                rigid_bodies.append(dict(path=str(prim.GetPath()), mass=str(UsdPhysics.MassAPI(prim).GetMassAttr().Get()),
                                         inertia=str(UsdPhysics.MassAPI(prim).GetDiagonalInertiaAttr().Get())))
            if prim.IsA(UsdPhysics.Joint):
                joints.append(dict(path=str(prim.GetPath()), type=prim.GetTypeName(),
                                   attributes={a.GetName():str(a.Get()) for a in prim.GetAttributes() if a.GetName().startswith('physics:')}))
            if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
                articulation_roots.append(str(prim.GetPath()))
        asset_rows.append(dict(asset=str(urdf_path), source_links=[p.attrib['name'] for p in source.findall('link')],
                               source_joints=[dict(p.attrib, limit=p.find('limit').attrib if p.find('limit') is not None else None) for p in source.findall('joint')],
                               rigid_bodies=rigid_bodies, usd_joints=joints, articulation_roots=articulation_roots))
    out.with_suffix('.assets.json').write_text(json.dumps(asset_rows, indent=2))
out.write_text(json.dumps(result, indent=2))
print('RESULT', str(out), flush=True)
env.close()
app.close()

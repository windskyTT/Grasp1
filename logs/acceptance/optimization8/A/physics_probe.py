"""优化8物理子步复现（沿用优化7fixture）；诊断参数只影响本次进程，不修改正式配置或USD。"""
import argparse
import csv
import json
from pathlib import Path
import sys
import time

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument('--decimation', type=int, choices=(2, 4), default=4)
parser.add_argument('--episode', type=float, choices=(4., 10.), default=10.)
parser.add_argument('--duration', type=float, default=10.)
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--num_envs', type=int, default=16)
parser.add_argument('--modes', nargs='+', choices=('zero', 'random', 'fixed', 'replay'), default=['zero', 'random', 'fixed'])
parser.add_argument('--collision_variant', choices=('original', 'base_filter', 'box'), default='original')
parser.add_argument('--hand_stiffness', type=float)
parser.add_argument('--hand_friction', type=float)
parser.add_argument('--legacy_joint_friction', type=float, help='Single-variable USD jointFriction audit before physics initialization.')
parser.add_argument('--hand_armature', type=float)
parser.add_argument('--arm_armature', type=float)
parser.add_argument('--object_armature', type=float)
parser.add_argument('--velocity_iterations', type=int)
parser.add_argument('--object_velocity_iterations', type=int)
parser.add_argument('--contact_detail', action='store_true', help='Record named body/filter contact points for the selected environment.')
parser.add_argument('--isolation', choices=('baseline', 'no_object', 'no_table', 'no_contacts'), default='baseline')
parser.add_argument('--initial_state', type=Path)
parser.add_argument('--source_env', type=int, default=7)
parser.add_argument('--replay', type=Path, default=Path('logs/acceptance/optimization6/diagnosis/zero_physics_trace.pt'))
parser.add_argument('--limit_case', choices=('lower', 'inward', 'outward'))
parser.add_argument('--force_perturbation', action='store_true', help='2.5N finger-link force at 2.0-2.1s, diagnostic only.')
parser.add_argument('--solver',choices=('tgs','pgs'),default='tgs')
parser.add_argument('--output_dir', type=Path, required=True)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import gymnasium as gym
import torch
from pxr import Usd, UsdPhysics
import Grasp1.tasks
from Grasp1.tasks.manager_based.grasp1.config.ur5_allegro.teacher_env_cfg import UR5AllegroTeacherEnvCfg
from Grasp1.utils.paths import object_asset_dir
from isaaclab.managers import EventTermCfg
from pxr import PhysxSchema

out = args.output_dir
out.mkdir(parents=True, exist_ok=True)
cfg = UR5AllegroTeacherEnvCfg(decimation=args.decimation, episode_length_s=args.episode)
cfg.seed = args.seed
cfg.set_num_envs(args.num_envs)
cfg.sim.physx.solver_type = 1 if args.solver == 'tgs' else 0
initial = torch.load(args.initial_state, map_location='cpu', weights_only=False) if args.initial_state else None
if initial is not None:
    assert args.num_envs == 1
    names = (initial['names'][args.source_env],)
    cfg.object_names = names
    cfg.scene.object.spawn.usd_path = [str(object_asset_dir(cfg.object_dataset, names[0]) / f'{names[0]}.usd')]
    cfg.events.initialize_teacher_data.params['object_names'] = names
    cfg.events.reset_teacher.params['object_names'] = names
if args.velocity_iterations is not None:
    cfg.scene.robot.spawn.articulation_props.solver_velocity_iteration_count = args.velocity_iterations
if args.object_velocity_iterations is not None:
    cfg.scene.object.spawn.articulation_props.solver_velocity_iteration_count = args.object_velocity_iterations
if args.isolation in ('no_table', 'no_contacts'):
    pos = cfg.scene.table.init_state.pos
    cfg.scene.table.init_state.pos = (pos[0], pos[1], pos[2]-2.0)
if args.legacy_joint_friction is not None:
    def change_legacy_friction(env, env_ids, value):
        for prim in env.sim.stage.Traverse():
            path = str(prim.GetPath())
            if '/Robot/joints/joint_' in path and prim.IsA(UsdPhysics.RevoluteJoint):
                PhysxSchema.PhysxJointAPI(prim).GetJointFrictionAttr().Set(value)
    cfg.events.diagnostic_legacy_friction = EventTermCfg(func=change_legacy_friction,mode='prestartup',params={'value':args.legacy_joint_friction})
from physics_runtime_audit import prepare_collision_variant, runtime_audit, subscribe_contacts
prepare_collision_variant(cfg, args.collision_variant)
if args.hand_stiffness is not None:
    cfg.scene.robot.actuators['hand'].stiffness = args.hand_stiffness
cfg.synchronize_control_timing()
assert cfg.sim.dt == 1.0 / 120.0
env = gym.make('Grasp1-UR5-Allegro-Teacher-v0', cfg=cfg).unwrapped
try:
    robot, obj = env.scene['robot'], env.scene['object']
    term = env.action_manager.get_term('teacher')
    joints = list(term._joint_ids)
    hand = joints[6:]
    if args.hand_friction is not None:
        robot.write_joint_friction_coefficient_to_sim(args.hand_friction, joint_ids=hand)
    if args.hand_armature is not None:
        robot.write_joint_armature_to_sim(args.hand_armature, joint_ids=hand)
    if args.arm_armature is not None:
        robot.write_joint_armature_to_sim(args.arm_armature, joint_ids=joints[:6])
    if args.object_armature is not None:
        obj.write_joint_armature_to_sim(args.object_armature)
    # 查询加载后的实际参数；包括导入器保留的joint摩擦，不能只审查Cfg。
    def parameters(asset):
        return dict(joints=asset.joint_names, bodies=asset.body_names,
                    stiffness=asset.data.joint_stiffness.tolist(), damping=asset.data.joint_damping.tolist(),
                    effort_limit=asset.root_physx_view.get_dof_max_forces().tolist(),
                    velocity_limit=asset.root_physx_view.get_dof_max_velocities().tolist(),
                    friction=asset.root_physx_view.get_dof_friction_properties().tolist(),
                    legacy_friction=asset.root_physx_view.get_dof_friction_coefficients().tolist(),
                    armature=asset.root_physx_view.get_dof_armatures().tolist(),
                    masses=asset.root_physx_view.get_masses().tolist(),
                    inertias=asset.root_physx_view.get_inertias().tolist(),
                    limits=asset.data.joint_pos_limits.tolist())
    audit = dict(robot=parameters(robot), object=parameters(obj), overrides=vars(args).copy())
    audit['overrides'] = {k: str(v) if isinstance(v, Path) else v for k, v in audit['overrides'].items()}
    audit['usd_joints'] = []
    audit['usd_colliders'] = []
    # Robot碰撞mesh是instance proxy；普通Traverse会遗漏其CollisionAPI。
    for prim in env.sim.stage.Traverse(Usd.TraverseInstanceProxies()):
        path = str(prim.GetPath())
        if not path.startswith('/World/envs/env_0/'):
            continue
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            audit['usd_colliders'].append(dict(path=path, type=prim.GetTypeName(),
                attributes={a.GetName(): str(a.Get()) for a in prim.GetAttributes() if a.GetName().startswith(('physics:', 'physx'))}))
        if prim.IsA(UsdPhysics.Joint):
            audit['usd_joints'].append(dict(path=path,
                attributes={a.GetName(): str(a.Get()) for a in prim.GetAttributes() if a.GetName().startswith(('physics:', 'physx', 'drive:'))},
                bodies={r.GetName(): [str(p) for p in r.GetTargets()] for r in prim.GetRelationships()}))
    (out/'loaded_parameters.json').write_text(json.dumps(audit, indent=2))
    (out/'loaded_colliders.json').write_text(json.dumps(runtime_audit(env), indent=2))
    contact_step = [0]
    contact_reports, collect_contact_reports = subscribe_contacts(contact_step)
    sensors = list(env.scene.sensors.items())
    sensor_meta = [dict(name=n, bodies=s.body_names, filters=s.cfg.filter_prim_paths_expr) for n, s in sensors]
    original_update, original_reset = env.scene.update, env._reset_idx
    replay = torch.load(args.replay, map_location=env.device, weights_only=False) if 'replay' in args.modes else None
    rows, summaries = [], []
    for mode in args.modes:
        env.reset(seed=args.seed)
        if initial is not None:
            src = args.source_env
            q, v = initial['robot_q'][src:src+1].to(env.device), initial['robot_v'][src:src+1].to(env.device)
            robot.write_joint_state_to_sim(q, v)
            pose = initial['object_root'][src:src+1].to(env.device).clone()
            pose[:, :3] += env.scene.env_origins - initial['origins'][src:src+1].to(env.device)
            obj.write_root_pose_to_sim(pose)
            obj.write_root_velocity_to_sim(initial['object_velocity'][src:src+1].to(env.device))
            obj.write_joint_state_to_sim(initial['object_q'][src:src+1].to(env.device), initial['object_v'][src:src+1].to(env.device))
            for key, value in initial['reset_data'].items():
                env._teacher_reset_data[key][:] = value[src:src+1].to(env.device)
                if key == 'object_pose':
                    env._teacher_reset_data[key][:, :3] += env.scene.env_origins - initial['origins'][src:src+1].to(env.device)
            term.reset()
            env._teacher_runtime_features.invalidate()
            env.sim.forward()
        state = dict(names=cfg.object_names, robot_q=robot.data.joint_pos.clone().cpu(), robot_v=robot.data.joint_vel.clone().cpu(),
                     object_root=obj.data.root_pose_w.clone().cpu(), object_velocity=obj.data.root_vel_w.clone().cpu(),
                     object_q=obj.data.joint_pos.clone().cpu(), object_v=obj.data.joint_vel.clone().cpu(), origins=env.scene.env_origins.cpu(),
                     reset_data={k:v.clone().cpu() for k,v in env._teacher_reset_data.items() if isinstance(v,torch.Tensor) and v.shape[:1] == (env.num_envs,)})
        torch.save(state, out/f'initial_{mode}.pt')
        if args.isolation in ('no_object', 'no_contacts'):
            pose = obj.data.root_pose_w.clone()
            pose[:, 0] += 2.0
            obj.write_root_pose_to_sim(pose)
            env._teacher_reset_data['collision_pending'][:] = False
            env._teacher_reset_data['bias_pending'][:] = False
        if args.limit_case:
            q = robot.data.joint_pos.clone()
            q[:, joints[-1]] = robot.data.joint_pos_limits[:, joints[-1], 0] + 0.001
            robot.write_joint_state_to_sim(q, torch.zeros_like(q))
            term.reset()
        fixed = term.target.clone()
        if args.limit_case == 'inward': fixed[:, -1] += 0.01
        if args.limit_case == 'outward': fixed[:, -1] -= 0.01
        limits = robot.data.joint_pos_limits[:, joints]
        fixed.clamp_(limits[..., 0], limits[..., 1])
        frames, reset_events = [], []
        pending_reset = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
        previous = robot.data.joint_pos[:, joints].clone()
        physics_step = 0
        def reset(ids):
            reset_events.append(dict(step=counter[0], ids=ids.clone(), reasons={n:env.termination_manager.get_term(n)[ids].clone() for n in env.termination_manager.active_terms}))
            pending_reset[ids] = True
            original_reset(ids)
            if mode == 'fixed':
                fixed[ids] = term.target[ids]
        def update(dt):
            original_update(dt)
            if dt != env.physics_dt: return
            # 外围脚本作用域的容器计数避免每个物理步同步GPU。
            collect_contact_reports()
            counter[0] += 1
            contact_step[0] = counter[0]
            q = robot.data.joint_pos[:, joints]
            normals = torch.stack([s.data.force_matrix_w[:, 0].sum(1) for _,s in sensors], 1)
            friction = torch.stack([s.data.friction_forces_w[:, 0].sum(1) for _,s in sensors], 1)
            frames.append(dict(q=q.clone(), v=robot.data.joint_vel[:, joints].clone(),
                delta_v=(q-previous)/env.physics_dt, target=term.target.clone(), raw=term.raw_actions.clone(),
                processed=term.processed_actions.clone(), torque_approx=robot.data.applied_torque[:, joints].clone(),
                velocity_target=robot.data.joint_vel_target[:, joints].clone(),
                robot_poses=robot.data.body_link_pose_w.clone(), object_poses=obj.data.body_link_pose_w.clone(), object_q=obj.data.joint_pos.clone(), object_v=obj.data.joint_vel.clone(),
                normal=normals.clone(), friction=friction.clone(),
                net_normal=torch.stack([s.data.net_forces_w[:,0] for _,s in sensors],1).clone(),
                reset_before=pending_reset.clone()))
            if args.contact_detail:
                details = []
                selected = args.source_env if env.num_envs > 1 else 0
                # 两侧刚体可由sensor和filter解析；Tensor API不暴露碰撞shape身份。
                for prefix in ('teacher_af_contact', 'teacher_table_contact'):
                    for index in (7, 10, 11, 12):
                        sensor = env.scene[f'{prefix}_{index}']
                        force, point, normal, distance, count, start = sensor.contact_physx_view.get_contact_data(env.physics_dt)
                        slots = torch.arange(sensor.cfg.max_contact_data_count_per_prim, device=env.device)
                        ix = (start[selected, 0] + slots).long().clamp(0, len(force)-1)
                        valid = slots < count[selected, 0]
                        values = torch.cat((force[ix].reshape(-1, 1), point[ix], normal[ix], distance[ix].reshape(-1, 1)), 1)
                        details.append(torch.where(valid[:, None], values, torch.zeros_like(values)))
                frames[-1]['contact_detail'] = torch.stack(details)
            previous.copy_(q)
            pending_reset.zero_()
        counter = [0]
        env._reset_idx, env.scene.update = reset, update
        contact_reports.clear()
        contact_step[0] = 0
        torch.manual_seed(args.seed)
        zero = torch.zeros((env.num_envs, 22), device=env.device)
        actions = []
        perturb_body = robot.body_names.index('link_15_0') if args.force_perturbation else None
        torch.cuda.synchronize()
        started = time.perf_counter()
        try:
            for control in range(round(args.duration/env.step_dt)):
                physics_step = counter[0]
                if args.force_perturbation and control in (round(2./env.step_dt),round(2.1/env.step_dt)):
                    force = torch.zeros((env.num_envs,1,3),device=env.device)
                    if control == round(2./env.step_dt): force[:,:,0] = -2.5
                    position = torch.zeros_like(force); position[:,:,2] = 0.0423
                    robot.set_external_force_and_torque(force,torch.zeros_like(force),positions=position,
                                                         body_ids=[perturb_body],is_global=False)
                action = zero if mode == 'zero' else torch.randn_like(zero) if mode == 'random' else (fixed-robot.data.joint_pos[:, joints])/term._scale
                if mode == 'replay':
                    target = replay['target'][min(control*args.decimation, len(replay['target'])-1)]
                    if args.num_envs == 1: target = target[args.source_env:args.source_env+1]
                    action = (target-robot.data.joint_pos[:, joints])/term._scale
                actions.append(action.clone())
                env.step(action)
        finally:
            env._reset_idx, env.scene.update = original_reset, original_update
        torch.cuda.synchronize()
        wall = time.perf_counter()-started
        data = {k:torch.stack([f[k] for f in frames]).cpu() for k in frames[0]}
        data.update(physics_dt=env.physics_dt, control_dt=env.step_dt, names=cfg.object_names,
                    joints=term._joint_names, sensors=sensor_meta, actions=torch.stack(actions).cpu(),
                    reset_events=[dict(step=e['step'],ids=e['ids'].tolist(),reasons={k:v.tolist() for k,v in e['reasons'].items()}) for e in reset_events])
        if args.contact_detail:
            data['contact_detail_sensors'] = [f'{p}_{i}' for p in ('teacher_af_contact','teacher_table_contact') for i in (7,10,11,12)]
            data['contact_detail_columns'] = ['normal_force_N','point_x','point_y','point_z','normal_x','normal_y','normal_z','distance_m']
            data['contact_detail_env'] = args.source_env if env.num_envs > 1 else 0
            data['shape_identity'] = 'unavailable in the installed RigidContactView Tensor API; collider paths in loaded_parameters.json'
        torch.save(data, out/f'trace_{mode}.pt')
        (out/f'contact_reports_{mode}.json').write_text(json.dumps(contact_reports, indent=2))
        lo, hi = limits.cpu().unbind(-1)
        outside = torch.maximum((lo-data['q']).clamp_min(0), (data['q']-hi).clamp_min(0))
        hand_v = data['v'][:, :, 6:].abs()
        peak = int(hand_v.argmax()); step, remainder = divmod(peak, env.num_envs*16); idx, joint = divmod(remainder, 16); joint += 6
        row = dict(mode=mode,physics_step=step+1,env_id=idx,object=cfg.object_names[idx],joint=term._joint_names[joint],
                   qvel_rad_s=float(data['v'][step,idx,joint]),delta_velocity_rad_s=float(data['delta_v'][step,idx,joint]),
                   q_rad=float(data['q'][step,idx,joint]),target_rad=float(data['target'][step,idx,joint]),
                   outside_rad=float(outside[step,idx,joint]),contact_peak_N=float(data['normal'][step,idx].norm(dim=-1).max()))
        rows.append(row)
        summary=dict(**row,hand_velocity_peak=float(hand_v.max()),velocity_ratio_to_7=float(hand_v.max()/7),
            hand_outside_peak_rad=float(outside[:,:,6:].max()),hand_outside_env_joint_frames=int((outside[:,:,6:]>1e-6).sum()),
            outside_duration_env_joint_s=float((outside[:,:,6:]>1e-6).sum()*env.physics_dt),
            normal_peak_N=float(data['normal'].norm(dim=-1).max()),friction_peak_N=float(data['friction'].norm(dim=-1).max()),
            net_normal_peak_N=float(data['net_normal'].norm(dim=-1).max()),
            velocity_difference_peak=float((data['v']-data['delta_v']).abs()[~data['reset_before']].max()),
            finite=all(bool(torch.isfinite(data[k]).all()) for k in ('q','v','object_poses','object_q','object_v','normal','friction')),
            wall_s=wall,physics_steps=len(frames),num_envs=env.num_envs,decimation=args.decimation,episode_s=args.episode,
            seed=args.seed,reset_events=data['reset_events'],trace=str(out/f'trace_{mode}.pt'))
        targets = data['target'].reshape(-1,args.decimation,env.num_envs,22)
        summary['joint_violation_p9999_rad'] = float(torch.quantile(outside.flatten(), 0.9999))
        summary['joint_violation_max_rad'] = float(outside.max())
        summary['contact_report_pairs'] = len(contact_reports)
        summary['target_substeps_equal'] = bool((targets==targets[:,0:1]).all())
        summary['target_outside_fraction'] = float(((data['target']<lo)|(data['target']>hi)).float().mean())
        reference = (112.90217590332031 if args.decimation==4 and mode=='zero' else
                     76.92410278320312 if args.decimation==2 and mode=='random' else None)
        summary['optimization6_reference_rad_s'] = reference
        summary['reference_difference_rad_s'] = float(hand_v.max())-reference if reference is not None else None
        summary['exact_reference_peak_reproduced'] = abs(float(hand_v.max())-reference)<1e-4 if reference is not None else None
        summaries.append(summary)
        print('OPT8_PHYSICS',json.dumps(summary),flush=True)
    with (out/'peak_cases.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    (out/'result.json').write_text(json.dumps(dict(parameters=str(out/'loaded_parameters.json'),runs=summaries),indent=2))
finally:
    env.close()
    app.close()

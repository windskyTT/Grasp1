"""诊断脚本基准：机器人IK接近/闭合/抬升；物体只由真实接触运动。"""
import argparse
import csv
import json
from pathlib import Path
import time

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--task', default='Grasp1-UR5-Allegro-GraspTrain-v2')
parser.add_argument('--dataset', choices=['new_training_set'], default='new_training_set')
parser.add_argument('--object', default='wood_block_oriented')
parser.add_argument('--num_envs', type=int, default=1)
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--trials', type=int, default=10)
parser.add_argument('--mode', choices=['scripted', 'zero', 'random', 'replay', 'interface'], default='scripted')
parser.add_argument('--solver', choices=['tgs', 'pgs'], default='tgs')
parser.add_argument('--legacy_joint_friction', type=float)
parser.add_argument('--hand_friction', type=float)
parser.add_argument('--position_iterations', type=int)
parser.add_argument('--hand_stiffness', type=float)
parser.add_argument('--approach_distance', type=float, default=0.04)
parser.add_argument('--episode_length_s', type=float)
parser.add_argument('--arm_action_speed', type=float)
parser.add_argument('--decimation', type=int, choices=[2,4], default=4)
AppLauncher.add_app_launcher_args(parser)
parser.add_argument('--output_dir', type=Path, required=True)
args = parser.parse_args()
app = AppLauncher(args).app

import gymnasium as gym
import torch
from isaaclab_tasks.utils import parse_env_cfg
import Grasp1.tasks
from Grasp1.tasks.manager_based.grasp1.mdp.full_grasp_metrics import contact_forces


def synthetic_checks(cfg):
    from types import SimpleNamespace
    from Grasp1.tasks.manager_based.grasp1.mdp.full_grasp_metrics import hold_transition, phase_from_state, relative_velocity
    from Grasp1.tasks.manager_based.grasp1.mdp.full_grasp_rewards import lift_progress, top_down_alignment
    from Grasp1.tasks.manager_based.grasp1.config.ur5_allegro.agents.rsl_rl_full_grasp_ppo_cfg import UR5AllegroFullGraspPPORunnerCfg
    from checkpoint_contract import build_contract, compare_contract
    from Grasp1.tasks.manager_based.grasp1.config.ur5_allegro.teacher_env_cfg import UR5AllegroTeacherEnvCfg
    from Grasp1.tasks.manager_based.grasp1.config.ur5_allegro.grasp_train_env_cfg import UR5AllegroGraspTrainEnvCfg
    from Grasp1.tasks.manager_based.grasp1.mdp.actions import TeacherRelativeJointPositionAction
    for legacy in (UR5AllegroTeacherEnvCfg(), UR5AllegroGraspTrainEnvCfg()):
        assert legacy.actions.teacher.class_type is TeacherRelativeJointPositionAction
        assert legacy.scene.robot.spawn.articulation_props.solver_position_iteration_count == 32
        assert legacy.episode_length_s == 4
    n = 8
    phase = phase_from_state(torch.tensor([.12,.03,.03,.03]), torch.tensor([False,False,True,True]),
        torch.tensor([False,False,False,True]))
    assert phase.tolist() == [0,1,2,3]
    relative = relative_velocity(torch.tensor([[0.,.2,0.,0.,0.,2.]]), torch.tensor([[0.,0.,0.,0.,0.,2.]]),
        torch.tensor([[.1,0.,0.]]), torch.tensor([[1.,0.,0.,0.]]))
    assert torch.allclose(relative, torch.zeros_like(relative), atol=1e-7)
    score=top_down_alignment(torch.tensor([[2**-.5,0.,2**-.5,0.],[1.,0.,0.,0.],[2**-.5,0.,-2**-.5,0.]]))
    assert torch.allclose(score,torch.tensor([1.,0.,0.]),atol=1e-6)
    contact = torch.ones(n, dtype=torch.bool)
    opposing = contact.clone()
    unsupported = contact.clone()
    safe = contact.clone()
    lift = torch.full((n,), 0.11)
    drift = torch.zeros(n)
    angle = torch.zeros(n)
    speed = torch.zeros(n)
    contact[1], opposing[2], unsupported[3], safe[7] = False, False, False, False
    drift[4], angle[5], speed[6] = 0.05, 0.5, 0.4
    elapsed = torch.zeros(n)
    for step in range(61):
        holding, elapsed, success = hold_transition(contact, opposing, lift, unsupported, drift, angle, speed, safe,
            elapsed, 1/60, .1, 1)
        if step < 59:
            assert not bool(success.any())
    assert success.tolist() == [True,False,False,False,False,False,False,False]
    contact[0] = False
    _, elapsed, success = hold_transition(contact, opposing, lift, unsupported, drift, angle, speed, safe, elapsed, 1/60, .1, 1)
    assert float(elapsed[0]) == 0 and not bool(success.any())
    progress = lift_progress(torch.tensor([0.,.05,.1,.2]), torch.tensor([True]*4), torch.tensor([True]*4), .1)
    assert torch.allclose(progress, torch.tensor([0.,.5,1.,1.]))
    assert not bool(lift_progress(torch.tensor([.2]), torch.tensor([False]), torch.tensor([True]), .1).any())
    scene = {}
    for index in range(4):
        matrix = torch.zeros(2,1,3,3)
        matrix[0,0,0,2] = index+1
        matrix[0,0,2,2] = 2*(index+1)
        scene[f'full_tip_{index}'] = SimpleNamespace(data=SimpleNamespace(force_matrix_w=matrix, friction_forces_w=torch.zeros_like(matrix)))
    for name in ('top','bottom'):
        scene[f'full_support_{name}'] = SimpleNamespace(data=SimpleNamespace(force_matrix_w=torch.zeros(2,1,1,3)))
    forces, table, support, _ = contact_forces(SimpleNamespace(scene=scene))
    assert forces[0].tolist() == [1.,2.,3.,4.] and table[0].tolist() == [2.,4.,6.,8.]
    assert not bool(forces[1].any()) and not bool(table[1].any())
    contract = build_contract(args.task, cfg, UR5AllegroFullGraspPPORunnerCfg())
    normalized = json.loads(json.dumps(contract))
    assert not compare_contract(normalized, normalized, resume=True)
    for key in ('episode_length_s','full_grasp_parameters','observation_semantics_version','curriculum_stage'):
        changed = dict(normalized)
        changed[key] = 'changed'
        assert key in compare_contract(normalized, changed, resume=False)
    cfg_dict = cfg.rewards.to_dict()
    assert 'obj_displacement_reward' not in cfg_dict and 'obj_vel_reward_' not in cfg_dict
    return dict(status='pass', kind='synthetic_not_grasp_success',
        cases=['continuous_hold','contact_loss','opposing_contact','table_support','relative_drift','relative_angle',
            'relative_speed','corotating_velocity','orientation_direction','unsafe','lift_direction','contact_positive_negative','phase_transitions','checkpoint_semantic_mismatch'], contract=contract)


def scripted_actions(env, palm_start, open_hand, time_s):
    """仅此诊断入口提供控制器；train/play从不调用本函数。"""
    robot, memory = env.scene['robot'], env._full_grasp_state
    term = env.action_manager.get_term('teacher')
    q = robot.data.joint_pos[:, term._joint_ids]
    goal = palm_start.clone()
    goal[:, 2] -= args.approach_distance*min(max((time_s-1.5)/2, 0), 1)
    goal[:, 2] += 0.15*min(max((time_s-4)/2.5, 0), 1)
    palm = robot.data.body_link_pos_w[:, memory.palm_id]
    jacobian = robot.root_physx_view.get_jacobians()[:, memory.palm_id-1, :3, term._joint_ids[:6]]
    error = (goal-palm).clamp(-0.005, 0.005)
    system = jacobian @ jacobian.transpose(1,2) + torch.eye(3, device=env.device)[None]*0.001
    delta = (jacobian.transpose(1,2) @ torch.linalg.solve(system, error.unsqueeze(-1))).squeeze(-1)
    closed = q.new_tensor([0.0,1.1,1.1,1.2]*3+[1.4,0.45,0.5,1.0])
    fraction = min(max((time_s-0.5)/2.5, 0), 1)
    hand_goal = open_hand+(closed-open_hand)*fraction
    q_goal = torch.cat((q[:, :6]+delta, hand_goal), -1)
    actions = (q_goal-term.target)/term._scale
    return actions.clamp(-1, 1)


def main():
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    (out/'command.json').write_text(json.dumps(vars(args), default=str, indent=2))
    cfg = parse_env_cfg(args.task, device=args.device or 'cuda:0', num_envs=args.num_envs)
    cfg.seed = args.seed
    cfg.object_dataset = args.dataset
    cfg.training_objects = (args.object,)
    cfg.set_num_envs(args.num_envs)
    cfg.decimation = args.decimation
    if args.arm_action_speed is not None:
        cfg.arm_action_speed = args.arm_action_speed
    if args.episode_length_s is not None:
        cfg.episode_length_s = args.episode_length_s
    cfg.synchronize_control_timing()
    cfg.sim.physx.solver_type = 1 if args.solver == 'tgs' else 0
    if args.legacy_joint_friction is not None:
        cfg.legacy_joint_friction = args.legacy_joint_friction
    if args.hand_friction is not None:
        cfg.hand_static_friction = args.hand_friction
    if args.position_iterations is not None:
        cfg.scene.robot.spawn.articulation_props.solver_position_iteration_count = args.position_iterations
    if args.hand_stiffness is not None:
        cfg.scene.robot.actuators['hand'].stiffness = args.hand_stiffness
    assert cfg.sim.dt == 1/120
    env = gym.make(args.task, cfg=cfg).unwrapped
    try:
        robot, obj, memory = env.scene['robot'], env.scene['object'], env._full_grasp_state
        term = env.action_manager.get_term('teacher')
        sensor_table = {name:dict(bodies=sensor.body_names, num_bodies=sensor.num_bodies,
                                filters=sensor.cfg.filter_prim_paths_expr, shape=list(sensor.data.force_matrix_w.shape))
                        for name,sensor in env.scene.sensors.items()}
        (out/'sensors.json').write_text(json.dumps(sensor_table, indent=2))
        (out/'collision_local_vertices.json').write_text(json.dumps({robot.body_names[body]:points.cpu().tolist()
            for body,points in memory.hand_geometry}, indent=2))
        (out/'physics.json').write_text(json.dumps(dict(robot_bodies=robot.body_names,
            legacy_friction=robot.root_physx_view.get_dof_friction_coefficients().tolist(),
            friction=robot.root_physx_view.get_dof_friction_properties().tolist(),
            masses=robot.root_physx_view.get_masses().tolist(), inertias=robot.root_physx_view.get_inertias().tolist(),
            limits=robot.data.joint_pos_limits.tolist(), robot_cfg=cfg.scene.robot.to_dict(),
            object_joint_limits=obj.data.joint_pos_limits.tolist(), object_masses=obj.root_physx_view.get_masses().tolist(),
            config=cfg.to_dict()), default=str, indent=2))
        if args.mode == 'interface':
            tests = synthetic_checks(cfg)
            (out/'synthetic.json').write_text(json.dumps(tests, default=str, indent=2))
            from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
            from rsl_rl.runners import OnPolicyRunner
            from Grasp1.tasks.manager_based.grasp1.config.ur5_allegro.agents.rsl_rl_full_grasp_ppo_cfg import UR5AllegroFullGraspPPORunnerCfg
            agent = UR5AllegroFullGraspPPORunnerCfg()
            agent.synchronize_control_timing(env.step_dt)
            wrapped = RslRlVecEnvWrapper(env, clip_actions=agent.clip_actions)
            runner = OnPolicyRunner(wrapped, agent.to_dict(), log_dir=None, device=env.device)
            with torch.inference_mode():
                forward = runner.get_inference_policy()(wrapped.get_observations())
            assert forward.shape == (env.num_envs, 22) and bool(torch.isfinite(forward).all())
            obs, _ = env.reset(seed=args.seed)
            target_before = term.target.clone()
            term.process_actions(torch.zeros((env.num_envs,22), device=env.device))
            assert torch.equal(term.target, target_before)
            nonzero = torch.full((env.num_envs,22), .1, device=env.device)
            limits = robot.data.joint_pos_limits[:,term._joint_ids]
            expected = (target_before+nonzero*term._scale).clamp(limits[...,0],limits[...,1])
            term.process_actions(nonzero)
            assert torch.allclose(term.target, expected, atol=1e-7)
            for substep in range(cfg.decimation):
                term.apply_actions()
                assert torch.equal(term.target, expected)
            before = (memory.phase.clone(), memory.hold.clone(), term.target.clone())
            selected = torch.tensor([0], device=env.device)
            memory.hold[:] = 0.5
            memory.phase[:] = 2
            env._reset_idx(selected)
            assert float(memory.hold[0]) == 0 and int(memory.phase[0]) == 0
            assert torch.equal(term.target[0], robot.data.joint_pos[0, term._joint_ids])
            if env.num_envs > 1:
                assert bool((memory.hold[1:] == 0.5).all()) and bool((memory.phase[1:] == 2).all())
                assert torch.equal(term.target[1:], before[2][1:])
            (out/'reset.json').write_text(json.dumps(dict(criteria=env._full_grasp_reset_diagnostics.tolist(),
                q=robot.data.joint_pos.tolist(), poses=robot.data.body_link_pose_w.tolist(),
                object_pose=obj.data.body_link_pose_w.tolist()), indent=2))
            (out/'result.json').write_text(json.dumps(dict(status='pass', kind='real_interface',
                observation_dim=list(obs['policy'].shape), action_dim=env.action_manager.total_action_dim,
                partial_reset='pass', num_envs=env.num_envs, reset_valid=memory.valid_reset.tolist(),
                physics_dt=env.physics_dt, decimation=cfg.decimation, standard_runner_forward='pass',
                ppo_updates=0, checkpoint=None, zero_holds_target='pass', all_22_outputs='pass',
                target_substep_cache='pass'), indent=2))
            return
        frames, rows, violations, controls = [], [], [], []
        original_update = env.scene.update
        previous = robot.data.joint_pos[:, term._joint_ids].clone()
        trial_number = [0]
        frame_number = [0]
        unsafe_step = [False]
        active_trial = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)

        def record(dt):
            original_update(dt)
            if dt != env.physics_dt:
                return
            q = robot.data.joint_pos[:, term._joint_ids].clone()
            qd = robot.data.joint_vel[:, term._joint_ids].clone()
            limits = robot.data.joint_pos_limits[:, term._joint_ids]
            violation = torch.maximum(limits[..., 0]-q, q-limits[..., 1]).clamp_min(0)
            forces, table, support, normal = contact_forces(env)
            net_tip_normal = torch.stack([env.scene[f'full_tip_{i}'].data.net_forces_w[:,0] for i in range(4)],1)
            filtered_tip_normal = torch.stack([env.scene[f'full_tip_{i}'].data.force_matrix_w[:,0].sum(1) for i in range(4)],1)
            clearance, object_clearance, bounds = memory.geometry()
            frames.append(dict(trial=torch.full((env.num_envs,), trial_number[0]), q=q, qd=qd,
                active_trial=active_trial.clone(),
                delta_qd=(q-previous)/dt, target=term.target.clone(), force=forces.clone(), table=table.clone(),
                raw_actions=term.raw_actions.clone(),
                normal_object=normal.clone(),
                net_tip_normal=net_tip_normal.clone(), other_tip_normal_N=(net_tip_normal-filtered_tip_normal).norm(dim=-1),
                table_normal=torch.stack([env.scene[f'full_tip_{i}'].data.force_matrix_w[:,0,2].clone() for i in range(4)], 1),
                support=support.clone(), top=obj.data.body_link_pose_w[:, memory.top_id].clone(),
                palm=robot.data.body_link_pose_w[:, memory.palm_id].clone(),
                body_velocity=robot.data.body_link_vel_w.clone(), clearance=clearance.clone(),
                object_q=obj.data.joint_pos.clone(), object_qd=obj.data.joint_vel.clone(),
                top_velocity=obj.data.body_link_vel_w[:,memory.top_id].clone(),
                tip_bounds=bounds.clone(), object_clearance=object_clearance.clone(), violation=violation))
            if args.mode == 'replay':
                force, point, normal, distance, count, start = env.scene['full_tip_3'].contact_physx_view.get_contact_data(dt)
                slots = torch.arange(32, device=env.device)
                ix = (start[0, 2]+slots).long().clamp(0, len(force)-1)
                detail = torch.cat((force[ix].reshape(-1,1), point[ix], normal[ix], distance[ix].reshape(-1,1)), -1)
                frames[-1]['thumb_table_contacts'] = torch.where((slots<count[0,2])[:,None], detail, torch.zeros_like(detail))
            violations.append(violation)
            previous.copy_(q)
            frame_number[0] += 1
            # 出现真实越限/速度异常立即结束本次试验，保留异常物理帧。
            unsafe_step[0] |= bool((violation > 0.005).any() or (qd[:, 6:].abs() > 14).any())

        env.scene.update = record
        started = time.perf_counter()
        trial_lengths = []
        try:
            for trial in range(args.trials):
                trial_number[0] = trial
                active_trial[:] = True
                env.reset(seed=args.seed+trial)
                previous.copy_(robot.data.joint_pos[:, term._joint_ids])
                memory.snapshot = None
                initial_z = memory.initial_z.clone()
                initial_palm = robot.data.body_link_pos_w[:, memory.palm_id].clone()
                open_hand = robot.data.joint_pos[:, term._joint_ids[6:]].clone()
                valid = memory.valid_reset.clone()
                attempts = memory.reset_attempts.clone()
                if args.mode == 'replay':
                    fixture = torch.load('logs/acceptance/optimization9/A/original_replay/initial_replay.pt', map_location=env.device, weights_only=False)
                    replay = torch.load('logs/acceptance/optimization9/A/original_replay/trace_replay.pt', map_location=env.device, weights_only=False)
                    robot.write_joint_state_to_sim(fixture['robot_q'], fixture['robot_v'])
                    pose = fixture['object_root'].clone()
                    # 回放保留与历史相同的object-off诊断；新任务试验始终object-on。
                    pose[:, 0] += 2
                    obj.write_root_pose_to_sim(pose)
                    obj.write_root_velocity_to_sim(fixture['object_velocity'])
                    term.reset()
                    env.sim.forward()
                    previous.copy_(robot.data.joint_pos[:, term._joint_ids])
                done = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
                max_lift = torch.zeros(env.num_envs, device=env.device)
                max_hold = torch.zeros(env.num_envs, device=env.device)
                failure = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
                succeeded = torch.zeros_like(done)
                elapsed = torch.zeros(env.num_envs, device=env.device)
                unsafe_step[0] = False
                start_index = len(frames)
                steps = round(cfg.episode_length_s/env.step_dt) if args.mode != 'replay' else len(replay['target'])//cfg.decimation
                for step in range(steps):
                    if args.mode == 'scripted':
                        actions = scripted_actions(env, initial_palm, open_hand, step*env.step_dt)
                    elif args.mode == 'random':
                        actions = torch.rand((env.num_envs, 22), device=env.device)*2-1
                    elif args.mode == 'replay':
                        target = replay['target'][step*cfg.decimation]
                        actions = (target-term.target)/term._scale
                    else:
                        actions = torch.zeros((env.num_envs, 22), device=env.device)
                    actions[done] = 0
                    if args.mode == 'replay':
                        # 历史兼容性诊断直接重放固定目标；不以新任务reset遮住旧tip/Table异常。
                        env.action_manager.process_action(actions)
                        for substep in range(cfg.decimation):
                            env.action_manager.apply_action()
                            env.scene.write_data_to_sim()
                            env.sim.step(render=False)
                            env.scene.update(env.physics_dt)
                        env.common_step_counter += 1
                        memory.snapshot = None
                        terminated = torch.zeros_like(done)
                        timeout = torch.zeros_like(done)
                    else:
                        _, _, terminated, timeout, _ = env.step(actions)
                    # reset后使用本次episode已保存的终态，不能读新物体状态作成功。
                    just_done = (terminated | timeout) & ~done
                    s = memory.read()
                    if args.mode != 'replay':
                        for index in range(env.num_envs):
                            if done[index]:
                                continue
                            controls.append(dict(trial=trial, env=index, time_s=(step+1)*env.step_dt,
                                phase=int(memory.last_phase[index] if just_done[index] else memory.phase[index]),
                                top_lift_m=float(memory.last_lift[index] if just_done[index] else s['lift'][index]),
                                hold_s=float(memory.last_hold[index] if just_done[index] else memory.hold[index]),
                                done=bool(just_done[index]),
                                **{'reward_'+name:float(env.reward_manager._step_reward[index,i]*env.step_dt)
                                   for i,name in enumerate(env.reward_manager.active_terms)}))
                    if not args.headless and step % 15 == 0:
                        print('FULL_GRASP_STATE '+json.dumps(dict(time_s=(step+1)*env.step_dt,
                            top_lift_m=float(s['lift'][0]), hold_s=float(memory.hold[0]), phase=int(memory.phase[0]),
                            controller=args.mode)), flush=True)
                    max_lift = torch.maximum(max_lift, torch.where(just_done, memory.last_lift, s['lift']))
                    max_hold = torch.maximum(max_hold, torch.where(just_done, memory.last_hold, memory.hold))
                    succeeded[just_done] = memory.last_success[just_done]
                    failure[just_done] = memory.last_failure[just_done]
                    elapsed[just_done] = (step+1)*env.step_dt
                    previous[just_done] = robot.data.joint_pos[:, term._joint_ids][just_done]
                    done |= just_done
                    active_trial[:] = ~done
                    if bool(done.all()) or (unsafe_step[0] and args.mode != 'replay'):
                        break
                trial_lengths.append(len(frames)-start_index)
                for index in range(env.num_envs):
                    row = dict(trial=trial, env=index, seed=args.seed+trial, object=args.object,
                        controller=args.mode, reset_valid=bool(valid[index]), reset_attempts=int(attempts[index]),
                        success=bool(succeeded[index]), max_lift_m=float(max_lift[index]), max_hold_s=float(max_hold[index]),
                        failure=('physical_violation' if unsafe_step[0] else str(int(failure[index]))),
                        simulated_s=float(elapsed[index]) if done[index] else (step+1)*env.step_dt, initial_top_z=float(initial_z[index]),
                        initial_palm=json.dumps(initial_palm[index].tolist()))
                    rows.append(row)
                    print('FULL_GRASP_TRIAL '+json.dumps(row), flush=True)
        finally:
            env.scene.update = original_update
            if frames:
                data = {key:torch.stack([frame[key].detach().cpu() for frame in frames]) for key in frames[0]}
                data.update(physics_dt=env.physics_dt, joints=term._joint_names, tip_names=['link_3_0_tip','link_7_0_tip','link_11_0_tip','link_15_0_tip'])
                torch.save(data, out/'trace.pt')
                with (out/'steps.csv').open('w', newline='') as handle:
                    writer = csv.writer(handle)
                    writer.writerow(['physics_step','trial','env','q','qd','delta_qd','target','tip_object_N','tip_table_N','object_table_N','top_pose','palm_pose','tip_minmax_z','limit_max_rad'])
                    for step in range(len(frames)):
                        for index in range(env.num_envs):
                            writer.writerow([step,int(data['trial'][step,index]),index,*[json.dumps(data[key][step,index].tolist()) for key in ('q','qd','delta_qd','target','force','table','support','top','palm','tip_bounds')],float(data['violation'][step,index].max())])
            if rows:
                if controls:
                    with (out/'control_rewards.csv').open('w',newline='') as handle:
                        writer=csv.DictWriter(handle,fieldnames=list(controls[0]))
                        writer.writeheader()
                        writer.writerows(controls)
                with (out/'trials.csv').open('w', newline='') as handle:
                    writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                    writer.writeheader()
                    writer.writerows(rows)
                violation = torch.stack(violations).flatten()
                successes = sum(row['success'] for row in rows)
                physical_pass = (float(violation.max()) <= 0.005 and float(torch.quantile(violation, .9999)) <= .001
                    and bool(torch.isfinite(data['q']).all()) and bool(torch.isfinite(data['qd']).all())
                    and float(data['qd'][:,:,6:].abs().max()) <= 14
                    and not any(row['failure']=='physical_violation' or int(row['failure']) & 3 for row in rows))
                group_status = ('pass' if physical_pass and (args.mode != 'scripted' or successes >= 3)
                    else 'partial' if physical_pass and successes > 0 and len(rows) < 3 else 'fail')
                result = dict(status=group_status,
                    physics_status='pass' if physical_pass else 'fail',
                    attempts=len(rows), successes=successes, success_rate=successes/len(rows), controller=args.mode,
                    reset_valid_count=sum(row['reset_valid'] for row in rows), physical_steps=len(frames),
                    completed_physics_steps_per_trial=trial_lengths, wall_s=time.perf_counter()-started,
                    limit_max_rad=float(violation.max()), limit_p9999_rad=float(torch.quantile(violation, .9999)),
                    qd_peak=float(data['qd'][:,:,6:].abs().max()), delta_qd_peak=float(data['delta_qd'][:,:,6:].abs().max()),
                    tip_object_peak_N=data['force'].amax(dim=(0,1)).tolist(), tip_table_peak_N=data['table'].amax(dim=(0,1)).tolist(),
                    tip_other_normal_peak_N=data['other_tip_normal_N'].amax(dim=(0,1)).tolist(),
                    observation_dim=env.observation_manager.group_obs_dim, action_dim=22,
                    vram_peak_bytes=torch.cuda.max_memory_allocated(), task=args.task, seed=args.seed,
                    action_semantics=type(term).__name__, episode_length_s=cfg.episode_length_s,
                    candidate=dict(solver=args.solver, legacy_joint_friction=cfg.legacy_joint_friction, hand_static_friction=cfg.hand_static_friction,
                        position_iterations=cfg.scene.robot.spawn.articulation_props.solver_position_iteration_count,
                        hand_stiffness=cfg.scene.robot.actuators['hand'].stiffness))
                (out/'result.json').write_text(json.dumps(result, indent=2))
                print('FULL_GRASP_RESULT '+json.dumps(result), flush=True)
    finally:
        env.close()


try:
    main()
except Exception:
    import traceback
    traceback.print_exc()
    raise
finally:
    app.close()

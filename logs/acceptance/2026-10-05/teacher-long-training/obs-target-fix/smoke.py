"""优化4：冻结配置的zero/random smoke与真实target observation验收。"""
import argparse
from pathlib import Path
from isaaclab.app import AppLauncher
parser = argparse.ArgumentParser()
parser.add_argument('--num_envs', type=int, default=16)
parser.add_argument('--steps', type=int, default=300)
parser.add_argument('--scale_factor', type=float, default=1.0)
parser.add_argument('--velocity_iterations', type=int, default=0)
parser.add_argument('--contact_last', action='store_true')
parser.add_argument('--passive', action='store_true')
parser.add_argument('--reward_semantics', choices=['step','time'])
parser.add_argument('--action_checks', action='store_true')
parser.add_argument('--armature', type=float)
parser.add_argument('--friction', type=float)
parser.add_argument('--object_velocity', type=float)
parser.add_argument('--dataset', default='new_training_set')
parser.add_argument('--zero_action', action='store_true')
parser.add_argument('--unclamped', action='store_true')
parser.add_argument('--output', required=True)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app
import json
import torch
import gymnasium as gym
import Grasp1.tasks
from isaaclab_tasks.utils import parse_env_cfg

cfg = parse_env_cfg('Grasp1-UR5-Allegro-Teacher-v0',num_envs=args.num_envs)
cfg.set_num_envs(args.num_envs)
if args.dataset != 'new_training_set':
    from Grasp1.data.object_set import object_names
    from Grasp1.tasks.manager_based.grasp1.config.ur5_allegro.teacher_env_cfg import _build_teacher_object_cfg
    names=object_names(args.dataset)
    cfg.object_dataset=args.dataset
    cfg.object_names=tuple(names[i%len(names)] for i in range(args.num_envs))
    cfg.scene.object=_build_teacher_object_cfg(cfg.object_names,dataset_name=args.dataset)
    for event in [cfg.events.initialize_teacher_data,cfg.events.reset_teacher]:
        event.params.update(dataset_name=args.dataset,object_names=cfg.object_names)
cfg.seed = 42
if args.unclamped:
    from isaaclab.envs.mdp import RelativeJointPositionAction
    cfg.actions.teacher.class_type=RelativeJointPositionAction
cfg.sim.physx.solve_articulation_contact_last = args.contact_last
cfg.scene.object.spawn.articulation_props.solver_velocity_iteration_count = args.velocity_iterations
cfg.actions.teacher.scale = {k:v*args.scale_factor for k,v in cfg.actions.teacher.scale.items()}
if args.reward_semantics is not None:
    from Grasp1.tasks.manager_based.grasp1.config.ur5_allegro.teacher_env_cfg import _isaaclab_reward_weight
    desired_weight=1.0/(0.2 if args.reward_semantics=='time' else cfg.sim.dt*cfg.decimation)
    for term_cfg in vars(cfg.rewards).values():
        term_cfg.weight *= desired_weight/_isaaclab_reward_weight(1.0)
env = gym.make('Grasp1-UR5-Allegro-Teacher-v0',cfg=cfg)
e = env.unwrapped
robot,obj=e.scene['robot'],e.scene['object']
term=e.action_manager.get_term('teacher')
result=dict(options=vars(args), object_names=cfg.object_names, joint_names=robot.joint_names,
    action_joint_names=[robot.joint_names[i] for i in term._joint_ids],
    runtime_velocity_limits=robot.root_physx_view.get_dof_max_velocities().tolist(),
    object_joint_limits=obj.data.joint_pos_limits[0].tolist(),
    object_stiffness=obj.root_physx_view.get_dof_stiffnesses().tolist(),
    object_damping=obj.root_physx_view.get_dof_dampings().tolist(),
    object_velocity_limits=obj.root_physx_view.get_dof_max_velocities().tolist(),
    object_solver=cfg.scene.object.spawn.articulation_props.to_dict())
if args.passive:
    obj.write_joint_stiffness_to_sim(0.0)
    obj.write_joint_damping_to_sim(0.0)
if args.armature is not None:
    obj.write_joint_armature_to_sim(args.armature)
if args.friction is not None:
    obj.write_joint_friction_coefficient_to_sim(args.friction)
if args.object_velocity is not None:
    obj.write_joint_velocity_limit_to_sim(args.object_velocity)
result['loaded_object_armature']=obj.root_physx_view.get_dof_armatures().tolist()

with torch.inference_mode():
    obs,_=env.reset(seed=42)
    torch.testing.assert_close(obs['policy'][:,22:44], torch.zeros_like(term.target),atol=0,rtol=0)
    qmin=obj.data.joint_pos.clone(); qmax=qmin.clone()
    velocity=torch.zeros_like(robot.data.joint_vel[0]); contact=effort=delta=0.0
    count_outside=total_targets=clipped=0
    done_counts={name:0 for name in e.termination_manager.active_terms}
    sums=torch.zeros((e.num_envs,len(e.reward_manager.active_terms)),device=e.device)
    episode_running=torch.zeros_like(sums); episode_sums=torch.zeros_like(sums[0]); episode_count=0
    mean_velocity=0.0
    policy_velocity=torch.zeros_like(velocity)
    peak_velocity_state={}
    previous_joint_pos=robot.data.joint_pos.clone()
    previous_joint_vel=robot.data.joint_vel.clone()
    original_apply=term.apply_actions
    def apply():
        global clipped,delta
        current=robot.data.joint_pos[:,term._joint_ids]
        limits=robot.data.joint_pos_limits[:,term._joint_ids]
        proposed=current+term.processed_actions
        clipped+=int(((proposed<limits[...,0])|(proposed>limits[...,1])).sum())
        original_apply()
        delta=max(delta,float((robot.data.joint_pos_target[:,term._joint_ids]-current).abs().max()))
    term.apply_actions=apply
    original_update=e.scene.update
    def update(dt):
        global qmin,qmax,velocity,count_outside,total_targets,contact,effort,peak_velocity_state,previous_joint_pos,previous_joint_vel
        original_update(dt)
        qmin=torch.minimum(qmin,obj.data.joint_pos); qmax=torch.maximum(qmax,obj.data.joint_pos)
        if float(robot.data.joint_vel.abs().max())>float(velocity.max()):
            flat=int(robot.data.joint_vel.abs().argmax()); env_id,joint_id=divmod(flat,robot.num_joints)
            peak_velocity_state=dict(env_id=env_id,joint=robot.joint_names[joint_id],control_step=e.common_step_counter,
                joint_vel=robot.data.joint_vel[env_id].tolist(),applied_torque=robot.data.applied_torque[env_id].tolist(),
                joint_pos=robot.data.joint_pos[env_id].tolist(),joint_pos_target=robot.data.joint_pos_target[env_id].tolist(),
                previous_joint_pos=previous_joint_pos[env_id].tolist(),previous_joint_vel=previous_joint_vel[env_id].tolist(),
                contact_force=[float(e.scene[f'teacher_af_contact_{i}'].data.net_forces_w[env_id].norm()) for i in range(13)])
        velocity=torch.maximum(velocity,robot.data.joint_vel.abs().max(0).values)
        targets=robot.data.joint_pos_target[:,term._joint_ids]
        limits=robot.data.joint_pos_limits[:,term._joint_ids]
        count_outside+=int(((targets<limits[...,0])|(targets>limits[...,1])).sum())
        total_targets+=targets.numel()
        contact=max(contact,max(float(e.scene[f'teacher_af_contact_{i}'].data.net_forces_w.norm(dim=-1).max()) for i in range(13)),
                    max(float(e.scene[f'teacher_arm_contact_{i}'].data.net_forces_w.norm(dim=-1).max()) for i in range(6)))
        effort=max(effort,float(robot.data.applied_torque.abs().max()))
        assert torch.isfinite(robot.data.joint_vel).all() and torch.isfinite(obj.data.body_state_w).all()
        previous_joint_pos=robot.data.joint_pos.clone()
        previous_joint_vel=robot.data.joint_vel.clone()
    e.scene.update=update
    actions=torch.zeros((e.num_envs,22),device=e.device)
    action_generator=torch.Generator(device=e.device).manual_seed(20261005)
    for _ in range(args.steps):
        if not args.zero_action: actions.uniform_(-1,1,generator=action_generator)
        obs,reward,terminated,truncated,_=env.step(actions)
        assert obs['policy'].shape==(e.num_envs,153)
        assert torch.isfinite(obs['policy']).all() and torch.isfinite(reward).all()
        torch.testing.assert_close(obs['policy'][:,:22],robot.data.joint_pos[:,term._joint_ids])
        torch.testing.assert_close(obs['policy'][:,22:44],term.target-robot.data.joint_pos[:,term._joint_ids])
        mean_velocity+=float(robot.data.joint_vel.abs().mean())/args.steps
        policy_velocity=torch.maximum(policy_velocity,robot.data.joint_vel.abs().max(0).values)
        values=e.reward_manager._step_reward*e.step_dt
        sums+=values; episode_running+=values
        done=terminated|truncated
        episode_sums+=episode_running[done].sum(0); episode_count+=int(done.sum()); episode_running[done]=0
        for name in done_counts: done_counts[name]+=int(e.termination_manager.get_term(name).sum())
    means=sums.mean(0)/args.steps
    result.update(object_joint_min_by_env=qmin[:,0].tolist(),object_joint_max_by_env=qmax[:,0].tolist(),
        max_velocity_by_joint=dict(zip(robot.joint_names,velocity.tolist())), mean_joint_velocity=mean_velocity,
        max_policy_velocity_by_joint=dict(zip(robot.joint_names,policy_velocity.tolist())),
        peak_velocity_state=peak_velocity_state,
        max_contact_force=contact,max_contact_impulse=contact*e.physics_dt,max_effort=effort,max_action_delta=delta,
        target_outside_limits_fraction=count_outside/total_targets, target_clipping_fraction=clipped/total_targets,
        done_counts=done_counts,finite=True,
        reward_terms={name:dict(per_step=float(means[i]),per_second=float(means[i]/e.step_dt),
            completed_episode=float(episode_sums[i]/episode_count) if episode_count else None)
            for i,name in enumerate(e.reward_manager.active_terms)})
    # partial reset 只改选中环境，且清除相对动作残差。
    ids=torch.tensor([3,7,11],device=e.device)
    keep=torch.ones(e.num_envs,dtype=torch.bool,device=e.device); keep[ids]=False
    before=robot.data.joint_pos.clone(); residual=term.processed_actions.clone(); object_before=obj.data.joint_pos.clone(); target_before=term.target.clone()
    e._reset_idx(ids)
    assert torch.equal(before[keep],robot.data.joint_pos[keep])
    assert torch.equal(object_before[keep],obj.data.joint_pos[keep])
    assert torch.equal(residual[keep],term.processed_actions[keep])
    assert (term.processed_actions[ids]==0).all() and (obj.data.joint_pos[ids]==0).all()
    assert torch.equal(target_before[keep],term.target[keep])
    torch.testing.assert_close(term.target[ids],robot.data.joint_pos[ids][:,term._joint_ids],atol=0,rtol=0)
    result['partial_reset']=True
    if args.action_checks:
        saved=robot.data.joint_pos.clone(); limits=robot.data.joint_pos_limits[:,term._joint_ids]
        checks=[]
        for position in ['current','upper','lower']:
            state=saved.clone()
            if position=='current': state[:,term._joint_ids]=(limits[...,0]+limits[...,1])/2
            else: state[:,term._joint_ids]=limits[...,1 if position=='upper' else 0]+(-1e-6 if position=='upper' else 1e-6)
            robot.write_joint_state_to_sim(state,torch.zeros_like(state))
            for sign in [0,1,-1]:
                for joint in range(22):
                    actions.zero_(); actions[:,joint]=sign
                    term.process_actions(actions); term.apply_actions()
                    expected=(robot.data.joint_pos[:,term._joint_ids]+term.processed_actions).clamp(limits[...,0],limits[...,1])
                    torch.testing.assert_close(robot.data.joint_pos_target[:,term._joint_ids],expected)
                    torch.testing.assert_close(term.target,expected)
                    checked=e.observation_manager.compute_group('policy')
                    torch.testing.assert_close(checked[:,22:44],expected-robot.data.joint_pos[:,term._joint_ids])
                    if position=='current':
                        torch.testing.assert_close(checked[:,22:44],term.processed_actions,atol=3e-7,rtol=1e-4)
                    checks.append((position,sign,joint))
        robot.write_joint_state_to_sim(saved,torch.zeros_like(saved))
        result['target_clamp_checks']=len(checks)
    obs,_=env.reset()
    torch.testing.assert_close(obs['policy'][:,22:44],torch.zeros_like(term.target),atol=0,rtol=0)
    result['full_reset']=True
Path(args.output).write_text(json.dumps(result,indent=2))
print('ROLLOUT_PROBE',args.output,flush=True)
env.close(); app.close()

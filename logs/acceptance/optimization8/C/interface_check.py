"""优化8独立接口/Reward/契约检查；不执行任何PPO update。"""
import argparse
import copy
import csv
import json
from pathlib import Path
import sys
from isaaclab.app import AppLauncher
parser=argparse.ArgumentParser()
parser.add_argument('--num_envs',type=int,default=1)
parser.add_argument('--decimation',type=int,choices=(2,4),default=4)
parser.add_argument('--task',choices=('Teacher-v0','GraspTrain-v1'),default='GraspTrain-v1')
parser.add_argument('--static_checks',action='store_true')
parser.add_argument('--output_dir',type=Path,required=True)
AppLauncher.add_app_launcher_args(parser)
args=parser.parse_args()
app=AppLauncher(args).app
import gymnasium as gym
import torch
from isaaclab.assets import Articulation
import Grasp1.tasks
from Grasp1.tasks.manager_based.grasp1.config.ur5_allegro.teacher_env_cfg import UR5AllegroTeacherEnvCfg
from Grasp1.tasks.manager_based.grasp1.config.ur5_allegro.grasp_train_env_cfg import UR5AllegroGraspTrainEnvCfg
from Grasp1.tasks.manager_based.grasp1.config.ur5_allegro.agents.rsl_rl_ppo_cfg import UR5AllegroTeacherPPORunnerCfg
from Grasp1.tasks.manager_based.grasp1.config.ur5_allegro.agents.rsl_rl_grasp_train_ppo_cfg import UR5AllegroGraspTrainPPORunnerCfg
from Grasp1.tasks.manager_based.grasp1.mdp import rewards,rewards_grasp_v1
sys.path.insert(0,str(Path('scripts/rsl_rl').resolve()))
sys.path.insert(0,str(Path('logs/acceptance/optimization8/B').resolve()))
from isaaclab_tasks.utils import load_cfg_from_registry
from checkpoint_contract import build_contract,compare_contract,save_metadata,validate_checkpoint,asset_contents
from reward_direction_tests import run as run_reward_tests
out=args.output_dir;out.mkdir(parents=True,exist_ok=True)
task_id='Grasp1-UR5-Allegro-'+args.task
cfg=(UR5AllegroTeacherEnvCfg if args.task=='Teacher-v0' else UR5AllegroGraspTrainEnvCfg)()
cfg.decimation=args.decimation
cfg.set_num_envs(args.num_envs)
if args.task=='GraspTrain-v1':cfg.synchronize_control_timing()
else:cfg.synchronize_control_timing()
agent=(UR5AllegroTeacherPPORunnerCfg if args.task=='Teacher-v0' else UR5AllegroGraspTrainPPORunnerCfg)()
agent.synchronize_control_timing(cfg.control_dt())
assert cfg.sim.dt==1/120
assert gym.spec(task_id).entry_point=='isaaclab.envs:ManagerBasedRLEnv'
assert type(load_cfg_from_registry(task_id,'env_cfg_entry_point')) is type(cfg)
assert type(load_cfg_from_registry(task_id,'rsl_rl_cfg_entry_point')) is type(agent)
if args.static_checks:
    validation=run_reward_tests()
    Path('logs/acceptance/optimization8/B/reward_validation.json').write_text(json.dumps(validation,indent=2)+'\n')
    contract=build_contract(task_id,cfg,agent)
    normalized=json.loads(json.dumps(contract))
    cases={'identical_resume':not compare_contract(normalized,normalized,resume=True)}
    changes={'decimation':2,'action_scale_hand':[.00375]*16,'reward_profile':'source_legacy','reward_terms':{},'curriculum_stage':'changed','normalizer_configuration':{'actor':True,'critic':True},'policy_architecture':{'activation':'relu'},'asset_contents':{'modified.usd':'changed'},'object_asset_contents':{'new_evaluation_object.usd':'new'}}
    for key,value in changes.items():
        other=copy.deepcopy(normalized);other[key]=value
        cases[key+'_resume_rejected']=bool(compare_contract(normalized,other,resume=True))
    other=copy.deepcopy(normalized);other['object_asset_contents']={'new_evaluation_object.usd':'new'}
    cases['heldout_object_change_allowed_for_eval']=not compare_contract(normalized,other,resume=False)
    other=copy.deepcopy(normalized)
    path=next(iter(other['object_asset_contents']))
    other['object_asset_contents'][path]='changed_same_object'
    cases['same_object_asset_change_rejected_for_eval']=bool(compare_contract(normalized,other,resume=False))
    checkpoint=out/'contract_fixture.pt';torch.save({'test_kind':'synthetic_metadata_check'},checkpoint)
    save_metadata(checkpoint,contract,0)
    audit,_=validate_checkpoint(checkpoint,contract,resume=True)
    cases['sidecar_round_trip']=audit['compatible']
    with checkpoint.open('ab') as f:f.write(b'changed')
    try:validate_checkpoint(checkpoint,contract,resume=True)
    except ValueError:cases['checkpoint_bytes_change_rejected']=True
    else:cases['checkpoint_bytes_change_rejected']=False
    # 实际USD依赖字节变化；只修改本次目录的独立合成fixture。
    from types import SimpleNamespace as NS
    asset=out/'asset_fixture.usda';asset.write_text('#usda 1.0\ndef Xform "Fixture" {}\n')
    fake=NS(scene=NS(robot=NS(spawn=NS(usd_path=str(asset))),table=NS(spawn=NS(usd_path=str(asset))),object=NS(spawn=NS(usd_path=[str(asset)]))))
    before=asset_contents(fake);asset.write_text('#usda 1.0\ndef Xform "Changed" {}\n');after=asset_contents(fake)
    cases['actual_usd_byte_change_detected']=before!=after
    assert all(cases.values()),cases
    (out/'checkpoint_contract_tests.json').write_text(json.dumps(dict(test_kind='static_synthetic_contract',cases=cases,pass_=True,asset_dependency_count=len(contract['asset_contents'])+len(contract['object_asset_contents'])),indent=2)+'\n')
    (out/'candidate_contract.json').write_text(json.dumps(contract,indent=2)+'\n')
    matrix=[]
    for cls in [UR5AllegroTeacherEnvCfg,UR5AllegroGraspTrainEnvCfg]:
        for d in [2,4]:
            c=cls();c.decimation=d
            if isinstance(c,UR5AllegroGraspTrainEnvCfg):c.synchronize_control_timing()
            else:c.synchronize_control_timing()
            a=UR5AllegroGraspTrainPPORunnerCfg() if isinstance(c,UR5AllegroGraspTrainEnvCfg) else UR5AllegroTeacherPPORunnerCfg();a.synchronize_control_timing(c.control_dt())
            matrix.append(dict(cfg=cls.__name__,physics_dt=c.sim.dt,decimation=d,control_dt=c.control_dt(),episode_s=c.episode_length_s,arm_scale=c.actions.teacher.scale[c.actions.teacher.joint_names[0]],hand_scale=c.actions.teacher.scale[c.actions.teacher.joint_names[6]],gamma=a.algorithm.gamma,lam=a.algorithm.lam,reward_profile=getattr(c,'reward_profile','source_legacy'),object_type=type(c.scene.object).__name__,candidate_qualified=False))
    Path('logs/acceptance/optimization8/C/task_config_matrix.json').write_text(json.dumps(matrix,indent=2)+'\n')
    changed=UR5AllegroGraspTrainEnvCfg();changed.sim.dt=.01
    try:changed.synchronize_control_timing()
    except AssertionError:dt_rejected=True
    else:dt_rejected=False
    assert dt_rejected
    (out/'dt_override_check.json').write_text(json.dumps({'forbidden_dt_rejected':True})+'\n')
# 真实接口短运行：所有产出均标为诊断，不作为Gate A/B/C通过证据。
env=gym.make(task_id,cfg=cfg).unwrapped
try:
    obs,_=env.reset(seed=8)
    robot,obj=env.scene['robot'],env.scene['object']
    term=env.action_manager.get_term('teacher')
    assert obs['policy'].shape==(args.num_envs,153)
    assert isinstance(obj,Articulation) and 'top' in obj.body_names and 'bottom' in obj.body_names
    captured=[];distributions=[]
    original_apply=term.apply_actions
    def apply():
        captured.append(term.target.clone())
        original_apply()
    term.apply_actions=apply
    actions=torch.zeros(args.num_envs,22,device=env.device)
    for step in range(20):
        obs,reward,terminated,truncated,_=env.step(actions)
        assert torch.isfinite(obs['policy']).all() and torch.isfinite(reward).all()
        limits=robot.data.joint_pos_limits[:,term._joint_ids]
        assert ((term.target>=limits[...,0])&(term.target<=limits[...,1])).all()
        # 同一PhysX末帧比较两个定义；仅诊断分布，不进行训练Reward选型。
        for name in env.reward_manager.active_terms:
            tc=env.reward_manager.get_term_cfg(name)
            raw=tc.func(env,**tc.params)
            distributions.append(dict(step=step,name=name,profile=getattr(cfg,'reward_profile','source_legacy'),raw=raw.clone().cpu(),weight=tc.weight))
        for profile,func in [('source_legacy',rewards.affordance_impulse_reward),('source_physics_aligned',rewards_grasp_v1.affordance_impulse_reward)]:
            tc=env.reward_manager.get_term_cfg('affordance_impulse_reward')
            distributions.append(dict(step=step,name='affordance_impulse_reward_comparison',profile=profile,raw=func(env,**tc.params).clone().cpu(),weight=tc.weight))
    term.apply_actions=original_apply
    targets=torch.stack(captured).reshape(20,args.decimation,args.num_envs,22)
    assert torch.equal(targets,targets[:,:1].expand_as(targets))
    ids=torch.tensor([0],device=env.device)
    others_before=term.target[1:].clone()
    env._reset_idx(ids)
    assert torch.equal(term.target[0],robot.data.joint_pos[0,term._joint_ids])
    assert torch.equal(term.target[1:],others_before)
    params={}
    for name,asset in [('robot',robot),('object',obj)]:
        inertias=asset.root_physx_view.get_inertias().reshape(args.num_envs,-1,3,3)
        params[name]=dict(bodies=asset.body_names,joints=asset.joint_names,masses=asset.root_physx_view.get_masses().tolist(),coms=asset.root_physx_view.get_coms().tolist(),inertias=inertias.tolist(),min_inertia_eigenvalue=float(torch.linalg.eigvalsh(inertias).min()),stiffness=asset.data.joint_stiffness.tolist(),damping=asset.data.joint_damping.tolist(),limits=asset.data.joint_pos_limits.tolist())
    (out/'loaded_parameters.json').write_text(json.dumps(params,indent=2)+'\n')
    rows=[]
    keys=sorted(set((d['name'],d['profile']) for d in distributions))
    for name,profile in keys:
        entries=[d for d in distributions if (d['name'],d['profile'])==(name,profile)]
        raw=torch.cat([d['raw'].flatten() for d in entries]);weight=entries[0]['weight'];weighted=raw*weight
        quant=torch.quantile(raw.double(),torch.tensor([.1,.9,.99],dtype=torch.double))
        rows.append(dict(name=name,profile=profile,test_kind='unqualified_real_interface_diagnostic',samples=raw.numel(),raw_mean=float(raw.mean()),raw_std=float(raw.std(unbiased=False)),raw_p10=float(quant[0]),raw_p90=float(quant[1]),raw_p99=float(quant[2]),weight=weight,weighted_step_mean=float(weighted.mean())*env.step_dt,weighted_per_second_mean=float(weighted.mean()),positive_step_mean=float((weighted*env.step_dt).clamp_min(0).mean()),negative_step_mean=float((weighted*env.step_dt).clamp_max(0).mean())))
    with (out/'reward_component_distribution.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    (out/'interface_result.json').write_text(json.dumps(dict(status='pass_interface_only',task_id=task_id,num_envs=args.num_envs,physics_dt=env.physics_dt,control_dt=env.step_dt,observation_shape=list(obs['policy'].shape),action_shape=list(actions.shape),target_substeps_equal=True,partial_reset_target_equal_measured=True,other_partial_reset_targets_unchanged=True,object_top_reference=obj.body_names.index('top'),robot_inertia_min=params['robot']['min_inertia_eigenvalue'],object_inertia_min=params['object']['min_inertia_eigenvalue'],formal_candidate_qualified=False,ppo_updates=0),indent=2)+'\n')
    print('OPT8_INTERFACE_PASS',task_id,args.num_envs,args.decimation,flush=True)
finally:
    env.close()
    app.close()

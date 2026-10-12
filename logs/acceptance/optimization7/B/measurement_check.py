"""优化7真实多环境：top/root计量、首次失败冻结、post-reset隔离和接触审计。"""
import argparse
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from isaaclab.app import AppLauncher
parser=argparse.ArgumentParser()
parser.add_argument('--decimation',type=int,choices=(2,4),required=True)
parser.add_argument('--num_envs',type=int,default=16)
parser.add_argument('--copies',action='store_true')
parser.add_argument('--output_dir',type=Path,required=True)
AppLauncher.add_app_launcher_args(parser)
args=parser.parse_args()
app=AppLauncher(args).app
import gymnasium as gym
import torch
import Grasp1.tasks
from Grasp1.tasks.manager_based.grasp1.config.ur5_allegro.teacher_env_cfg import UR5AllegroTeacherEnvCfg
from Grasp1.tasks.manager_based.grasp1.config.ur5_allegro.agents.rsl_rl_ppo_cfg import UR5AllegroTeacherPPORunnerCfg
from Grasp1.utils.paths import object_asset_dir
sys.path.insert(0,str(Path('scripts/rsl_rl').resolve()))
from grasp_diagnostics import EvaluationTrajectories, PhysicsDiagnostics, configure_diagnostics
from checkpoint_contract import build_contract, save_metadata, validate_checkpoint
out=args.output_dir; out.mkdir(parents=True,exist_ok=True)
cfg=UR5AllegroTeacherEnvCfg(decimation=args.decimation,episode_length_s=10.)
cfg.set_num_envs(args.num_envs)
if args.copies:
    name='wood_block_oriented'
    cfg.object_names=(name,)*args.num_envs
    cfg.scene.object.spawn.usd_path=[str(object_asset_dir(cfg.object_dataset,name)/f'{name}.usd')]*args.num_envs
    cfg.events.initialize_teacher_data.params['object_names']=cfg.object_names
    cfg.events.reset_teacher.params['object_names']=cfg.object_names
cfg.synchronize_control_timing()
configure_diagnostics(cfg.scene)
env=gym.make('Grasp1-UR5-Allegro-Teacher-v0',cfg=cfg).unwrapped
trials=physics=None
raw_scene_update=env.scene.update
try:
    env.reset(seed=1)
    obj,robot=env.scene['object'],env.scene['robot']
    term=env.action_manager.get_term('teacher')
    zero=torch.zeros((env.num_envs,22),device=env.device)
    settings=SimpleNamespace(diagnostics=True,paper_stability=False,success_height=0.1,
              max_relative_translation_drift_m=0.02,max_relative_rotation_drift_rad=0.2)
    trials=EvaluationTrajectories(env,cfg.object_names,settings)
    trials.start_round(0); trials.phase='grasp'
    initial=trials.initial_z.clone()
    original_height=env.termination_manager.get_term_cfg('invalid_hand_height')
    original_func=original_height.func
    # 在真正物理积分后、termination/reset前插入一个已知top高度。
    last={}
    injection={'terminal_height':True,'post_reset_velocity':False}
    def inject_terminal_height(dt):
        raw_scene_update(dt)
        if dt==env.physics_dt:
            # 用官方root写入+forward获得top link新姿态，位置有限，不改变关节结构。
            if injection['terminal_height'] and env._sim_step_counter % args.decimation == 0:
                pose=obj.data.root_pose_w[0:1].clone(); pose[:,2]+=0.2
                obj.write_root_pose_to_sim(pose,env_ids=torch.tensor([0],device=env.device))
                env.sim.forward()
                last['top']=obj.data.body_link_pose_w[0,trials.top_id].clone()
            if injection['post_reset_velocity']:
                # 只注入当前读取帧的GPU测量值，物理solver下一帧重新产生状态。
                robot.data.joint_vel[0,term._joint_ids[-1]]=123.0
    env.scene.update=inject_terminal_height
    physics=PhysicsDiagnostics(env,trials)
    original_height.func=lambda e,**kw:torch.arange(e.num_envs,device=e.device)==0
    env.termination_manager.set_term_cfg('invalid_hand_height',original_height)
    try:
        _,_,done,_,_=env.step(zero)
        assert bool(done[0])
    finally:
        injection['terminal_height']=False
        original_height.func=original_func
        env.termination_manager.set_term_cfg('invalid_hand_height',original_height)
    trials.sample('grasp')
    frozen=trials.final_object_pose[0].clone()
    assert torch.equal(frozen,last['top']) and not bool(trials.active[0])
    assert bool(trials.active[1:].all())
    assert float(frozen[2]-initial[0])>0.1
    bank=physics.trial_peaks['active_trial_peak']
    active_before=bank.velocity[0].clone()
    failed=physics.trial_peaks['failed_before_reset_peak']
    assert int(failed.count[0])==1 and int(failed.count[1:].sum())==0
    # 新episode可控速度注入用于诊断隔离测试；不作为物理安全样本。
    injection['post_reset_velocity']=True
    trials.phase='lift'
    try:
        env.step(zero)
    finally:
        injection['post_reset_velocity']=False
    trials.sample('lift')
    assert torch.equal(bank.velocity[0],active_before)
    assert float(physics.trial_peaks['post_reset_peak'].velocity[0,-1])==123.
    assert torch.equal(trials.final_object_pose[0],frozen)
    source=trials.final_object_pose[:,2]-trials.initial_z>0.1
    trials.finish_round(source,trials.active & source)
    assert trials.rows[0]['source_height_success'] and not trials.rows[0]['active_height_success']
    trials.save(out)
    measured_diff=(obj.data.body_link_pose_w[:,trials.top_id,2]-obj.data.root_pos_w[:,2]).tolist()
    result=dict(status='pass',decimation=args.decimation,num_envs=env.num_envs,copies=args.copies,
                first_failed_top_frozen=True,other_environments_alive=True,active_peak_unchanged=True,
                failed_last_frame_recorded=True,post_reset_injected_peak_rad_s=123.,
                spike_kind='GPU metric injection after a real early termination and automatic reset; not a physical safety sample',
                actual_top_minus_root_z_m=measured_diff,physics=physics.summary())
    # 非零offset且body顺序反转的合成输入，运行实际EvaluationTrajectories消费者。
    synthetic_link=torch.zeros((2,2,7),device=env.device);synthetic_link[:,:,3]=1.
    synthetic_link[:,0,2]=.75;synthetic_link[:,1,2]=.70
    synthetic_root=synthetic_link[:,1].clone()
    wrist=torch.zeros((2,1,7),device=env.device);wrist[:,:,3]=1.
    synthetic_scene={'object':SimpleNamespace(body_names=['top','bottom'],data=SimpleNamespace(
        body_link_pose_w=synthetic_link,root_pose_w=synthetic_root,root_pos_w=synthetic_root[:,:3])),
        'robot':SimpleNamespace(data=SimpleNamespace(body_link_pose_w=wrist))}
    for index in range(13):
        synthetic_scene[f'teacher_af_contact_{index}']=SimpleNamespace(data=SimpleNamespace(
            force_matrix_w=torch.zeros((2,1,1,3),device=env.device)))
    synthetic_env=SimpleNamespace(scene=synthetic_scene,device=env.device,num_envs=2,step_dt=env.step_dt,
        _teacher_body_indices=SimpleNamespace(wrist=0),_reset_idx=lambda ids:None,
        termination_manager=SimpleNamespace(active_terms=[]))
    synthetic_args=SimpleNamespace(diagnostics=False,paper_stability=False,success_height=.1,
        max_relative_translation_drift_m=.02,max_relative_rotation_drift_rad=.2)
    synthetic=EvaluationTrajectories(synthetic_env,('offset_0','offset_1'),synthetic_args)
    try:
        synthetic.start_round(0)
        synthetic_link[:,0,2]+=.2
        synthetic.sample('grasp')
        top_success=synthetic.final_object_pose[:,2]-synthetic.initial_z>.1
        root_success=synthetic.final_root_pose[:,2]-synthetic.initial_root_z>.1
        assert top_success.all() and not root_success.any()
        synthetic.finish_round(top_success,synthetic.active & top_success)
        result['top_root_distinct_classifier']=dict(status='pass',body_order=['top','bottom'],rows=synthetic.rows)
    finally:
        synthetic.close()
    # 从当前真实配置建立可JSON序列化的完整契约，并执行同语义正例。
    agent=UR5AllegroTeacherPPORunnerCfg(); agent.synchronize_control_timing(env.step_dt)
    contract=build_contract('Grasp1-UR5-Allegro-Teacher-v0',cfg,agent)
    test_checkpoint=out/'contract_only_model.pt'
    save_metadata(test_checkpoint,contract,99)
    audit,_=validate_checkpoint(test_checkpoint,contract,resume=True)
    result['actual_configuration_contract']=audit
    # 世界方向/单位测试；运行时接触样本单列，不改变默认Reward。
    n=torch.tensor([[0.,0.,12.],[12.,0.,0.],[8.485281,0.,8.485281],[0.,0.,-12.]],device=env.device)
    f=torch.tensor([[3.,0.,0.],[0.,0.,3.],[2.121320,0.,-2.121320],[3.,0.,0.]],device=env.device)
    impulse=(n+f)*env.physics_dt
    current=torch.stack((f.norm(dim=-1),n.norm(dim=-1)),1)*env.physics_dt
    source_components=torch.stack((impulse[:,:2].norm(dim=-1),impulse[:,2]),1)
    assert torch.allclose(current[0],source_components[0])
    assert not torch.allclose(current[1:],source_components[1:])
    result['contact_projection']={'current_local_norm_impulses':current.tolist(),'source_world_xy_signed_z':source_components.tolist(),
                                  'directions':['axis_aligned_z','axis_aligned_x','inclined_45_degrees','negative_world_z'],
                                  'force_to_impulse_dt':env.physics_dt,'default_reward_changed':False}
    (out/'measurement_regression.json').write_text(json.dumps(result,indent=2,allow_nan=False))
    print('OPT7_MEASUREMENT_PASS',args.decimation,args.num_envs,flush=True)
finally:
    if physics is not None: physics.close()
    env.scene.update=raw_scene_update
    if trials is not None: trials.close()
    env.close(); app.close()

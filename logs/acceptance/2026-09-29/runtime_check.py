"""本轮验收的诊断记录脚本，不参与训练。"""
import argparse
from isaaclab.app import AppLauncher
parser=argparse.ArgumentParser()
parser.add_argument('--num_envs',type=int,default=1)
parser.add_argument('--steps',type=int,default=140)
parser.add_argument('--mode',choices=['zero','random'],default='zero')
AppLauncher.add_app_launcher_args(parser)
args=parser.parse_args()
app=AppLauncher(args).app
import time
import torch
import gymnasium as gym
import Grasp1.tasks
from isaaclab_tasks.utils import parse_env_cfg
try:
    cfg=parse_env_cfg('Grasp1-UR5-Allegro-Teacher-v0',device='cuda:0',num_envs=args.num_envs)
    cfg.set_num_envs(args.num_envs)
    cfg.viewer.eye=(2.2,-3.0,2.0)
    cfg.viewer.lookat=(0.2,-0.65,0.85)
    env=gym.make('Grasp1-UR5-Allegro-Teacher-v0',cfg=cfg)
    e=env.unwrapped
    obs,_=env.reset()
    print('RESET_PASS',flush=True)
    print('JOINTS',e.scene['robot'].joint_names,flush=True)
    print('BODIES',e.scene['robot'].body_names,flush=True)
    for name,term in e.action_manager._terms.items():
        print('ACTION',name,term._joint_ids,term._scale,flush=True)
    for name,sensor in e.scene.sensors.items():
        print('SENSOR',name,tuple(sensor.data.force_matrix_w.shape),flush=True)
    start=time.monotonic()
    done_count=0
    term_counts={name:0 for name in e.termination_manager.active_terms}
    reward_min=torch.full((len(e.reward_manager.active_terms),),float('inf'),device=e.device)
    reward_max=-reward_min.clone()
    for step in range(args.steps):
        with torch.inference_mode():
            actions=torch.zeros((args.num_envs,e.action_manager.total_action_dim),device=e.device)
            if args.mode=='random':
                actions.uniform_(-1,1)
            obs,reward,terminated,truncated,extra=env.step(actions)
            assert obs['policy'].shape==(args.num_envs,153)
            assert obs['policy'].dtype==torch.float32 and obs['policy'].device.type=='cuda'
            for label,x in [('obs',obs['policy']),('reward',reward),('robot',e.scene['robot'].data.joint_pos),('robot_vel',e.scene['robot'].data.joint_vel),('object',e.scene['object'].data.root_state_w)]:
                assert torch.isfinite(x).all(),(step,label)
            for name in term_counts:
                term_counts[name]+=int(e.termination_manager.get_term(name).sum())
            done_count+=int((terminated|truncated).sum())
            terms=e.reward_manager._step_reward
            assert torch.isfinite(terms).all()
            reward_min=torch.minimum(reward_min,terms.min(dim=0).values)
            reward_max=torch.maximum(reward_max,terms.max(dim=0).values)
            if step%20==0:
                print('STEP',step,'obs_range',float(obs['policy'].min()),float(obs['policy'].max()),'reward',reward[:2].tolist(),'done',done_count,'object',e.scene['object'].data.root_pos_w[:2].tolist(),flush=True)
    print('RUN_PASS',args.mode,args.num_envs,args.steps,'seconds',time.monotonic()-start,'done',done_count,'terminations',term_counts,flush=True)
    print('REWARD_RANGES',list(zip(e.reward_manager.active_terms,reward_min.tolist(),reward_max.tolist())),flush=True)
    for name in e.reward_manager.active_terms:
        term=e.reward_manager.get_term_cfg(name)
        value=term.func(e,**term.params)
        assert value.shape==(args.num_envs,) and value.dtype==torch.float32 and value.device.type=='cuda'
        assert torch.isfinite(value).all()
        print('RAW_REWARD',name,'weight',term.weight,'range',float(value.min()),float(value.max()),flush=True)
    for asset_name in ['robot','object']:
        asset=e.scene[asset_name]
        masses=asset.root_physx_view.get_masses()
        inertia=asset.root_physx_view.get_inertias()
        assert torch.isfinite(masses).all() and (masses>=0).all() and (masses>0).any()
        assert torch.isfinite(inertia).all()
        print('MASS_INERTIA',asset_name,'mass_range',float(masses.min()),float(masses.max()),'inertia_finite',True,flush=True)
        q=asset.data.root_quat_w
        assert torch.allclose(q.norm(dim=-1),torch.ones(args.num_envs,device=e.device),atol=1e-4)
    # 单步动作映射检查：缩放后的残差应按已解析的关节索引加入当前关节位置。
    for name,term in e.action_manager._terms.items():
        action=torch.ones_like(term.raw_actions)
        before=term._asset.data.joint_pos[:,term._joint_ids].clone()
        term.process_actions(action)
        expected=(before+term._scale).clamp(term._joint_pos_limits[...,0],term._joint_pos_limits[...,1])
        assert torch.allclose(term.processed_actions,expected)
    print('ACTION_MAPPING_PASS',flush=True)
    from isaaclab.utils.math import quat_apply,quat_mul
    from Grasp1.tasks.manager_based.grasp1.mdp.keypoints import hand_keypoints_w
    term=e.observation_manager._group_obs_term_cfgs['policy'][0].func
    wrist_q=quat_mul(e.scene['robot'].data.body_link_pose_w[:,term._wrist_body_id,3:7],term._wrist_offset_quat)
    local_center=hand_keypoints_w(e)[:,0]+quat_apply(wrist_q,term._hand_center.expand(args.num_envs,-1))-e.scene.env_origins
    current=e.observation_manager.compute_group('policy')
    assert torch.allclose(current[:,93:96],local_center,atol=1e-5)
    print('LOCAL_OBSERVATION_PASS',current[:2,93:96].tolist(),flush=True)
    if args.num_envs == 1:
        from Grasp1.tasks.manager_based.grasp1.mdp import rewards as rw
        from Grasp1.tasks.manager_based.grasp1.config.ur5_allegro.teacher_env_cfg import FINGER_REWARD_WEIGHTS, CONTACT_REWARD_WEIGHTS
        from types import SimpleNamespace
        # 构造输入只用于奖励方向单元检查，不替换运行环境中的 reward/termination。
        fixture=torch.zeros((1,153),device=e.device)
        fake=SimpleNamespace(num_envs=1,observation_manager=SimpleNamespace(compute_group=lambda name: fixture))
        fixture[:,102:153]=1.0
        far=rw.affordance_reward(fake,FINGER_REWARD_WEIGHTS)
        fixture[:,102:153]=0.01
        near=rw.affordance_reward(fake,FINGER_REWARD_WEIGHTS)
        assert (near>far).all()
        print('REACH_DIRECTION_PASS',far.tolist(),near.tolist(),flush=True)
        saved=rw.hand_keypoints_w
        points=torch.zeros((1,17,3),device=e.device)
        rw.hand_keypoints_w=lambda env:points
        points[...,2]=0.791
        clear=rw.table_reward(fake,FINGER_REWARD_WEIGHTS,0.771)
        points[...,2]=0.772
        close=rw.table_reward(fake,FINGER_REWARD_WEIGHTS,0.771)
        rw.hand_keypoints_w=saved
        assert (close>clear).all()  # 实际 weight 为负，靠近桌面的贡献更低。
        saved=rw._filtered_impulses
        impulses=torch.zeros((1,13),device=e.device)
        rw._filtered_impulses=lambda *args:(impulses,impulses,impulses)
        no_contact=rw.affordance_contact_reward(fake,CONTACT_REWARD_WEIGHTS)
        impulses[:]=0.02
        contact=rw.affordance_contact_reward(fake,CONTACT_REWARD_WEIGHTS)
        assert (contact>no_contact).all()
        assert (rw.affordance_impulse_reward(fake,CONTACT_REWARD_WEIGHTS)>0).all()
        assert (rw.table_contact_reward(fake,CONTACT_REWARD_WEIGHTS)>0).all()
        assert (rw.table_impulse_reward(fake,CONTACT_REWARD_WEIGHTS)>0).all()
        rw._filtered_impulses=saved
        print('TABLE_CONTACT_DIRECTION_PASS',flush=True)
    if args.num_envs >= 16:
        ids=torch.tensor([3,7,11],device=e.device)
        keep=torch.ones(args.num_envs,dtype=torch.bool,device=e.device);keep[ids]=False
        snapshots={}
        for asset_name in ['robot','object']:
            asset=e.scene[asset_name]
            for field in ['root_state_w','joint_pos','joint_vel']:
                snapshots[(asset_name,field)]=getattr(asset.data,field).clone()
        buffers={name:value.clone() for name,value in e._teacher_reset_data.items() if isinstance(value,torch.Tensor) and value.ndim and value.shape[0]==args.num_envs}
        episode=e.episode_length_buf.clone()
        with torch.inference_mode():
            e._reset_idx(ids)
        for (asset_name,field),before in snapshots.items():
            assert torch.equal(getattr(e.scene[asset_name].data,field)[keep],before[keep]),(asset_name,field)
        for name,before in buffers.items():
            assert torch.equal(e._teacher_reset_data[name][keep],before[keep]),name
        assert torch.equal(e.episode_length_buf[keep],episode[keep])
        assert (e.episode_length_buf[ids]==0).all()
        for asset_name in ['robot','object']:
            assert (e.scene[asset_name].data.joint_vel[ids]==0).all()
        assert (e.scene['object'].data.root_vel_w[ids]==0).all()
        print('PARTIAL_RESET_PASS',[3,7,11],flush=True)
        from Grasp1.tasks.manager_based.grasp1.mdp import terminations as done_terms
        from isaaclab.envs import mdp as common_mdp
        from types import SimpleNamespace
        fake=SimpleNamespace(num_envs=args.num_envs,device=e.device,obs_buf={'policy':torch.zeros((args.num_envs,153),device=e.device)},episode_length_buf=torch.zeros(args.num_envs,device=e.device),max_episode_length=e.max_episode_length)
        fake.observation_manager=SimpleNamespace(compute_group=lambda name:fake.obs_buf[name])
        fake.obs_buf['policy'][3,0]=float('nan')
        invalid=done_terms.invalid_observation(fake,('policy',))
        assert invalid.nonzero().flatten().tolist()==[3]
        fake.episode_length_buf[7]=e.max_episode_length
        assert common_mdp.time_out(fake).nonzero().flatten().tolist()==[7]
        points=torch.ones((args.num_envs,17,3),device=e.device)
        points[11,0,2]=0.7
        low=done_terms.invalid_hand_height(fake,lambda env:points,0.771)
        assert low.nonzero().flatten().tolist()==[11]
        print('TERMINATION_CASES_PASS',flush=True)

    env.close()
except BaseException:
    import traceback
    traceback.print_exc()
    env.close()
    raise
finally:
    app.close()

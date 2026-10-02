"""桌边安装任务的实际 Isaac Sim 验收；不参与训练。"""
import argparse
import faulthandler
faulthandler.dump_traceback_later(60, repeat=True)
from isaaclab.app import AppLauncher
parser=argparse.ArgumentParser()
parser.add_argument('--num_envs',type=int,default=1)
parser.add_argument('--resets',type=int,default=100)
parser.add_argument('--baseline',action='store_true')
parser.add_argument('--output',required=True)
parser.add_argument('--screenshot',action='store_true')
AppLauncher.add_app_launcher_args(parser)
args=parser.parse_args()
app=AppLauncher(args).app
import json
from pathlib import Path
import numpy as np
import torch
import gymnasium as gym
from pxr import Usd, UsdGeom, Gf, UsdPhysics
import Grasp1.tasks
from isaaclab_tasks.utils import parse_env_cfg
from Grasp1.tasks.manager_based.grasp1.mdp import events,rewards
from isaaclab.utils.math import quat_apply
cfg=parse_env_cfg('Grasp1-UR5-Allegro-Teacher-v0',device='cuda:0',num_envs=args.num_envs)
cfg.set_num_envs(args.num_envs)
env=gym.make('Grasp1-UR5-Allegro-Teacher-v0',cfg=cfg,render_mode='rgb_array' if args.screenshot else None)
e=env.unwrapped
print("ENV_READY",flush=True)
result={'num_envs':args.num_envs,'resets':args.resets,'baseline':args.baseline}
collision_counts=[]
original_fallback=events._apply_collision_fallback
def record_fallback(*call_args,**kwargs):
    collision_counts.append(int(call_args[3].sum()))
    return original_fallback(*call_args,**kwargs)
events._apply_collision_fallback=record_fallback
feasible=[]
relative=[]
support_errors=[]
post_collision=[]
robot=e.scene['robot']
obj=e.scene['object']
# 静态 USD 中的每个物体碰撞网格转换到其 articulation root frame。
mesh_points=[]
for path in cfg.scene.object.spawn.usd_path:
    asset_stage=Usd.Stage.Open(path)
    xf=UsdGeom.XformCache()
    pts=[]
    for prim in Usd.PrimRange.Stage(asset_stage,Usd.TraverseInstanceProxies()):
        if prim.IsA(UsdGeom.Mesh) and '/collisions/' in str(prim.GetPath()):
            matrix=xf.GetLocalToWorldTransform(prim)
            pts.extend([matrix.Transform(Gf.Vec3d(*p)) for p in UsdGeom.Mesh(prim).GetPointsAttr().Get()])
    mesh_points.append(torch.tensor(np.asarray(pts),device=e.device,dtype=torch.float32))
    print('OBJECT_COLLISION_POINTS',path,len(pts),flush=True)
try:
    with torch.inference_mode():
        for i in range(args.resets):
            obs,_=env.reset(seed=1000+i)
            data=e._teacher_reset_data
            feasible.extend(e._teacher_last_ik_feasible.cpu().tolist())
            rel=obj.data.root_pos_w-robot.data.root_pos_w
            relative.extend(rel.cpu().tolist())
            planar=torch.linalg.vector_norm(rel[:,:2],dim=-1)
            assert (rel[:,0]>0).all() and (planar>=.45-1e-5).all() and (planar<=.75+1e-5).all()
            from Grasp1.tasks.manager_based.grasp1.grasp1_env_cfg import TABLE_XY_BOUNDS
            local_object=obj.data.root_pos_w-e.scene.env_origins
            assert (local_object[:,0]>=TABLE_XY_BOUNDS[0]).all() and (local_object[:,0]<=TABLE_XY_BOUNDS[1]).all()
            assert (local_object[:,1]>=TABLE_XY_BOUNDS[2]).all() and (local_object[:,1]<=TABLE_XY_BOUNDS[3]).all()
            for j,points in enumerate(mesh_points):
                world=quat_apply(obj.data.root_quat_w[j].expand(len(points),-1),points)+obj.data.root_pos_w[j]
                support_errors.append(float(world[:,2].min()-e.scene.env_origins[j,2]-0.771))
            obs,reward,terminated,truncated,extra=env.step(torch.zeros((args.num_envs,22),device=e.device))
            assert obs['policy'].shape==(args.num_envs,153)
            assert torch.isfinite(obs['policy']).all() and torch.isfinite(reward).all()
            e.sim.step(render=False)
            e.scene.update(e.physics_dt)
            post_collision.extend(events._current_arm_collision(e,torch.arange(args.num_envs,device=e.device)).cpu().tolist())
            if i%10==0:
                print('RESET',i,'IK',sum(feasible),'collision_fallback',sum(collision_counts),flush=True)
        result.update(ik_success=sum(feasible),ik_total=len(feasible),collision_fallback=sum(collision_counts),post_fallback_collision=sum(post_collision),object_support_error_min=min(support_errors),object_support_error_max=max(support_errors),relative_vectors=relative)
        stage=e.sim.stage
        result['root_positions']=(robot.data.root_pos_w-e.scene.env_origins).tolist()
        result['root_quaternions']=robot.data.root_quat_w.tolist()
        result['sensors']={name:{'filters':sensor.cfg.filter_prim_paths_expr,'shape':list(sensor.data.force_matrix_w.shape)} for name,sensor in e.scene.sensors.items()}
        assert not any(p.GetName() in ('Mat','WoodenTable') for p in stage.Traverse())
        if not args.baseline:
            assert not stage.GetPrimAtPath('/World/envs/env_0/Robot/platform_base_link')
            from Grasp1.tasks.manager_based.grasp1.grasp1_env_cfg import TABLE_XY_BOUNDS,SUPPORT_HEIGHT,ROBOT_ROOT_POSITION
            base_bounds=[]
            for j in range(args.num_envs):
                base=stage.GetPrimAtPath(f'/World/envs/env_{j}/Robot/base_link')
                xf=UsdGeom.XformCache()
                matrix=xf.GetLocalToWorldTransform(base)
                asset_stage=Usd.Stage.Open(cfg.scene.robot.spawn.usd_path)
                asset_xf=UsdGeom.XformCache()
                asset_base=asset_stage.GetPrimAtPath('/ur5/base_link')
                pts=[]
                for p in Usd.PrimRange(asset_base,Usd.TraverseInstanceProxies()):
                    if p.IsA(UsdGeom.Mesh):
                        local=asset_xf.GetLocalToWorldTransform(p)*asset_xf.GetLocalToWorldTransform(asset_base).GetInverse()
                        pts.extend([(local*matrix).Transform(Gf.Vec3d(*v)) for v in UsdGeom.Mesh(p).GetPointsAttr().Get()])
                arr=np.asarray(pts)-np.asarray(e.scene.env_origins[j].cpu())
                lo,hi=arr.min(0),arr.max(0)
                assert abs(lo[2]-SUPPORT_HEIGHT)<.002,(lo,hi)
                assert lo[0]>=TABLE_XY_BOUNDS[0]-.002 and hi[0]<=TABLE_XY_BOUNDS[1]+.002
                assert lo[1]>=TABLE_XY_BOUNDS[2]-.002 and hi[1]<=TABLE_XY_BOUNDS[3]+.002
                base_bounds.append({'min':lo.tolist(),'max':hi.tolist()})
            assert torch.allclose(robot.data.root_pos_w-e.scene.env_origins,torch.tensor(ROBOT_ROOT_POSITION,device=e.device).expand(args.num_envs,-1),atol=1e-5)
            result['base_bounds']=base_bounds
            if not args.screenshot:
                # 实际接触验收：把手部移入桌面，再执行物理步，读取真实传感器和奖励原始项。
                root_pose=robot.data.root_link_pose_w.clone()
                from Grasp1.tasks.manager_based.grasp1.mdp.keypoints import hand_keypoints_w
                from Grasp1.tasks.manager_based.grasp1.grasp1_env_cfg import TABLE_CENTER_XY
                hand_points=hand_keypoints_w(e)
                test_pose=root_pose.clone()
                test_pose[:,:2]+=test_pose.new_tensor(TABLE_CENTER_XY)+e.scene.env_origins[:,:2]-hand_points.mean(dim=1)[:,:2]
                test_pose[:,2]+=SUPPORT_HEIGHT+e.scene.env_origins[:,2]-hand_points[:,:,2].amin(dim=1)-.015
                robot.write_root_pose_to_sim(test_pose)
                contacts=torch.zeros(args.num_envs,device=e.device)
                impulses=torch.zeros_like(contacts)
                for _ in range(10):
                    e.sim.step(render=False);e.scene.update(e.physics_dt)
                    contacts=torch.maximum(contacts,rewards.table_contact_reward(e,cfg.rewards.table_contact_reward.params['contact_weights']))
                    impulses=torch.maximum(impulses,rewards.table_impulse_reward(e,cfg.rewards.table_impulse_reward.params['contact_weights']))
                result['support_contact_raw']=contacts.tolist()
                result['support_impulse_raw']=impulses.tolist()
                robot.write_root_pose_to_sim(root_pose)
                assert max(result['support_contact_raw'])>0 and max(result['support_impulse_raw'])>0,result
        if args.screenshot:
            obs,_=env.reset(seed=1005)
            env.step(torch.zeros((args.num_envs,22),device=e.device))
            e.sim.set_camera_view(eye=(1.8,.9,2.5),target=(.1,-.7,.75))
            frame=env.render()
            for _ in range(20):
                e.sim.render()
                frame=env.render()
            assert frame.max()>0,'RGB render has no scene pixels'
            from PIL import Image
            Image.fromarray(frame).save(str(Path(args.output).with_name('scene-close.png')))
        Path(args.output).write_text(json.dumps(result,indent=2))
        print('ACCEPTANCE_PASS',json.dumps({k:v for k,v in result.items() if k not in ('relative_vectors','sensors')}),flush=True)
except BaseException:
    import traceback
    traceback.print_exc()
    raise
finally:
    env.close()
    faulthandler.cancel_dump_traceback_later()
    app.close()

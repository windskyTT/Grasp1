"""优化3 Phase 1：源资产审计与单环境逐物理步约束测试。"""
import argparse
from pathlib import Path
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='new_training_set')
parser.add_argument('--object', default='003_cracker_box')
parser.add_argument('--condition', choices=['free', 'table', 'contact'], default='free')
parser.add_argument('--velocity_iterations', type=int, default=0)
parser.add_argument('--output', required=True)
parser.add_argument('--audit', action='store_true')
parser.add_argument('--contact_last', action='store_true')
parser.add_argument('--passive', action='store_true')
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import json
import math
import xml.etree.ElementTree as ET
import torch
import gymnasium as gym
from pxr import Usd, UsdPhysics
import Grasp1.tasks
from isaaclab_tasks.utils import parse_env_cfg
from Grasp1.tasks.manager_based.grasp1.config.ur5_allegro.teacher_env_cfg import _build_teacher_object_cfg
from Grasp1.tasks.manager_based.grasp1.mdp import keypoints
from isaaclab.utils.math import quat_apply

out = Path(args.output)
if args.audit:
    rows = []
    for path in sorted(Path('assets/objects').glob('*/*/*.urdf')):
        if path.stem != path.parent.name:
            continue
        urdf = ET.parse(path).getroot()
        source = Path('/home/windsky/project/RobustDexGrasp/rsc') / path.relative_to('assets/objects')
        joint = urdf.find('joint')
        source_joint = ET.parse(source).getroot().find('joint')
        stage = Usd.Stage.Open(str(path.with_suffix('.usd')))
        joints, roots, bodies = [], [], []
        for prim in stage.Traverse():
            if prim.IsA(UsdPhysics.Joint):
                joints.append(dict(path=str(prim.GetPath()), type=prim.GetTypeName(),
                    attributes={a.GetName():str(a.Get()) for a in prim.GetAttributes()},
                    relationships={r.GetName():[str(t) for t in r.GetTargets()] for r in prim.GetRelationships()},
                    apis=list(prim.GetAppliedSchemas())))
                j = UsdPhysics.RevoluteJoint(prim)
                assert math.isclose(math.radians(j.GetUpperLimitAttr().Get()), float(joint.find('limit').get('upper')), abs_tol=1e-9)
                assert math.isclose(math.radians(j.GetLowerLimitAttr().Get()), float(joint.find('limit').get('lower')), abs_tol=1e-9)
            if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
                roots.append(str(prim.GetPath()))
            if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                bodies.append(dict(path=str(prim.GetPath()), attributes={a.GetName():str(a.Get()) for a in prim.GetAttributes() if a.GetName().startswith('physics:')}))
        assert len(roots) == 1 and roots[0].endswith('/bottom')
        assert source_joint.find('limit').attrib == joint.find('limit').attrib
        rows.append(dict(asset=str(path), source_asset=str(source), urdf_joint=dict(joint.attrib,
            parent=joint.find('parent').attrib, child=joint.find('child').attrib, axis=joint.find('axis').attrib,
            limit=joint.find('limit').attrib, dynamics=joint.find('dynamics').attrib),
            usd_joints=joints, articulation_roots=roots, bodies=bodies))
    out.with_suffix('.assets.json').write_text(json.dumps(rows, indent=2))
    print('ASSET_AUDIT', len(rows), flush=True)

cfg = parse_env_cfg('Grasp1-UR5-Allegro-Teacher-v0', num_envs=1)
cfg.set_num_envs(1)
cfg.seed = 42
cfg.sim.physx.solve_articulation_contact_last = args.contact_last
cfg.object_dataset = args.dataset
cfg.object_names = (args.object,)
cfg.scene.object = _build_teacher_object_cfg(cfg.object_names, dataset_name=args.dataset)
cfg.scene.object.spawn.articulation_props.solver_velocity_iteration_count = args.velocity_iterations
for event in [cfg.events.initialize_teacher_data, cfg.events.reset_teacher]:
    event.params.update(dataset_name=args.dataset, object_names=cfg.object_names)
env = gym.make('Grasp1-UR5-Allegro-Teacher-v0', cfg=cfg)
e = env.unwrapped
robot, obj = e.scene['robot'], e.scene['object']
term = e.action_manager.get_term('teacher')
result = dict(dataset=args.dataset, object=args.object, condition=args.condition,
    velocity_iterations=args.velocity_iterations, joint_limits=obj.data.joint_pos_limits.tolist(),
    contact_last=args.contact_last, passive=args.passive,
    body_names=obj.body_names, joint_names=obj.joint_names, samples=[])
result['loaded_stiffness'] = obj.root_physx_view.get_dof_stiffnesses().tolist()
result['loaded_damping'] = obj.root_physx_view.get_dof_dampings().tolist()
result['loaded_armature'] = obj.root_physx_view.get_dof_armatures().tolist()
if args.passive:
    obj.write_joint_stiffness_to_sim(0.0)
    obj.write_joint_damping_to_sim(0.0)
with torch.inference_mode():
    env.reset(seed=42)
    pose = obj.data.root_pose_w.clone()
    if args.condition == 'free':
        pose[:, 0] += 5.0
        pose[:, 2] = 600.0  # 10秒自由落体仍不触地。
    elif args.condition == 'table':
        robot_pose = robot.data.root_pose_w.clone()
        robot_pose[:, 0] += 5.0
        robot.write_root_pose_to_sim(robot_pose)
    else:
        body = robot.body_names.index('link_3_0')
        center = e._teacher_affordance_points_o.mean(1)
        pose[:, :3] = robot.data.body_com_pos_w[:, body] - quat_apply(pose[:, 3:7], center)
        pose[:, 0] += 0.01
    obj.write_root_pose_to_sim(pose)
    obj.write_root_velocity_to_sim(torch.zeros((1,6), device=e.device))
    obj.write_joint_state_to_sim(torch.zeros_like(obj.data.joint_pos), torch.zeros_like(obj.data.joint_vel))
    e._teacher_runtime_features.invalidate()
    e.scene.write_data_to_sim()
    e.sim.forward()
    term.process_actions(torch.zeros((1,22), device=e.device))
    minimum, maximum, max_velocity, max_contact = 0.0, 0.0, 0.0, 0.0
    for step in range(1200):
        term.apply_actions()
        e.scene.write_data_to_sim()
        e.sim.step(render=False)
        e.scene.update(e.physics_dt)
        q, v = obj.data.joint_pos, obj.data.joint_vel
        assert torch.isfinite(q).all() and torch.isfinite(v).all() and torch.isfinite(obj.data.body_state_w).all()
        minimum = min(minimum, float(q.min()))
        maximum = max(maximum, float(q.max()))
        max_velocity = max(max_velocity, float(v.abs().max()))
        force = max(float(e.scene[f'teacher_af_contact_{i}'].data.force_matrix_w.norm(dim=-1).max()) for i in range(13))
        max_contact = max(max_contact, force)
        if step < 10 or step + 1 in [120,480,1200]:
            result['samples'].append(dict(time=(step+1)*e.physics_dt, q=q.tolist(), v=v.tolist(),
                body_pose=obj.data.body_link_pose_w.tolist(), body_velocity=obj.data.body_vel_w.tolist(),
                contact_force=force, min_so_far=minimum, max_so_far=maximum))
    result.update(min=minimum, max=maximum, max_joint_velocity=max_velocity, max_contact_force=max_contact,
                  max_limit_violation_rad=max(-minimum,maximum-0.001,0))
    env.reset(seed=42)
    obs,reward,*_ = env.step(torch.zeros((1,22), device=e.device))
    assert torch.isfinite(obs['policy']).all() and torch.isfinite(reward).all()
    result['reset_reward_finite'] = True
out.write_text(json.dumps(result, indent=2))
print('OBJECT_PROBE', json.dumps({k:v for k,v in result.items() if k != 'samples'}), flush=True)
env.close()
app.close()

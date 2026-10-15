"""源URDF、实际USD composition和运行时安装变换；不运行策略或修改资产。"""
import argparse
import csv
import json
from pathlib import Path
import xml.etree.ElementTree as ET

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--task', default='Grasp1-UR5-Allegro-GraspTrain-v2')
parser.add_argument('--seed', type=int, default=21)
parser.add_argument('--source_urdf', type=Path, default=Path('/home/windsky/project/RobustDexGrasp/rsc/ur5_allegro/ur5_allegro.urdf'))
parser.add_argument('--output_dir', type=Path, required=True)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import gymnasium as gym
import numpy as np
from scipy.spatial.transform import Rotation
import torch
from pxr import Usd, UsdGeom, UsdPhysics
from isaaclab_tasks.utils import parse_env_cfg
from isaaclab.utils.math import matrix_from_quat, quat_apply
import Grasp1.tasks
from Grasp1.robots.robot_profile import HAND_CENTER


def urdf_mount(path):
    root = ET.parse(path).getroot()
    names = ('base_link-base_fixed_joint', 'wrist_3_link-tool0_fixed_joint',
             'tool02Flange_fixed_joint', 'Flange2hand_fixed_joint')
    joints = {}
    for joint in root.findall('joint'):
        if joint.get('name') in names:
            origin = joint.find('origin')
            joints[joint.get('name')] = dict(parent=joint.find('parent').get('link'),
                child=joint.find('child').get('link'), xyz=list(map(float, origin.get('xyz').split())),
                rpy=list(map(float, origin.get('rpy').split())), type=joint.get('type'))
    palm = root.find("link[@name='Allegro_base_link']")
    return dict(path=str(path), joints=joints, palm_origins={kind:dict(palm.find(kind+'/origin').attrib)
        for kind in ('visual', 'collision', 'inertial')})


def pose_record(position, quaternion):
    p, q = position.cpu().numpy(), quaternion.cpu().numpy()
    rotation = Rotation.from_quat(q[[1, 2, 3, 0]])
    return dict(xyz_m=p.tolist(), quaternion_wxyz=q.tolist(), rpy_rad=rotation.as_euler('xyz').tolist(),
        axes_world_columns=rotation.as_matrix().tolist())


def relative_record(robot, parent, child):
    a, b = (robot.body_names.index(name) for name in (parent, child))
    poses = robot.data.body_link_pose_w[0]
    ra, rb = matrix_from_quat(poses[a, 3:7]), matrix_from_quat(poses[b, 3:7])
    position = ra.T @ (poses[b, :3]-poses[a, :3])
    rotation = (ra.T @ rb).cpu().numpy()
    xyzw = Rotation.from_matrix(rotation).as_quat()
    return dict(parent=parent, child=child, xyz_m=position.cpu().tolist(),
                quaternion_wxyz=xyzw[[3, 0, 1, 2]].tolist(),
                rpy_rad=Rotation.from_matrix(rotation).as_euler('xyz').tolist())


def main():
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    cfg = parse_env_cfg(args.task, device=args.device or 'cuda:0', num_envs=1)
    cfg.seed = args.seed
    cfg.set_num_envs(1)
    env = gym.make(args.task, cfg=cfg).unwrapped
    try:
        env.reset(seed=args.seed)
        robot = env.scene['robot']
        term = env.action_manager.get_term('teacher')
        saved_q = robot.data.joint_pos.clone()
        source = urdf_mount(args.source_urdf)
        current = urdf_mount(Path('assets/robots/ur5_allegro/ur5_allegro.urdf'))
        stage = env.sim.stage
        usd_joints = {}
        for prim in stage.Traverse():
            if '/Robot/joints/' in str(prim.GetPath()) and prim.IsA(UsdPhysics.FixedJoint):
                joint = UsdPhysics.FixedJoint(prim)
                usd_joints[prim.GetName()] = dict(body0=list(map(str, joint.GetBody0Rel().GetTargets())),
                    body1=list(map(str, joint.GetBody1Rel().GetTargets())),
                    localPos0=list(joint.GetLocalPos0Attr().Get()), localPos1=list(joint.GetLocalPos1Attr().Get()),
                    contributing_layers=[s.layer.identifier for s in prim.GetPrimStack()])
                for index in (0, 1):
                    quat = (joint.GetLocalRot0Attr() if index == 0 else joint.GetLocalRot1Attr()).Get()
                    usd_joints[prim.GetName()][f'localRot{index}_wxyz'] = [quat.GetReal(), *quat.GetImaginary()]
        cache = UsdGeom.XformCache()
        body = stage.GetPrimAtPath('/World/envs/env_0/Robot/Allegro_base_link')
        inverse = cache.GetLocalToWorldTransform(body).GetInverse()
        meshes = []
        for prim in Usd.PrimRange(body, Usd.TraverseInstanceProxies()):
            if prim.IsA(UsdGeom.Mesh):
                transform = cache.GetLocalToWorldTransform(prim)*inverse
                ancestor = prim
                collision = False
                while ancestor.GetPath() != body.GetPath():
                    collision |= ancestor.HasAPI(UsdPhysics.CollisionAPI)
                    ancestor = ancestor.GetParent()
                meshes.append(dict(path=str(prim.GetPath()), local_matrix_row_vectors=[list(row) for row in transform],
                    collision=collision, layers=[s.layer.identifier for s in prim.GetPrimStack()]))
        samples = []
        for name, arm_q in (('C1_seed21', saved_q[:, term._joint_ids[:6]]),
                            ('zero_arm', torch.zeros((1, 6), device=env.device)),
                            ('other_arm', saved_q[:, term._joint_ids[:6]]+saved_q.new_tensor([.15, -.1, .1, .05, -.1, .1]))):
            q = saved_q.clone()
            q[:, term._joint_ids[:6]] = arm_q
            robot.write_joint_state_to_sim(q, torch.zeros_like(q))
            term.reset()
            env.sim.forward()
            env.scene.update(0.0)
            frames = {link:pose_record(robot.data.body_link_pos_w[0, robot.body_names.index(link)],
                       robot.data.body_link_quat_w[0, robot.body_names.index(link)])
                for link in ('base_link', 'base', 'wrist_3_link', 'tool0', 'Flange_base_link', 'Allegro_base_link',
                             'link_3_0_tip', 'link_7_0_tip', 'link_11_0_tip', 'link_15_0_tip')}
            palm = robot.data.body_link_pose_w[:, robot.body_names.index('Allegro_base_link')]
            offset = palm.new_tensor(HAND_CENTER)[None]
            samples.append(dict(name=name, q=q[0, term._joint_ids].cpu().tolist(), frames=frames,
                flange_to_hand=relative_record(robot, 'Flange_base_link', 'Allegro_base_link'),
                tool_to_flange=relative_record(robot, 'tool0', 'Flange_base_link'),
                legacy_center_w=(palm[:, :3]+quat_apply(palm[:, 3:7], offset))[0].cpu().tolist(),
                geometric_center_w=(palm[:, :3]-quat_apply(palm[:, 3:7], offset))[0].cpu().tolist()))
        transforms = [sample['flange_to_hand'] for sample in samples]
        max_position = max(np.linalg.norm(t['xyz_m']) for t in transforms)
        max_angle = max(np.linalg.norm(t['rpy_rad']) for t in transforms)
        result = dict(task=args.task, quaternion_order='wxyz', source=source, current=current,
            source_urdf_matches_current=source['joints']==current['joints'] and source['palm_origins']==current['palm_origins'],
            actual_loaded_usd=cfg.scene.robot.spawn.usd_path,
            usd_layers=[layer.identifier for layer in stage.GetUsedLayers() if 'ur5_allegro' in layer.identifier],
            usd_fixed_joints=usd_joints, palm_mesh_transforms=meshes, runtime=samples,
            flange_hand_identity_position_error_m=float(max_position), flange_hand_identity_angle_error_rad=float(max_angle),
            numerical_mount_status='pass' if max_position < .001 and max_angle < np.deg2rad(1) else 'fail',
            visual_reference_status='needs_reference_images',
            asset_changed=False, position_tolerance_m=.001, angle_tolerance_deg=1,
            tolerance_basis='float32 PhysX body transforms; gate is 1mm/1deg',
            robot_joint_names=robot.joint_names, masses=robot.root_physx_view.get_masses().tolist(),
            inertias=robot.root_physx_view.get_inertias().tolist())
        (out/'mount_transforms.json').write_text(json.dumps(result, indent=2))
        with (out/'mount_transforms.csv').open('w', newline='') as handle:
            writer = csv.writer(handle)
            writer.writerow(['pose', 'frame', 'xyz_m', 'quaternion_wxyz', 'rpy_rad', 'axes_world_columns'])
            for sample in samples:
                for link, frame in sample['frames'].items():
                    writer.writerow([sample['name'], link, *[json.dumps(frame[k]) for k in
                        ('xyz_m', 'quaternion_wxyz', 'rpy_rad', 'axes_world_columns')]])
        robot.write_joint_state_to_sim(saved_q, torch.zeros_like(saved_q))
        term.reset()
        env.sim.forward()
        env.scene.update(0.0)
        if not args.headless:
            from Grasp1.tasks.manager_based.grasp1.mdp.mount_frames import MountFrames
            from omni.kit.viewport.utility import get_active_viewport, capture_viewport_to_file
            frames = MountFrames(env)
            frames.update()
            for label, eye in (('front', (1.35, -1.5, 1.45)), ('side', (.3, .5, 1.3)), ('top', (.3, -.7, 2.5))):
                env.sim.set_camera_view(eye, (.25, -.75, .98))
                for _ in range(15):
                    env.sim.render()
                capture = capture_viewport_to_file(get_active_viewport(), str(out/f'{label}.png'))
                for _ in range(20):
                    app.update()
        print('MOUNT_AUDIT '+json.dumps(dict(numerical=result['numerical_mount_status'], max_position_m=max_position,
              max_angle_rad=max_angle, source_matches=result['source_urdf_matches_current'])), flush=True)
    finally:
        env.close()


try:
    main()
finally:
    app.close()

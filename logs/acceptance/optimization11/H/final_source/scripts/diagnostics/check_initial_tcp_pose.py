"""v3桌心TCP IK/FK、真实物理静置和动作时间预算；不训练Policy。"""
import argparse
import csv
import json
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--seed', type=int, default=21)
parser.add_argument('--heights', type=float, nargs='+', default=[.5, .45, .4, .35, .3])
parser.add_argument('--settle_s', type=float, default=1.)
parser.add_argument('--approach_probe', action='store_true', help='C3 diagnostic joint-rate-limited approach; never policy success.')
parser.add_argument('--output_dir', type=Path, required=True)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import gymnasium as gym
import torch
from isaaclab_tasks.utils import parse_env_cfg
from isaaclab.utils.math import quat_apply, matrix_from_quat, quat_error_magnitude
import Grasp1.tasks
from Grasp1.tasks.manager_based.grasp1.grasp1_env_cfg import SUPPORT_HEIGHT
from Grasp1.tasks.manager_based.grasp1.mdp import events, geometry


def main():
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    cfg = parse_env_cfg('Grasp1-UR5-Allegro-GraspTrain-v3', device=args.device or 'cuda:0', num_envs=1)
    cfg.seed = args.seed
    cfg.set_num_envs(1)
    env = gym.make('Grasp1-UR5-Allegro-GraspTrain-v3', cfg=cfg).unwrapped
    rows = []
    try:
        robot, memory = env.scene['robot'], env._full_grasp_state
        term = env.action_manager.get_term('teacher')
        offset = torch.tensor(cfg.tcp_offset_h, device=env.device)[None]
        desired_quat = torch.tensor(cfg.palm_quaternion_w, device=env.device)[None]
        ids = torch.tensor([0], device=env.device)
        # 旧C1的几何公式与显式TCP反解必须等价；不改变原函数消费者。
        base_pos, base_rotation = events._solver_base_pose(robot, ids)
        cloud = torch.tensor([[[.35, -.75, .82], [.36, -.74, .81]]], device=env.device)
        direction = torch.tensor([[0., 0., 1.]], device=env.device)
        rotation = matrix_from_quat(desired_quat)
        old = geometry.wrist_poses_for_ur5_ik(cloud, direction, rotation[:, None], base_pos, base_rotation,
                                             pregrasp_distance=cfg.pregrasp_distance)[:, 0]
        new = geometry.tcp_pose_for_ur5_ik(cloud.mean(1)+direction*cfg.pregrasp_distance,
            rotation, offset, base_pos, base_rotation)
        assert torch.allclose(old, new, atol=1e-6)
        initial_poses = {}
        for label, height in [('C1', None), *[(f'table_center_{h:.2f}m', h) for h in args.heights]]:
            cfg.set_curriculum('C1' if height is None else 'C3')
            cfg.tcp_initial_height = height
            cfg.events.reset_teacher.params['tcp_height'] = height
            # EventManager在构造时复制了TermCfg；诊断切换课程必须显式更新实际消费者。
            env.event_manager.set_term_cfg('reset_teacher', cfg.events.reset_teacher)
            env.reset(seed=args.seed)
            palm = robot.data.body_link_pose_w[:, memory.palm_id]
            tcp_before = (palm[:, :3]+quat_apply(palm[:, 3:7], offset)).clone()
            if height is None:
                data = env._teacher_reset_data
                pose = data['object_pose']
                cloud_w = geometry.object_points_to_world(data['points'], pose[:, :3], pose[:, 3:7])
                target = cloud_w.mean(1)+direction*cfg.pregrasp_distance
            else:
                target = tcp_before.new_tensor([[.2, -.75152, SUPPORT_HEIGHT+height]])
            before_error = float((tcp_before-target).norm())
            angle_error = float(quat_error_magnitude(palm[:, 3:7], desired_quat).max())
            q_initial = robot.data.joint_pos[:, term._joint_ids].clone()
            reset_valid = bool(memory.valid_reset[0])
            target_near = tcp_before.clone()
            target_near[:, 2] = SUPPORT_HEIGHT+.18
            near_pose = geometry.tcp_pose_for_ur5_ik(target_near, rotation, offset, base_pos, base_rotation)
            near_q, near_feasible = geometry.ur5_inverse_kinematics(near_pose, reference_joints=q_initial[:, :6])
            lower_bound = float((near_q-q_initial[:, :6]).abs().max()/cfg.arm_action_speed) if bool(near_feasible[0]) else None
            initial_poses[label] = dict(q=q_initial[0].cpu().tolist(), palm=palm[0].cpu().tolist(),
                tcp_w=tcp_before[0].cpu().tolist(), robot_root=robot.data.root_link_pose_w[0].cpu().tolist())
            table_peak = 0.
            arm_peak = 0.
            violation_peak = 0.
            velocity_peak = 0.
            min_clearance = float(memory.geometry()[0][0])
            for _ in range(round(args.settle_s/env.physics_dt)):
                env.action_manager.apply_action()
                env.scene.write_data_to_sim()
                env.sim.step(render=False)
                env.scene.update(env.physics_dt)
                env.common_step_counter += 1
                memory.snapshot = None
                s = memory.read()
                table_peak = max(table_peak, float(s['table'].max()))
                arm_peak = max(arm_peak, float(s['arm_force'].max()))
                limits = robot.data.joint_pos_limits[:, term._joint_ids]
                q = robot.data.joint_pos[:, term._joint_ids]
                violation_peak = max(violation_peak, float(torch.maximum(limits[..., 0]-q, q-limits[..., 1]).clamp_min(0).max()))
                velocity_peak = max(velocity_peak, float(robot.data.joint_vel[:, term._joint_ids].abs().max()))
                min_clearance = min(min_clearance, float(s['clearance'][0]))
            palm = robot.data.body_link_pose_w[:, memory.palm_id]
            tcp = palm[:, :3]+quat_apply(palm[:, 3:7], offset)
            valid = reset_valid and before_error < .001 and angle_error < .0174533 and min_clearance >= -.002 and table_peak <= 20 and arm_peak <= 5 and violation_peak <= .005
            row = dict(label=label, height_m=height, reset_valid=reset_valid,
                tcp_target_w=target[0].cpu().tolist(), tcp_fk_w=tcp_before[0].cpu().tolist(),
                fk_position_error_m=before_error, fk_angle_error_rad=angle_error,
                settled_tcp_w=tcp[0].cpu().tolist(), settled_position_error_m=float((tcp-target).norm()),
                min_hand_table_clearance_m=min_clearance, tip_table_peak_N=table_peak,
                arm_contact_peak_N=arm_peak, joint_violation_peak_rad=violation_peak, reported_qd_peak=velocity_peak,
                close_pose_ik_feasible=bool(near_feasible[0]), minimum_target_travel_s=lower_bound,
                episode_s=cfg.episode_length_s, reset_pose_status='pass' if valid else 'fail',
                time_budget_status='insufficient' if lower_bound is not None and lower_bound+1 > cfg.episode_length_s else 'not_proven',
                policy_success=None)
            rows.append(row)
            print('TCP_CHECK '+json.dumps(row), flush=True)
            if not args.headless and height == .5:
                from Grasp1.tasks.manager_based.grasp1.mdp.mount_frames import MountFrames
                from omni.kit.viewport.utility import get_active_viewport, capture_viewport_to_file
                frames = MountFrames(env, cfg.tcp_offset_h)
                frames.update()
                env.sim.set_camera_view((1.35, -1.8, 1.7), (.2, -.75, 1.1))
                for _ in range(20):
                    env.sim.render()
                capture_viewport_to_file(get_active_viewport(), str(out/'C3_initial.png'))
                for _ in range(20):
                    app.update()
            if args.approach_probe and height == .5:
                approach_rows = []
                q_goal = torch.cat((near_q, q_initial[:, 6:]), -1)
                reached = False
                unsafe = False
                for step in range(round(cfg.episode_length_s/env.step_dt)):
                    actions = ((q_goal-term.target)/term._scale).clamp(-1, 1)
                    env.action_manager.process_action(actions)
                    for _ in range(cfg.decimation):
                        env.action_manager.apply_action()
                        env.scene.write_data_to_sim()
                        env.sim.step(render=False)
                        env.scene.update(env.physics_dt)
                        env.common_step_counter += 1
                        memory.snapshot = None
                        s = memory.read()
                        unsafe |= bool(s['unsafe'][0])
                    palm = robot.data.body_link_pose_w[:, memory.palm_id]
                    tcp = palm[:, :3]+quat_apply(palm[:, 3:7], offset)
                    error = float((tcp-target_near).norm())
                    orientation_error = float(quat_error_magnitude(palm[:, 3:7], desired_quat).max())
                    reached = error < .005 and orientation_error < .05
                    approach_rows.append(dict(time_s=(step+1)*env.step_dt, tcp_w=tcp[0].cpu().tolist(),
                        position_error_m=error, angle_error_rad=orientation_error,
                        q=s['q'][0].cpu().tolist(), qd=s['qd'][0].cpu().tolist(), target=term.target[0].cpu().tolist(),
                        clearance_m=float(s['clearance'][0]), tip_table_N=s['table'][0].cpu().tolist(),
                        violation_rad=float(s['violation'].max()), projection_fraction=float(term.projection_fraction[0])))
                    if reached or unsafe:
                        break
                approach = dict(controller='diagnostic_joint_target_approach', policy_success=None,
                    reached=reached and not unsafe, unsafe=unsafe, elapsed_s=(step+1)*env.step_dt,
                    target_tcp_w=target_near[0].cpu().tolist(), final_tcp_w=tcp[0].cpu().tolist(),
                    position_error_m=error, angle_error_rad=orientation_error,
                    lift_hold_time_still_required_s=3.5, episode_s=cfg.episode_length_s)
                (out/'C3_approach.json').write_text(json.dumps(approach, indent=2))
                with (out/'C3_approach.csv').open('w', newline='') as handle:
                    writer = csv.DictWriter(handle, fieldnames=list(approach_rows[0]))
                    writer.writeheader()
                    writer.writerows(approach_rows)
                print('C3_APPROACH '+json.dumps(approach), flush=True)
        (out/'initial_poses.json').write_text(json.dumps(initial_poses, indent=2))
        (out/'tcp_poses.json').write_text(json.dumps(dict(task='Grasp1-UR5-Allegro-GraspTrain-v3',
            tcp_offset_h=cfg.tcp_offset_h, quaternion_order='wxyz', rows=rows,
            checks=dict(old_C1_formula_equivalent=True), source_visual_confirmation=False), indent=2))
        with (out/'tcp_poses.csv').open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    finally:
        env.close()


try:
    main()
finally:
    app.close()

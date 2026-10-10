"""评估专用接触过程统计；不参与Teacher观测、奖励或训练场景。"""

import csv
import json
from pathlib import Path

import torch
from isaaclab.sensors import ContactSensorCfg

from Grasp1.robots.robot_profile import HAND_CONTACT_LINK_NAMES
from Grasp1.tasks.manager_based.grasp1.mdp.keypoints import hand_keypoints_o


def configure_diagnostics(scene_cfg):
    """复用原top/table传感器，仅在评估场景增加bottom过滤传感器。"""
    for index, name in enumerate(HAND_CONTACT_LINK_NAMES):
        setattr(scene_cfg, f"diagnostic_bottom_{index}", ContactSensorCfg(
            prim_path=f"{{ENV_REGEX_NS}}/Robot/.*{name.replace('.', '_')}",
            filter_prim_paths_expr=["{ENV_REGEX_NS}/Object/.*bottom"],
            update_period=0.0,
            max_contact_data_count_per_prim=512,
        ))


class GraspDiagnostics:
    """以0.1N法向过滤接触统计force-closure proxy，并记录抓取与lift两阶段。"""

    def __init__(self, env, names):
        self.env = env
        self.names = names
        self.rows = []

    def distance(self):
        return torch.cdist(hand_keypoints_o(self.env), self.env._teacher_affordance_points_o).amin(dim=(1, 2))

    def start_round(self, round_index):
        self.round_index = round_index
        self.initial_z = self.env.scene['object'].data.root_pos_w[:, 2].clone()
        term = self.env.action_manager.get_term('teacher')
        self.initial_q = self.env.scene['robot'].data.joint_pos[:, term._joint_ids].clone()
        self.pregrasp_distance = self.distance()
        self.states = {}
        for phase in ('grasp', 'lift'):
            self.states[phase] = {
                'steps': 0,
                'counts': torch.zeros((self.env.num_envs, 10), device=self.env.device),
                'peaks': torch.zeros((self.env.num_envs, 3), device=self.env.device),
                'max_bodies': torch.zeros(self.env.num_envs, device=self.env.device),
                'max_tips': torch.zeros(self.env.num_envs, device=self.env.device),
                'streak': torch.zeros((self.env.num_envs, 13), device=self.env.device),
                'longest': torch.zeros((self.env.num_envs, 13), device=self.env.device),
                'any_streak': torch.zeros(self.env.num_envs, device=self.env.device),
                'any_longest': torch.zeros(self.env.num_envs, device=self.env.device),
                'action_sum': torch.zeros((self.env.num_envs, 2), device=self.env.device),
                'height_peak': torch.zeros(self.env.num_envs, device=self.env.device),
                'retained_body_sum': torch.zeros(self.env.num_envs, device=self.env.device),
            }

    def sample(self, phase, actions, active):
        e = self.env
        forces = []
        for prefix in ('teacher_af_contact', 'diagnostic_bottom', 'teacher_table_contact'):
            forces.append(torch.stack([
                torch.nan_to_num(e.scene[f'{prefix}_{i}'].data.force_matrix_w[:, 0, 0]).norm(dim=-1)
                for i in range(13)
            ], dim=1))
        top, bottom, table = [(f > 0.1) & active[:, None] for f in forces]
        contact = top | bottom
        bodies = contact.sum(1)
        tips = contact[:, [3, 6, 9, 12]].sum(1)
        fingers = torch.stack([contact[:, 1 + 3*i:4 + 3*i].any(1) for i in range(4)], 1)
        thumb = fingers[:, 3]
        indicators = torch.stack((contact.any(1), top.any(1), bottom.any(1), table.any(1),
                                  bodies >= 2, bodies >= 3,
                                  thumb & fingers[:, 0], thumb & fingers[:, 1],
                                  thumb & fingers[:, 2], thumb & (fingers[:, :3].sum(1) >= 2)), 1)
        state = self.states[phase]
        state['steps'] += 1
        state['counts'] += indicators
        state['peaks'] = torch.maximum(state['peaks'], torch.stack([f.max(1).values for f in forces], 1) * active[:, None])
        state['max_bodies'] = torch.maximum(state['max_bodies'], bodies)
        state['max_tips'] = torch.maximum(state['max_tips'], tips)
        state['streak'] = (state['streak'] + 1) * contact
        state['longest'] = torch.maximum(state['longest'], state['streak'])
        state['any_streak'] = (state['any_streak'] + 1) * contact.any(1)
        state['any_longest'] = torch.maximum(state['any_longest'], state['any_streak'])
        state['action_sum'] += torch.stack([actions[:, :6].abs().mean(1), actions[:, 6:].abs().mean(1)], 1) * active[:, None]
        state['height_peak'] = torch.maximum(state['height_peak'], torch.where(active, e.scene['object'].data.root_pos_w[:, 2] - self.initial_z, state['height_peak']))
        if phase == 'lift':
            state['retained_body_sum'] += (contact & self.end_contact).sum(1)
            target = e.action_manager.get_term('teacher').target[:, 6:]
            drift = (target - self.end_hand_target).abs().max(1).values * active
            self.hand_target_drift = torch.maximum(self.hand_target_drift, drift)
        self.last_contact = contact

    def end_grasp(self, active):
        e = self.env
        term = e.action_manager.get_term('teacher')
        self.grasp_active = active.clone()
        self.end_distance = self.distance()
        self.hand_movement = (e.scene['robot'].data.joint_pos[:, term._joint_ids][:, 6:] - self.initial_q[:, 6:]).abs().mean(1)
        self.end_contact = self.last_contact.clone()
        self.end_hand_target = term.target[:, 6:].clone()
        self.hand_target_drift = torch.zeros(e.num_envs, device=e.device)

    def finish_round(self, height_gain, source, strict, lift_target, arm_ids, active):
        labels = ('object_contact', 'top_contact', 'bottom_contact', 'table_contact',
                  'two_plus_contacts', 'three_plus_contacts', 'thumb_index', 'thumb_middle',
                  'thumb_ring', 'thumb_multiple')
        residual = (self.env.scene['robot'].data.joint_pos[:, arm_ids] - lift_target).abs().max(1).values
        for i, name in enumerate(self.names):
            row = dict(object=name, round=self.round_index + 1,
                       pregrasp_distance_m=float(self.pregrasp_distance[i]),
                       grasp_end_distance_m=float(self.end_distance[i]) if bool(self.grasp_active[i]) else None,
                       hand_joint_movement_rad=float(self.hand_movement[i]) if bool(self.grasp_active[i]) else None,
                       height_gain_m=float(height_gain[i]), source_success=bool(source[i]),
                       strict_success=bool(strict[i]), active_height_success=bool(strict[i]),
                       lift_arm_target_residual_rad=float(residual[i]) if bool(active[i]) else None,
                       lift_max_hand_target_drift_rad=float(self.hand_target_drift[i]),
                       grasp_end_contact_bodies=int(self.end_contact[i].sum()),
                       lift_final_retained_bodies=int((self.end_contact[i] & self.last_contact[i]).sum()))
            for phase, state in self.states.items():
                for j, label in enumerate(labels):
                    row[f'{phase}_{label}_steps'] = int(state['counts'][i, j])
                for j, label in enumerate(('top', 'bottom', 'table')):
                    row[f'{phase}_{label}_peak_N'] = float(state['peaks'][i, j])
                row[f'{phase}_max_simultaneous_bodies'] = int(state['max_bodies'][i])
                row[f'{phase}_max_simultaneous_fingertips'] = int(state['max_tips'][i])
                row[f'{phase}_longest_body_contact_s'] = float(state['longest'][i].max()) * self.env.step_dt
                row[f'{phase}_longest_any_contact_s'] = float(state['any_longest'][i]) * self.env.step_dt
                row[f'{phase}_arm_raw_action_mean_abs'] = float(state['action_sum'][i, 0]) / state['steps']
                row[f'{phase}_hand_raw_action_mean_abs'] = float(state['action_sum'][i, 1]) / state['steps']
                row[f'{phase}_max_height_gain_m'] = float(state['height_peak'][i])
            row['lift_contact_retention_fraction'] = row['lift_object_contact_steps'] / self.states['lift']['steps'] if row['grasp_end_contact_bodies'] else None
            row['lift_same_body_retention_fraction'] = float(self.states['lift']['retained_body_sum'][i]) / (self.states['lift']['steps'] * row['grasp_end_contact_bodies']) if row['grasp_end_contact_bodies'] else None
            self.rows.append(row)

    def save(self, output_dir):
        output_dir = Path(output_dir)
        with (output_dir / 'grasp_diagnostics.csv').open('w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=list(self.rows[0]))
            writer.writeheader()
            writer.writerows(self.rows)
        (output_dir / 'grasp_diagnostics.json').write_text(json.dumps(self.rows, indent=2, allow_nan=False))


class PhysicsDiagnostics:
    """诊断入口按每个PhysX步聚合GPU峰值；只在输出摘要时转CPU。"""

    def __init__(self, env):
        self.env = env
        self.steps = 0
        self.q_min = torch.full((22,), torch.inf, device=env.device)
        self.q_max = torch.full((22,), -torch.inf, device=env.device)
        self.target_min = self.q_min.clone()
        self.target_max = self.q_max.clone()
        self.velocity = torch.zeros(22, device=env.device)
        self.torque = self.velocity.clone()
        self.tracking = self.velocity.clone()
        self.object_min = torch.tensor(torch.inf, device=env.device)
        self.object_max = -self.object_min
        self.object_velocity = torch.tensor(0., device=env.device)
        self.force_peak = torch.tensor(0., device=env.device)
        self.friction_peak = self.force_peak.clone()
        self.total_filtered_peak = self.force_peak.clone()
        self.finite = torch.tensor(True, device=env.device)
        self.clip_count = torch.tensor(0, device=env.device)
        self.target_count = 0
        self.raw_sum = torch.tensor(0., device=env.device)
        self.raw_max = self.raw_sum.clone()
        self.raw_gt1 = self.clip_count.clone()
        self.raw_gt2 = self.clip_count.clone()
        term = env.action_manager.get_term('teacher')
        original_process = term.process_actions
        self.term = term
        self.original_process = original_process

        def process(actions):
            original_process(actions)
            proposed = env.scene['robot'].data.joint_pos[:, term._joint_ids] + term.processed_actions
            limits = env.scene['robot'].data.joint_pos_limits[:, term._joint_ids]
            self.clip_count += ((proposed < limits[..., 0]) | (proposed > limits[..., 1])).sum()
            self.target_count += actions.numel()
            self.raw_sum += actions.abs().sum()
            self.raw_max = torch.maximum(self.raw_max, actions.abs().max())
            self.raw_gt1 += (actions.abs() > 1).sum()
            self.raw_gt2 += (actions.abs() > 2).sum()

        term.process_actions = process
        original_update = env.scene.update
        self.original_update = original_update

        def update(dt):
            original_update(dt)
            if dt != env.physics_dt:
                return
            self.steps += 1
            robot, obj = env.scene['robot'], env.scene['object']
            q = robot.data.joint_pos[:, term._joint_ids]
            v = robot.data.joint_vel[:, term._joint_ids]
            effort = robot.data.applied_torque[:, term._joint_ids]
            self.q_min = torch.minimum(self.q_min, q.amin(0))
            self.q_max = torch.maximum(self.q_max, q.amax(0))
            self.target_min = torch.minimum(self.target_min, term.target.amin(0))
            self.target_max = torch.maximum(self.target_max, term.target.amax(0))
            self.velocity = torch.maximum(self.velocity, v.abs().amax(0))
            self.torque = torch.maximum(self.torque, effort.abs().amax(0))
            self.tracking = torch.maximum(self.tracking, (term.target - q).abs().amax(0))
            self.object_min = torch.minimum(self.object_min, obj.data.joint_pos.min())
            self.object_max = torch.maximum(self.object_max, obj.data.joint_pos.max())
            self.object_velocity = torch.maximum(self.object_velocity, obj.data.joint_vel.abs().max())
            forces = torch.stack([sensor.data.net_forces_w.norm(dim=-1).max()
                                  for sensor in env.scene.sensors.values()])
            self.force_peak = torch.maximum(self.force_peak, forces.max())
            filtered = [sensor for sensor in env.scene.sensors.values() if sensor.cfg.track_friction_forces]
            normals = torch.stack([sensor.data.force_matrix_w[:, 0].sum(1) for sensor in filtered], 1)
            friction = torch.stack([sensor.data.friction_forces_w[:, 0].sum(1) for sensor in filtered], 1)
            self.friction_peak = torch.maximum(self.friction_peak, friction.norm(dim=-1).max())
            self.total_filtered_peak = torch.maximum(self.total_filtered_peak, (normals + friction).norm(dim=-1).max())
            self.finite &= (torch.isfinite(q).all() & torch.isfinite(v).all() & torch.isfinite(effort).all()
                            & torch.isfinite(obj.data.body_state_w).all() & torch.isfinite(obj.data.joint_pos).all()
                            & torch.isfinite(obj.data.joint_vel).all() & torch.isfinite(forces).all()
                            & torch.isfinite(normals).all() & torch.isfinite(friction).all())

        env.scene.update = update

    def close(self):
        self.term.process_actions = self.original_process
        self.env.scene.update = self.original_update

    def summary(self):
        term = self.env.action_manager.get_term('teacher')
        return dict(physics_steps=self.steps, finite=bool(self.finite),
                    joint_names=list(term._joint_names), joint_position_min=self.q_min.tolist(),
                    joint_position_max=self.q_max.tolist(), target_min=self.target_min.tolist(),
                    target_max=self.target_max.tolist(), velocity_peak=self.velocity.tolist(),
                    torque_peak=self.torque.tolist(), torque_kind='IsaacLab implicit PD torque approximation; not measured PhysX drive torque',
                    tracking_error_peak=self.tracking.tolist(),
                    arm_velocity_peak=float(self.velocity[:6].max()),
                    hand_velocity_peak=float(self.velocity[6:].max()), contact_force_peak_N=float(self.force_peak),
                    friction_force_peak_N=float(self.friction_peak), total_filtered_force_peak_N=float(self.total_filtered_peak),
                    object_joint_min_rad=float(self.object_min), object_joint_max_rad=float(self.object_max),
                    object_joint_velocity_peak=float(self.object_velocity),
                    target_clipping_fraction=int(self.clip_count) / self.target_count,
                    raw_action_mean_abs=float(self.raw_sum) / self.target_count, raw_action_max_abs=float(self.raw_max),
                    raw_action_fraction_gt1=int(self.raw_gt1) / self.target_count,
                    raw_action_fraction_gt2=int(self.raw_gt2) / self.target_count)


class EvaluationTrajectories:
    """冻结首次终止前状态，保存连续高度和物体相对手腕位姿，防止reset污染trial。"""

    def __init__(self, env, names, args):
        self.env, self.names, self.args = env, names, args
        self.rows, self.rounds = [], []
        self.active = None
        self.original_reset = env._reset_idx

        def reset(ids):
            if self.active is not None:
                first = ids[self.active[ids]]
                self.final_object_pose[first] = env.scene['object'].data.root_pose_w[first]
                self.failure_step[first] = self.step + 1
                for name, mask in self.failures.items():
                    mask[first] = env.termination_manager.get_term(name)[first]
                self.active[first] = False
            self.original_reset(ids)

        env._reset_idx = reset

    def start_round(self, index):
        self.index, self.step = index, 0
        e = self.env
        self.active = torch.ones(e.num_envs, device=e.device, dtype=torch.bool)
        self.initial_z = e.scene['object'].data.root_pos_w[:, 2].clone()
        self.final_object_pose = e.scene['object'].data.root_pose_w.clone()
        self.failure_step = torch.full((e.num_envs,), -1, device=e.device, dtype=torch.long)
        self.failures = {name: torch.zeros_like(self.active) for name in e.termination_manager.active_terms}
        self.trace = []

    def sample(self, phase):
        from isaaclab.utils.math import subtract_frame_transforms
        e = self.env
        self.step += 1
        obj = e.scene['object'].data.root_pose_w
        wrist = e.scene['robot'].data.body_link_pose_w[:, e._teacher_body_indices.wrist]
        relative_p, relative_q = subtract_frame_transforms(wrist[:, :3], wrist[:, 3:7], obj[:, :3], obj[:, 3:7])
        self.final_object_pose[self.active] = obj[self.active]
        contacts = torch.stack([e.scene[f'teacher_af_contact_{i}'].data.force_matrix_w[:, 0, 0].norm(dim=-1)
                                for i in range(13)], 1) > 0.1
        if self.args.diagnostics:
            contacts |= torch.stack([e.scene[f'diagnostic_bottom_{i}'].data.force_matrix_w[:, 0, 0].norm(dim=-1)
                                     for i in range(13)], 1) > 0.1
        height = self.final_object_pose[:, 2] - self.initial_z
        self.trace.append(torch.cat([height[:, None], relative_p, relative_q, self.active[:, None],
                                     (contacts.any(1) & self.active)[:, None]], 1))

    def finish_round(self, source, active_height):
        trace = torch.stack(self.trace)
        paper = torch.zeros_like(self.active)
        translation = rotation = None
        if self.args.paper_stability:
            window_steps = round(5.0 / self.env.step_dt)
            window = trace[-window_steps - 1:]
            translation = (window[:, :, 1:4] - window[0, :, 1:4]).norm(dim=-1).amax(0)
            qdot = (window[:, :, 4:8] * window[0, :, 4:8]).sum(-1).abs().clamp(max=1.0)
            rotation = (2 * torch.acos(qdot)).amax(0)
            paper = (self.active & (window[:, :, 0] > self.args.success_height).all(0)
                     & window[:, :, 8:10].bool().all(dim=(0, 2))
                     & (translation <= self.args.max_relative_translation_drift_m)
                     & (rotation <= self.args.max_relative_rotation_drift_rad))
        for i, name in enumerate(self.names):
            cause = [key for key, mask in self.failures.items() if bool(mask[i])]
            valid = trace[:, i, 8].bool()
            heights = trace[valid, i, 0]
            self.rows.append(dict(object=name, round=self.index + 1, source_success=bool(source[i]),
                                  active_height_success=bool(active_height[i]),
                                  paper_stable_success=bool(paper[i]) if self.args.paper_stability else None,
                                  final_height_gain_m=float(trace[-1, i, 0]),
                                  max_active_height_gain_m=float(heights.max()) if heights.numel() else None,
                                  failure_step=int(self.failure_step[i]), failure_reason='+'.join(cause) or
                                  ('height_below_threshold' if not bool(active_height[i]) else ''),
                                  dropped_after_crossing=bool((heights > self.args.success_height).any()) and
                                  not bool(active_height[i]),
                                  hold_translation_drift_m=float(translation[i]) if translation is not None else None,
                                  hold_rotation_drift_rad=float(rotation[i]) if rotation is not None else None))
        self.rounds.append(trace.cpu())
        # 下一轮显式reset不属于上一trial的提前终止。
        self.active = None
        return paper

    def save(self, output_dir):
        path = Path(output_dir)
        torch.save(dict(names=self.names, step_dt=self.env.step_dt,
                        columns=['height_gain_m', 'relative_x', 'relative_y', 'relative_z',
                                 'relative_qw', 'relative_qx', 'relative_qy', 'relative_qz', 'active', 'contact'],
                        rounds=self.rounds), path / 'height_relative_trajectories.pt')
        with (path / 'trials.csv').open('w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=list(self.rows[0]))
            writer.writeheader()
            writer.writerows(self.rows)
        (path / 'stability_protocol.json').write_text(json.dumps(dict(
            enabled=self.args.paper_stability, hold_duration_s=5.0,
            translation_threshold_m=self.args.max_relative_translation_drift_m,
            rotation_threshold_rad=self.args.max_relative_rotation_drift_rad,
            threshold_basis='Engineering thresholds: 2cm is 20% of 10cm lift; 0.2rad approximately 11.5deg. '
                            'Not an original paper threshold; inspect saved sample trajectories.',
        ), indent=2))

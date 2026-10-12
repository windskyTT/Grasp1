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
        self.top_id, _ = object_body_ids(env.scene['object'])

    def distance(self):
        return torch.cdist(hand_keypoints_o(self.env), self.env._teacher_affordance_points_o).amin(dim=(1, 2))

    def start_round(self, round_index):
        self.round_index = round_index
        self.initial_z = self.env.scene['object'].data.body_link_pose_w[:,self.top_id,2].clone()
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
        state['height_peak'] = torch.maximum(state['height_peak'], torch.where(active, e.scene['object'].data.body_link_pose_w[:,self.top_id,2] - self.initial_z, state['height_peak']))
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
                       height_gain_m=float(height_gain[i]), source_height_success=bool(source[i]),
                       active_height_success=bool(strict[i]),
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


def object_body_ids(asset):
    """按精确body名称解析源指标top及诊断bottom，不依赖导出顺序。"""
    for name in ('top', 'bottom'):
        if asset.body_names.count(name) != 1:
            raise ValueError(f'Expected exactly one object body {name!r}, got {asset.body_names}.')
    return asset.body_names.index('top'), asset.body_names.index('bottom')


class TrialPhysicsPeaks:
    """GPU上按环境保留峰值及发生帧；空集合在输出时转为None。"""

    def __init__(self, env):
        self.env = env
        self.count = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        self.velocity = torch.zeros((env.num_envs, 22), device=env.device)
        self.velocity_step = torch.zeros_like(self.velocity, dtype=torch.long)
        self.velocity_phase = torch.full_like(self.velocity_step, -1)
        self.position_outside = torch.zeros_like(self.velocity)
        self.q_min = torch.full_like(self.velocity,torch.inf)
        self.q_max = torch.full_like(self.velocity,-torch.inf)
        self.object_q_min = torch.full((env.num_envs,),torch.inf,device=env.device)
        self.object_q_max = torch.full((env.num_envs,),-torch.inf,device=env.device)
        self.object_velocity = torch.zeros(env.num_envs,device=env.device)
        self.outside_count = torch.zeros_like(self.velocity_step)
        self.force = torch.zeros(env.num_envs, device=env.device)
        self.force_step = torch.zeros_like(self.count)
        self.force_phase = torch.full_like(self.count, -1)
        self.friction = torch.zeros_like(self.force)
        self.finite = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)

    def sample(self, values, mask, step, phase):
        v, outside, force, friction, finite, q, object_q, object_v = values
        greater = (v > self.velocity) & mask[:, None]
        self.velocity = torch.maximum(self.velocity, torch.where(mask[:, None], v, 0.))
        self.velocity_step = torch.where(greater, step, self.velocity_step)
        self.velocity_phase = torch.where(greater, phase, self.velocity_phase)
        self.position_outside = torch.maximum(self.position_outside, torch.where(mask[:, None], outside, 0.))
        self.q_min = torch.minimum(self.q_min,torch.where(mask[:,None],q,torch.inf))
        self.q_max = torch.maximum(self.q_max,torch.where(mask[:,None],q,-torch.inf))
        self.object_q_min = torch.minimum(self.object_q_min,torch.where(mask,object_q.amin(1),torch.inf))
        self.object_q_max = torch.maximum(self.object_q_max,torch.where(mask,object_q.amax(1),-torch.inf))
        self.object_velocity = torch.maximum(self.object_velocity,torch.where(mask,object_v.abs().amax(1),0.))
        self.outside_count += ((outside > 1e-6) & mask[:, None])
        greater_force = (force > self.force) & mask
        self.force = torch.maximum(self.force, torch.where(mask, force, 0.))
        self.force_step = torch.where(greater_force, step, self.force_step)
        self.force_phase = torch.where(greater_force, phase, self.force_phase)
        self.friction = torch.maximum(self.friction, torch.where(mask, friction, 0.))
        self.finite &= ~mask | finite
        self.count += mask

    def summary(self, names, resets, reasons):
        term = self.env.action_manager.get_term('teacher')
        rows = []
        for i, name in enumerate(names):
            count = int(self.count[i])
            joint = int(self.velocity[i, 6:].argmax())+6
            rows.append(dict(env_id=i, object=name, samples=count,
                hand_velocity_peak_rad_s=float(self.velocity[i, joint]) if count else None,
                joint=term._joint_names[joint] if count else None,
                physics_step=int(self.velocity_step[i, joint]) if count else None,
                phase={-1:'unspecified',0:'grasp',1:'lift'}[int(self.velocity_phase[i,joint])],
                hand_outside_peak_rad=float(self.position_outside[i,6:].max()) if count else None,
                joint_position_min=self.q_min[i].tolist() if count else None,
                joint_position_max=self.q_max[i].tolist() if count else None,
                object_joint_min_rad=float(self.object_q_min[i]) if count else None,
                object_joint_max_rad=float(self.object_q_max[i]) if count else None,
                object_joint_velocity_peak_rad_s=float(self.object_velocity[i]) if count else None,
                outside_env_joint_physics_frames=int(self.outside_count[i,6:].sum()),
                outside_env_joint_s=float(self.outside_count[i,6:].sum())*self.env.physics_dt,
                contact_normal_peak_N=float(self.force[i]) if count else None,
                contact_physics_step=int(self.force_step[i]) if count else None,
                contact_phase={-1:'unspecified',0:'grasp',1:'lift'}[int(self.force_phase[i])],
                friction_peak_N=float(self.friction[i]) if count else None,
                finite=bool(self.finite[i]) if count else None,
                reset_count=int(resets[i]), termination_reasons=reasons[i]))
        valid = [r for r in rows if r['samples']]
        worst = max(valid, key=lambda r:r['hand_velocity_peak_rad_s']) if valid else None
        return dict(hand_velocity_peak_rad_s=worst['hand_velocity_peak_rad_s'] if worst else None,
                    velocity_case=worst,
                    contact_normal_peak_N=max(r['contact_normal_peak_N'] for r in valid) if valid else None,
                    per_environment=rows)


class PhysicsDiagnostics:
    """诊断入口按每个PhysX步聚合GPU峰值；只在输出摘要时转CPU。"""

    def __init__(self, env, trajectories=None):
        self.env = env
        self.trajectories = trajectories
        if trajectories is not None:
            trajectories.physics = self
        self.trial_peaks = {n:TrialPhysicsPeaks(env) for n in
                            ('active_trial_peak','failed_before_reset_peak','post_reset_peak','all_scene_peak')}
        self.last_values = None
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
            limits = robot.data.joint_pos_limits[:, term._joint_ids]
            outside = torch.maximum((limits[...,0]-q).clamp_min(0), (q-limits[...,1]).clamp_min(0))
            force_env = torch.stack([s.data.net_forces_w.norm(dim=-1).amax(1)
                                     for s in env.scene.sensors.values()],1).amax(1)
            finite_env = (torch.isfinite(q).all(1) & torch.isfinite(v).all(1)
                          & torch.isfinite(effort).all(1)
                          & torch.isfinite(normals).all(dim=(1,2)) & torch.isfinite(friction).all(dim=(1,2))
                          & torch.isfinite(obj.data.body_state_w).all(dim=(1,2))
                          & torch.isfinite(obj.data.joint_pos).all(1) & torch.isfinite(obj.data.joint_vel).all(1)
                          & torch.isfinite(force_env))
            self.last_values = (v.abs(), outside, force_env, friction.norm(dim=-1).amax(1), finite_env,
                                q, obj.data.joint_pos,obj.data.joint_vel)
            phase = {'grasp':0,'lift':1}.get(trajectories.phase if trajectories is not None else None,-1)
            all_mask = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
            self.trial_peaks['all_scene_peak'].sample(self.last_values, all_mask, env._sim_step_counter, phase)
            if trajectories is not None and trajectories.active is not None:
                self.trial_peaks['active_trial_peak'].sample(self.last_values, trajectories.active, env._sim_step_counter, phase)
                self.trial_peaks['post_reset_peak'].sample(self.last_values, ~trajectories.active, env._sim_step_counter, phase)

        env.scene.update = update

    def close(self):
        self.term.process_actions = self.original_process
        self.env.scene.update = self.original_update
        if self.trajectories is not None:
            self.trajectories.physics = None

    def capture_failed(self, ids):
        if self.last_values is not None:
            mask = torch.zeros(self.env.num_envs, dtype=torch.bool, device=self.env.device)
            mask[ids] = True
            phase = {'grasp':0,'lift':1}.get(self.trajectories.phase,-1)
            self.trial_peaks['failed_before_reset_peak'].sample(self.last_values, mask, self.env._sim_step_counter, phase)

    def summary(self):
        term = self.env.action_manager.get_term('teacher')
        result = dict(physics_steps=self.steps, finite=bool(self.finite),
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
        names = self.trajectories.names if self.trajectories is not None else self.env.cfg.object_names
        resets = self.trajectories.reset_count if self.trajectories is not None else torch.zeros(self.env.num_envs, device=self.env.device)
        reasons = self.trajectories.all_failure_reasons() if self.trajectories is not None else [[] for _ in names]
        result.update({n:bank.summary(names,resets,reasons) for n,bank in self.trial_peaks.items()})
        result['legacy_peak_scope'] = 'all_scene_peak; includes failed and automatically reset environments'
        return result


class EvaluationTrajectories:
    """冻结首次终止前状态，保存连续高度和物体相对手腕位姿，防止reset污染trial。"""

    def __init__(self, env, names, args):
        self.env, self.names, self.args = env, names, args
        self.rows, self.rounds = [], []
        self.active = None
        self.phase = None
        self.physics = None
        self.top_id, self.bottom_id = object_body_ids(env.scene['object'])
        self.reset_count = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        self.failure_history = {name: torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
                                for name in env.termination_manager.active_terms}
        self.original_reset = env._reset_idx

        def reset(ids):
            if self.active is not None:
                first = ids[self.active[ids]]
                self.capture(first)
                if self.physics is not None:
                    self.physics.capture_failed(first)
                self.reset_count[ids] += 1
                self.failure_step[first] = self.step + 1
                for name, mask in self.failures.items():
                    mask[first] = env.termination_manager.get_term(name)[first]
                    self.failure_history[name][first] |= mask[first]
                self.active[first] = False
            self.original_reset(ids)

        env._reset_idx = reset

    def close(self):
        self.env._reset_idx = self.original_reset

    def all_failure_reasons(self):
        return [[name for name,mask in self.failure_history.items() if bool(mask[i])]
                for i in range(self.env.num_envs)]

    def capture(self, ids):
        """积分结束后、reset写状态前读取top/root/bottom和相对手腕位姿。"""
        from isaaclab.utils.math import subtract_frame_transforms
        obj = self.env.scene['object']
        top = obj.data.body_link_pose_w[:,self.top_id]
        bottom = obj.data.body_link_pose_w[:,self.bottom_id]
        wrist = self.env.scene['robot'].data.body_link_pose_w[:, self.env._teacher_body_indices.wrist]
        p, q = subtract_frame_transforms(wrist[:,:3],wrist[:,3:7],top[:,:3],top[:,3:7])
        self.final_object_pose[ids] = top[ids]
        self.final_root_pose[ids] = obj.data.root_pose_w[ids]
        self.final_bottom_pose[ids] = bottom[ids]
        self.final_relative_pose[ids] = torch.cat((p,q),1)[ids]
        height = top[:,2]-self.initial_z
        self.height_peak[ids] = torch.maximum(self.height_peak[ids],height[ids])

    def start_round(self, index):
        self.index, self.step = index, 0
        e = self.env
        self.active = torch.ones(e.num_envs, device=e.device, dtype=torch.bool)
        obj = e.scene['object']
        self.initial_z = obj.data.body_link_pose_w[:,self.top_id,2].clone()
        self.initial_root_z = obj.data.root_pos_w[:,2].clone()
        self.initial_bottom_z = obj.data.body_link_pose_w[:,self.bottom_id,2].clone()
        self.final_object_pose = obj.data.body_link_pose_w[:,self.top_id].clone()
        self.final_root_pose = obj.data.root_pose_w.clone()
        self.final_bottom_pose = obj.data.body_link_pose_w[:,self.bottom_id].clone()
        self.final_relative_pose = torch.zeros((e.num_envs,7),device=e.device)
        self.height_peak = torch.zeros(e.num_envs,device=e.device)
        self.capture(torch.arange(e.num_envs,device=e.device))
        self.round_initial_resets = self.reset_count.clone()
        self.failure_step = torch.full((e.num_envs,), -1, device=e.device, dtype=torch.long)
        self.failures = {name: torch.zeros_like(self.active) for name in e.termination_manager.active_terms}
        self.trace = []

    def sample(self, phase):
        e = self.env
        self.step += 1
        self.phase = phase
        self.capture(self.active.nonzero().flatten())
        contacts = torch.stack([e.scene[f'teacher_af_contact_{i}'].data.force_matrix_w[:, 0, 0].norm(dim=-1)
                                for i in range(13)], 1) > 0.1
        if self.args.diagnostics:
            contacts |= torch.stack([e.scene[f'diagnostic_bottom_{i}'].data.force_matrix_w[:, 0, 0].norm(dim=-1)
                                     for i in range(13)], 1) > 0.1
        height = self.final_object_pose[:, 2] - self.initial_z
        self.trace.append(torch.cat([height[:, None], self.final_relative_pose, self.active[:, None],
                                     (contacts.any(1) & self.active)[:, None],
                                     self.final_object_pose[:,2:3],self.final_root_pose[:,2:3],self.final_bottom_pose[:,2:3]], 1))

    def finish_round(self, source, active_height):
        trace = torch.stack(self.trace)
        engineering = torch.zeros_like(self.active)
        translation = rotation = None
        subchecks = None
        if self.args.paper_stability:
            window_steps = round(5.0 / self.env.step_dt)
            window = trace[-window_steps - 1:]
            translation = (window[:, :, 1:4] - window[0, :, 1:4]).norm(dim=-1).amax(0)
            qdot = (window[:, :, 4:8] * window[0, :, 4:8]).sum(-1).abs().clamp(max=1.0)
            rotation = (2 * torch.acos(qdot)).amax(0)
            subchecks = dict(height=(window[:,:,0] > self.args.success_height).all(0),
                             alive=window[:,:,8].bool().all(0), contact=window[:,:,9].bool().all(0),
                             translation=translation <= self.args.max_relative_translation_drift_m,
                             rotation=rotation <= self.args.max_relative_rotation_drift_rad)
            subchecks['window_complete'] = torch.full_like(self.active,len(window)==window_steps+1)
            engineering = torch.stack(list(subchecks.values())).all(0)
        for i, name in enumerate(self.names):
            cause = [key for key, mask in self.failures.items() if bool(mask[i])]
            valid = trace[:, i, 8].bool()
            heights = trace[valid, i, 0]
            row = dict(object=name, env_id=i, round=self.index + 1, source_height_success=bool(source[i]),
                                  active_height_success=bool(active_height[i]),
                                  engineering_stable_success=bool(engineering[i]) if self.args.paper_stability else None,
                                  final_height_gain_m=float(trace[-1, i, 0]),
                                  initial_top_z_m=float(self.initial_z[i]),final_top_z_m=float(self.final_object_pose[i,2]),
                                  initial_root_z_m=float(self.initial_root_z[i]),final_root_z_m=float(self.final_root_pose[i,2]),
                                  initial_bottom_z_m=float(self.initial_bottom_z[i]),final_bottom_z_m=float(self.final_bottom_pose[i,2]),
                                  top_minus_root_z_m=float(self.final_object_pose[i,2]-self.final_root_pose[i,2]),
                                  max_active_height_gain_m=float(self.height_peak[i]),
                                  failure_step=int(self.failure_step[i]), failure_reason='+'.join(cause) or
                                  ('height_below_threshold' if not bool(active_height[i]) else ''),
                                  dropped_after_crossing=bool(self.height_peak[i] > self.args.success_height) and
                                  float(trace[-1,i,0]) <= self.args.success_height,
                                  failed_after_crossing=bool(self.height_peak[i] > self.args.success_height) and
                                  int(self.failure_step[i]) >= 0,
                                  hold_translation_drift_m=float(translation[i]) if translation is not None else None,
                                  hold_rotation_drift_rad=float(rotation[i]) if rotation is not None else None,
                                  reset_count=int(self.reset_count[i]-self.round_initial_resets[i]))
            for name in ('height','alive','contact','translation','rotation','window_complete'):
                row[f'hold_{name}_pass'] = bool(subchecks[name][i]) if subchecks is not None else None
            self.rows.append(row)
        self.rounds.append(trace.cpu())
        # 下一轮显式reset不属于上一trial的提前终止。
        self.active = None
        return engineering

    def save(self, output_dir):
        path = Path(output_dir)
        if hasattr(self.args,'checkpoint_compatibility'):
            audit = self.args.checkpoint_compatibility
            for row in self.rows:
                row.update(checkpoint_compatible=audit['compatible'],comparability_warning=audit['comparability_warning'],
                           checkpoint_differences=json.dumps(audit['differences'],sort_keys=True))
        torch.save(dict(names=self.names, step_dt=self.env.step_dt,
                        columns=['height_gain_m', 'relative_x', 'relative_y', 'relative_z',
                                 'relative_qw', 'relative_qx', 'relative_qy', 'relative_qz', 'active', 'contact',
                                 'top_z_m','root_z_m','bottom_z_m'],
                        object_reference_body='top',
                        rounds=self.rounds), path / 'height_relative_trajectories.pt')
        with (path / 'trials.csv').open('w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=list(self.rows[0]))
            writer.writeheader()
            writer.writerows(self.rows)
        (path / 'stability_protocol.json').write_text(json.dumps(dict(
            enabled=self.args.paper_stability, protocol='engineering_stability_20_20_ramp14',hold_duration_s=5.0,
            translation_threshold_m=self.args.max_relative_translation_drift_m,
            rotation_threshold_rad=self.args.max_relative_rotation_drift_rad,
            threshold_basis='Engineering thresholds: 2cm is 20% of 10cm lift; 0.2rad approximately 11.5deg. '
                            'Not an original paper threshold; inspect saved sample trajectories.',
        ), indent=2))

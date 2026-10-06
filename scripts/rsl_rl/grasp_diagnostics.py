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
                'height_peak': torch.full((self.env.num_envs,), -torch.inf, device=self.env.device),
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
        state['peaks'] = torch.maximum(state['peaks'], torch.stack([f.max(1).values for f in forces], 1))
        state['max_bodies'] = torch.maximum(state['max_bodies'], bodies)
        state['max_tips'] = torch.maximum(state['max_tips'], tips)
        state['streak'] = (state['streak'] + 1) * contact
        state['longest'] = torch.maximum(state['longest'], state['streak'])
        state['any_streak'] = (state['any_streak'] + 1) * contact.any(1)
        state['any_longest'] = torch.maximum(state['any_longest'], state['any_streak'])
        state['action_sum'] += torch.stack([actions[:, :6].abs().mean(1), actions[:, 6:].abs().mean(1)], 1)
        state['height_peak'] = torch.maximum(state['height_peak'], e.scene['object'].data.root_pos_w[:, 2] - self.initial_z)
        if phase == 'lift':
            state['retained_body_sum'] += (contact & self.end_contact).sum(1)
            target = e.action_manager.get_term('teacher').target[:, 6:]
            drift = (target - self.end_hand_target).abs().max(1).values * active
            self.hand_target_drift = torch.maximum(self.hand_target_drift, drift)
        self.last_contact = contact

    def end_grasp(self):
        e = self.env
        term = e.action_manager.get_term('teacher')
        self.end_distance = self.distance()
        self.hand_movement = (e.scene['robot'].data.joint_pos[:, term._joint_ids][:, 6:] - self.initial_q[:, 6:]).abs().mean(1)
        self.end_contact = self.last_contact.clone()
        self.end_hand_target = term.target[:, 6:].clone()
        self.hand_target_drift = torch.zeros(e.num_envs, device=e.device)

    def finish_round(self, height_gain, source, strict, lift_target, arm_ids):
        labels = ('object_contact', 'top_contact', 'bottom_contact', 'table_contact',
                  'two_plus_contacts', 'three_plus_contacts', 'thumb_index', 'thumb_middle',
                  'thumb_ring', 'thumb_multiple')
        residual = (self.env.scene['robot'].data.joint_pos[:, arm_ids] - lift_target).abs().max(1).values
        for i, name in enumerate(self.names):
            row = dict(object=name, round=self.round_index + 1,
                       pregrasp_distance_m=float(self.pregrasp_distance[i]),
                       grasp_end_distance_m=float(self.end_distance[i]),
                       hand_joint_movement_rad=float(self.hand_movement[i]),
                       height_gain_m=float(height_gain[i]), source_success=bool(source[i]),
                       strict_success=bool(strict[i]), lift_arm_target_residual_rad=float(residual[i]),
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

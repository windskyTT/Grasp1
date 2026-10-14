"""汇总已执行的物理回放，保留失败与负对照，生成原始状态对照图。"""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch

root = Path('logs/acceptance/optimization9/A')
cases = ['original_replay', 'all_body_contact', 'tip_table_pair_isolation', 'friction005_replay',
         'position128_replay', 'merged_thumb_retry', 'position128_velocity8_replay',
         'position128_velocity32_replay', 'external_each_iteration_replay',
         'position255_velocity32_replay', 'position255_velocity32_external_replay',
         'full_scene_replay', 'D2_historical_random_replay']
reference = torch.load(root/'original_replay/trace_replay.pt', weights_only=False)
rows = []
for name in cases:
    result = json.loads((root/name/'result.json').read_text())['runs'][0]
    trace = torch.load(root/name/'trace_replay.pt', weights_only=False)
    source = torch.load('logs/acceptance/optimization7/C/E2_screen/trace_random.pt', weights_only=False) if name.startswith('D2') else reference
    target = source['target'][:len(trace['target']), 7:8] if name.startswith('D2') else source['target']
    difference = float((target - trace['target']).abs().max())
    constraint = root/name/'contact_constraint_trace.pt'
    contacts = torch.load(constraint, weights_only=False) if constraint.exists() else None
    rows.append(dict(experiment=name, decimation=result['decimation'], physics_steps=result['physics_steps'],
                     wall_s=result['wall_s'], hand_qd_peak_rad_s=result['hand_velocity_peak'],
                     q_limit_p9999_rad=result['joint_violation_p9999_rad'], q_limit_max_rad=result['joint_violation_max_rad'],
                     qd_reported_at_peak=result['qvel_rad_s'], qd_difference_at_peak=result['delta_velocity_rad_s'],
                     table_force_peak_N=float(contacts['table_force'].norm(dim=-1).max()) if contacts else None,
                     max_target_difference_rad=difference, same_target_within_1e_6_rad=difference < 1e-6,
                     numerical_q_limit_pass=result['joint_violation_p9999_rad'] <= .001 and result['joint_violation_max_rad'] <= .005,
                     accepted_fix=False, category='diagnostic_negative_control' if name == 'tip_table_pair_isolation' else 'failed_physics_candidate'))
with (root/'before_after_physics.csv').open('w') as output:
    writer = csv.DictWriter(output, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)
(root/'physics_gate.json').write_text(json.dumps(dict(status='fail', physics_dt=1/120,
    accepted_fix=None, physics_stable=False, trigger_identified='link_15_0_tip versus Table',
    constraint_solver_cause_resolved=False, q_limit_thresholds=dict(p9999_rad=.001, max_rad=.005),
    completed_replays=len(rows), physics_steps=sum(row['physics_steps'] for row in rows),
    diagnostic_wall_s=sum(row['wall_s'] for row in rows),
    safety_matrix_1_16_128_multiple_seeds='blocked',
    scripted_grasp='blocked', ppo='blocked',
    experiments=rows), ensure_ascii=False, indent=2) + '\n')

fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
for name, label in [('all_body_contact', 'Original: tip/table contact'),
                    ('tip_table_pair_isolation', 'Diagnostic only: tip/table filtered'),
                    ('position255_velocity32_replay', '255 position / 32 velocity iterations')]:
    trace = torch.load(root/name/'trace_replay.pt', weights_only=False)
    contact = torch.load(root/name/'contact_constraint_trace.pt', weights_only=False)
    joint = trace['joints'].index('joint_15_0')
    body = contact['body_paths'].index('/World/envs/env_0/Robot/link_15_0_tip')
    time = torch.arange(1, len(trace['q']) + 1) / 120
    axes[0].plot(time, trace['q'][:, 0, joint], label=label, linewidth=1)
    axes[1].plot(time, trace['v'][:, 0, joint], linewidth=.7, label=label)
    axes[2].plot(time, contact['table_force'][:, body].norm(dim=-1), linewidth=1, label=label)
axes[0].axhline(-.162, color='black', linestyle='--', label='Original joint lower limit')
axes[0].set_ylabel('Actual joint position (rad)')
axes[1].set_ylabel('Reported joint velocity (rad/s)')
axes[2].set_ylabel('Tip/table normal force (N)')
axes[2].set_xlabel('Physics time (s)')
for axis in axes:
    axis.grid(alpha=.25)
axes[0].legend(fontsize=8)
fig.suptitle('Same saved initial state and targets; 120 Hz PhysX; diagnostic controls are not accepted fixes')
fig.tight_layout()
fig.savefig(root/'contact_causality.png', dpi=160)
print(json.dumps({'completed_replays':len(rows), 'physics_gate':'fail', 'plot':str(root/'contact_causality.png')}))

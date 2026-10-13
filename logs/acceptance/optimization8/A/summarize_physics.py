"""从保存的真实120Hz轨迹导出峰值、窗口与复现边界。"""
import csv
import json
from pathlib import Path
import torch
root=Path('logs/acceptance/optimization8/A')
rows=[]
for folder in sorted(root.iterdir()):
    if not folder.is_dir() or not (folder/'result.json').exists():continue
    run=json.loads((folder/'result.json').read_text())['runs'][0]
    trace=torch.load(folder/'trace_replay.pt',map_location='cpu',weights_only=False)
    params=json.loads((folder/'loaded_parameters.json').read_text())
    lo,hi=torch.tensor(params['robot']['limits'])[...,0],torch.tensor(params['robot']['limits'])[...,1]
    ids=[params['robot']['joints'].index(n) for n in trace['joints']]
    lo,hi=lo[:,ids],hi[:,ids]
    outside=torch.maximum((lo-trace['q']).clamp_min(0),(trace['q']-hi).clamp_min(0))
    flat=int(outside.argmax());frame,rem=divmod(flat,22);env,joint=divmod(rem,22)
    longest=torch.zeros(22,dtype=torch.long);current=torch.zeros_like(longest)
    for mask in outside[:,0]>.001:
        current=torch.where(mask,current+1,0);longest=torch.maximum(longest,current)
    row=dict(experiment_id=folder.name,velocity_peak_frame=run['physics_step'],velocity_peak_joint=run['joint'],qvel_peak_rad_s=run['qvel_rad_s'],velocity_peak_difference_rad_s=run['delta_velocity_rad_s'],max_violation_rad=float(outside.max()),max_violation_frame=frame+1,max_violation_joint=trace['joints'][joint],max_violation_q_rad=float(trace['q'][frame,env,joint]),max_violation_target_rad=float(trace['target'][frame,env,joint]),longest_over_001_env_joint_s=float(longest.max())/120,normal_force_peak_N=run['normal_peak_N'],reset_at_frame=[r['step'] for r in trace['reset_events']],trace=str(folder/'trace_replay.pt'))
    rows.append(row)
    peak=run['physics_step']-1;j=trace['joints'].index(run['joint'])
    body_name=run['joint'].replace('joint_', 'link_',1)
    body=params['robot']['bodies'].index(body_name)
    window=[]
    for frame in range(max(1,peak-50),min(len(trace['q']),peak+51)):
        position=trace['robot_poses'][frame,0,body,:3]
        speed=(position-trace['robot_poses'][frame-1,0,body,:3]).norm()*120
        window.append(dict(physics_step=frame+1,time_s=(frame+1)/120,q_rad=float(trace['q'][frame,0,j]),qvel_rad_s=float(trace['v'][frame,0,j]),delta_q_dt_rad_s=float(trace['delta_v'][frame,0,j]),target_rad=float(trace['target'][frame,0,j]),actual_violation_rad=float(outside[frame,0,j]),link_x_m=float(position[0]),link_y_m=float(position[1]),link_z_m=float(position[2]),link_position_difference_speed_m_s=float(speed),reset_before=bool(trace['reset_before'][frame,0])))
    with (folder/'peak_window.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(window[0]));w.writeheader();w.writerows(window)
with (root/'peak_cases.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
fixture=torch.load('logs/acceptance/optimization7/A/baseline_D4/initial_zero.pt',map_location='cpu',weights_only=False)
reference=torch.load('logs/acceptance/optimization6/diagnosis/zero_physics_trace.pt',map_location='cpu',weights_only=False)
d0=torch.load(root/'original_table_on_object_off/trace_replay.pt',map_location='cpu',weights_only=False)
d1=torch.load(root/'original_table_on_object_off_repeat/trace_replay.pt',map_location='cpu',weights_only=False)
assert torch.equal(d0['q'],d1['q']) and torch.equal(d0['v'],d1['v']) and torch.equal(d0['target'],d1['target'])
replay_error=float((d0['target']-reference['target'][:len(d0['target']),7:8]).abs().max())
(root/'replay_equivalence.json').write_text(json.dumps(dict(repeated_no_object_q_v_target_exact=True,old_requested_target_max_error_rad=replay_error,initial_state_source='optimization7/A/baseline_D4/initial_zero.pt env7',target_source='optimization6/diagnosis/zero_physics_trace.pt env7',D2_input='optimization7/C/E2_screen initial_random and saved target; no fresh Gaussian claim',historical_16env_peaks_reproduced_exactly=False),indent=2)+'\n')
print('peak cases',len(rows),'target replay error',replay_error)

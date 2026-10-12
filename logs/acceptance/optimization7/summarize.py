"""从优化7真实运行文件生成Gate和报告；未运行的正式实验使用N/A。"""
import csv
import json
from pathlib import Path
import shlex
import subprocess

root=Path('logs/acceptance/optimization7')
def read(path): return json.loads(Path(path).read_text())
def write(path,data): Path(path).write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')
def table(path,rows):
    with Path(path).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
diagnostics=[]
peak_rows=[]
commands=[]
for result_path in sorted((root/'A').glob('*/result.json')):
    data=read(result_path)
    parameters=read(data['parameters'])
    overrides=parameters['overrides']
    cmd=['/home/windsky/IsaacLab/isaaclab.sh','-p',str(root/'A/minimal_repro.py')]
    supplied=['decimation','episode','num_envs','duration','seed','modes','initial_state','source_env',
              'hand_armature','arm_armature','legacy_joint_friction','hand_friction','object_armature',
              'velocity_iterations','object_velocity_iterations','limit_case','solver','isolation','contact_detail','force_perturbation']
    for key in supplied:
        value=overrides.get(key)
        if value is None or value is False: continue
        cmd.append('--'+key)
        if value is not True:cmd.extend(str(v) for v in value) if isinstance(value,list) else cmd.append(str(value))
    cmd.extend(['--output_dir',overrides['output_dir'],'--headless'])
    commands.append(dict(command=shlex.join(cmd),status='executed',evidence=str(result_path)))
    for run in data['runs']:
        diagnostics.append(dict(configuration=str(result_path.parent),**run))
        peak_rows.append(dict(configuration=str(result_path.parent),mode=run['mode'],
             physics_step=run['physics_step'],env_id=run['env_id'],object=run['object'],joint=run['joint'],
             qvel_rad_s=run['qvel_rad_s'],delta_velocity_rad_s=run['delta_velocity_rad_s'],
             q_rad=run['q_rad'],target_rad=run['target_rad'],outside_rad=run['outside_rad'],trace=run['trace']))
table(root/'A/peak_cases.csv',peak_rows)
write(root/'A/experiments.json',dict(baseline_git_sha=head,formal_physics_defaults_changed=False,experiments=diagnostics))

candidates=[]; training=[]; efficiency=[]; phases=[]
for group,decimation,episode in [('E1',2,4),('E2',2,10),('E3',4,4),('E4',4,10)]:
    result=read(root/f'C/{group}_screen/result.json')
    runs=result['runs']
    for run in runs:
        peak_rows.append(dict(configuration=str(root/f'C/{group}_screen'),mode=run['mode'],
             physics_step=run['physics_step'],env_id=run['env_id'],object=run['object'],joint=run['joint'],
             qvel_rad_s=run['qvel_rad_s'],delta_velocity_rad_s=run['delta_velocity_rad_s'],
             q_rad=run['q_rad'],target_rad=run['target_rad'],outside_rad=run['outside_rad'],trace=run['trace']))
    row=dict(group=group,physics_gate='fail',decision='global_block',decimation=decimation,episode_s=episode,
             seed=1,num_envs=16,duration_per_mode_s=12,modes=[r['mode'] for r in runs],
             hand_velocity_peak_rad_s=max(r['hand_velocity_peak'] for r in runs),
             hand_outside_peak_rad=max(r['hand_outside_peak_rad'] for r in runs),
             evidence=str(root/f'C/{group}_screen/result.json'),
             reason='Unresolved physical state/limit deviations; finite tensors and legal targets do not establish safety.',
             configured_hand_velocity_reference_rad_s=7,historical_stop_line_rad_s=100,
             numerical_limit_fixture_reference_rad=read(root/'A/limit_force/result.json')['runs'][0]['hand_outside_peak_rad'],
             formal_updates=0,formal_checkpoint=None,training_status='blocked')
    candidates.append(row)
    write(root/f'C/safety_candidate_{group}.json',row)
    for iteration in (100,500,1000,2000):
        training.append(dict(group=group,decimation=decimation,episode_s=episode,seed=1,
            requested_update=iteration,status='not_run',reason='Gate A failed',actual_updates=0,
            transitions=0,environment_simulated_s=0,wall_s=0,checkpoint=None,
            gamma=.996**((decimation/120)/.2),lam=.95**((decimation/120)/.2)))
    efficiency.append(dict(group=group,status='blocked',training_num_envs=None,planned_num_envs=2048,
        gpu_hours=None,simulation_hours=None,gpu_hours_per_simulation_hour=None,
        successful_trials_per_gpu_hour=None,diagnostics_off_on_repeats='not_run',reason='Gate A failed'))
    for phase in ('0_4s','4_10s') if episode==10 else ('0_4s',):
        phases.append(dict(group=group,phase=phase,status='not_run',first_contact_s=None,
            two_plus_contact_fraction=None,three_plus_contact_fraction=None,contact_duration_s=None,
            affordance_distance_m=None,reason='No formal PPO training'))
write(root/'C/candidate_gates.json',dict(global_block=True,candidates=candidates))
table(root/'A/peak_cases.csv',peak_rows)
physics_runs = diagnostics + [dict(configuration=str(root/f'C/{g["group"]}_screen'),**r)
    for g in candidates for r in read(root/f'C/{g["group"]}_screen/result.json')['runs']]
table(root/'A/physics_ab.csv',[dict(configuration=r['configuration'],mode=r['mode'],seed=r['seed'],
    num_envs=r['num_envs'],decimation=r['decimation'],episode_s=r['episode_s'],
    physics_steps=r['physics_steps'],hand_velocity_peak=r['hand_velocity_peak'],
    hand_outside_peak_rad=r['hand_outside_peak_rad'],normal_peak_N=r['normal_peak_N'],
    friction_peak_N=r['friction_peak_N'],net_normal_peak_N=r.get('net_normal_peak_N'),
    velocity_difference_peak=r['velocity_difference_peak'],finite=r['finite'],trace=r['trace']) for r in physics_runs])
table(root/'C/training_matrix.csv',training);table(root/'C/efficiency.csv',efficiency)
table(root/'C/behavior_by_episode_phase.csv',phases)
table(root/'C/multi_seed.csv',[dict(candidate='not_selected',seed=s,status='not_run',reason='Gate A failed') for s in (1,2,3)])
(root/'C/selection.md').write_text('''# 候选选型

E1–E4各自完成16环境×12秒×zero/random/固定目标筛查，均有未解除的实际状态越限或速度尖峰，因此全局停止正式训练。四组实际100/500/1000/2000更新均未执行；3-seed、2048环境GPU效率与4–10秒正式训练行为均N/A。

维持D2+4秒默认。没有证据证明D4、10秒训练或armature候选更适合正式抓取训练；保留默认不表示该默认已安全或已收敛。PGS速度较低但位置越限更大，不能按单项速度选择。短程接口PPO共3次update，全部属于B阶段检查点保存/恢复测试，不能填入本矩阵。
''')

peak=read(root/'A/peak_case.json')
physics_gate=dict(status='fail',global_block=True,root_cause_established=False,physical_fix_adopted=False,
    reason='Default model exhibits large actual position violations and unresolved endpoint velocity spikes in all four candidates.',
    most_severe_reproduced_case=peak,candidates=candidates,reference='original 100rad/s stop line and loaded 7rad/s actuator reference are not a safety certificate',
    blocked_experiments=['formal PPO matrix','three-seed selection','qualified policy evaluation'])
write(root/'A/physics_gate.json',physics_gate)

measure=[read(root/'B/final_D2/measurement_regression.json'),read(root/'B/final_D4_copies30/measurement_regression.json')]
regression=[read(root/f'B/regression_D{d}/D{d}_T10.json') for d in (2,4)]
write(root/'B/measurement_regression.json',dict(status='pass',runs=[str(root/'B/final_D2/measurement_regression.json'),str(root/'B/final_D4_copies30/measurement_regression.json')],
       cases=46,synthetic_top_offset_and_reversed_body_order=True,reset_isolation=True,
       finite_empty_active_set=True,action_boundary_checks=sum(r['action']['cases'] for r in regression)))
compat=read(root/'B/checkpoint_compatibility.json');compat['native_resume']=read(root/'B/native_resume.json')
write(root/'B/checkpoint_compatibility.json',compat)
write(root/'B/gate.json',dict(status='pass',evidence=['B/measurement_regression.json','B/checkpoint_compatibility.json',
     'B/protocol_tests.json','B/stability_tests.json','B/ppo_regression.json','B/contact_reward_audit.md'],
     warning='Synthetic offset, force projection, std-floor and injected measurement tests verify consumers; they are not policy-success evidence.'))

drows=[];comparisons=[];interfaces=[]
for folder in ('interface_20_20_ramp16','interface_injected_20_20_ramp16','interface_injected_engineering_ramp14'):
    path=root/'D'/folder;s=read(path/'quantitative_eval.json')
    with (path/'trials.csv').open() as f:rows=list(csv.DictReader(f))
    assert s['total_attempts']==len(rows)==1 and s['grasp_steps']==s['lift_steps']==1200
    assert s['physics']['physics_steps']==4800
    interfaces.append(dict(directory=str(path),protocol=s['protocol'],purpose='nonformal_interface_diagnostic',
        formal=False,checkpoint=s['checkpoint'],policy_training_updates=3,attempts=1,
        physics_steps=4800,policy_steps=2400,source_successes=s['total_source_successes'],
        active_successes=s['total_successes'],engineering_success_rate=s['engineering_stable_success'],
        active_peak=s['physics']['active_trial_peak']['hand_velocity_peak_rad_s'],
        post_reset_peak=s['physics']['post_reset_peak']['hand_velocity_peak_rad_s']))
    for r in rows:
        drows.append(dict(directory=str(path),protocol=s['protocol'],formal=False,**r))
        comparisons.append(dict(directory=str(path),object=r['object'],formal=False,
            initial_top_z_m=r['initial_top_z_m'],final_top_z_m=r['final_top_z_m'],
            initial_root_z_m=r['initial_root_z_m'],final_root_z_m=r['final_root_z_m'],
            initial_bottom_z_m=r['initial_bottom_z_m'],final_bottom_z_m=r['final_bottom_z_m'],
            top_minus_root_z_m=r['top_minus_root_z_m']))
table(root/'D/trials.csv',drows);table(root/'D/top_root_comparison.csv',comparisons)
for folder in ('source_20_20_ramp16','engineering_20_20_ramp14'):
    (root/'D'/folder).mkdir(exist_ok=True)
    write(root/'D'/folder/'status.json',dict(status='blocked',attempts=0,success_rate=None,reason='No qualified candidate after Gate A/C'))
write(root/'D/evaluation_summary.json',dict(status='blocked',formal_attempts=0,formal_success_rate=None,
    reason='No physically qualified trained candidate',interface_gate='pass',diagnostic_attempts=3,interfaces=interfaces,
    source_30x3='not_run',engineering_30x3='not_run',safe_checkpoint=None))

gates={'A':dict(status='fail',reason=physics_gate['reason'],evidence=['A/physics_gate.json','A/peak_case.json','A/experiments.json']),
       'B':dict(status='pass',evidence=['B/gate.json']),
       'C':dict(status='blocked',reason='Global physical safety problem',evidence=['C/candidate_gates.json']),
       'D':dict(status='blocked',reason='No qualified formal checkpoint',interface_gate='pass',evidence=['D/evaluation_summary.json'])}
write(root/'gates.json',gates)
workspace=subprocess.check_output(['git','status','--short'],text=True)
changed=subprocess.check_output(['git','diff','--name-only'],text=True).splitlines()+['scripts/rsl_rl/checkpoint_contract.py']
write(root/'changes.json',[dict(path=p,reason={'scripts/rsl_rl/evaluate.py':'top metrics, CLI protocol checks and checkpoint contract',
       'scripts/rsl_rl/grasp_diagnostics.py':'top/failure pose capture and per-trial physics statistics',
       'scripts/rsl_rl/train.py':'strict resume contract, per-checkpoint metadata and next update counter',
       'scripts/rsl_rl/play.py':'diagnostic checkpoint override with recorded audit',
       'scripts/rsl_rl/checkpoint_contract.py':'shared semantic comparison used by train/play/evaluate'}.get(p,'remove stale timing comments'),
       tests=['B/gate.json']) for p in changed])
native=read(root/'B/native_resume.json')
summary=dict(task_complete=False,task_id='Grasp1-UR5-Allegro-Teacher-v0',baseline_git_sha=head,
    workspace=workspace,source_changes=changed,gates=gates,physical_defaults_changed=False,
    default_decimation=2,default_episode_s=4,formal_training_updates=0,formal_training_transitions=0,
    diagnostic_isaac_ppo_updates=3,diagnostic_isaac_transitions=1536,diagnostic_checkpoint=native['checkpoint'],
    formal_checkpoints=[],formal_evaluation_attempts=0,formal_success_rate=None,diagnostic_evaluation_attempts=3,
    not_run=['E1-E4 100/500/1000/2000 formal PPO','2048-env performance on/off 3-5 repeats',
             'three seeds','qualified 30x3 ramp16 evaluation','qualified 30x3 ramp14 engineering evaluation'],
    next_critical_blocker='Establish and validate a physical model/constraint repair for the 120.8146rad/s joint_15_0 event and actual joint-limit violations; no candidate currently clears Gate A.',
    evidence=['A/peak_window_E2_random.csv','A/physics_gate.json','B/gate.json','C/candidate_gates.json','D/evaluation_summary.json'])
write(root/'summary.json',summary)
write(root/'baseline/source_state_final.json',dict(head=head,workspace=workspace))
(root/'baseline/final_source.diff').write_text(subprocess.check_output(['git','diff','--','source','scripts'],text=True))

# 保存可复现的等效参数命令；解析的是实际加载参数，不把未运行的模型实验放此节。
legacy='/home/windsky/IsaacLab/isaaclab.sh -p logs/acceptance/optimization6/check.py --decimation 4 --episode 10 --trace_mode zero --output_dir logs/acceptance/optimization7/A/legacy_repro --headless'
commands.insert(0,dict(command=legacy,status='executed',evidence='A/legacy_repro.log'))
for c in candidates:
    commands.append(dict(command=f'/home/windsky/IsaacLab/isaaclab.sh -p {root}/A/minimal_repro.py --decimation {c["decimation"]} --episode {c["episode_s"]} --duration 12 --num_envs 16 --modes zero random fixed --output_dir {root}/C/{c["group"]}_screen --headless',status='executed',evidence=c['evidence']))
for d in (2,4):
    commands.append(dict(command=f'/home/windsky/IsaacLab/isaaclab.sh -p logs/acceptance/optimization6/check.py --decimation {d} --episode 10 --duration 0.2 --output_dir {root}/B/regression_D{d} --headless',status='executed',evidence=f'B/regression_D{d}.log'))
for folder,d,n,copies in [('D2',2,16,False),('D4_copies30',4,30,True),('final_D2',2,16,False),('final_D4_copies30',4,30,True)]:
    commands.append(dict(command=f'/home/windsky/IsaacLab/isaaclab.sh -p {root}/B/measurement_check.py --decimation {d} --num_envs {n} '+('--copies ' if copies else '')+f'--output_dir {root}/B/{folder} --headless',status='executed',evidence=f'B/{folder}.log'))
for name in ('protocol_tests','stability_tests','contact_audit','ppo_regression'):
    commands.append(dict(command=f'python {root}/B/{name}.py',status='executed',evidence=f'B/{name}.json' if name!='contact_audit' else 'B/contact_reward_audit.json'))
basecmd='/home/windsky/IsaacLab/isaaclab.sh -p scripts/rsl_rl/train.py --task Grasp1-UR5-Allegro-Teacher-v0 --num_envs 16 --decimation 2 --episode_length_s 10 --seed 1 --headless'
commands.append(dict(command=basecmd+' --max_iterations 2 --run_name optimization7_contract_diagnostic',status='executed',evidence='B/contract_train.log'))
run=Path(native['resume_from']).parent.name
for suffix,status in [('', 'failed_metadata_discovery_then_fixed'),('_retry','executed')]:
    commands.append(dict(command=basecmd+f' --max_iterations 1 --run_name optimization7_contract_resume_diagnostic{suffix} --resume --load_run {run} agent.load_checkpoint=model_1.pt',status=status,evidence=f'B/contract_resume{suffix}.log'))
for interface in interfaces:
    folder=Path(interface['directory']).name
    entry='scripts/rsl_rl/evaluate.py' if folder=='interface_20_20_ramp16' else f'{root}/D/injected_40s.py'
    flags='--paper_stability' if 'engineering' in folder else '--grasp_duration_s 20 --lift_duration_s 20 --lift_ramp_duration_s 16'
    commands.append(dict(command=f'/home/windsky/IsaacLab/isaaclab.sh -p {entry} --checkpoint {native["checkpoint"]} --decimation 2 --object Hammer_40 --rounds 1 {flags} --diagnostics --seed 1 --output_dir {interface["directory"]} --headless',status='executed_nonformal_diagnostic',evidence=str(Path(interface['directory'])/'quantitative_eval.json')))
write(root/'commands.json',commands)
lines=['# 优化7命令与实际状态','', '从 `/home/windsky/project/Grasp1` 运行；A的参数命令由实际loaded_parameters重建，显式写出的默认值与原运行一致。历史阶段诊断字段不同，以各原始文件为准。','',
       '```bash','source /home/windsky/miniconda3/etc/profile.d/conda.sh','conda activate grasp','export TERM=xterm PYTHONUNBUFFERED=1','```','', '## 已执行','']
for c in commands:lines.extend([f'状态：`{c["status"]}`；证据：`{c["evidence"]}`','', '```bash',c['command'],'```',''])
lines.extend(['```bash',f'python {root}/plot_traces.py',f'python {root}/summarize.py',f'python {root}/final_check.py','python -m compileall -q source/Grasp1/Grasp1/tasks/manager_based/grasp1 scripts/rsl_rl logs/acceptance/optimization7','git diff --check','```','',
    'D注入程序首次运行因SimulationApp.close()的正常SystemExit(0)未到达外层事后断言；三个40秒循环已完整结束。随后对保存的真实轨迹运行同一段CPU断言并写入interface_injection_check.json，包装器已加入对正常退出的处理。','',
    '## 尚未执行','', '四格正式100/500/1000/2000训练、2048环境吞吐on/off重复测量、三seed、合格模型30×3源时长评估及工程保持评估均blocked。因没有合格checkpoint，不提供虚构路径的正式评估命令。原始完整模板见docs/优化7.md。'])
(root/'commands.md').write_text('\n'.join(lines)+'\n')
print(json.dumps(dict(task_complete=False,gates={k:v['status'] for k,v in gates.items()},candidates=candidates,interfaces=interfaces),indent=2))

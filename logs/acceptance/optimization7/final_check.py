"""最终检查源码可编译及交付统计一致性，未执行实验不转换为失败率。"""
import csv
import json
from pathlib import Path
import subprocess
import sys

root = Path('logs/acceptance/optimization7')
def read(path): return json.loads(Path(path).read_text())
compile_result = subprocess.run([sys.executable,'-m','compileall','-q',
    'source/Grasp1/Grasp1/tasks/manager_based/grasp1','scripts/rsl_rl',str(root)],capture_output=True,text=True)
diff_result = subprocess.run(['git','diff','--check'],capture_output=True,text=True)
assert compile_result.returncode == diff_result.returncode == 0, compile_result.stdout+compile_result.stderr+diff_result.stdout
summary = read(root/'summary.json')
assert summary['task_complete'] is False
assert summary['formal_training_updates'] == summary['formal_evaluation_attempts'] == 0
assert summary['formal_success_rate'] is None
gates = read(root/'gates.json')
assert {k:v['status'] for k,v in gates.items()} == dict(A='fail',B='pass',C='blocked',D='blocked')
for result in ('protocol_tests','checkpoint_compatibility','stability_tests','ppo_regression','measurement_regression','gate'):
    assert read(root/f'B/{result}.json')['status'] == 'pass'
for folder in ('final_D2','final_D4_copies30'):
    m = read(root/f'B/{folder}/measurement_regression.json')
    assert m['status']=='pass' and m['first_failed_top_frozen'] and m['active_peak_unchanged']
    assert m['physics']['post_reset_peak']['hand_velocity_peak_rad_s'] == 123.
native = read(root/'B/native_resume.json')
assert Path(native['checkpoint']).is_file() and native['optimizer_steps_before']==[32.] and native['optimizer_steps_after']==[48.]
assert read(native['metadata'])['completed_updates']==3
with (root/'C/training_matrix.csv').open() as f: training=list(csv.DictReader(f))
assert len(training)==16 and all(r['status']=='not_run' and int(r['actual_updates'])==0 for r in training)
for folder in ('source_20_20_ramp16','engineering_20_20_ramp14'):
    formal=read(root/f'D/{folder}/status.json')
    assert formal['status']=='blocked' and formal['attempts']==0 and formal['success_rate'] is None
with (root/'D/trials.csv').open() as f: rows=list(csv.DictReader(f))
assert len(rows)==3 and all(r['evaluation_role']=='nonformal_interface_diagnostic' for r in rows)
for folder in ('interface_20_20_ramp16','interface_injected_20_20_ramp16','interface_injected_engineering_ramp14'):
    d=read(root/f'D/{folder}/quantitative_eval.json')
    assert d['total_attempts']==1 and d['grasp_steps']==d['lift_steps']==1200 and d['physics']['physics_steps']==4800
    assert not d['formal_candidate_qualified'] and d['policy_training_updates']==3
    if folder!='interface_20_20_ramp16': assert read(root/f'D/{folder}/interface_injection_check.json')['status']=='pass'
assert len(read(root/'A/collider_audit_replay/loaded_parameters.json')['usd_colliders'])==27
for artifact in ('A/diagnosis.md','A/physics_source_audit.md','A/physics_gate.json','A/peak_cases.csv',
                 'A/physics_event_curves.png','D/interface_40s_height_curves.png','commands.md'):
    assert (root/artifact).is_file(), artifact
assert Path('docs/优化7_执行报告.md').is_file()
result=dict(status='pass',compileall_exit_code=compile_result.returncode,diff_check_exit_code=diff_result.returncode,
            artifact_consistency='pass',task_complete=False,formal_updates=0,formal_attempts=0,diagnostic_updates=3,
            nonformal_40s_attempts=3,report='docs/优化7_执行报告.md')
(root/'final_validation.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
print(json.dumps(result,indent=2))

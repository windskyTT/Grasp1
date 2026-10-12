"""非正式40秒入口诊断：微型接口策略，提前终止后制造测量峰值。"""
import json
from pathlib import Path
import runpy
import sys
import gymnasium as gym

sys.path.insert(0,str(Path('scripts/rsl_rl').resolve()))
original_make=gym.make
output=Path(sys.argv[sys.argv.index('--output_dir')+1])
restores=[]
def make_with_injection(*args,**kwargs):
    result=original_make(*args,**kwargs)
    env=result.unwrapped
    import torch
    old_update=env.scene.update
    term_cfg=env.termination_manager.get_term_cfg('invalid_observation')
    old_func=term_cfg.func
    robot=env.scene['robot']
    ids=env.action_manager.get_term('teacher')._joint_ids
    failure_at=round(2./env.step_dt)
    def termination(e,**params):
        mask=old_func(e,**params)
        if e.common_step_counter==failure_at:
            mask=mask.clone(); mask[0]=True
        return mask
    def update(dt):
        old_update(dt)
        if dt==env.physics_dt and env.common_step_counter>=failure_at:
            # 后续新episode的读数用于隔离断言；不写入PhysX真实速度。
            robot.data.joint_vel[0,ids[-1]]=123.
    term_cfg.func=termination
    env.termination_manager.set_term_cfg('invalid_observation',term_cfg)
    env.scene.update=update
    restores.append((env,old_update,term_cfg,old_func))
    return result
gym.make=make_with_injection
try:
    print('OPT7_NONFORMAL_INTERFACE_DIAGNOSTIC: 3-update interface policy, injected failure, excluded from formal evaluation.',flush=True)
    runpy.run_path('scripts/rsl_rl/evaluate.py',run_name='__main__')
except SystemExit as error:
    # IsaacSim 5.1 SimulationApp.close()正常退出也抛SystemExit(0)。
    if error.code not in (None,0):
        raise
finally:
    gym.make=original_make
    # evaluate.py已在finally恢复自身回调。环境关闭后只恢复原Python函数绑定。
    for env,old_update,term_cfg,old_func in restores:
        env.scene.update=old_update
        term_cfg.func=old_func
import csv
import torch
summary=json.loads((output/'quantitative_eval.json').read_text())
row=list(csv.DictReader((output/'trials.csv').open()))[0]
trace=torch.load(output/'height_relative_trajectories.pt',map_location='cpu',weights_only=False)['rounds'][0]
assert summary['grasp_duration_s']==summary['lift_duration_s']==20.
assert len(trace)==summary['grasp_steps']+summary['lift_steps']
failure=int(row['failure_step'])
assert failure>0 and row['failure_reason']=='invalid_observation'
assert (trace[failure-1:,0,10]==trace[failure-1,0,10]).all()
assert summary['physics']['active_trial_peak']['hand_velocity_peak_rad_s']<123.
assert summary['physics']['post_reset_peak']['hand_velocity_peak_rad_s']>=123.
summary.update(evaluation_role='nonformal_interface_diagnostic',policy_training_updates=3,formal_candidate_qualified=False,injected_failure=True)
(output/'quantitative_eval.json').write_text(json.dumps(summary,indent=2))
with (output/'trials.csv').open() as file:
    rows=list(csv.DictReader(file))
for trial in rows:
    trial.update(evaluation_role='nonformal_interface_diagnostic',policy_training_updates=3,injected_failure=True)
with (output/'trials.csv').open('w',newline='') as file:
    writer=csv.DictWriter(file,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
check=dict(status='pass',purpose='nonformal_40s_interface_injection',formal=False,policy_training_updates=3,
           attempts=1,simulated_duration_s=40.,physics_steps=summary['physics']['physics_steps'],
           policy_steps=len(trace),failure_step=failure,top_frozen=True,
           active_trial_peak=summary['physics']['active_trial_peak']['hand_velocity_peak_rad_s'],
           post_reset_peak=summary['physics']['post_reset_peak']['hand_velocity_peak_rad_s'],comparability_warning=summary['comparability_warning'])
(output/'interface_injection_check.json').write_text(json.dumps(check,indent=2))
print('OPT7_40S_INTERFACE_PASS',json.dumps(check),flush=True)

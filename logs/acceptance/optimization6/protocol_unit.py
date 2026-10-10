"""源评估默认参数和独立5秒保持分类器的实际函数测试。"""
import argparse,ast,json,math,sys
from pathlib import Path
from types import SimpleNamespace
from isaaclab.app import AppLauncher
parser=argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args=parser.parse_args()
launcher=AppLauncher(args)
app=launcher.app
import torch
sys.path.insert(0,str(Path('scripts/rsl_rl').resolve()))
from grasp_diagnostics import EvaluationTrajectories
from isaaclab.envs import ManagerBasedRLEnv
root=Path('logs/acceptance/optimization6')
try:
    tree=ast.parse(Path('scripts/rsl_rl/evaluate.py').read_text())
    defaults={n.args[0].value:next(k.value.value for k in n.keywords if k.arg=='default') for n in ast.walk(tree)
              if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='add_argument'
              and n.args and isinstance(n.args[0],ast.Constant) and any(k.arg=='default' and isinstance(k.value,ast.Constant) for k in n.keywords)}
    assert (defaults['--grasp_duration_s'],defaults['--lift_duration_s'],defaults['--lift_ramp_duration_s'],defaults['--success_height'])==(20.,20.,16.,.1)
    cases=[]
    for decimation in (2,4):
        dt=decimation/120
        steps=round(5/dt)
        timeout=ManagerBasedRLEnv.max_episode_length.fget(SimpleNamespace(max_episode_length_s=(round(40/dt)+2)*dt,step_dt=dt))
        assert timeout>round(40/dt)
        for failure in ('none','height_drop','contact_gap','translation_slip','rotation_slip','terminated'):
            trial=EvaluationTrajectories.__new__(EvaluationTrajectories)
            trial.env=SimpleNamespace(step_dt=dt)
            trial.args=SimpleNamespace(paper_stability=True,success_height=.1,max_relative_translation_drift_m=.02,max_relative_rotation_drift_rad=.2)
            trial.names=('stable_sample','comparison_sample')
            trial.rows=[];trial.rounds=[];trial.index=0
            trial.active=torch.ones(2,dtype=torch.bool)
            trial.failures={'time_out':torch.zeros(2,dtype=torch.bool)}
            trial.failure_step=torch.full((2,),-1)
            samples=torch.zeros((steps+1,2,10))
            samples[:,:,0]=.2
            samples[:,:,4]=1.
            samples[:,:,8:10]=1.
            if failure=='height_drop': samples[steps//2,1,0]=.09
            if failure=='contact_gap': samples[steps//2,1,9]=0.
            if failure=='translation_slip': samples[-1,1,1]=.03
            if failure=='rotation_slip': samples[-1,1,4]=math.cos(.4/2);samples[-1,1,5]=math.sin(.4/2)
            if failure=='terminated': trial.active[1]=False
            trial.trace=list(samples)
            outcome=trial.finish_round(torch.ones(2,dtype=torch.bool),trial.active)
            assert bool(outcome[0])
            assert bool(outcome[1])==(failure=='none')
            cases.append(dict(decimation=decimation,hold_intervals=steps,hold_s=steps*dt,failure=failure,outcomes=outcome.tolist(),timeout_steps=timeout))
    result=dict(defaults=defaults,stability_cases=cases,all_pass=True,
                limitation='Synthetic classifier tests and runtime timeout property; not physical grasp or a 40s policy evaluation.')
    (root/'protocol_unit.json').write_text(json.dumps(result,indent=2))
    print('PROTOCOL_UNIT_PASS',flush=True)
except Exception:
    import traceback
    (root/'protocol_unit.error.txt').write_text(traceback.format_exc())
    print(traceback.format_exc(),flush=True)
    raise
finally:
    app.close()

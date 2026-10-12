"""最后五秒工程保持的独立合成轨迹测试；不代表物理抓取成功。"""
import ast
import json
from pathlib import Path
from types import SimpleNamespace
import torch

module=ast.parse(Path('scripts/rsl_rl/grasp_diagnostics.py').read_text())
node=next(n for n in module.body if isinstance(n,ast.ClassDef) and n.name=='EvaluationTrajectories')
scope={'torch':torch}
exec(compile(ast.Module(body=[node],type_ignores=[]),'<EvaluationTrajectories>','exec'),scope)
Cls=scope['EvaluationTrajectories']
rows=[]
for decimation in (2,4):
    dt=decimation/120
    for case in ('pass','height_drop','contact_loss','translation','rotation','early_done','short_window'):
        obj=object.__new__(Cls)
        obj.env=SimpleNamespace(step_dt=dt)
        obj.args=SimpleNamespace(paper_stability=True,success_height=0.1,
            max_relative_translation_drift_m=0.02,max_relative_rotation_drift_rad=0.2)
        steps=round(40/dt) if case!='short_window' else round(4/dt)
        trace=torch.zeros((steps,1,13)); trace[:,:,0]=0.2; trace[:,:,4]=1.; trace[:,:,8:10]=1.
        if case=='height_drop': trace[-1,:,0]=0.05
        if case=='contact_loss': trace[-1,:,9]=0.
        if case=='translation': trace[-1,:,1]=0.03
        if case=='rotation': trace[-1,:,4]=torch.cos(torch.tensor(.15)); trace[-1,:,7]=torch.sin(torch.tensor(.15))
        if case=='early_done': trace[-1,:,8]=0.
        obj.trace=list(trace); obj.active=trace[-1,:,8].bool(); obj.names=('synthetic',); obj.index=0
        obj.rows=[]; obj.rounds=[]; obj.failures={}; obj.failure_step=torch.tensor([-1])
        obj.initial_z=obj.initial_root_z=obj.initial_bottom_z=torch.tensor([.75])
        obj.final_object_pose=obj.final_root_pose=obj.final_bottom_pose=torch.tensor([[0.,0.,.95,1.,0.,0.,0.]])
        obj.height_peak=torch.tensor([.2]); obj.reset_count=obj.round_initial_resets=torch.tensor([0])
        stable=obj.finish_round(torch.tensor([True]),obj.active)
        assert bool(stable[0])==(case=='pass'),(decimation,case,obj.rows)
        rows.append(dict(decimation=decimation,case=case,status='pass',result=obj.rows[0]))
Path('logs/acceptance/optimization7/B/stability_tests.json').write_text(json.dumps(dict(status='pass',synthetic=True,cases=rows),indent=2))
print('OPT7_STABILITY_PASS',len(rows))

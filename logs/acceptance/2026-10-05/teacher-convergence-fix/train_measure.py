"""执行现有train.py，记录优化3所需终止、物体限位和有限性数据。"""
import json
from pathlib import Path
import runpy
import sys
from isaaclab.app import AppLauncher

run_name=sys.argv[sys.argv.index('--run_name')+1]
root=Path('logs/acceptance/2026-10-05/teacher-convergence-fix')
counts=names=qmin=qmax=finite=None
launcher_init=AppLauncher.__init__
def launch_and_measure(self,*args,**kwargs):
    launcher_init(self,*args,**kwargs)
    import torch
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from rsl_rl.runners import OnPolicyRunner
    original_step=RslRlVecEnvWrapper.step
    def measured_step(self,actions):
        global counts,names,qmin,qmax,finite
        e=self.unwrapped
        if counts is None:
            names=e.termination_manager.active_terms
            counts=torch.zeros(len(names),device=e.device,dtype=torch.long)
            qmin=e.scene['object'].data.joint_pos.clone(); qmax=qmin.clone()
            finite=torch.ones((),device=e.device,dtype=torch.bool)
            original_update=e.scene.update
            def measured_update(dt):
                global qmin,qmax,finite
                original_update(dt)
                q=e.scene['object'].data.joint_pos
                qmin=torch.minimum(qmin,q); qmax=torch.maximum(qmax,q)
                finite &= torch.isfinite(q).all() & torch.isfinite(e.scene['robot'].data.joint_vel).all()
            e.scene.update=measured_update
        result=original_step(self,actions)
        counts.add_(torch.stack([e.termination_manager.get_term(name).sum() for name in names]))
        finite &= torch.isfinite(actions).all() & torch.isfinite(result[1]).all()
        for obs in result[0].values(): finite &= torch.isfinite(obs).all()
        return result
    RslRlVecEnvWrapper.step=measured_step
    original_learn=OnPolicyRunner.learn
    def measured_learn(self,*args,**kwargs):
        value=original_learn(self,*args,**kwargs)
        result=dict(terminations=dict(zip(names,counts.tolist())),finite=bool(finite),
            object_joint_min=float(qmin.min()),object_joint_max=float(qmax.max()),
            cuda_peak_allocated_mib=torch.cuda.max_memory_allocated()/1024**2)
        (root/(run_name+'-runtime.json')).write_text(json.dumps(result,indent=2))
        print('TRAINING_RUNTIME',json.dumps(result),flush=True)
        return value
    OnPolicyRunner.learn=measured_learn
AppLauncher.__init__=launch_and_measure
sys.path.insert(0,str(Path('scripts/rsl_rl').resolve()))
sys.argv[0]='scripts/rsl_rl/train.py'
runpy.run_path('scripts/rsl_rl/train.py',run_name='__main__')

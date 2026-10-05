"""执行现有 train.py，并仅在本次 smoke 进程记录逐步终止次数。"""
import json
from pathlib import Path
import runpy
import sys
from isaaclab.app import AppLauncher

counts = None
names = None
launcher_init = AppLauncher.__init__
def launch_and_measure(self, *args, **kwargs):
    launcher_init(self, *args, **kwargs)
    import torch
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    original_step = RslRlVecEnvWrapper.step
    def measured_step(self, actions):
        global counts, names
        value = original_step(self, actions)
        manager = self.unwrapped.termination_manager
        if counts is None:
            names = manager.active_terms
            counts = torch.zeros(len(names), device=self.unwrapped.device, dtype=torch.long)
        counts.add_(torch.stack([manager.get_term(name).sum() for name in names]))
        return value
    RslRlVecEnvWrapper.step = measured_step
    from rsl_rl.runners import OnPolicyRunner
    original_learn = OnPolicyRunner.learn
    def measured_learn(self, *args, **kwargs):
        value = original_learn(self, *args, **kwargs)
        result = dict(zip(names, counts.tolist()))
        Path('logs/acceptance/2026-10-02/dexsuite-teacher-modification/ppo-termination-counts.json').write_text(json.dumps(result, indent=2))
        print('PPO_TERMINATION_COUNTS', json.dumps(result), flush=True)
        return value
    OnPolicyRunner.learn = measured_learn
AppLauncher.__init__ = launch_and_measure
sys.path.insert(0, str(Path('scripts/rsl_rl').resolve()))
sys.argv[0] = 'scripts/rsl_rl/train.py'
runpy.run_path('scripts/rsl_rl/train.py', run_name='__main__')

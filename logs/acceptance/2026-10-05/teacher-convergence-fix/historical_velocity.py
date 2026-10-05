"""按优化2原验收流程复现29.98 rad/s，并补记录具体关节。"""
import json
from pathlib import Path
import runpy
from isaaclab.app import AppLauncher

peak=None
launcher_init=AppLauncher.__init__
def launch_baseline(self,*args,**kwargs):
    launcher_init(self,*args,**kwargs)
    import torch
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.envs.mdp import RelativeJointPositionAction
    import isaaclab_tasks.utils as utils
    original_parse=utils.parse_env_cfg
    def parse_baseline(*args,**kwargs):
        cfg=original_parse(*args,**kwargs)
        cfg.actions.teacher.class_type=RelativeJointPositionAction
        cfg.actions.teacher.scale={k:v*4 for k,v in cfg.actions.teacher.scale.items()}
        cfg.scene.object.spawn.joint_drive_props=None
        cfg.events.configure_object_joint.params['armature']=0.0
        for reward in vars(cfg.rewards).values(): reward.weight*=12
        return cfg
    utils.parse_env_cfg=parse_baseline
    original_step=ManagerBasedRLEnv.step
    def measured_step(self,actions):
        global peak
        result=original_step(self,actions)
        velocity=self.scene['robot'].data.joint_vel.abs().max(0).values
        peak=velocity.clone() if peak is None else torch.maximum(peak,velocity)
        Path('logs/acceptance/2026-10-05/teacher-convergence-fix/historical-velocity-by-joint.json').write_text(json.dumps(dict(zip(self.scene['robot'].joint_names,peak.tolist())),indent=2))
        return result
    ManagerBasedRLEnv.step=measured_step
AppLauncher.__init__=launch_baseline
runpy.run_path('logs/acceptance/2026-10-02/dexsuite-teacher-modification/benchmark.py',run_name='__main__')

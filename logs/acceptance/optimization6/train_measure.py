"""运行既有train.py；优化6训练诊断只读取状态，Gate失败保存现场并结束。"""
import argparse
import ast
import json
from pathlib import Path
import runpy
import sys
import time
from isaaclab.app import AppLauncher
parser=argparse.ArgumentParser(add_help=False)
parser.add_argument('--measurement_dir',type=Path,required=True)
args,train_args=parser.parse_known_args()
stage=args.measurement_dir.resolve()
stage.mkdir(parents=True,exist_ok=True)
sys.path.insert(0,str(Path('scripts/rsl_rl').resolve()))
sys.argv=['scripts/rsl_rl/train.py',*train_args]
original_launch=AppLauncher.__init__
state={}

def launch(self,*args,**kwargs):
    original_launch(self,*args,**kwargs)
    import gymnasium as gym
    import pynvml
    import torch
    from rsl_rl.runners import OnPolicyRunner
    from rsl_rl.algorithms import PPO
    from grasp_diagnostics import PhysicsDiagnostics,configure_diagnostics
    from Grasp1.tasks.manager_based.grasp1.mdp.keypoints import hand_keypoints_o
    pynvml.nvmlInit()
    gpu=pynvml.nvmlDeviceGetHandleByIndex(0)
    original_make=gym.make
    def make(task,**kwargs):
        configure_diagnostics(kwargs['cfg'].scene)
        return original_make(task,**kwargs)
    gym.make=make
    # 使用优化5已验证的只读KL测量函数；不改算法更新、损失或optimizer。
    old=Path('logs/acceptance/2026-10-06/teacher-ppo-stability/train_measure.py')
    tree=ast.parse(old.read_text())
    node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='instrument_ppo_update')
    import inspect,textwrap
    ns=dict(inspect=inspect,textwrap=textwrap,stage=stage)
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(old),'exec'),ns)
    PPO.update=ns['instrument_ppo_update'](PPO.update)
    original_learn=OnPolicyRunner.learn
    original_log=OnPolicyRunner.log
    original_load=OnPolicyRunner.load
    def load(runner,*args,**kwargs):
        value=original_load(runner,*args,**kwargs)
        runner.current_learning_iteration+=1
        runner.alg.learning_rate=runner.alg.optimizer.param_groups[0]['lr']
        return value
    OnPolicyRunner.load=load
    def learn(runner,*args,**kwargs):
        e=runner.env.unwrapped
        physics=PhysicsDiagnostics(e)
        state.update(runner=runner,physics=physics,started=time.perf_counter(),records=[],gate_stop=None)
        counts={name:torch.zeros((),device=e.device,dtype=torch.long) for name in e.termination_manager.active_terms}
        behavior=torch.zeros((2,12),device=e.device)
        episodes=torch.zeros((),device=e.device,dtype=torch.long)
        contact_seen=torch.zeros(e.num_envs,device=e.device,dtype=torch.bool)
        two_seen=contact_seen.clone()
        contact_time=torch.zeros((),device=e.device)
        two_time=contact_time.clone()
        contact_episodes=torch.zeros((),device=e.device)
        two_episodes=contact_episodes.clone()
        actual_elapsed=torch.zeros(e.num_envs,device=e.device)
        streak=torch.zeros(e.num_envs,device=e.device)
        longest=streak.clone()
        original_reward=e.reward_manager.compute
        reward_sums=torch.zeros(len(e.reward_manager.active_terms),device=e.device)
        def reward(dt):
            value=original_reward(dt)
            fresh=actual_elapsed==0
            actual_elapsed.add_(e.step_dt)
            elapsed=actual_elapsed
            contact_seen[fresh]=False
            two_seen[fresh]=False
            streak[fresh]=0
            top=torch.stack([e.scene[f'teacher_af_contact_{i}'].data.force_matrix_w[:,0,0].norm(dim=-1)>0.1 for i in range(13)],1)
            bottom=torch.stack([e.scene[f'diagnostic_bottom_{i}'].data.force_matrix_w[:,0,0].norm(dim=-1)>0.1 for i in range(13)],1)
            table=torch.stack([e.scene[f'teacher_table_contact_{i}'].data.force_matrix_w[:,0,0].norm(dim=-1)>0.1 for i in range(13)],1)
            contact=top|bottom
            bodies=contact.sum(1)
            fingers=torch.stack([contact[:,1+3*i:4+3*i].any(1) for i in range(4)],1)
            distance=torch.cdist(hand_keypoints_o(e),e._teacher_affordance_points_o).amin(dim=(1,2))
            first=contact.any(1)&~contact_seen
            first_two=(bodies>=2)&~two_seen
            contact_time.add_(elapsed[first].sum())
            two_time.add_(elapsed[first_two].sum())
            contact_episodes.add_(first.sum())
            two_episodes.add_(first_two.sum())
            contact_seen[:] |= contact.any(1)
            two_seen[:] |= bodies>=2
            streak[:] = (streak+e.step_dt)*contact.any(1)
            longest[:] = torch.maximum(longest,streak)
            t=e.action_manager.get_term('teacher')
            tracking=(t.target-e.scene['robot'].data.joint_pos[:,t._joint_ids]).abs().mean(1)
            metrics=torch.stack([torch.ones_like(distance),contact.any(1),top.any(1),bottom.any(1),table.any(1),bodies>=2,
                                 bodies>=3,fingers[:,3] & fingers[:,:3].any(1),distance,tracking,
                                 e.scene['object'].data.root_pos_w[:,2]-e._teacher_reset_data['object_pose'][:,2],value],1)
            for phase,mask in enumerate((elapsed<=4.0,elapsed>4.0)):
                behavior[phase] += metrics[mask].sum(0)
            reward_sums.add_(e.reward_manager._step_reward.sum(0)*dt)
            for name in counts:
                counts[name]+=e.termination_manager.get_term(name).sum()
            episodes.add_(e.reset_buf.sum())
            actual_elapsed[e.reset_buf]=0
            return value
        e.reward_manager.compute=reward
        state.update(counts=counts,behavior=behavior,episodes=episodes,reward_sums=reward_sums,
                     contact_time=contact_time,two_time=two_time,contact_episodes=contact_episodes,
                     two_episodes=two_episodes,longest=longest)
        try:
            return original_learn(runner,*args,**kwargs)
        finally:
            save(runner)
    OnPolicyRunner.learn=learn
    def save(runner):
        e=runner.env.unwrapped
        p=state['physics'].summary()
        wall=time.perf_counter()-state['started']
        updates=len(state['records'])
        transitions=updates*e.num_envs*runner.num_steps_per_env
        phase=[]
        for name,row in zip(('0_4s','4_10s'),state['behavior'].cpu().tolist()):
            denom=row[0]
            labels=['samples','contact','top','bottom','table','two_plus','three_plus','thumb_other','distance_m','tracking_rad','height_gain_m','reward']
            phase.append(dict(phase=name,**{key:(value/denom if key!='samples' and denom else value if key=='samples' else None) for key,value in zip(labels,row)}))
        records=state['records']
        summary=dict(run_path=runner.log_dir,checkpoint=str(Path(runner.log_dir)/f'model_{runner.current_learning_iteration}.pt'),
                     total_updates=runner.current_learning_iteration+1,stage_updates=updates,transitions=transitions,
                     aggregate_simulated_s=transitions*e.step_dt,parallel_simulated_s=updates*32*e.step_dt,
                     environment_physx_steps=transitions*e.cfg.decimation,gpu_wall_s=wall,
                     fps=transitions/wall,wall_s_per_million_transitions=wall/transitions*1e6 if transitions else None,
                     wall_s_per_parallel_simulated_s=wall/(updates*32*e.step_dt) if updates else None,
                     effective_transitions_per_gpu_hour=transitions/wall*3600,
                     success_per_gpu_hour=None,physics=p,behavior=phase,
                     terminations={name:int(value) for name,value in state['counts'].items()},
                     reward_terms={name:float(value)/transitions if transitions else None for name,value in zip(e.reward_manager.active_terms,state['reward_sums'])},
                     first_contact_mean_s=float(state['contact_time']/state['contact_episodes']) if int(state['contact_episodes']) else None,
                     first_two_contact_mean_s=float(state['two_time']/state['two_episodes']) if int(state['two_episodes']) else None,
                     longest_contact_s=float(state['longest'].max()),
                     vram_peak_mib=max((r['vram_mib'] for r in records),default=0),
                     gate_stop=state['gate_stop'],step_dt=e.step_dt,decimation=e.cfg.decimation,episode_s=e.cfg.episode_length_s,
                     seed=e.cfg.seed,gamma=runner.alg.gamma,lam=runner.alg.lam,num_envs=e.num_envs,
                     last20_value_loss=sum(r['value_function'] for r in records[-20:])/len(records[-20:]) if records else None,
                     peak_value_loss=max((r['value_function'] for r in records),default=None))
        (stage/'runtime.json').write_text(json.dumps(summary,indent=2,allow_nan=False))
    def log(runner,locs,*args,**kwargs):
        original_log(runner,locs,*args,**kwargs)
        p=state['physics'].summary()
        row=dict(iteration=locs['it']+1,collection_time=locs['collection_time'],learning_time=locs['learn_time'],
                 mean_reward=sum(locs['rewbuffer'])/len(locs['rewbuffer']) if locs['rewbuffer'] else None,
                 std_min=float(runner.alg.policy.std.min()),std_max=float(runner.alg.policy.std.max()),
                 vram_mib=pynvml.nvmlDeviceGetMemoryInfo(gpu).used/1024**2,
                 **locs['loss_dict'],hand_velocity_peak=p['hand_velocity_peak'],contact_force_peak_N=p['contact_force_peak_N'],
                 clipping_fraction=p['target_clipping_fraction'])
        state['records'].append(row)
        with (stage/'iterations.jsonl').open('a') as file:
            file.write(json.dumps(row)+'\n')
        if (not p['finite'] or p['hand_velocity_peak']>=100. or p['contact_force_peak_N']>=10000.
            or p['object_joint_min_rad'] < -0.001 or p['object_joint_max_rad'] > 0.002):
            state['gate_stop']='physics anomaly: screening limit >=100rad/s hand or >=10kN normal contact, or nonfinite'
            runner.save(str(Path(runner.log_dir)/f'model_{locs["it"]}.pt'))
            save(runner)
            raise RuntimeError(state['gate_stop'])
    OnPolicyRunner.log=log
AppLauncher.__init__=launch
try:
    runpy.run_path('scripts/rsl_rl/train.py',run_name='__main__')
except Exception:
    import traceback
    (stage/'exception.txt').write_text(traceback.format_exc())
    print(traceback.format_exc(),flush=True)
    raise

"""本轮验收的稳定性/吞吐测量；不修改任务配置或训练实现。"""
import argparse
from isaaclab.app import AppLauncher
parser=argparse.ArgumentParser()
parser.add_argument('--num_envs',type=int,required=True)
parser.add_argument('--steps',type=int,default=140)
parser.add_argument('--seconds',type=float,default=0)
parser.add_argument('--profile',action='store_true')
AppLauncher.add_app_launcher_args(parser)
args=parser.parse_args()
app=AppLauncher(args).app
import json
import subprocess
import time
import torch
import gymnasium as gym
import Grasp1.tasks
from isaaclab_tasks.utils import parse_env_cfg

def gpu_sample():
    result=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
    return [float(x) for x in result.strip().split(',')]

try:
    cfg=parse_env_cfg('Grasp1-UR5-Allegro-Teacher-v0',device='cuda:0',num_envs=args.num_envs)
    cfg.set_num_envs(args.num_envs)
    env=gym.make('Grasp1-UR5-Allegro-Teacher-v0',cfg=cfg)
    e=env.unwrapped
    done_counts=torch.zeros((),device=e.device,dtype=torch.long)
    invalid_counts=torch.zeros((),device=e.device,dtype=torch.long)
    samples=[]
    with torch.inference_mode():
        env.reset()
        actions=torch.zeros((args.num_envs,22),device=e.device)
        for _ in range(10):
            env.step(actions)
        torch.cuda.synchronize()
        start=time.monotonic()
        count=0
        next_sample=0.0
        while count<args.steps or (args.seconds and time.monotonic()-start<args.seconds):
            actions.uniform_(-1,1)
            obs,reward,terminated,truncated,_=env.step(actions)
            done_counts+=(terminated|truncated).sum()
            invalid_counts+=e.termination_manager.get_term('invalid_observation').sum()
            if args.seconds:
                for value in (obs['policy'],reward,e.scene['robot'].data.joint_pos,e.scene['robot'].data.joint_vel,e.scene['object'].data.root_state_w):
                    assert torch.isfinite(value).all()
            count+=1
            elapsed=time.monotonic()-start
            if elapsed>=next_sample:
                for value in (obs['policy'],reward,e.scene['robot'].data.joint_pos,e.scene['object'].data.root_state_w):
                    assert torch.isfinite(value).all()
                memory,util=gpu_sample()
                samples.append([elapsed,memory,util])
                print('MONITOR',count,elapsed,memory,util,flush=True)
                next_sample=elapsed+10
        torch.cuda.synchronize()
        elapsed=time.monotonic()-start
        assert int(invalid_counts)==0
        result=dict(invalid_observations=int(invalid_counts),num_envs=args.num_envs,steps=count,seconds=elapsed,env_steps_per_second=count*args.num_envs/elapsed,done=int(done_counts),gpu_samples=samples,stable=True)
        print('BENCHMARK_RESULT',json.dumps(result),flush=True)
        if args.profile:
            from torch.profiler import profile,ProfilerActivity,record_function
            def wrap(obj,name,label):
                original=getattr(obj,name)
                def measured(*a,**kw):
                    with record_function(label):
                        return original(*a,**kw)
                setattr(obj,name,measured)
            wrap(e.sim,'step','acceptance/physics_step')
            wrap(e.scene,'update','acceptance/scene_update')
            wrap(e.observation_manager,'compute','acceptance/observations')
            wrap(e.reward_manager,'compute','acceptance/rewards')
            wrap(e.event_manager,'apply','acceptance/events')
            with profile(activities=[ProfilerActivity.CPU,ProfilerActivity.CUDA]) as prof:
                for _ in range(5):
                    env.step(actions)
            prof.export_chrome_trace('logs/acceptance/2026-09-29/torch-profile.json')
            print(prof.key_averages().table(sort_by='cpu_time_total',row_limit=25),flush=True)
    env.close()
except BaseException:
    import traceback
    traceback.print_exc()
    raise
finally:
    app.close()

"""只从保存的真实轨迹绘图；旧/新异常是不同输入案例，不能当作修复前后。"""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch

root = Path('logs/acceptance/optimization7')
old = torch.load('logs/acceptance/optimization6/diagnosis/zero_physics_trace.pt',map_location='cpu',weights_only=False)
new = torch.load(root/'A/trace_peak_E2_random.pt',map_location='cpu',weights_only=False)
fig, axes = plt.subplots(3,2,figsize=(12,8),sharex='col')
for col,(data,peak,title) in enumerate(((old,892,'Original D4/T10 zero'),(new,776,'Current D2/T10 random'))):
    first = data.get('first_physics_step',1)
    ix = torch.arange(peak-50-first,peak+51-first)
    joint = data['joints'].index('joint_15_0')
    q = data['q'][:,7,joint]
    delta = data['delta_v'][:,7,joint] if 'delta_v' in data else torch.cat((torch.zeros(1),(q[1:]-q[:-1])/data['physics_dt']))
    time = (ix+first-peak)*data['physics_dt']
    axes[0,col].plot(time,q[ix],label='actual q')
    axes[0,col].plot(time,data['target'][ix,7,joint],label='cached target')
    axes[1,col].plot(time,data['v'][ix,7,joint],label='reported qvel')
    axes[1,col].plot(time,delta[ix],label='position difference / dt')
    contact = data['contacts'].amax(-1) if 'contacts' in data else data['normal'].norm(dim=-1).amax(-1)
    axes[2,col].plot(time,contact[ix,7],label='sensor maximum')
    axes[0,col].set_title(title+'; wood / env 7 / joint 15')
    for row,label in enumerate(('Joint position (rad)','Velocity (rad/s)','Normal force (N)')):
        axes[row,col].set_ylabel(label);axes[row,col].grid(alpha=.3);axes[row,col].legend(fontsize=8)
    axes[2,col].set_xlabel('Seconds relative to velocity event')
fig.suptitle('Different input cases; no physical repair adopted')
fig.tight_layout();fig.savefig(root/'A/physics_event_curves.png',dpi=160);plt.close(fig)

fig,axes = plt.subplots(3,1,figsize=(10,8),sharex=True)
folders = ('interface_20_20_ramp16','interface_injected_20_20_ramp16','interface_injected_engineering_ramp14')
for ax,folder in zip(axes,folders):
    data = torch.load(root/'D'/folder/'height_relative_trajectories.pt',map_location='cpu',weights_only=False)
    trace = data['rounds'][0][:,0]
    t = torch.arange(1,len(trace)+1)*data['step_dt']
    ax.plot(t,trace[:,0],label='top height gain')
    ax.axhline(.1,color='black',linestyle='--',label='0.1m criterion')
    ax.axvline(20,color='gray',linestyle=':')
    ax.set_title(folder);ax.set_ylabel('Height gain (m)');ax.grid(alpha=.3);ax.legend(fontsize=8)
axes[-1].set_xlabel('Real simulation time (s)')
fig.suptitle('Three-update diagnostic policy; nonformal interface tests')
fig.tight_layout();fig.savefig(root/'D/interface_40s_height_curves.png',dpi=160);plt.close(fig)
print('Saved physical event and 40s diagnostic curves.')

"""用真实逐物理帧世界力重算接触方向分量；不改默认Reward。"""
import csv
import json
from pathlib import Path
import torch

source=Path('logs/acceptance/optimization7/A/single_wood_D4/trace_replay.pt')
data=torch.load(source,map_location='cpu',weights_only=False)
n,f=data['normal'][:,:,:13],data['friction'][:,:,:13]
total=(n+f)*data['physics_dt']
current_xy=f.norm(dim=-1)*data['physics_dt']; current_z=n.norm(dim=-1)*data['physics_dt']
world_z=total[:,:,:,2]
# 最后一维必须是世界坐标XYZ；XY范数只在最后一维投影。
world_xy=total[...,:2].norm(dim=-1)
valid=n.norm(dim=-1)>0.1
assert bool(valid.any())
report=dict(status='pass',trace=str(source),physics_dt=data['physics_dt'],force_frame='world',
    samples=int(valid.sum()),default_reward_changed=False,
    current_tangential_mean_Ns=float(current_xy[valid].mean()),world_xy_mean_Ns=float(world_xy[valid].mean()),
    current_normal_mean_Ns=float(current_z[valid].mean()),world_signed_z_mean_Ns=float(world_z[valid].mean()),
    xy_mean_absolute_difference_Ns=float((current_xy-world_xy).abs()[valid].mean()),
    z_mean_absolute_difference_Ns=float((current_z-world_z).abs()[valid].mean()),
    negative_world_z_samples=int(((world_z<0)&valid).sum()),
    individual_contact_threshold='Source ignores total contact impulses below 0.001Ns before body accumulation; filtered aggregate trace cannot reproduce that filter exactly.')
ratio=(n[...,2].abs()/n.norm(dim=-1).clamp_min(1e-12))
examples=[]
for label,expected in [('axis_z',1.),('inclined_45',2**-.5),('axis_xy',0.)]:
    candidates=(ratio-expected).abs().masked_fill(~valid,float('inf'))
    flat=int(candidates.argmin()); step,env,sensor=torch.unravel_index(torch.tensor(flat),valid.shape)
    i,j,k=int(step),int(env),int(sensor)
    a,b=float(current_xy[i,j,k]),float(world_xy[i,j,k])
    c,d=float(current_z[i,j,k]),float(world_z[i,j,k])
    examples.append(dict(case=label,physics_step=i+1,env_id=j,sensor=data['sensors'][k]['name'],
        normal_x_N=float(n[i,j,k,0]),normal_y_N=float(n[i,j,k,1]),normal_z_N=float(n[i,j,k,2]),
        friction_x_N=float(f[i,j,k,0]),friction_y_N=float(f[i,j,k,1]),friction_z_N=float(f[i,j,k,2]),
        current_local_xy_Ns=a,world_xy_Ns=b,current_normal_Ns=c,world_signed_z_Ns=d,
        xy_relative_difference=(a-b)/abs(b) if b else None,z_relative_difference=(c-d)/abs(d) if d else None,
        measured_abs_normal_z_fraction=float(ratio[i,j,k])))
out=Path('logs/acceptance/optimization7/B')
report['examples']=examples
(out/'contact_reward_audit.json').write_text(json.dumps(report,indent=2,allow_nan=False))
with (out/'contact_projection.csv').open('w',newline='') as file:
    writer=csv.DictWriter(file,fieldnames=list(examples[0]));writer.writeheader();writer.writerows(examples)
(out/'contact_reward_audit.md').write_text('''# 接触Reward实测审计

源Environment.hpp 796–810：先过滤对象top、dynamic配对与小于0.001 N·s的单个接触；用接触坐标变换、Object A/B符号恢复世界总冲量，逐body累加后读取世界XY模与有符号Z。

本机ContactSensor最新物理帧的force_matrix_w和friction_forces_w已在世界坐标，Tensor API用physics_dt将冲量换算为N。runtime.py对filter维求和后，摩擦模作为切向量、法向模作为法向量，再乘一次physics_dt；它们与世界XY/Z不同。单步缓存是在积分后common_step_counter递增时失效，消费控制周期末帧；没有按decimation重复累加冲量。

真实斜向接触与轴向近似样本及当前/世界分量差异见contact_projection.csv；统计见contact_reward_audit.json。B/D2和B/D4_copies30另含轴向/45度合成力测试。形状来自A的sensor元数据，af是一body对一top，arm过滤top/bottom/table三个body。传感器聚合后不能精确重现源逐contact的小冲量过滤，因此重算值仅对齐世界方向和单位，不宣称完整源Reward等价。

保持17项Reward与源时间换算不变。本轮没有证据证明方向近似是主要训练失败原因，没有引入world_xy_z_source_like正式候选。论文稳定指标要求抬升并至少5秒不落下；接触冲量阈值不是该指标的直接定义。2cm/0.2rad仍是工程代理标准。
''')
print(json.dumps(report,indent=2))

"""仅CPU：原源码MLP与本机RSL-RL网络的严格参数/forward对照。"""
import hashlib
import importlib.util
import json
from pathlib import Path

import torch
from tensordict import TensorDict
from rsl_rl.modules import ActorCritic

root = Path('/home/windsky/project/RobustDexGrasp/raisimGymTorch')
source = root/'raisimGymTorch/algo/ppo/module.py'
checkpoint = root/'data_all/teacher/teacher_ckpt/full_12500_r.pt'
out = Path('logs/acceptance/optimization8/E')
spec = importlib.util.spec_from_file_location('source_ppo_module', source)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
weights = torch.load(checkpoint, map_location='cpu', weights_only=False)
torch.manual_seed(8)
x = torch.randn(64, 153)
obs = TensorDict({'policy':x}, batch_size=[64])
rsl = ActorCritic(obs, {'policy':['policy'], 'critic':['policy']}, 22,
                 actor_hidden_dims=[128,128], critic_hidden_dims=[128,128], activation='lrelu',
                 actor_obs_normalization=False, critic_obs_normalization=False)
source_actor = module.MLP([128,128], torch.nn.LeakyReLU, 153, 22)
source_critic = module.MLP([128,128], torch.nn.LeakyReLU, 153, 1)
source_actor.load_state_dict(weights['actor_architecture_state_dict'], strict=True)
source_critic.load_state_dict(weights['critic_architecture_state_dict'], strict=True)
mapped = {}
for prefix,key in [('actor','actor_architecture_state_dict'),('critic','critic_architecture_state_dict')]:
    mapped.update({prefix+'.'+k.removeprefix('architecture.'):v for k,v in weights[key].items()})
mapped['std'] = weights['actor_distribution_state_dict']['std']
rsl.load_state_dict(mapped, strict=True)
with torch.no_grad():
    actor_error = (source_actor.architecture(x)-rsl.actor(x)).abs().max().item()
    critic_error = (source_critic.architecture(x)-rsl.critic(x)).abs().max().item()
assert actor_error == 0 and critic_error == 0
assert torch.equal(rsl.std, weights['actor_distribution_state_dict']['std'])
audit = dict(status='cpu_network_compatible', checkpoint=str(checkpoint),
    sha256=hashlib.file_digest(checkpoint.open('rb'),'sha256').hexdigest(),
    source_module=str(source), input_dim=153, action_dim=22, hidden_dims=[128,128],
    activation='LeakyReLU', negative_slope=0.01,
    key_shapes={k:{n:list(v.shape) for n,v in d.items()} for k,d in weights.items() if k!='optimizer_state_dict'},
    action_order='6 UR5 then 16 Allegro from robot_profile and source hardware contract',
    observation_contract='102D source state + 51D affordance vectors; source-to-Isaac field formulas require physical validation',
    normalization='train.py uses RaisimGymVecEnvOther.observe_vision_new which directly concatenates raw102D and rotated51D without normalizer; mean/var CSVs are not consumed in this path',
    std_min=float(rsl.std.min()),std_max=float(rsl.std.max()),
    optimizer_migrated=False, physical_transfer_status='blocked_gate_A', attempts=0)
(out/'source_checkpoint_audit.json').write_text(json.dumps(audit,indent=2)+'\n')
(out/'forward_equivalence.json').write_text(json.dumps(dict(test_kind='synthetic_cpu',samples=64,
    actor_max_abs_error=actor_error,critic_max_abs_error=critic_error,strict_load=True,pass_=True),indent=2)+'\n')
(out/'transfer_trials.csv').write_text('trial,object,seed,short_lift_success\n')
(out/'decision.md').write_text('# 源权重可行性\n\nCPU网络严格映射与64个固定153D向量的actor/critic前向误差均为0。仅证明网络可映射；不证明观测分布/动力学可迁移。\n\n物理迁移被Gate A阻断，尝试数0；未产生fine-tune或示范数据。normalizer与观测字段逐项部署审计留待物理通过。\n')
print(json.dumps({'actor_max_error':actor_error,'critic_max_error':critic_error,'strict_load':True}))

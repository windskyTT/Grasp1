"""真实RSL-RL更新的std投影等价性；合成rollout仅测试算法接口，不是抓取训练。"""
import ast
import copy
import json
from pathlib import Path
from types import SimpleNamespace
import torch
from tensordict import TensorDict
from rsl_rl.modules import ActorCritic
from rsl_rl.algorithms import PPO

torch.set_num_threads(1)
tree=ast.parse(Path('scripts/rsl_rl/train.py').read_text())
node=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='update_with_minimum_std')
results=[]
for start_std in (1.,.1):
    torch.manual_seed(1)
    obs=TensorDict({'policy':torch.randn(16,153)},batch_size=[16])
    policy=ActorCritic(obs,{'policy':['policy'],'critic':['policy']},22,
        actor_hidden_dims=[128,128],critic_hidden_dims=[128,128],activation='lrelu',init_noise_std=start_std)
    algorithm=PPO(policy,num_learning_epochs=4,num_mini_batches=4,clip_param=.2,
        gamma=.9996660539892578,lam=.9957346812224394,value_loss_coef=.5,entropy_coef=0.,
        learning_rate=5e-4,max_grad_norm=.5,use_clipped_value_loss=True,schedule='fixed',device='cpu')
    algorithm.init_storage('rl',16,32,obs,[22])
    raw=[]
    with torch.no_grad():
        for _ in range(32):
            actions=algorithm.act(obs)
            raw.append(actions.clone())
            next_obs=TensorDict({'policy':torch.randn(16,153)},batch_size=[16])
            algorithm.process_env_step(next_obs,torch.randn(16),torch.zeros(16,dtype=torch.bool),{})
            obs=next_obs
        algorithm.compute_returns(obs)
    baseline=copy.deepcopy(algorithm)
    wrapped=copy.deepcopy(algorithm)
    scope={'torch':torch,'original_update':wrapped.update,'runner':SimpleNamespace(alg=wrapped),
           'agent_cfg':SimpleNamespace(minimum_action_std=.2)}
    exec(compile(ast.Module(body=[node],type_ignores=[]),'<current_train_std_hook>','exec'),scope)
    torch.manual_seed(4);baseline_losses=baseline.update()
    torch.manual_seed(4);wrapped_losses=scope['update_with_minimum_std']()
    assert baseline_losses==wrapped_losses
    for name,param in baseline.policy.state_dict().items():
        if name=='std':
            assert torch.equal(wrapped.policy.state_dict()[name],param.clamp(min=.2))
        else:
            assert torch.equal(wrapped.policy.state_dict()[name],param)
    for key,state in baseline.optimizer.state_dict()['state'].items():
        for name,value in state.items():
            other=wrapped.optimizer.state_dict()['state'][key][name]
            assert torch.equal(value,other) if isinstance(value,torch.Tensor) else value==other
    samples=torch.stack(raw)
    if start_std==1.:assert (samples.abs()>1).any()
    results.append(dict(start_std=start_std,status='pass',baseline_std_min=float(baseline.policy.std.min()),
        projected_std_min=float(wrapped.policy.std.min()),losses_equal=True,other_parameters_equal=True,
        optimizer_equal=True,raw_action_max_abs=float(samples.abs().max()),raw_fraction_gt1=float((samples.abs()>1).float().mean())))
Path('logs/acceptance/optimization7/B/ppo_regression.json').write_text(json.dumps(
    dict(status='pass',synthetic_rollout=True,native_rsl_rl='3.1.2',formal_training=False,cases=results),indent=2))
print('OPT7_PPO_REGRESSION_PASS',json.dumps(results))

"""检查指定验收 run 的全部 checkpoint 张量和 TensorBoard scalars。"""
import argparse
import json
import math
from pathlib import Path
import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

parser = argparse.ArgumentParser()
parser.add_argument('--run_name', required=True)
parser.add_argument('--output', required=True)
args = parser.parse_args()
run = sorted(Path('logs/rsl_rl/grasp1_ur5_allegro_teacher').glob('*_'+args.run_name))[-1]
checkpoints = {}
def check(value):
    if isinstance(value, torch.Tensor):
        assert torch.isfinite(value).all()
        return value.numel()
    if isinstance(value, dict):
        return sum(check(v) for v in value.values())
    if isinstance(value, (list,tuple)):
        return sum(check(v) for v in value)
    if isinstance(value, float):
        assert math.isfinite(value)
    return 0
for path in sorted(run.glob('model_*.pt')):
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    checkpoints[path.name] = dict(finite=True, tensor_elements=check(checkpoint), iteration=checkpoint['iter'])
assert checkpoints
events = EventAccumulator(str(run), size_guidance={'scalars':0})
events.Reload()
scalars = {}
for tag in events.Tags()['scalars']:
    rows = events.Scalars(tag)
    values = [row.value for row in rows]
    assert values and all(math.isfinite(v) for v in values), tag
    scalars[tag] = dict(count=len(values), first=values[0], last=values[-1], minimum=min(values), maximum=max(values), mean=sum(values)/len(values))
assert scalars
result = dict(run=str(run), checkpoints=checkpoints, scalar_tags=scalars, all_finite=True,
              reward_terms=sum(tag.startswith('Episode_Reward/') for tag in scalars))
assert result['reward_terms'] == 17
Path(args.output).write_text(json.dumps(result,indent=2))
print('TRAINING_CHECK', json.dumps({key:result[key] for key in ['run','checkpoints','all_finite','reward_terms']}), flush=True)

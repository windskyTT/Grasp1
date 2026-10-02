"""记录本次 baseline/validation 的 PPO checkpoint 有限性与 TensorBoard 指标。"""
from pathlib import Path
import json
import math
import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

root = Path('logs/rsl_rl/grasp1_ur5_allegro_teacher')
results = {}
for label, suffix in [('before','perf_cache_baseline'),('after','perf_cache_validation')]:
    run = sorted(root.glob('*_' + suffix))[-1]
    checkpoint = run / 'model_0.pt'
    state = torch.load(checkpoint,map_location='cpu',weights_only=False)
    tensor_count = [0]
    def check(value):
        if isinstance(value,torch.Tensor):
            assert torch.isfinite(value).all()
            tensor_count[0] += 1
        elif isinstance(value,dict):
            for child in value.values(): check(child)
        elif isinstance(value,(tuple,list)):
            for child in value: check(child)
    check(state)
    events = EventAccumulator(str(run),size_guidance={'scalars':0}).Reload()
    metrics = {}
    for tag in events.Tags()['scalars']:
        values = events.Scalars(tag)
        assert all(math.isfinite(x.value) for x in values), tag
        metrics[tag] = [x.value for x in values]
    results[label] = dict(run=str(run),checkpoint=str(checkpoint),finite_tensors=tensor_count[0],
                          iteration=state.get('iter'),metrics=metrics)
    print(label,'CHECKPOINT_FINITE',tensor_count[0],{k:v for k,v in metrics.items() if k.startswith('Perf/')})
Path('logs/acceptance/2026-10-01/perf-cache/checkpoints.json').write_text(json.dumps(results,indent=2))

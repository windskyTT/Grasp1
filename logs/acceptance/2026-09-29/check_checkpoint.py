import json
import math
import sys
from pathlib import Path
import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
run=Path(sys.argv[1])
checkpoint=run/'model_19.pt'
state=torch.load(checkpoint,map_location='cpu',weights_only=False)
count=0
def check(value,path='checkpoint'):
    global count
    if isinstance(value,torch.Tensor):
        assert torch.isfinite(value).all(),path
        count+=1
    elif isinstance(value,dict):
        for key,item in value.items():check(item,f'{path}/{key}')
    elif isinstance(value,(tuple,list)):
        for i,item in enumerate(value):check(item,f'{path}/{i}')
check(state)
events=EventAccumulator(str(run),size_guidance={'scalars':0}).Reload()
metrics={}
for tag in events.Tags()['scalars']:
    values=events.Scalars(tag)
    assert all(math.isfinite(x.value) for x in values),tag
    metrics[tag]=dict(count=len(values),first=values[0].value,last=values[-1].value,min=min(x.value for x in values),max=max(x.value for x in values))
result=dict(checkpoint=str(checkpoint),tensor_count=count,iteration=state.get('iter'),metrics=metrics)
Path('logs/acceptance/2026-09-29/ppo-metrics.json').write_text(json.dumps(result,indent=2))
print('CHECKPOINT_FINITE',checkpoint,count,state.get('iter'))
for tag,value in metrics.items():
    if tag.startswith(('Loss','Policy','Perf','Train')):print(tag,value)

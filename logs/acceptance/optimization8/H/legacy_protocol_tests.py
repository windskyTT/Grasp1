"""无需仿真的checkpoint字段对照和CLI负例；真实时序另在measurement_check.py验证。"""
import ast
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import os
import re

sys.path.insert(0,str(Path('scripts/rsl_rl').resolve()))
from checkpoint_contract import compare_contract, validate_checkpoint, metadata_path, SEMANTIC_FIELDS, RESUME_FIELDS

out=Path('logs/acceptance/optimization8/H')
base={key:0 for key in RESUME_FIELDS}
base.update(schema_version=1,task_id='Grasp1-UR5-Allegro-Teacher-v0',decimation=2,episode_length_s=10,
            training_git_sha='first',next_update_index=100,checkpoint_iteration=99)
cases=[]
with tempfile.TemporaryDirectory() as directory:
    checkpoint=Path(directory)/'model_99.pt'
    checkpoint.touch()
    metadata_path(checkpoint).write_text(json.dumps(base))
    audit,_=validate_checkpoint(checkpoint,base,resume=True)
    assert audit['compatible']
    cases.append(dict(case='same_semantics_resume',status='pass',audit=audit))
    changed=copy.deepcopy(base); changed['training_git_sha']='other_commit'
    assert not compare_contract(base,changed,resume=True)
    changed['decimation']=4
    for resume in (False,True):
        try: validate_checkpoint(checkpoint,changed,resume=resume,allow_mismatch=resume)
        except ValueError as error: cases.append(dict(case=f'D2_to_D4_resume_{resume}',status='pass',error=str(error)))
        else: raise AssertionError('Mismatched checkpoint accepted')
    audit,_=validate_checkpoint(checkpoint,changed,allow_mismatch=True)
    assert audit['comparability_warning'] and 'decimation' in audit['differences']
    cases.append(dict(case='diagnostic_override',status='pass',audit=audit))
    changed=copy.deepcopy(base); changed['episode_length_s']=40
    assert not compare_contract(base,changed,resume=False)
    assert 'episode_length_s' in compare_contract(base,changed,resume=True)
    cases.append(dict(case='eval_horizon_is_protocol_resume_horizon_is_contract',status='pass'))
    metadata_path(checkpoint).unlink()
    try: validate_checkpoint(checkpoint,base)
    except ValueError as error: cases.append(dict(case='legacy_missing_metadata',status='pass',error=str(error)))
    else: raise AssertionError('Legacy checkpoint accepted')
    parsed=ast.parse(Path('/home/windsky/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/utils/parse_cfg.py').read_text())
    node=next(n for n in parsed.body if isinstance(n,ast.FunctionDef) and n.name=='get_checkpoint_path')
    scope={'os':os,'re':re}
    exec(compile(ast.Module(body=[node],type_ignores=[]),'<installed_get_checkpoint_path>','exec'),scope)
    run=Path(directory)/'run_1';run.mkdir()
    (run/'model_1.pt').touch();metadata_path(run/'model_1.pt').write_text('{}')
    assert scope['get_checkpoint_path'](directory,'run_1','model_1.pt')==str(run/'model_1.pt')
    cases.append(dict(case='native_checkpoint_discovery_excludes_metadata',status='pass'))
(out/'checkpoint_compatibility.json').write_text(json.dumps(dict(status='pass',synthetic=True,cases=cases),indent=2))

tests=[['--paper_stability','--grasp_duration_s','10'],['--lift_ramp_duration_s','21'],
       ['--lift_ramp_duration_s','0'],['--lift_duration_s','nan'],['--success_height','inf'],
       ['--max_relative_translation_drift_m','-0.01'],['--max_relative_rotation_drift_rad','0'],['--rounds','0']]
rows=[]
for flags in tests:
    result=subprocess.run([sys.executable,'scripts/rsl_rl/evaluate.py',*flags],capture_output=True,text=True)
    assert result.returncode==2, (flags,result.stdout,result.stderr)
    assert '[INFO][AppLauncher]' not in result.stdout
    rows.append(dict(flags=flags,status='pass',exit_code=result.returncode,message=result.stderr.splitlines()[-1]))

# 高度差语义与body导出顺序独立；同一物体的root不能替代top。
module=ast.parse(Path('scripts/rsl_rl/grasp_diagnostics.py').read_text())
node=next(n for n in module.body if isinstance(n,ast.FunctionDef) and n.name=='object_body_ids')
scope={}; exec(compile(ast.Module(body=[node],type_ignores=[]),'<body_ids>','exec'),scope)
from types import SimpleNamespace
assert scope['object_body_ids'](SimpleNamespace(body_names=['top','bottom']))==(0,1)
assert scope['object_body_ids'](SimpleNamespace(body_names=['bottom','top']))==(1,0)
try: scope['object_body_ids'](SimpleNamespace(body_names=['bottom','other']))
except ValueError: pass
else: raise AssertionError('Missing top accepted')
(out/'protocol_tests.json').write_text(json.dumps(dict(status='pass',cli=rows,body_name_resolution='pass'),indent=2))
print('OPT8_LEGACY_PROTOCOL_PASS',len(rows),'CLI cases and',len(cases),'checkpoint cases')

"""固定训练集Reset种子比较真实Policy；复用play.py严格契约，不调用脚本控制器。"""
import argparse
import csv
import json
import os
from pathlib import Path
import shutil
import subprocess

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--checkpoints', type=Path, nargs='+', required=True)
parser.add_argument('--task', default='Grasp1-UR5-Allegro-GraspTrain-v3')
parser.add_argument('--curriculum', choices=('C1', 'C2', 'C3'), default='C1')
parser.add_argument('--seed', type=int, default=21)
parser.add_argument('--trials', type=int, default=20)
parser.add_argument('--output_dir', type=Path, required=True)
args = parser.parse_args()
out = args.output_dir.resolve()
out.mkdir(parents=True, exist_ok=True)
root = Path(__file__).resolve().parents[2]
rows, rejected = [], []
for index, checkpoint in enumerate(args.checkpoints):
    checkpoint = checkpoint.resolve()
    metadata = checkpoint.with_name('env_contract_'+checkpoint.name+'.json')
    saved = json.loads(metadata.read_text()) if metadata.exists() else {}
    # task不符已足够拒绝；同task的完整Reset/TCP/外参验证由play统一执行。
    if saved.get('task_id') != args.task:
        rejected.append(dict(checkpoint=str(checkpoint), reason='missing_metadata_or_task_mismatch',
                             saved_task=saved.get('task_id')))
        continue
    trial_rows = []
    complete = True
    for seed in range(args.seed, args.seed+args.trials):
        run_out = out/f'checkpoint_{index}'/f'seed_{seed}'
        run_out.mkdir(parents=True, exist_ok=True)
        command = ['/home/windsky/IsaacLab/isaaclab.sh', '-p', 'scripts/rsl_rl/play.py',
            '--task', args.task, '--checkpoint', str(checkpoint), '--curriculum', args.curriculum,
            '--dataset', 'new_training_set', '--object', 'wood_block_oriented', '--num_envs', '1',
            '--episodes', '1', '--seed', str(seed), '--headless', '--output_dir', str(run_out)]
        with (run_out/'play.log').open('w') as log:
            result = subprocess.run(command, cwd=root, env=dict(os.environ, TERM='xterm'),
                                    stdout=log, stderr=subprocess.STDOUT)
        (run_out/'command.json').write_text(json.dumps(command, indent=2))
        if result.returncode:
            rejected.append(dict(checkpoint=str(checkpoint), reason='play_failed', seed=seed,
                                 returncode=result.returncode, log=str(run_out/'play.log')))
            complete = False
            break
        with (run_out/'full_grasp_trials.csv').open() as handle:
            trial_rows.extend(csv.DictReader(handle))
    successes = sum(row['success']=='True' for row in trial_rows)
    unsafe = sum(int(row['failure_bits']) & 3 != 0 for row in trial_rows)
    rows.append(dict(checkpoint=str(checkpoint), completed_updates=saved['completed_updates'],
        status='complete' if complete else 'failed',
        successes=successes, attempts=len(trial_rows), success_rate=successes/len(trial_rows) if trial_rows else None,
        unsafe=unsafe, dropped=sum(bool(int(row['failure_bits']) & 4) for row in trial_rows),
        timeouts=sum(bool(int(row['failure_bits']) & 8) for row in trial_rows),
        mean_lift_m=sum(float(row['lift_m']) for row in trial_rows)/len(trial_rows) if trial_rows else None,
        mean_hold_s=sum(float(row['hold_s']) for row in trial_rows)/len(trial_rows) if trial_rows else None,
        mean_elapsed_s=sum(float(row['elapsed_s']) for row in trial_rows)/len(trial_rows) if trial_rows else None))
with (out/'checkpoints.csv').open('w', newline='') as handle:
    writer = csv.DictWriter(handle, fieldnames=['checkpoint', 'completed_updates', 'status', 'successes', 'attempts',
        'success_rate', 'unsafe', 'dropped', 'timeouts', 'mean_lift_m', 'mean_hold_s', 'mean_elapsed_s'])
    writer.writeheader()
    writer.writerows(rows)
eligible = [row for row in rows if row['status']=='complete' and row['unsafe']==0 and row['successes']>0]
best = max(eligible, key=lambda row:(row['success_rate'], row['mean_hold_s'])) if eligible else None
if best:
    checkpoint = Path(best['checkpoint'])
    shutil.copy2(checkpoint, out/'best.pt')
    shutil.copy2(checkpoint.with_name('env_contract_'+checkpoint.name+'.json'), out/'env_contract_best.pt.json')
summary = dict(status='pass' if best else 'blocked', controller='policy', task=args.task,
    curriculum=args.curriculum, seed_range=[args.seed, args.seed+args.trials-1],
    results=rows, rejected=rejected, best_checkpoint=str(out/'best.pt') if best else None,
    policy_attempts=sum(row['attempts'] for row in rows),
    reason=None if best else 'no_compatible_successful_policy_in_fixed_protocol')
(out/'comparison.json').write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))

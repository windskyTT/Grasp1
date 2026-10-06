"""按优化4顺序运行阶段、A/B评估和学习曲线；不改动MDP/PPO配置。"""
import argparse
import csv
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

ROOT = Path('logs/acceptance/2026-10-05/teacher-long-training').resolve()
TASK = 'Grasp1-UR5-Allegro-Teacher-v0'
MODES = {'A': 'current-relative-repeat', 'B': 'hold-grasp-posture'}
progress = []
evaluations = []
diagnostic_rows = []


def read_json(path):
    return json.loads(Path(path).read_text())


def write_csv(path, rows):
    if rows:
        fields = list(dict.fromkeys(k for row in rows for k in row))
        with path.open('w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)


def export(status):
    (ROOT / 'training_progress.json').write_text(json.dumps(dict(status=status, stages=progress), indent=2, allow_nan=False))
    write_csv(ROOT / 'training_progress.csv', [{k: v for k, v in row.items() if not isinstance(v, (dict, list))} for row in progress])
    write_csv(ROOT / 'evaluation_progress.csv', evaluations)
    write_csv(ROOT / 'grasp_diagnostics.csv', diagnostic_rows)
    lines = ['# 优化4执行报告', '', f'状态：{status}', '',
             '## A. 已证明', '',
             '已保存Optimization-3 baseline；原HEAD为a79fd6d82f3ed72427314f3c38e7aba0b2125e44。',
             'Observation 22:44读取实际发送的clamped target减当前测量joint position。',
             '16 env zero/random各300步、每组198组全22关节target/observation检查通过；full/partial reset通过，未重置行保留，153维观测与22维动作保持。',
             'Smoke target outside fraction=0；random UR5/Allegro qvel峰值2.40931/6.02800 rad/s；物体q范围-1.02439e-6～0.0010000742 rad；无invalid termination及NaN/Inf。',
             '只修改action保存目标、observation读取目标和reset同步，以及评估诊断；Physics/PPO/reward/action scale配置保持Optimization-3。',
             '额外bottom传感器仅在--diagnostics评估创建；原reward的top/table sensors不变。接触阈值0.1N为法向过滤接触的行为统计，采样周期1/60秒。指尖按USD合并后的四个distal body统计。',
             'Mode A默认保持原relative-repeat；Mode B逐物理子步保持grasp-end实际手部target。',
             '训练从头开始；阶段resume时将RSL-RL保存的0起始编号加1，避免重复最后一次iteration，并从已加载optimizer同步实际adaptive LR，避免恢复为初始LR。KL来自既有adaptive调度的16个minibatch均值，仅作记录。', '',
             '| iteration | origin | reward | std | value loss | LR | source A/B | strict A/B |',
             '|---|---|---|---|---|---|---|---|']
    for row in progress:
        stage_evals = {r['mode']: r for r in evaluations if r['iteration'] == row['iteration'] and r['origin'] == row['origin']}
        source = ' / '.join(str(stage_evals[m]['source_successes']) + '/30' if m in stage_evals else '未测' for m in ('A', 'B'))
        strict = ' / '.join(str(stage_evals[m]['strict_successes']) + '/30' if m in stage_evals else '未测' for m in ('A', 'B'))
        lines.append(f"| {row['iteration']} | {row['origin']} | {row.get('mean_reward')} | {row.get('policy_std')} | {row.get('value_loss')} | {row.get('learning_rate')} | {source} | {strict} |")
    lines += ['', '已完成的各阶段run/checkpoint、transitions、耗时、FPS、VRAM、17项奖励、终止和有限性数据详见training_progress.json及对应stage目录。未完成阶段不填入结果。', '',
              '## B. 观察趋势', '',
              '学习曲线同时记录距离、object/top/bottom/table接触、多指与拇指对向接触、lift接触保留及高度。历史100点为旧Observation，仅供历史参考；本轮100点与后续点使用修正后的Observation。',
              '是否继续5000/10000/20000由decision文件及多指标趋势决定，单个0/30不作为停止条件。', '',
              '## C. 尚不能证明', '',
              '运行有限性与目标语义通过不能证明最终策略收敛。旧100 iteration的0/30不能证明算法失败、reward错误或PPO需要重写。',
              '接触proxy不能证明完整force closure；1 round评估存在采样变动。未达到阶段的数据不能作为训练结果。',
              '本轮未修改lambda、LR配置、reward或object armature。', '']
    (ROOT / 'REPORT.md').write_text('\n'.join(lines))


def checkpoint_finite(checkpoint):
    state = torch.load(checkpoint, map_location='cpu', weights_only=False)
    def visit(value):
        if isinstance(value, torch.Tensor):
            return bool(torch.isfinite(value).all())
        if isinstance(value, dict):
            return all(visit(v) for v in value.values())
        if isinstance(value, (list, tuple)):
            return all(visit(v) for v in value)
        if isinstance(value, float):
            return math.isfinite(value)
        return True
    assert visit(state), f'Non-finite checkpoint: {checkpoint}'


def scalar_summary(run, end_iteration):
    events = EventAccumulator(str(run), size_guidance={'scalars': 0}).Reload()
    elapsed_s = sum(v.value for tag in ('Perf/collection time', 'Perf/learning_time')
                    for v in events.Scalars(tag) if v.step < end_iteration)
    summary = {}
    for tag in events.Tags()['scalars']:
        records = events.Scalars(tag)
        assert all(math.isfinite(v.value) for v in records), f'Non-finite TensorBoard: {tag}'
        # /time曲线的step是秒数，不能按iteration截断。
        values = [v.value for v in records if (v.step <= elapsed_s if tag.endswith('/time') else v.step < end_iteration)]
        assert all(math.isfinite(v) for v in values), f'Non-finite TensorBoard: {tag}'
        if values:
            summary[tag] = dict(count=len(values), last=values[-1], minimum=min(values), maximum=max(values),
                                mean=statistics.mean(values), last50_mean=statistics.mean(values[-50:]))
    return summary


def add_training_point(iteration, runtime, origin, checkpoint=None):
    checkpoint = checkpoint or runtime['checkpoint']
    checkpoint_finite(checkpoint)
    for saved in Path(runtime['run_path']).glob('model_*.pt'):
        checkpoint_finite(saved)
    scalars = scalar_summary(runtime['run_path'], iteration)
    (ROOT / f'stage{iteration}').mkdir(exist_ok=True)
    (ROOT / f'stage{iteration}' / 'tensorboard_summary.json').write_text(json.dumps(scalars, indent=2))
    metrics = {'mean_reward': 'Train/mean_reward', 'episode_length': 'Train/mean_episode_length',
               'value_loss': 'Loss/value_function', 'surrogate_loss': 'Loss/surrogate', 'entropy': 'Loss/entropy',
               'policy_std': 'Policy/mean_noise_std', 'learning_rate': 'Loss/learning_rate', 'KL': 'Loss/kl',
               'FPS': 'Perf/total_fps', 'collection_time': 'Perf/collection time', 'learning_time': 'Perf/learning_time'}
    row = dict(runtime, iteration=iteration, checkpoint=checkpoint, total_env_transitions=iteration*2048*32,
               origin=origin, checkpoint_finite=True, tensorboard_finite=True,
               reward_terms={k: v for k, v in scalars.items() if k.startswith('Episode_Reward/')})
    assert len(row['reward_terms']) == 17
    for key, tag in metrics.items():
        row[key] = scalars[tag]['last'] if tag in scalars else None
    row['mean_reward_last50'] = scalars['Train/mean_reward']['last50_mean']
    progress.append(row)
    export(f'{iteration}训练完成，准备评估')
    return row


def run_command(command, log):
    print('RUN', ' '.join(command), flush=True)
    with log.open('w') as file:
        result = subprocess.run(command, stdout=file, stderr=subprocess.STDOUT)
    (log.with_suffix('.exit')).write_text(str(result.returncode))
    assert result.returncode == 0, f'Command failed ({result.returncode}): {log}'


def evaluate(iteration, checkpoint, origin='obsfix'):
    for mode, name in MODES.items():
        out = ROOT / f'stage{iteration}' / f'mode{mode}'
        run_command([sys.executable, '-u', 'scripts/rsl_rl/evaluate.py', '--task', TASK,
                     '--checkpoint', str(checkpoint), '--rounds', '1', '--diagnostics',
                     '--lift_hand_mode', name, '--output_dir', str(out), '--headless'],
                    ROOT / f'stage{iteration}' / f'mode{mode}.log')
        result = read_json(out / 'quantitative_eval.json')
        rows = read_json(out / 'grasp_diagnostics.json')
        if mode == 'B':
            assert max(r['lift_max_hand_target_drift_rad'] for r in rows) == 0
        row = dict(iteration=iteration, origin=origin, mode=mode, checkpoint=str(checkpoint),
                   source_successes=result['total_source_successes'], strict_successes=result['total_successes'],
                   attempts=result['total_attempts'],
                   median_final_distance_m=statistics.median(r['grasp_end_distance_m'] for r in rows),
                   max_height_gain_m=max(r['height_gain_m'] for r in rows),
                   median_height_gain_m=statistics.median(r['height_gain_m'] for r in rows),
                   max_during_lift_height_gain_m=max(r['lift_max_height_gain_m'] for r in rows),
                   mean_lift_retention=statistics.mean([r['lift_contact_retention_fraction'] for r in rows if r['lift_contact_retention_fraction'] is not None]) if any(r['lift_contact_retention_fraction'] is not None for r in rows) else None,
                   mean_same_body_retention=statistics.mean([r['lift_same_body_retention_fraction'] for r in rows if r['lift_same_body_retention_fraction'] is not None]) if any(r['lift_same_body_retention_fraction'] is not None for r in rows) else None)
        for label in ('object_contact', 'top_contact', 'bottom_contact', 'table_contact', 'two_plus_contacts', 'three_plus_contacts', 'thumb_index', 'thumb_middle', 'thumb_ring', 'thumb_multiple'):
            row[f'objects_with_{label}'] = sum(r[f'grasp_{label}_steps'] > 0 for r in rows)
        row['objects_with_thumb_other'] = sum(any(r[f'grasp_thumb_{finger}_steps'] > 0 for finger in ('index', 'middle', 'ring')) for r in rows)
        evaluations.append(row)
        diagnostic_rows.extend(dict(iteration=iteration, origin=origin, mode=mode, **r) for r in rows)
        export(f'{iteration} Mode {mode}评估完成')


def continue_decision(iteration, previous):
    current = next(r for r in evaluations if r['iteration'] == iteration and r['mode'] == 'A' and r['origin'] == 'obsfix')
    before = next(r for r in evaluations if r['iteration'] == previous and r['mode'] == 'A' and r['origin'] == 'obsfix')
    gains = {key: current[key] > before[key] for key in ('objects_with_object_contact', 'objects_with_two_plus_contacts', 'objects_with_thumb_other', 'median_height_gain_m', 'mean_lift_retention') if current[key] is not None and before[key] is not None}
    gains['distance'] = current['median_final_distance_m'] < before['median_final_distance_m']
    reward_now = next(r['mean_reward_last50'] for r in progress if r['iteration'] == iteration and r['origin'] == 'obsfix')
    reward_before = next(r['mean_reward_last50'] for r in progress if r['iteration'] == previous and r['origin'] == 'obsfix')
    gains['reward'] = reward_now > reward_before
    # 负高度差变小、无手部接触时的数值位移，都不能作为lift学习进展。
    gains['median_height_gain_m'] = gains['median_height_gain_m'] and current['median_height_gain_m'] > 0 and current['objects_with_object_contact'] > 0
    if iteration == 2000:
        early = next(r for r in evaluations if r['iteration'] == 500 and r['mode'] == 'A' and r['origin'] == 'obsfix')
        for key in ('objects_with_object_contact', 'objects_with_two_plus_contacts', 'objects_with_thumb_other', 'median_height_gain_m', 'mean_lift_retention'):
            if key in gains and early[key] is not None:
                gains[key] = gains[key] and before[key] >= early[key]
        gains['distance'] = gains['distance'] and before['median_final_distance_m'] <= early['median_final_distance_m']
        early_reward = next(r['mean_reward_last50'] for r in progress if r['iteration'] == 500 and r['origin'] == 'obsfix')
        gains['reward'] = gains['reward'] and reward_before >= early_reward
    result = dict(iteration=iteration, comparison=previous, improved_metrics=gains,
                  continue_training=current['source_successes'] > 0 or any(v for k, v in gains.items() if k != 'reward'),
                  criterion='source>0 or sustained behavior improvement; height progress requires positive median gain with hand-object contact; reward alone does not establish behavior improvement; 0/30 alone does not stop training')
    (ROOT / f'stage{iteration}' / 'decision.json').write_text(json.dumps(result, indent=2))
    return result['continue_training']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--wait_pid', type=int, required=True)
    args = parser.parse_args()
    old = ROOT.parent / 'teacher-convergence-fix'
    metrics = read_json(ROOT / 'baseline/baseline_metrics.json')
    scalars = read_json(old / 'phase7-train100-check.json')['scalar_tags']
    progress.append(dict(iteration=100, origin='optimization3-historical', total_env_transitions=100*2048*32,
                         mean_reward=scalars['Train/mean_reward']['last'], policy_std=scalars['Policy/mean_noise_std']['last'],
                         value_loss=scalars['Loss/value_function']['last'], learning_rate=scalars['Loss/learning_rate']['last'],
                         checkpoint=(old / 'phase7-checkpoint.txt').read_text().strip(), observation_semantics='processed_action'))
    old_diag = read_json(old / 'phase7-diagnosis-summary.json')
    evaluations.append(dict(iteration=100, origin='optimization3-historical', mode='A', source_successes=0, strict_successes=0,
                            attempts=30, median_final_distance_m=old_diag['grasp_end_distance_median_m'],
                            objects_with_top_contact=old_diag['objects_with_contact_above_01N'],
                            max_height_gain_m=old_diag['height_gain_max_m'], median_height_gain_m=old_diag['height_gain_median_m']))
    export('500 iteration训练运行中')
    while Path(f'/proc/{args.wait_pid}').exists():
        time.sleep(10)
    runtime = read_json(ROOT / 'stage500/runtime.json')
    assert runtime.get('status') == 'completed' and runtime['finite'] and runtime['iteration'] == 500
    assert runtime['target_outside_fraction'] == 0
    new100 = str(Path(runtime['run_path']) / 'model_99.pt')
    point100 = read_json(ROOT / 'stage100/runtime.json')
    add_training_point(100, point100, 'obsfix', new100)
    evaluate(100, new100)
    add_training_point(500, runtime, 'obsfix')
    evaluate(500, runtime['checkpoint'])
    previous = 500
    for iteration in (1000, 2000, 5000, 10000, 20000):
        if previous >= 2000 and not continue_decision(previous, 1000 if previous == 2000 else {5000: 2000, 10000: 5000}[previous]):
            export(f'{previous}阶段完成；多指标未显示继续改善，保留baseline供plateau分析')
            return
        stage = ROOT / f'stage{iteration}'
        stage.mkdir(exist_ok=True)
        export(f'从{previous} checkpoint resume至{iteration}运行中')
        run_command([sys.executable, '-u', str(ROOT / 'train_measure.py'), '--task', TASK, '--num_envs', '2048',
                     '--max_iterations', str(iteration-previous), '--run_name', f'teacher_obsfix_stage{iteration}',
                     '--resume', '--load_run', Path(runtime['run_path']).name,
                     '--checkpoint', Path(runtime['checkpoint']).name, '--headless'], stage / 'train.log')
        runtime = read_json(stage / 'runtime.json')
        assert runtime['status'] == 'completed' and runtime['finite'] and runtime['iteration'] == iteration
        assert runtime['target_outside_fraction'] == 0
        add_training_point(iteration, runtime, 'obsfix')
        evaluate(iteration, runtime['checkpoint'])
        previous = iteration
    export('20000 iteration与所有阶段A/B评估完成')


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        export(f'执行中断：{error}')
        raise

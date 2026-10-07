"""运行优化5的一个训练节点及90-trial A/B评估，导出已完成数据。"""
import argparse
import csv
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys

import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

ROOT = Path(__file__).resolve().parent
TASK = 'Grasp1-UR5-Allegro-Teacher-v0'


def read(path):
    return json.loads(Path(path).read_text())


def write_csv(path, rows):
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def export(stage, summary):
    (stage / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False))
    summaries = [read(p) for p in sorted(ROOT.glob('**/summary.json')) if 'smoke' not in p.parts]
    (ROOT / 'training_progress.json').write_text(json.dumps(summaries, indent=2, allow_nan=False))
    write_csv(ROOT / 'training_progress.csv', [dict(candidate=s['candidate_name'], **{k: v for k, v in s['training'].items() if not isinstance(v, (dict, list))}) for s in summaries])
    write_csv(ROOT / 'evaluation_progress.csv', [dict(candidate=s['candidate_name'], iteration=s['training']['iteration'], **e) for s in summaries for e in s['evaluations']])
    write_csv(ROOT / 'action_diagnostics.csv', [dict(candidate=s['candidate_name'], iteration=s['training']['iteration'], **s['actions']) for s in summaries])
    lines = ['# 优化5执行报告', '', 'Phase 0–6及四组单变量A/B已完成。以下仅列实际完成节点；最终from-scratch分阶段验收进行中，后续长训练按gate决定。', '',
             '## A. 已证明', '',
             'V1.4 baseline位于baseline/，HEAD=5919609fe3eb5091677fb415ca0476a83ab1a9fe。',
             'Source action range semantics：高斯采样不做[-1,1] clipping或tanh，仅最终joint target限位。clip1是工程A/B；源minimum std=0.2在完整PPO更新之后执行。详见source-action-semantics.md。',
             'PPO诊断在adaptive/fixed下与原更新的参数、optimizer及原损失完全相等，见smoke/ppo-instrumentation-equivalence.json。',
             '本轮None候选的100节点模型参数及optimizer与V1.4同节点完全一致，见smoke/v14-node100-equivalence.json。该结论限定于已直接对照的前100次更新。',
             'minimum std扩展在完整PPO更新后执行；16-env激活下界测试中首次更新的actor/critic参数及optimizer完全一致，仅std为精确clamp(min=0.2)，见smoke/minimum-std-projection-check.json。该smoke使用init std=0.1，正式实验保持1.0。',
             '训练保持2048环境、seed=1、32 rollout、原physics/reward/action scale/network/gamma；单变量candidate配置逐节点保存。',
             '正式评估每物体3 rounds，90 trials，Mode A/B；policy clipping作用于抓取策略及重复的手部动作，固定UR5抬升轨迹保持。', '',
             '| candidate | iteration | reward last50 | value final / peak | std min / median / max | target clip | UR5 / hand qvel |',
             '|---|---:|---:|---|---|---|---|']
    for s in summaries:
        t = s['training']
        label = s['candidate_name'] + (' (复用)' if 'reused_from' in t else '')
        lines.append(f"| {label} | {t['iteration']} | {t['reward_last50']:.5g} | {t['value_loss']:.5g} / {t['value_loss_peak']:.5g} | {t['std_min']:.4g} / {t['std_median']:.4g} / {t['std_max']:.4g} | {t['target_clipping_fraction']:.3%} | {t['ur5_qvel_peak']:.4g} / {t['allegro_qvel_peak']:.4g} |")
    lines += ['', '| candidate | iteration | mode | source / strict | distance median cm | contact / 2+ / 3+ / thumb-other trials | height gain median / max m |', '|---|---:|---|---|---:|---|---|']
    for s in summaries:
        for e in s['evaluations']:
            lines.append(f"| {s['candidate_name']} | {s['training']['iteration']} | {e['mode']} | {e['source_successes']}/{e['attempts']} / {e['strict_successes']}/{e['attempts']} | {100*e['median_final_distance_m']:.5g} | {e['trials_with_object_contact']} / {e['trials_with_two_plus_contacts']} / {e['trials_with_three_plus_contacts']} / {e['trials_with_thumb_other']} | {e['median_height_gain_m']:.4g} / {e['max_height_gain_m']:.4g} |")
    for decision_path in sorted(ROOT.glob('*/decision.json')):
        decision = read(decision_path)
        lines += ['', f"选择 {decision['variable']} = {decision['selected_value']}：", '']
        lines += [f'- {reason}' for reason in decision['reason']]
    lines += ['', '复用行指配置、seed和评估协议完全一致的已完成节点引用，不计为新增训练或独立重复；原始文件路径见各reference.json及training_progress.json的reused_from。']
    lines += ['', '## B. 观察趋势', '', '候选选择及节点继续决定必须同时看critic、reward、接触、距离、qvel和动作分布；单个0/90不是选择依据。选择理由另存各phase/decision.json。', '',
              '## C. 尚不能证明', '', 'finite checkpoint及TensorBoard不等于稳定学习或成功抓取。未完成节点没有结果；不预先宣称2000或20000成功。', '',
              '原始逐iteration动作、KL、std、VRAM记录在对应stage/iterations.jsonl；qvel为物理子步测量，contact peak为奖励读取时的过滤法向+切向总力，不能称为所有物理子步的contact上限。2+/3+来自现有诊断的同时接触body数，不能等同于已验证的不同手指闭合或force closure。',
              'raw与wrapper动作p95/p99按完整rollout精确计算；根action_diagnostics.csv的max_abs是阶段峰值，其余动作分布字段为最近50次rollout统计的均值，不是全历史样本的全局分位数。std median为torch.median的下中位数。wall time为learn循环，不含scene创建及独立评估。', '']
    (ROOT / 'REPORT.md').write_text('\n'.join(lines))


def finite_checkpoint(path):
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
    assert visit(torch.load(path, map_location='cpu', weights_only=False)), path


def run(command, log):
    (log.with_suffix('.command.json')).write_text(json.dumps(command, indent=2))
    print('RUN', ' '.join(command), flush=True)
    with log.open('w') as file:
        result = subprocess.run(command, stdout=file, stderr=subprocess.STDOUT)
    log.with_suffix('.exit').write_text(str(result.returncode))
    assert result.returncode == 0, f'Command failed: {log}'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--candidate', required=True)
    parser.add_argument('--iteration', type=int, required=True)
    parser.add_argument('--resume-runtime', type=Path)
    parser.add_argument('--evaluation-only', action='store_true')
    args = parser.parse_args()
    candidate_dir = ROOT / args.candidate
    config_path = candidate_dir / 'candidate.json'
    cfg = read(config_path)
    stage = candidate_dir / f'stage{args.iteration}'
    stage.mkdir(parents=True, exist_ok=True)
    if not args.evaluation_only:
        previous = read(args.resume_runtime) if args.resume_runtime else None
        command = [sys.executable, '-u', str(ROOT / 'train_measure.py'), '--measurement-dir', str(stage),
                   '--candidate-config', str(config_path), '--task', TASK, '--num_envs', '2048',
                   '--max_iterations', str(args.iteration - (previous['iteration'] if previous else 0)),
                   '--run_name', (f'teacher_stable_stage{args.iteration}' if args.candidate == 'selected-config'
                                  else f'teacher_stability_{args.candidate.replace("/", "_")}_stage{args.iteration}'), '--headless']
        if previous:
            assert previous['candidate'] == cfg, 'Resume必须保持同一candidate配置'
            command += ['--resume', '--load_run', Path(previous['run_path']).name, '--checkpoint', Path(previous['checkpoint']).name]
        run(command, stage / 'train.log')
    runtime = read(stage / 'runtime.json')
    assert runtime['status'] == 'completed' and runtime['iteration'] == args.iteration and runtime['finite']
    assert runtime['target_outside_fraction'] == 0
    for checkpoint in Path(runtime['run_path']).glob('model_*.pt'):
        finite_checkpoint(checkpoint)
    scalars = EventAccumulator(runtime['run_path'], size_guidance={'scalars': 0}).Reload()
    assert all(math.isfinite(v.value) for tag in scalars.Tags()['scalars'] for v in scalars.Scalars(tag))
    assert len([tag for tag in scalars.Tags()['scalars'] if tag.startswith('Episode_Reward/')]) == 17
    node_scalars = {tag: [v for v in scalars.Scalars(tag) if v.step < args.iteration] for tag in scalars.Tags()['scalars']}
    scalar_summary = {tag: dict(last=values[-1].value, peak=max(v.value for v in values)) for tag, values in node_scalars.items()}
    (stage / 'tensorboard_summary.json').write_text(json.dumps(scalar_summary, indent=2))
    rows = [json.loads(line) for line in (stage / 'iterations.jsonl').read_text().splitlines()]
    last = rows[-1]
    training = dict(runtime, checkpoint_finite=True, tensorboard_finite=True,
                    mean_reward=last['mean_reward'],
                    reward_last50=statistics.mean(r['mean_reward'] for r in rows[-50:]),
                    value_loss_peak=max(r['value_function'] for r in rows),
                    value_loss=last['value_function'], surrogate_loss=last['surrogate'], entropy=last['entropy'],
                    KL=last['kl'], learning_rate=last['learning_rate'], policy_std=last['policy_std'],
                    std_min=last['std_min'], std_median=last['std_median'], std_max=last['std_max'],
                    fps_mean=statistics.mean(2048*32/(r['collection_time']+r['learning_time']) for r in rows),
                    vram_last50_mean=statistics.mean(r['vram_mib'] for r in rows[-50:]))
    actions = {k: statistics.mean(r[k] for r in rows[-50:]) for k in last if k.startswith(('raw_action_', 'wrapper_action_'))}
    for prefix in ('raw_action', 'wrapper_action'):
        actions[f'{prefix}_max_abs'] = max(r[f'{prefix}_max_abs'] for r in rows)
    actions.update({k: runtime[k] for k in ('processed_relative_delta_mean_abs', 'processed_relative_delta_max_abs',
                                          'final_clamped_target_mean_abs', 'final_clamped_target_max_abs',
                                          'target_clipping_fraction', 'target_outside_fraction')})
    summary = dict(candidate_name=args.candidate, config=cfg, training=training, actions=actions, evaluations=[])
    export(stage, summary)
    clip = 'none' if cfg['clip_actions'] is None else str(cfg['clip_actions'])
    for mode, name in [('A', 'current-relative-repeat'), ('B', 'hold-grasp-posture')]:
        out = stage / f'mode{mode}'
        run([sys.executable, '-u', 'scripts/rsl_rl/evaluate.py', '--task', TASK, '--checkpoint', runtime['checkpoint'],
             '--rounds', '3', '--seed', '1', '--diagnostics', '--lift_hand_mode', name,
             '--output_dir', str(out), '--headless', '--clip_actions', clip], stage / f'mode{mode}.log')
        result = read(out / 'quantitative_eval.json')
        diagnostics = read(out / 'grasp_diagnostics.json')
        assert result['total_attempts'] == 90
        if mode == 'B':
            assert max(r['lift_max_hand_target_drift_rad'] for r in diagnostics) == 0
        retention = [r['lift_contact_retention_fraction'] for r in diagnostics if r['lift_contact_retention_fraction'] is not None]
        row = dict(mode=mode, source_successes=result['total_source_successes'], strict_successes=result['total_successes'],
                   attempts=result['total_attempts'], median_final_distance_m=statistics.median(r['grasp_end_distance_m'] for r in diagnostics),
                   max_height_gain_m=max(r['height_gain_m'] for r in diagnostics),
                   median_height_gain_m=statistics.median(r['height_gain_m'] for r in diagnostics),
                   max_during_lift_height_gain_m=max(r['lift_max_height_gain_m'] for r in diagnostics),
                   mean_lift_retention=statistics.mean(retention) if retention else None)
        for label in ('object_contact', 'top_contact', 'bottom_contact', 'table_contact', 'two_plus_contacts', 'three_plus_contacts'):
            row[f'trials_with_{label}'] = sum(r[f'grasp_{label}_steps'] > 0 for r in diagnostics)
        row['trials_with_thumb_other'] = sum(any(r[f'grasp_thumb_{finger}_steps'] > 0 for finger in ('index', 'middle', 'ring')) for r in diagnostics)
        summary['evaluations'].append(row)
        export(stage, summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()

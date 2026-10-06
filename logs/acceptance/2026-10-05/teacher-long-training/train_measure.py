"""执行现有train.py；只记录优化4数据，并纠正阶段resume的0起始计数。"""
import inspect
import json
import math
from pathlib import Path
import runpy
import sys
import textwrap
import time

from isaaclab.app import AppLauncher

run_name = sys.argv[sys.argv.index('--run_name') + 1]
stage_number = int(run_name.rsplit('stage', 1)[1])
root = Path('logs/acceptance/2026-10-05/teacher-long-training')
stage = root / f'stage{stage_number}'
stage.mkdir(parents=True, exist_ok=True)
runtime = {}
records = []
launcher_init = AppLauncher.__init__


def launch_and_measure(self, *args, **kwargs):
    launcher_init(self, *args, **kwargs)
    import torch
    import pynvml
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from rsl_rl.runners import OnPolicyRunner
    from rsl_rl.algorithms import PPO

    pynvml.nvmlInit()
    gpu = pynvml.nvmlDeviceGetHandleByIndex(0)

    # 原PPO已计算的每个minibatch KL只作记录；不修改更新公式或adaptive LR。
    original_update = PPO.update
    source = textwrap.dedent(inspect.getsource(original_update))
    source_lines = []
    for line in source.splitlines():
        source_lines.append(line)
        if line.strip() == 'kl_mean = torch.mean(kl)':
            indentation = line[:len(line) - len(line.lstrip())]
            source_lines.append(indentation + 'diagnostic_kl.append(kl_mean.detach().clone())')
    source = '\n'.join(source_lines) + '\n'
    source = source.replace('    mean_value_loss =', '    diagnostic_kl = []\n    mean_value_loss =', 1)
    source = source.replace('    return loss_dict',
                            '    loss_dict["kl"] = float(torch.stack(diagnostic_kl).mean())\n    return loss_dict')
    namespace = dict(original_update.__globals__)
    exec(compile(source, '<optimization4-kl-diagnostic>', 'exec'), namespace)
    PPO.update = namespace['update']
    (stage / 'measured_ppo_update.py').write_text(source)

    original_step = RslRlVecEnvWrapper.step

    def measured_step(self, actions):
        e = self.unwrapped
        if not runtime:
            robot, obj = e.scene['robot'], e.scene['object']
            runtime.update(
                start_time=time.monotonic(),
                names=e.termination_manager.active_terms,
                counts=torch.zeros(len(e.termination_manager.active_terms), device=e.device, dtype=torch.long),
                qmin=obj.data.joint_pos.clone(), qmax=obj.data.joint_pos.clone(),
                qvel=torch.zeros(robot.num_joints, device=e.device),
                finite=torch.ones((), device=e.device, dtype=torch.bool),
                target_outside=torch.zeros((), device=e.device, dtype=torch.long),
                target_clipped=torch.zeros((), device=e.device, dtype=torch.long), target_count=0,
            )
            term = e.action_manager.get_term('teacher')
            original_apply = term.apply_actions

            def measured_apply():
                limits = robot.data.joint_pos_limits[:, term._joint_ids]
                proposed = robot.data.joint_pos[:, term._joint_ids] + term.processed_actions
                runtime['target_clipped'] += ((proposed < limits[..., 0]) | (proposed > limits[..., 1])).sum()
                original_apply()
                runtime['target_outside'] += ((term.target < limits[..., 0]) | (term.target > limits[..., 1])).sum()
                runtime['target_count'] += term.target.numel()

            term.apply_actions = measured_apply
            original_scene_update = e.scene.update

            def measured_scene_update(dt):
                original_scene_update(dt)
                q = obj.data.joint_pos
                runtime['qmin'] = torch.minimum(runtime['qmin'], q)
                runtime['qmax'] = torch.maximum(runtime['qmax'], q)
                runtime['qvel'] = torch.maximum(runtime['qvel'], robot.data.joint_vel.abs().max(0).values)
                runtime['finite'] &= torch.isfinite(robot.data.joint_pos).all() & torch.isfinite(robot.data.joint_vel).all() & torch.isfinite(obj.data.body_state_w).all() & torch.isfinite(q).all()

            e.scene.update = measured_scene_update
        result = original_step(self, actions)
        runtime['counts'] += torch.stack([e.termination_manager.get_term(name).sum() for name in runtime['names']])
        runtime['finite'] &= torch.isfinite(actions).all() & torch.isfinite(result[1]).all()
        for obs in result[0].values():
            runtime['finite'] &= torch.isfinite(obs).all()
        return result

    RslRlVecEnvWrapper.step = measured_step
    original_load = OnPolicyRunner.load

    def measured_load(self, *args, **kwargs):
        result = original_load(self, *args, **kwargs)
        # 官方load恢复optimizer LR，但未同步adaptive调度使用的独立字段。
        self.alg.learning_rate = self.alg.optimizer.param_groups[0]['lr']
        # RSL-RL保存的是最后完成的0起始编号；下一阶段从下一iteration继续。
        self.current_learning_iteration += 1
        return result

    OnPolicyRunner.load = measured_load
    original_log = OnPolicyRunner.log

    def snapshot(runner):
        robot = runner.env.unwrapped.scene['robot']
        return dict(
            run_path=runner.log_dir, checkpoint=str(Path(runner.log_dir) / f'model_{runner.current_learning_iteration}.pt'),
            iteration=runner.current_learning_iteration + 1,
            total_env_transitions=(runner.current_learning_iteration + 1) * 2048 * 32,
            finite=bool(runtime['finite']), terminations=dict(zip(runtime['names'], runtime['counts'].tolist())),
            object_joint_min=float(runtime['qmin'].min()), object_joint_max=float(runtime['qmax'].max()),
            max_velocity_by_joint=dict(zip(robot.joint_names, runtime['qvel'].tolist())),
            target_outside_fraction=int(runtime['target_outside']) / runtime['target_count'],
            target_clipping_fraction=int(runtime['target_clipped']) / runtime['target_count'],
            cuda_peak_allocated_mib=torch.cuda.max_memory_allocated() / 1024**2,
            vram_peak_mib=max(r['vram_mib'] for r in records),
            from_scratch='--resume' not in sys.argv,
        )

    def measured_log(self, locs, *args, **kwargs):
        original_log(self, locs, *args, **kwargs)
        assert bool(runtime['finite']), 'Non-finite training runtime'
        assert all(math.isfinite(float(v)) for v in locs['loss_dict'].values()), 'Non-finite PPO scalar'
        assert torch.isfinite(self.alg.policy.action_std).all(), 'Non-finite policy std'
        row = dict(iteration=locs['it'] + 1, collection_time=locs['collection_time'],
                   learning_time=locs['learn_time'], learning_rate=self.alg.learning_rate,
                   policy_std=float(self.alg.policy.action_std.mean()), **locs['loss_dict'],
                   vram_mib=pynvml.nvmlDeviceGetMemoryInfo(gpu).used / 1024**2)
        records.append(row)
        with (stage / 'iterations.jsonl').open('a') as file:
            file.write(json.dumps(row, allow_nan=False) + '\n')
        if row['iteration'] == 100:
            self.save(str(Path(self.log_dir) / 'model_99.pt'))
            stage100 = root / 'stage100'
            stage100.mkdir(exist_ok=True)
            point = snapshot(self)
            point['wall_time_s'] = time.monotonic() - runtime['start_time']
            point['status'] = 'completed'
            (stage100 / 'runtime.json').write_text(json.dumps(point, indent=2, allow_nan=False))
        (stage / 'runtime.json').write_text(json.dumps(snapshot(self), indent=2, allow_nan=False))

    OnPolicyRunner.log = measured_log
    original_learn = OnPolicyRunner.learn

    def measured_learn(self, *args, **kwargs):
        start = time.monotonic()
        result = original_learn(self, *args, **kwargs)
        final = snapshot(self)
        final['wall_time_s'] = time.monotonic() - start
        final['status'] = 'completed'
        (stage / 'runtime.json').write_text(json.dumps(final, indent=2, allow_nan=False))
        self.writer.flush()
        return result

    OnPolicyRunner.learn = measured_learn


AppLauncher.__init__ = launch_and_measure
sys.path.insert(0, str(Path('scripts/rsl_rl').resolve()))
sys.argv[0] = 'scripts/rsl_rl/train.py'
runpy.run_path('scripts/rsl_rl/train.py', run_name='__main__')

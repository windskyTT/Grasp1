"""执行现有train.py，按优化5记录诊断并覆盖单变量candidate配置。"""
import argparse
import inspect
import json
import math
from pathlib import Path
import runpy
import statistics
import sys
import textwrap
import time

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument('--measurement-dir', type=Path, required=True)
parser.add_argument('--candidate-config', type=Path, required=True)
measurement, train_args = parser.parse_known_args()
from_scratch = '--resume' not in train_args
stage = measurement.measurement_dir.resolve()
stage.mkdir(parents=True, exist_ok=True)
candidate = json.loads(measurement.candidate_config.read_text())
sys.argv = [sys.argv[0], *train_args]
runtime = {}
records = []
launcher_init = AppLauncher.__init__


def instrument_ppo_update(original_update):
    """保存已安装PPO实现；仅增加KL测量，fixed时不执行LR调度。"""
    source = textwrap.dedent(inspect.getsource(original_update))
    source = source.replace('    mean_value_loss =', '    diagnostic_kl = []\n    mean_value_loss =', 1)
    lines = []
    for line in source.splitlines():
        if line.strip() == 'if self.desired_kl is not None and self.schedule == "adaptive":':
            lines.extend([
                '        if self.schedule != "adaptive":',
                '            with torch.inference_mode():',
                '                diagnostic_kl.append(torch.sum(',
                '                    torch.log(sigma_batch / old_sigma_batch + 1.0e-5)',
                '                    + (old_sigma_batch.square() + (old_mu_batch - mu_batch).square())',
                '                    / (2.0 * sigma_batch.square()) - 0.5, dim=-1).mean())',
            ])
        lines.append(line)
        if line.strip() == 'kl_mean = torch.mean(kl)':
            indentation = line[:len(line) - len(line.lstrip())]
            lines.append(indentation + 'diagnostic_kl.append(kl_mean.detach().clone())')
    source = '\n'.join(lines) + '\n'
    source = source.replace('    return loss_dict',
                            '    loss_dict["kl"] = float(torch.stack(diagnostic_kl).mean())\n    return loss_dict')
    namespace = dict(original_update.__globals__)
    exec(compile(source, '<optimization5-kl-measurement>', 'exec'), namespace)
    (stage / 'measured_ppo_update.py').write_text(source)
    return namespace['update']


def launch_and_measure(self, *args, **kwargs):
    launcher_init(self, *args, **kwargs)
    import cli_args
    import pynvml
    import torch
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from rsl_rl.algorithms import PPO
    from rsl_rl.runners import OnPolicyRunner
    from Grasp1.tasks.manager_based.grasp1.mdp.runtime import filtered_contact_impulses

    original_config = cli_args.update_rsl_rl_cfg

    def configure(agent_cfg, args_cli):
        cfg = original_config(agent_cfg, args_cli)
        cfg.clip_actions = candidate['clip_actions']
        cfg.algorithm.schedule = candidate['schedule']
        cfg.algorithm.learning_rate = 5.0e-4
        cfg.algorithm.lam = candidate['lam']
        cfg.minimum_action_std = candidate['minimum_std']
        return cfg

    cli_args.update_rsl_rl_cfg = configure
    pynvml.nvmlInit()
    gpu = pynvml.nvmlDeviceGetHandleByIndex(0)
    measured_update = instrument_ppo_update(PPO.update)
    learn_start = None

    def update(self):
        return measured_update(self)

    PPO.update = update
    original_step = RslRlVecEnvWrapper.step

    def measured_step(self, actions):
        e = self.unwrapped
        if not runtime:
            robot, obj = e.scene['robot'], e.scene['object']
            runtime.update(
                names=e.termination_manager.active_terms,
                counts=torch.zeros(len(e.termination_manager.active_terms), device=e.device, dtype=torch.long),
                qmin=obj.data.joint_pos.clone(), qmax=obj.data.joint_pos.clone(),
                qvel=torch.zeros(robot.num_joints, device=e.device),
                finite=torch.ones((), device=e.device, dtype=torch.bool),
                contact_force_peak=torch.zeros((), device=e.device),
                target_outside=torch.zeros((), device=e.device, dtype=torch.long),
                target_clipped=torch.zeros((), device=e.device, dtype=torch.long), target_count=0,
                final_target_abs_sum=torch.zeros((), device=e.device),
                final_target_abs_max=torch.zeros((), device=e.device),
                processed_abs_sum=torch.zeros((), device=e.device),
                processed_abs_max=torch.zeros((), device=e.device), processed_count=0,
            )
            term = e.action_manager.get_term('teacher')
            original_process = term.process_actions

            def measured_process(received):
                original_process(received)
                runtime['processed_abs_sum'] += term.processed_actions.abs().sum()
                runtime['processed_abs_max'] = torch.maximum(runtime['processed_abs_max'], term.processed_actions.abs().max())
                runtime['processed_count'] += term.processed_actions.numel()

            term.process_actions = measured_process
            original_apply = term.apply_actions

            def measured_apply():
                limits = robot.data.joint_pos_limits[:, term._joint_ids]
                proposed = robot.data.joint_pos[:, term._joint_ids] + term.processed_actions
                runtime['target_clipped'] += ((proposed < limits[..., 0]) | (proposed > limits[..., 1])).sum()
                original_apply()
                runtime['target_outside'] += ((term.target < limits[..., 0]) | (term.target > limits[..., 1])).sum()
                runtime['target_count'] += term.target.numel()
                runtime['final_target_abs_sum'] += term.target.abs().sum()
                runtime['final_target_abs_max'] = torch.maximum(runtime['final_target_abs_max'], term.target.abs().max())

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
            original_reward = e.reward_manager.compute

            def measured_reward(*args, **kwargs):
                reward = original_reward(*args, **kwargs)
                for prefix in e._teacher_runtime_features.contact_sensors:
                    impulse = filtered_contact_impulses(e, prefix)[0]
                    runtime['contact_force_peak'] = torch.maximum(runtime['contact_force_peak'], impulse.max() / e.physics_dt)
                    runtime['finite'] &= torch.isfinite(impulse).all()
                return reward

            e.reward_manager.compute = measured_reward
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
        self.alg.learning_rate = self.alg.optimizer.param_groups[0]['lr']
        self.current_learning_iteration += 1
        return result

    OnPolicyRunner.load = measured_load
    original_log = OnPolicyRunner.log

    def action_stats(actions):
        absolute = actions.abs().flatten()
        quantiles = torch.quantile(absolute, torch.tensor([0.95, 0.99], device=absolute.device))
        return dict(mean_abs=float(absolute.mean()), max_abs=float(absolute.max()),
                    p95_abs=float(quantiles[0]), p99_abs=float(quantiles[1]),
                    fraction_gt1=float((absolute > 1).float().mean()),
                    fraction_gt2=float((absolute > 2).float().mean()))

    def snapshot(runner):
        robot = runner.env.unwrapped.scene['robot']
        qvel = dict(zip(robot.joint_names, runtime['qvel'].tolist()))
        return dict(
            run_path=runner.log_dir, checkpoint=str(Path(runner.log_dir) / f'model_{runner.current_learning_iteration}.pt'),
            iteration=runner.current_learning_iteration + 1,
            total_env_transitions=(runner.current_learning_iteration + 1) * runner.env.num_envs * runner.num_steps_per_env,
            finite=bool(runtime['finite']), terminations=dict(zip(runtime['names'], runtime['counts'].tolist())),
            object_joint_min=float(runtime['qmin'].min()), object_joint_max=float(runtime['qmax'].max()),
            max_velocity_by_joint=qvel,
            ur5_qvel_peak=max(v for k, v in qvel.items() if not k.startswith('joint_')),
            allegro_qvel_peak=max(v for k, v in qvel.items() if k.startswith('joint_')),
            contact_force_peak=float(runtime['contact_force_peak']),
            target_outside_fraction=int(runtime['target_outside']) / runtime['target_count'],
            target_clipping_fraction=int(runtime['target_clipped']) / runtime['target_count'],
            final_clamped_target_mean_abs=float(runtime['final_target_abs_sum']) / runtime['target_count'],
            final_clamped_target_max_abs=float(runtime['final_target_abs_max']),
            processed_relative_delta_mean_abs=float(runtime['processed_abs_sum']) / runtime['processed_count'],
            processed_relative_delta_max_abs=float(runtime['processed_abs_max']),
            cuda_peak_allocated_mib=torch.cuda.max_memory_allocated() / 1024**2,
            vram_peak_mib=max(r['vram_mib'] for r in records),
            from_scratch=from_scratch, candidate=candidate,
        )

    def measured_log(self, locs, *args, **kwargs):
        original_log(self, locs, *args, **kwargs)
        assert bool(runtime['finite']), 'Non-finite training runtime'
        assert all(math.isfinite(float(v)) for v in locs['loss_dict'].values()), 'Non-finite PPO scalar'
        std = self.alg.policy.std.detach()
        assert torch.isfinite(std).all(), 'Non-finite policy std'
        raw = self.alg.storage.actions
        wrapper = raw if self.env.clip_actions is None else raw.clamp(-self.env.clip_actions, self.env.clip_actions)
        row = dict(iteration=locs['it'] + 1, collection_time=locs['collection_time'],
                   learning_time=locs['learn_time'], learning_rate=self.alg.learning_rate,
                   mean_reward=statistics.mean(locs['rewbuffer']) if locs['rewbuffer'] else None,
                   std_min=float(std.min()), std_median=float(std.median()), std_max=float(std.max()),
                   policy_std=float(std.mean()), **locs['loss_dict'],
                   vram_mib=pynvml.nvmlDeviceGetMemoryInfo(gpu).used / 1024**2)
        row.update({f'raw_action_{k}': v for k, v in action_stats(raw).items()})
        row.update({f'wrapper_action_{k}': v for k, v in action_stats(wrapper).items()})
        records.append(row)
        point = snapshot(self)
        row.update({k: point[k] for k in ('target_clipping_fraction', 'target_outside_fraction', 'ur5_qvel_peak', 'allegro_qvel_peak', 'contact_force_peak')})
        with (stage / 'iterations.jsonl').open('a') as file:
            file.write(json.dumps(row, allow_nan=False) + '\n')
        for key in ('std_min', 'std_median', 'std_max'):
            self.writer.add_scalar(f'Policy/{key}', row[key], locs['it'])
        for key, value in row.items():
            if key.startswith(('raw_action_', 'wrapper_action_')):
                self.writer.add_scalar(f'Action/{key}', value, locs['it'])
        if row['iteration'] == 100:
            self.save(str(Path(self.log_dir) / 'model_99.pt'))
            node100 = dict(point, wall_time_s=time.monotonic() - learn_start, status='completed')
            (stage / 'runtime100.json').write_text(json.dumps(node100, indent=2, allow_nan=False))
        (stage / 'runtime.json').write_text(json.dumps(point, indent=2, allow_nan=False))

    OnPolicyRunner.log = measured_log
    original_learn = OnPolicyRunner.learn

    def measured_learn(self, *args, **kwargs):
        nonlocal learn_start
        start = time.monotonic()
        learn_start = start
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

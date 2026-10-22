"""在唯一Teacher-v0中训练完整自主抓取；checkpoint契约保存在infos中。"""
import argparse
from datetime import datetime
from pathlib import Path
import cli_args
from isaaclab.app import AppLauncher


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=(cli_args.TASK,), default=cli_args.TASK)
    parser.add_argument("--num_envs", type=int, default=None)
    parser.add_argument("--max_iterations", type=int, default=None, help="Additional PPO updates in this invocation.")
    parser.add_argument("--seed", type=int, default=42)
    cli_args.add_rsl_rl_args(parser)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if args.num_envs is not None and args.num_envs < 1:
        parser.error("--num_envs must be positive")
    if args.max_iterations is not None and args.max_iterations < 1:
        parser.error("--max_iterations must be positive")
    if args.checkpoint and not args.resume:
        parser.error("Use --resume with --checkpoint")
    if args.no_log:
        args.kit_args = (args.kit_args or "") + " --/log/file=/dev/null --/app/settings/persistent=false --/crashreporter/enabled=false"
    app = AppLauncher(args).app
    env = None
    runner = None
    exit_code = 0
    try:
        import math
        import gymnasium as gym
        import torch
        from rsl_rl.runners import OnPolicyRunner
        from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
        import Grasp1.tasks  # noqa: F401
        from Grasp1.tasks.manager_based.grasp1.config.ur5_allegro.teacher_env_cfg import UR5AllegroTeacherEnvCfg
        from Grasp1.tasks.manager_based.grasp1.config.ur5_allegro.agents.rsl_rl_ppo_cfg import UR5AllegroTeacherPPORunnerCfg

        cfg, agent = UR5AllegroTeacherEnvCfg(), UR5AllegroTeacherPPORunnerCfg()
        cfg.set_num_envs(args.num_envs or cfg.scene.num_envs)
        cfg.synchronize_control_timing()
        agent.synchronize_control_timing(cfg.control_dt())
        cfg.seed = agent.seed = args.seed
        if args.device is not None:
            cfg.sim.device = agent.device = args.device
        iterations = args.max_iterations or agent.max_iterations
        task, training = cli_args.teacher_contract(cfg, agent)
        resume_path = cli_args.checkpoint_path(agent, args.checkpoint, args.load_run) if args.resume else None
        resume_info = cli_args.validate_teacher_checkpoint(resume_path, task, training) if resume_path else None
        suffix = f"_{args.run_name}" if args.run_name else ""
        log_dir = None if args.no_log else Path("logs/rsl_rl", agent.experiment_name, datetime.now().strftime("%Y-%m-%d_%H-%M-%S_%f")+suffix).resolve()
        cfg.log_dir = None if log_dir is None else str(log_dir)
        cfg.sim.save_logs_to_file = not args.no_log
        print(f"[INFO] Training objects: {dict((n, cfg.object_names.count(n)) for n in dict.fromkeys(cfg.object_names))}", flush=True)
        print(f"[INFO] physics_dt={cfg.sim.dt}, control_dt={cfg.control_dt()}, gamma={agent.algorithm.gamma}, lambda={agent.algorithm.lam}")
        env = gym.make(cli_args.TASK, cfg=cfg)
        env = RslRlVecEnvWrapper(env, clip_actions=agent.clip_actions)
        raw = env.unwrapped
        obs = env.get_observations()
        if obs["policy"].shape != (cfg.scene.num_envs, 153) or env.num_actions != 22:
            raise RuntimeError(f"Teacher dimensions: obs={obs['policy'].shape}, actions={env.num_actions}")
        if not torch.isfinite(obs["policy"]).all():
            raise RuntimeError("Nonfinite reset observation")
        print(f"[PASS] Reset: observation={tuple(obs['policy'].shape)}, action=({env.num_envs}, {env.num_actions})", flush=True)
        runner = OnPolicyRunner(env, agent.to_dict(), log_dir=cfg.log_dir, device=agent.device)
        # 标准runner默认收集Git状态；空列表确保本任务完全不执行Git操作。
        runner.git_status_repos = []
        # RSL-RL 3.1.2 在无日志时仍会调用代码快照；此开关同时跳过该写盘路径。
        runner.disable_logs = args.no_log
        completed = resume_info["completed_updates"] if resume_info else 0
        if resume_path:
            runner.load(resume_path, map_location=agent.device)
            runner.current_learning_iteration = completed
        starting_updates = completed
        original_step, original_update, original_save = env.step, runner.alg.update, runner.save
        totals = dict(reward=0., opposing=0., lift=0., steps=0,
                      successful_episodes=0, completed_episodes=0,
                      unsafe_episodes=0, timeout_episodes=0)
        reward_totals = torch.zeros(len(raw.reward_manager.active_terms), device=env.device)

        def step(actions):
            result = original_step(actions)
            observations, rewards, dones, extras = result
            if not torch.isfinite(observations["policy"]).all() or not torch.isfinite(rewards).all():
                raise RuntimeError("Nonfinite Teacher observation or reward")
            done = dones.bool()
            episode = extras["teacher_episode"]
            reward_totals.add_(raw.reward_manager._step_reward.mean(0)*raw.step_dt)
            totals["reward"] += rewards.mean().item()
            totals["opposing"] += extras["teacher_grip_rate"].item()
            totals["lift"] += episode["lift_height"].mean().item()
            totals["completed_episodes"] += done.sum().item()
            totals["successful_episodes"] += (done & episode["success"].bool()).sum().item()
            totals["unsafe_episodes"] += (done & episode["unsafe"].bool()).sum().item()
            totals["timeout_episodes"] += (done & episode["timeout"].bool()).sum().item()
            totals["steps"] += 1
            # 不改环境内部log；Runner只接收日志副本，Episode比率由metrics统一记录。
            runner_extras = dict(extras, log={key: value for key, value in extras["log"].items()
                                             if key not in ("success_rate", "unsafe_rate")})
            return observations, rewards, dones, runner_extras

        def update():
            nonlocal completed
            losses = original_update()
            completed += 1
            if not all(math.isfinite(float(v)) for v in losses.values()):
                raise RuntimeError(f"Nonfinite PPO losses at update {completed}: {losses}")
            if not all(torch.isfinite(p).all() for p in runner.alg.policy.parameters()):
                raise RuntimeError(f"Nonfinite policy weights at update {completed}")
            with torch.inference_mode():
                storage = runner.alg.storage
                means = runner.alg.policy.act_inference(storage.observations.flatten(0, 1))
                std = runner.alg.policy.log_std.exp().expand_as(means)
                old = torch.distributions.Normal(storage.mu.flatten(0, 1), storage.sigma.flatten(0, 1))
                new = torch.distributions.Normal(means, std)
                kl = torch.distributions.kl_divergence(old, new).sum(-1).mean().item()
            metrics = {key: totals[key]/max(totals["steps"], 1) for key in ("reward", "opposing", "lift")}
            metrics.update(successful_episodes=totals["successful_episodes"],
                           completed_episodes=totals["completed_episodes"],
                           unsafe_episodes=totals["unsafe_episodes"],
                           timeout_episodes=totals["timeout_episodes"])
            for rate, count in (("success_rate", "successful_episodes"),
                                ("unsafe_rate", "unsafe_episodes"), ("timeout_rate", "timeout_episodes")):
                metrics[rate] = totals[count]/totals["completed_episodes"] if totals["completed_episodes"] else None
            metrics.update(completed_updates=completed, kl=kl, **losses)
            reward_metrics = {name: value.item()/totals["steps"] for name, value in
                              zip(raw.reward_manager.active_terms, reward_totals)}
            for name, value in reward_metrics.items():
                if runner.writer is not None:
                    runner.writer.add_scalar(f"Teacher_Reward/{name}", value, completed)
            print(f"[REWARDS per control step] {reward_metrics}", flush=True)
            reward_totals.zero_()
            print(f"[PPO] {metrics}", flush=True)
            for key, value in metrics.items():
                if runner.writer is not None and value is not None:
                    runner.writer.add_scalar(f"Teacher/{key}", value, completed)
            for key in totals:
                totals[key] = 0
            return losses

        def save(path, infos=None):
            payload = dict(infos or {}, task_contract=task, training_contract=training,
                           completed_updates=completed, safety_failure=raw._teacher_grasp.safety_failure)
            original_save(path, payload)
            print(f"[CHECKPOINT] {path}; completed_updates={completed}", flush=True)

        env.step, runner.alg.update, runner.save = step, update, save
        try:
            runner.learn(num_learning_iterations=iterations, init_at_random_ep_len=False)
        except BaseException:
            if log_dir is not None and completed > starting_updates:
                runner.current_learning_iteration = completed-1
                runner.save(str(log_dir/f"model_{completed-1}.pt"))
            print(f"[STOP] completed_updates_this_run={completed-starting_updates}, total={completed}", flush=True)
            raise
        print(f"[DONE] completed_updates_this_run={completed-starting_updates}, total={completed}; log_dir={log_dir}", flush=True)
    except BaseException:
        import traceback
        exit_code = 1
        traceback.print_exc()
        raise
    finally:
        try:
            if runner is not None and runner.writer is not None:
                runner.writer.close()
        finally:
            try:
                if env is not None:
                    env.close()
            finally:
                if exit_code:
                    app.app.post_quit(exit_code)
                app.close()


if __name__ == "__main__":
    main()

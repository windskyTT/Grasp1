"""Teacher-v0纯Policy播放；固定训练集、153维观测和22维自主动作。"""
import argparse
import random
import time
import cli_args
from isaaclab.app import AppLauncher


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=(cli_args.TASK,), default=cli_args.TASK)
    parser.add_argument("--checkpoint", type=str)
    parser.add_argument("--episodes", type=int, default=None, help="Episodes per environment; default runs until the window closes.")
    parser.add_argument("--num_envs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--real-time", action="store_true")
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if not args.checkpoint:
        parser.error("--checkpoint is required")
    if args.num_envs < 1 or (args.episodes is not None and args.episodes < 1):
        parser.error("--num_envs and --episodes must be positive")
    app = AppLauncher(args).app
    env = None
    exit_code = 0
    try:
        import gymnasium as gym
        import torch
        from rsl_rl.runners import OnPolicyRunner
        from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
        import Grasp1.tasks  # noqa: F401
        from Grasp1.data.object_set import training_object_names
        from Grasp1.tasks.manager_based.grasp1.config.ur5_allegro.teacher_env_cfg import UR5AllegroTeacherEnvCfg
        from Grasp1.tasks.manager_based.grasp1.config.ur5_allegro.agents.rsl_rl_ppo_cfg import UR5AllegroTeacherPPORunnerCfg

        cfg, agent = UR5AllegroTeacherEnvCfg(), UR5AllegroTeacherPPORunnerCfg()
        cfg.seed = agent.seed = args.seed
        # 种子决定训练集中的物体；不固定为目录第一个，也不读取其他数据集。
        selected = random.Random(args.seed).choice(tuple(dict.fromkeys(training_object_names())))
        cfg.set_num_envs(args.num_envs, object_names=(selected,))
        cfg.viewer.eye = (1.4, -2.0, 1.4)
        cfg.viewer.lookat = (0.2, -0.75152, 0.95)
        cfg.synchronize_control_timing()
        agent.synchronize_control_timing(cfg.control_dt())
        if args.device is not None:
            cfg.sim.device = agent.device = args.device
        contract, _ = cli_args.teacher_contract(cfg, agent)
        path = cli_args.checkpoint_path(agent, args.checkpoint)
        infos = cli_args.validate_teacher_checkpoint(path, contract)
        print(f"[PLAY] new_training_set/{selected}, seed={args.seed}, completed_updates={infos['completed_updates']}", flush=True)
        env = gym.make(cli_args.TASK, cfg=cfg)
        env = RslRlVecEnvWrapper(env, clip_actions=agent.clip_actions)
        obs = env.get_observations()
        if obs["policy"].shape != (args.num_envs, 153) or env.num_actions != 22:
            raise RuntimeError("Teacher playback observation/action dimensions differ")
        runner = OnPolicyRunner(env, agent.to_dict(), log_dir=None, device=agent.device)
        runner.git_status_repos = []
        runner.load(path, load_optimizer=False, map_location=agent.device)
        policy = runner.get_inference_policy(device=env.device)
        episodes = torch.zeros(args.num_envs, dtype=torch.long, device=env.device)
        while app.is_running() and (args.episodes is None or bool((episodes < args.episodes).any())):
            started = time.perf_counter()
            with torch.inference_mode():
                actions = policy(obs)
                obs, _, dones, extras = env.step(actions)
            for index in dones.bool().nonzero().flatten().tolist():
                if args.episodes is None or episodes[index] < args.episodes:
                    episodes[index] += 1
                    result = {key: value[index].item() for key, value in extras["teacher_episode"].items()}
                    print(f"[EPISODE] env={index}, episode={episodes[index].item()}, {result}", flush=True)
            if args.real_time:
                delay = env.unwrapped.step_dt-(time.perf_counter()-started)
                if delay > 0:
                    time.sleep(delay)
    except BaseException:
        import traceback
        exit_code = 1
        traceback.print_exc()
        raise
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

"""Teacher PPO抓取 + 原始switch_root_guidance脚本抬升，默认显示Isaac Sim GUI。

源allegro_teacher/visual_eval.py:207-208、520-530和Environment.hpp:604-609、1047。
0.2s源控制周期换为1/60s：抓取70→840步(14s)，抬升30→360步(6s)，
完整插值80→960步(16s)。源评估只执行插值的30/80，不压缩到6秒内完成。
抬臂手指按源5Hz、0.015系数更新绝对目标，两个源控制时刻之间保持目标；
仅恢复目标更新语义，PhysX动力学及源随机单物理步延迟仍非完全等价。
"""
import argparse
import random
import time
import cli_args
from isaaclab.app import AppLauncher


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=(cli_args.TASK,), default=cli_args.TASK)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--num_envs", type=int, default=1)
    parser.add_argument("--episodes", type=int, default=1, help="Independent grasp/lift attempts per environment.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--real-time", action="store_true")
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if args.num_envs < 1 or args.episodes < 1:
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
        from Grasp1.tasks.manager_based.grasp1.grasp1_env_cfg import SOURCE_CONTROL_DT

        cfg, agent = UR5AllegroTeacherEnvCfg(), UR5AllegroTeacherPPORunnerCfg()
        cfg.seed = agent.seed = args.seed
        selected = random.Random(args.seed).choice(tuple(dict.fromkeys(training_object_names())))
        cfg.set_num_envs(args.num_envs, object_names=(selected,))
        cfg.viewer.eye = (1.4, -2.0, 1.4)
        cfg.viewer.lookat = (0.2, -0.75152, 0.95)
        cfg.synchronize_control_timing()
        agent.synchronize_control_timing(cfg.control_dt())
        if args.device is not None:
            cfg.sim.device = agent.device = args.device
        # 先按14s抓取训练契约严格验证；以下20s仅为本入口的完整评估时长。
        contract, _ = cli_args.teacher_contract(cfg, agent)
        path = cli_args.checkpoint_path(agent, args.checkpoint)
        infos = cli_args.validate_teacher_checkpoint(path, contract)
        grasp_steps = round(70 * SOURCE_CONTROL_DT / cfg.control_dt())
        lift_steps = round(30 * SOURCE_CONTROL_DT / cfg.control_dt())
        interpolation_steps = round(80 * SOURCE_CONTROL_DT / cfg.control_dt())
        finger_control_steps = round(SOURCE_CONTROL_DT / cfg.control_dt())
        cfg.episode_length_s = (grasp_steps + lift_steps) * cfg.control_dt()
        env = RslRlVecEnvWrapper(gym.make(cli_args.TASK, cfg=cfg), clip_actions=agent.clip_actions)
        raw = env.unwrapped
        obs = env.get_observations()
        if obs["policy"].shape != (args.num_envs, 153) or env.num_actions != 22:
            raise RuntimeError("Teacher evaluation observation/action dimensions differ")
        runner = OnPolicyRunner(env, agent.to_dict(), log_dir=None, device=agent.device)
        runner.git_status_repos = []
        runner.load(path, load_optimizer=False, map_location=agent.device)
        policy = runner.get_inference_policy(device=env.device)
        term = raw.action_manager.get_term("teacher")
        robot = raw.scene["robot"]
        arm_ids = term._joint_ids[:6]
        finger_ids = term._joint_ids[6:]
        theta0 = torch.tensor([0., -1.57, 1.57, 0., 1.57, -1.57], device=env.device)
        original_process = term.process_actions
        lifting = False
        lift_step = 0
        arm_start = robot.data.joint_pos[:, arm_ids].clone()
        finger_target = robot.data.joint_pos[:, finger_ids].clone()
        alive = torch.ones(args.num_envs, dtype=torch.bool, device=env.device)

        def process_actions(actions):
            nonlocal lift_step
            original_process(actions)
            if lifting:
                lift_step += 1
                # switch_root_guidance在切换时保存实测q；theta0为绝对目标。
                target = arm_start + (theta0 - arm_start) * min(lift_step / interpolation_steps, 1.)
                limits = robot.data.joint_pos_limits[:, arm_ids]
                term.target[alive, :6] = torch.clamp(target, limits[..., 0], limits[..., 1])[alive]
                # 源Environment.hpp:602-613每0.2s计算q_measured + 0.015*a，
                # 在整个源控制周期内保持；不能每60Hz步重算q + 0.1*a。
                if (lift_step - 1) % finger_control_steps == 0:
                    finger_limits = robot.data.joint_pos_limits[:, finger_ids]
                    finger_target.copy_(torch.clamp(
                        robot.data.joint_pos[:, finger_ids] + 0.015 * actions[:, 6:],
                        finger_limits[..., 0], finger_limits[..., 1],
                    ))
                term.target[alive, 6:] = finger_target[alive]
                # 已异常Reset的环境保留当前关节目标，不拼接为同一次抓取。
                term.target[~alive] = robot.data.joint_pos[:, term._joint_ids][~alive]

        term.process_actions = process_actions
        successful = 0
        attempted = 0
        print(f"[VISUAL_EVAL] object={selected}; completed_updates={infos['completed_updates']}; "
              f"grasp={grasp_steps}, lift={lift_steps}, interpolation={interpolation_steps}; "
              f"lift_fingers=source_5Hz, scale=0.015, hold_steps={finger_control_steps}; "
              f"GUI={raw.sim.has_gui()}", flush=True)
        for trial in range(args.episodes):
            if not app.is_running():
                break
            lifting = False
            lift_step = 0
            alive[:] = True
            if trial:
                with torch.inference_mode():
                    obs, _ = env.reset()
            final_actions = torch.zeros(args.num_envs, 22, device=env.device)
            lift_height = torch.zeros(args.num_envs, device=env.device)
            completed_sequence = False
            with torch.inference_mode():
                for step in range(grasp_steps + lift_steps):
                    if not app.is_running():
                        break
                    started = time.perf_counter()
                    if step < grasp_steps:
                        actions = policy(obs)
                        final_actions.copy_(actions)
                    else:
                        if step == grasp_steps:
                            lifting = True
                            arm_start.copy_(robot.data.joint_pos[:, arm_ids])
                            print(f"[LIFT] trial={trial + 1}, surviving_grasps={alive.sum().item()}", flush=True)
                        # 末次手指动作由process_actions按源5Hz设置并保持绝对目标。
                        actions = final_actions.clone()
                    actions[~alive] = 0.
                    obs, _, dones, extras = env.step(actions)
                    episode = extras["teacher_episode"]
                    lift_height.copy_(episode["lift_height"])
                    final_step = step == grasp_steps + lift_steps - 1
                    # 最后一步正常20s timeout允许完成；任何异常终止均失败。
                    failed = dones.bool() & (~episode["timeout"] | episode["invalid"] | episode["hand_height"])
                    if not final_step:
                        failed |= dones.bool()
                    alive &= ~failed
                    completed_sequence = final_step
                    if args.real_time:
                        delay = raw.step_dt - (time.perf_counter() - started)
                        if delay > 0:
                            time.sleep(delay)
            if not completed_sequence:
                print(f"[INCOMPLETE] trial={trial + 1}; window closed before evaluation ended", flush=True)
                break
            # 仅此独立评估入口统计原始top世界Z - reset root世界Z > 0.10m。
            success = alive & (lift_height > 0.10)
            successful += success.sum().item()
            attempted += args.num_envs
            for index in range(args.num_envs):
                print(f"[EVAL] trial={trial + 1}, env={index}, object={selected}, "
                      f"uninterrupted={alive[index].item()}, lift_m={lift_height[index].item():.6f}, "
                      f"success={success[index].item()}", flush=True)
        rate = successful / attempted if attempted else None
        print(f"[RESULT] scripted_lift_success={successful}/{attempted}, rate={rate}; "
              "PPO grasp + original scripted arm lift", flush=True)
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

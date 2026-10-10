# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""使用 RSL-RL 检查点在 Isaac Sim 中播放 UR5 + Allegro Teacher 策略。

脚本可从指定数据集选择一个物体，并将该物体分配给一个或多个并行环境；
也可录制视频，或在指定回合数后退出。
"""

"""先解析命令行参数并启动 Isaac Sim，再导入依赖仿真应用的运行模块。"""

import argparse
import random
import sys

from isaaclab.app import AppLauncher

# RSL-RL 命令行参数解析与配置覆盖工具。
import cli_args  # isort: skip


# 命令行参数：指定任务、数据集/物体、检查点、环境数量和播放方式。
# -----------------------------------------------------------------------------

parser = argparse.ArgumentParser(
    description="Play a UR5-Allegro Teacher checkpoint with RSL-RL."
)
parser.add_argument("--video",action="store_true",default=False,help="Record a video during playback.")
parser.add_argument("--video_length",type=int,default=200,help="Length of the recorded video in control steps.")
parser.add_argument("--disable_fabric",action="store_true",default=False,help="Disable Fabric and use USD I/O operations.")
parser.add_argument("--num_envs",type=int,default=1,help="Number of parallel environments using the selected object.")
parser.add_argument("--task",type=str,default="Grasp1-UR5-Allegro-Teacher-v0",help="Registered Gymnasium task id.")
parser.add_argument("--dataset",type=str,default="new_training_set",help="Object dataset under assets/objects.")
parser.add_argument("--object",dest="object_name",type=str,default=None,
                    help="Object name. Randomly select one from --dataset when omitted.",
                    )
parser.add_argument("--episodes",type=int,default=None,
                     help=("Stop after this many episodes per environment. "
                           "By default playback continues until Isaac Sim is closed."
                           ),
                    )
parser.add_argument("--agent",type=str,default="rsl_rl_cfg_entry_point",
                    help="Name of the RSL-RL agent configuration entry point.",
                    )
parser.add_argument("--seed",type=int,default=None,help="Seed used for environment reset and random object selection.")
parser.add_argument("--real-time",action="store_true",default=False,help="Run in real time, if possible.")

# 加入 RSL-RL 通用参数，例如检查点路径、训练运行目录和恢复选项。
cli_args.add_rsl_rl_args(parser)

# 加入 Isaac Sim 启动参数，例如 headless、设备和相机选项。
parser.add_argument("--decimation", type=int, choices=(2, 4), default=None, help="Teacher policy: 2=60Hz, 4=30Hz; physics stays 120Hz.")
parser.add_argument("--episode_length_s", type=float, default=None, help="Training/play episode duration in seconds (4 or 10 for the A/B).")
AppLauncher.add_app_launcher_args(parser)

args_cli, hydra_args = parser.parse_known_args()

if args_cli.video:
    # 录制视频时启用相机传感器。
    args_cli.enable_cameras = True

# 启动仿真前检查环境数和回合数必须为正数。
if args_cli.num_envs <= 0:
    parser.error("--num_envs must be greater than zero.")

if args_cli.episodes is not None and args_cli.episodes <= 0:
    parser.error("--episodes must be greater than zero.")

# 自定义参数已由 argparse 解析；只把剩余参数留给 Hydra。
sys.argv = [sys.argv[0]] + hydra_args


# -----------------------------------------------------------------------------
# 先启动 Isaac Sim；依赖仿真应用的运行模块必须在启动后导入。
# -----------------------------------------------------------------------------

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


"""仿真应用启动后，继续导入环境、RSL-RL 和任务运行依赖。"""

import os
import time

import gymnasium as gym
import torch
from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

# 导入 Grasp1 以执行 Gymnasium 任务注册。
import Grasp1.tasks  # noqa: F401
from Grasp1.data.object_set import object_names
from Grasp1.utils.paths import object_asset_dir


# 播放辅助函数。
# -----------------------------------------------------------------------------


def _configure_play_object(
    env_cfg: ManagerBasedRLEnvCfg,
    *,
    dataset_name: str,
    object_name: str,
    num_envs: int,
) -> tuple[str, ...]:
    """将 Teacher 训练物体池替换为指定物体，并返回各环境对应的物体名。

    将所选名称重复 ``num_envs`` 次，分配给各个并行环境；同时更新场景 USD 路径、
    环境物体池，以及启动数据准备和 reset 事件读取的数据集与物体元数据，确保
    场景生成的物体和 reset 加载的网格、最低点数据一致。

    ``env_cfg`` 是待修改的环境配置；其余参数指定数据集、物体名和并行环境数。
    返回值是按环境顺序重复的物体名称元组。
    """
    repeated_names = (object_name,) * num_envs
    env_cfg.scene.num_envs = num_envs
    object_usd = object_asset_dir(dataset_name, object_name) / f"{object_name}.usd"
    env_cfg.scene.object.spawn.usd_path = [str(object_usd)]
    env_cfg.object_dataset = dataset_name
    env_cfg.object_names = repeated_names
    env_cfg.events.initialize_teacher_data.params["dataset_name"] = dataset_name
    env_cfg.events.initialize_teacher_data.params["object_names"] = repeated_names
    env_cfg.events.reset_teacher.params["dataset_name"] = dataset_name
    env_cfg.events.reset_teacher.params["object_names"] = repeated_names

    return repeated_names


# -----------------------------------------------------------------------------
# 播放主流程。
# -----------------------------------------------------------------------------

@hydra_task_config(args_cli.task, args_cli.agent)
def main(
    env_cfg: ManagerBasedRLEnvCfg,
    agent_cfg: RslRlBaseRunnerCfg,
) -> None:
    """应用命令行设置、加载 Teacher 检查点，并在 Isaac Sim 中执行策略推理。"""

    # 将 RSL-RL 命令行选项覆盖到 Hydra 提供的 agent 配置。
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    if args_cli.decimation is not None:
        env_cfg.decimation = args_cli.decimation
    if args_cli.episode_length_s is not None:
        env_cfg.episode_length_s = args_cli.episode_length_s
    env_cfg.synchronize_control_timing()
    agent_cfg.synchronize_control_timing(env_cfg.control_dt())

    # 在场景和随机化初始化前设置环境随机种子与仿真设备。
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = (
        args_cli.device
        if args_cli.device is not None
        else env_cfg.sim.device
    )

    # 读取数据集物体；若未指定名称，则用种子随机选择一个物体。
    available_objects = object_names(args_cli.dataset)
    if args_cli.object_name is not None and args_cli.object_name not in available_objects:
        raise ValueError(f"Object {args_cli.object_name!r} is not in dataset {args_cli.dataset!r}.")
    object_name = args_cli.object_name or random.Random(agent_cfg.seed).choice(available_objects)

    # 用选中的物体替换训练物体池，并设置并行环境数量。
    selected_names = _configure_play_object(
        env_cfg,
        dataset_name=args_cli.dataset,
        object_name=object_name,
        num_envs=args_cli.num_envs,
    )

    print(f"[INFO] Task: {args_cli.task}")
    print(f"[INFO] Playing object: {args_cli.dataset}/{object_name}")
    print(f"[INFO] Number of environments: {len(selected_names)}")

    # 从显式路径或 RSL-RL 实验日志目录解析模型检查点。

    log_root_path = os.path.abspath(
        os.path.join(
            "logs",
            "rsl_rl",
            agent_cfg.experiment_name,
        )
    )
    print(f"[INFO] Loading experiment from directory: {log_root_path}")

    if args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(
            log_root_path,
            agent_cfg.load_run,
            agent_cfg.load_checkpoint,
        )

    log_dir = os.path.dirname(resume_path)
    env_cfg.log_dir = log_dir

    print(f"[INFO] Loading model checkpoint from: {resume_path}")

    # 创建注册的 Gymnasium 环境。RslRlVecEnvWrapper 会在构造时 reset，触发 Teacher
    # 的物体放置、预抓取姿态和 IK 初始化流程。

    env = gym.make(
        args_cli.task,
        cfg=env_cfg,
        render_mode="rgb_array" if args_cli.video else None,
    )

    if args_cli.video:
        # 从播放起始步录制指定长度的视频，并保存到本次运行的 videos/play 目录。
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording playback video.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # RSL-RL 向量环境包装器负责接口适配和动作裁剪，需作为最后一层包装器。
    env = RslRlVecEnvWrapper(
        env,
        clip_actions=agent_cfg.clip_actions,
    )

    # 检查 runner 类型，创建 OnPolicyRunner 并从检查点恢复策略参数。

    if agent_cfg.class_name != "OnPolicyRunner":
        raise ValueError(
            "UR5-Allegro Teacher expects OnPolicyRunner, "
            f"but received {agent_cfg.class_name!r}."
        )

    runner = OnPolicyRunner(
        env,
        agent_cfg.to_dict(),
        log_dir=None,
        device=agent_cfg.device,
    )
    runner.load(resume_path)

    policy = runner.get_inference_policy(
        device=env.unwrapped.device,
    )

    # 保留策略网络对象，以便在环境回合结束时重置策略内部状态。
    policy_nn = runner.alg.policy

    # 仿真控制步长，供实时播放时进行节奏控制。
    dt = env.unwrapped.step_dt

    # 向量包装器已执行 reset；读取策略推理的初始观测。
    obs = env.get_observations()

    # 记录总控制步数及各并行环境完成的回合数。
    timestep = 0
    episode_counts = torch.zeros(args_cli.num_envs, dtype=torch.long, device=env.unwrapped.device)

    # 推理循环：根据观测生成动作、推进环境并统计回合结束情况。

    while simulation_app.is_running():
        start_time = time.time()

        # 推理阶段不计算梯度，减少播放时不需要的训练计算。
        with torch.inference_mode():
            # 使用推理策略生成动作，并让所有并行环境前进一步。
            actions = policy(obs)
            obs, _, dones, _ = env.step(actions)

            # 清理结束回合环境对应的循环策略状态，避免状态延续到下一回合。
            policy_nn.reset(dones)

        timestep += 1
        episode_counts += dones

        # 达到视频长度，或所有环境均达到指定回合数时退出播放循环。
        if args_cli.video and timestep >= args_cli.video_length:
            break

        if args_cli.episodes is not None and bool(torch.all(episode_counts >= args_cli.episodes)):
            break

        # 实时模式下补足当前控制步剩余时间，使播放速度接近仿真实时速度。
        if args_cli.real_time:
            sleep_time = dt - (time.time() - start_time)
            if sleep_time > 0.0:
                time.sleep(sleep_time)

    # 汇总各环境完成的回合数，并关闭环境释放仿真资源。
    print(
        "[INFO] Playback finished. "
        f"Completed environment episodes: {int(episode_counts.sum().item())}."
    )

    env.close()


if __name__ == "__main__":
    # 运行播放流程，完成后关闭 Isaac Sim 应用。
    main()
    simulation_app.close()

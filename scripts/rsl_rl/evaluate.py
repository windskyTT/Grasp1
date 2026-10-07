# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""使用 UR5 + Allegro Teacher 检查点评估 ShapeNet 物体抓取后的抬升成功率。

脚本先执行策略控制的抓取阶段，再将 UR5 手臂移动到指定抬升姿态，并根据物体
抬升高度判断成功。Isaac Sim 启动后才导入任务和策略运行模块。
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

import cli_args  # isort: skip


# 命令行参数：任务、评估轮数、环境复制数、抓取/抬升秒数及结果路径。
# -----------------------------------------------------------------------------

parser = argparse.ArgumentParser(
    description="Evaluate UR5-Allegro Teacher lift success."
)
parser.add_argument(
    "--task",
    type=str,
    default="Grasp1-UR5-Allegro-Teacher-v0",
    help="Registered Gymnasium task id.",
)
parser.add_argument(
    "--agent",
    type=str,
    default="rsl_rl_cfg_entry_point",
    help="RSL-RL agent configuration entry point.",
)
parser.add_argument(
    "--rounds",
    type=int,
    default=5,
    help="Evaluation trials per object copy.",
)
parser.add_argument(
    "--repeat_per_object",
    type=int,
    default=1,
    help="Parallel copies of each ShapeNet evaluation object.",
)
parser.add_argument(
    "--grasp_duration_s",
    type=float,
    default=4.0,
    help="Policy-controlled grasp duration in seconds; matches the training horizon.",
)
parser.add_argument(
    "--lift_duration_s",
    type=float,
    default=2.0,
    help="Lift test duration in seconds.",
)
parser.add_argument(
    "--lift_ramp_duration_s",
    type=float,
    default=1.5,
    help="Duration in seconds used to interpolate the UR5 arm to the lift target.",
)
parser.add_argument(
    "--success_height",
    type=float,
    default=0.1,
    help="Required object height increase in meters.",
)
parser.add_argument(
    "--output_dir",
    type=str,
    default=None,
    help="Directory for JSON/CSV results. Defaults to <run>/evaluation.",
)
parser.add_argument(
    "--seed",
    type=int,
    default=None,
    help="Evaluation seed.",
)
parser.add_argument('--lift_hand_mode', choices=('current-relative-repeat', 'hold-grasp-posture'),
                    default='current-relative-repeat', help='Lift hand control A/B diagnostic.')
parser.add_argument('--diagnostics', action='store_true', help='Record top/bottom/table and multi-finger contacts.')
parser.add_argument('--clip_actions', choices=('none', '1.0'), default=None,
                    help='Override policy action clipping for the stability A/B experiment.')

# 加入 RSL-RL 通用参数，例如检查点路径和训练运行目录。
cli_args.add_rsl_rl_args(parser)
# 加入 Isaac Sim 启动参数，例如运行设备和 headless 模式。
AppLauncher.add_app_launcher_args(parser)

args_cli, hydra_args = parser.parse_known_args()

# 在启动仿真前检查评估轮数、环境复制数和各阶段秒数为正数。
for name in (
    "rounds",
    "repeat_per_object",
    "grasp_duration_s",
    "lift_duration_s",
    "lift_ramp_duration_s",
):
    if getattr(args_cli, name) <= 0:
        parser.error(f"--{name} must be greater than zero.")

if args_cli.success_height <= 0.0:
    parser.error("--success_height must be greater than zero.")

# 保留未被 argparse 消费的参数给 Hydra 解析任务和 agent 配置。
sys.argv = [sys.argv[0]] + hydra_args

# 启动 Isaac Sim；之后再导入需要仿真运行时的模块。
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


# -----------------------------------------------------------------------------
# 仿真启动后的环境、RSL-RL 和 Grasp1 运行时依赖。
# -----------------------------------------------------------------------------

import os

import gymnasium as gym
import torch
from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.utils.assets import retrieve_file_path
from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import isaaclab_tasks  # noqa: F401
import Grasp1.tasks  # noqa: F401

from Grasp1.data.object_set import (
    EVALUATION_DATASET,
    evaluation_object_names,
)
from Grasp1.robots.robot_profile import UR5_JOINT_NAMES
from Grasp1.tasks.manager_based.grasp1.config.ur5_allegro.teacher_env_cfg import (
    ARM_ACTION_SCALE,
    TEACHER_ACTION_DIM,
)
from Grasp1.utils.paths import object_asset_dir


# 旧版定量评估使用的 UR5 抬升目标关节角。
# -----------------------------------------------------------------------------

LIFT_TARGET_ARM_JOINTS = (
    0.0,
    -1.57,
    1.57,
    0.0,
    1.57,
    -1.57,
)

# 评估环境和文件路径辅助函数。
# -----------------------------------------------------------------------------

def _configure_evaluation(
    env_cfg: ManagerBasedRLEnvCfg,
    object_names: tuple[str, ...],
) -> None:
    """将 ShapeNet USD 和 reset 元数据按相同顺序绑定到并行环境槽位。

    同时设置环境数量、关闭物理复制、禁用物体随机选择，并把评估物体列表传给
    场景及 Teacher 数据初始化/reset 事件。最后按抓取和抬升总步数设置回合时长。
    """
    if not object_names:
        raise ValueError("Evaluation object list must not be empty.")

    env_cfg.scene.num_envs = len(object_names)
    env_cfg.scene.replicate_physics = False
    env_cfg.scene.object.spawn.usd_path = [
        str(object_asset_dir(EVALUATION_DATASET, name) / f"{name}.usd")
        for name in object_names
    ]
    env_cfg.scene.object.spawn.random_choice = False
    env_cfg.object_dataset = EVALUATION_DATASET
    env_cfg.object_names = object_names
    env_cfg.events.initialize_teacher_data.params["dataset_name"] = EVALUATION_DATASET
    env_cfg.events.initialize_teacher_data.params["object_names"] = object_names
    env_cfg.events.reset_teacher.params["dataset_name"] = EVALUATION_DATASET
    env_cfg.events.reset_teacher.params["object_names"] = object_names

    # 评估完整抓取/抬升流程，并额外留一个策略步，避免结束前 timeout。
    step_dt = env_cfg.sim.dt * env_cfg.decimation
    env_cfg.episode_length_s = (
        round(args_cli.grasp_duration_s / step_dt)
        + round(args_cli.lift_duration_s / step_dt) + 1
    ) * step_dt


def _resolve_checkpoint(
    agent_cfg: RslRlBaseRunnerCfg,
) -> tuple[str, str]:
    """根据命令行路径或 agent 配置解析检查点，并返回检查点路径和运行目录。"""
    log_root = os.path.abspath(
        os.path.join(
            "logs",
            "rsl_rl",
            agent_cfg.experiment_name,
        )
    )

    if args_cli.checkpoint:
        checkpoint = retrieve_file_path(args_cli.checkpoint)
    else:
        checkpoint = get_checkpoint_path(
            log_root,
            agent_cfg.load_run,
            agent_cfg.load_checkpoint,
        )

    return checkpoint, os.path.dirname(checkpoint)


def _save_results(
    *,
    output_dir: Path,
    summary: dict,
    rows: list[dict],
) -> None:
    """将总体评估摘要写为 JSON，并将逐物体统计写为 CSV。"""
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / "quantitative_eval.json"
    json_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    csv_path = output_dir / "per_object_success.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "object",
                "attempts",
                "successes",
                "failures",
                "success_rate",
                "source_successes",
                "strict_successes",
                "source_success_rate",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"[INFO] Saved evaluation summary: {json_path}")
    print(f"[INFO] Saved per-object results: {csv_path}")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

@hydra_task_config(args_cli.task, args_cli.agent)
def main(
    env_cfg: ManagerBasedRLEnvCfg,
    agent_cfg: RslRlBaseRunnerCfg,
) -> None:
    """加载 Teacher 策略，对每个 ShapeNet 环境重复执行抓取和抬升评估。"""

    # 合并标准 RSL-RL 命令行覆盖，并在环境初始化前设置种子和设备。
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    if args_cli.clip_actions is not None:
        agent_cfg.clip_actions = None if args_cli.clip_actions == 'none' else 1.0

    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = (
        args_cli.device
        if args_cli.device is not None
        else env_cfg.sim.device
    )

    # 构建 ShapeNet 物体列表；repeat_per_object 控制每个物体并行复制数量。
    names = tuple(evaluation_object_names(args_cli.repeat_per_object))
    _configure_evaluation(env_cfg, names)
    if args_cli.diagnostics:
        from grasp_diagnostics import configure_diagnostics, GraspDiagnostics
        configure_diagnostics(env_cfg.scene)

    # 解析检查点并将运行目录传给环境，供日志及评估结果路径使用。
    checkpoint, log_dir = _resolve_checkpoint(agent_cfg)
    env_cfg.log_dir = log_dir

    print(f"[INFO] Task: {args_cli.task}")
    print(f"[INFO] Dataset: {EVALUATION_DATASET}")
    print(f"[INFO] Unique objects: {len(dict.fromkeys(names))}")
    print(f"[INFO] Environments: {len(names)}")
    print(f"[INFO] Evaluation rounds: {args_cli.rounds}")
    print(f"[INFO] Loading checkpoint: {checkpoint}")

    # 创建并包装向量环境；包装器提供 RSL-RL 所需的观测、动作和 reset 接口。
    env = RslRlVecEnvWrapper(
        gym.make(args_cli.task, cfg=env_cfg),
        clip_actions=agent_cfg.clip_actions,
    )

    if agent_cfg.class_name != "OnPolicyRunner":
        raise ValueError(
            f"Expected OnPolicyRunner, got {agent_cfg.class_name!r}."
        )

    # 创建 RSL-RL runner，加载检查点并取得用于推理的策略。
    runner = OnPolicyRunner(
        env,
        agent_cfg.to_dict(),
        log_dir=None,
        device=agent_cfg.device,
    )
    runner.load(checkpoint)

    policy = runner.get_inference_policy(
        device=env.unwrapped.device,
    )

    policy_nn = runner.alg.policy

    # 用真实环境控制周期将秒数换算为策略步，不依赖旧 5 Hz 配置。
    grasp_steps = round(args_cli.grasp_duration_s / env.unwrapped.step_dt)
    lift_steps = round(args_cli.lift_duration_s / env.unwrapped.step_dt)
    lift_ramp_steps = round(args_cli.lift_ramp_duration_s / env.unwrapped.step_dt)
    print(f"[INFO] Evaluation timing: step_dt={env.unwrapped.step_dt}, "
          f"grasp={grasp_steps} steps, lift={lift_steps} steps, ramp={lift_ramp_steps} steps")

    # 确认运行环境动作维数与 Teacher 策略定义一致。
    if env.num_actions != TEACHER_ACTION_DIM:
        raise RuntimeError(
            "Teacher action dimension mismatch: expected "
            f"{TEACHER_ACTION_DIM}, got {env.num_actions}."
        )

    # 获取机器人和物体 articulation，供控制抬升动作及计算高度变化使用。
    robot = env.unwrapped.scene["robot"]
    object_asset = env.unwrapped.scene["object"]
    diagnostic = GraspDiagnostics(env.unwrapped, names) if args_cli.diagnostics else None
    action_term = env.unwrapped.action_manager.get_term('teacher')

    # 按 robot_profile 中的顺序解析 UR5 关节索引，以正确构造抬升目标。
    arm_joint_ids, resolved_joint_names = robot.find_joints(
        list(UR5_JOINT_NAMES),
        preserve_order=True,
    )
    if len(arm_joint_ids) != len(UR5_JOINT_NAMES):
        raise RuntimeError(
            "Failed to resolve all UR5 joints. "
            f"Resolved: {resolved_joint_names}"
        )

    # 将旧 Teacher 的抬升目标关节角转换为设备和精度匹配的 Torch 张量。
    lift_target = torch.tensor(
        LIFT_TARGET_ARM_JOINTS,
        dtype=robot.data.joint_pos.dtype,
        device=env.device,
    ).unsqueeze(0)

    # 按物体名称累计尝试次数和成功次数；重复环境会共同累计到同一物体项。
    attempts = {name: 0 for name in names}
    successes = {name: 0 for name in names}
    source_successes = {name: 0 for name in names}

    try:
        with torch.inference_mode():
            # 每轮开始时重置所有并行环境及策略循环状态，并记录物体初始高度。
            for round_idx in range(args_cli.rounds):
                if not simulation_app.is_running():
                    raise RuntimeError(
                        "Isaac Sim closed before evaluation finished."
                    )

                obs, _ = env.reset()

                policy_nn.reset(
                    torch.ones(
                        len(names),
                        dtype=torch.bool,
                        device=env.device,
                    ),
                )

                initial_z = object_asset.data.root_pos_w[:, 2].clone()
                if diagnostic is not None:
                    diagnostic.start_round(round_idx)

                # 标记仍参与本轮评估的环境；发生回合终止的环境不再执行动作。
                active = torch.ones(
                    len(names),
                    dtype=torch.bool,
                    device=env.device,
                )

                # 保存每个环境抓取阶段最后使用的动作，抬升阶段继续沿用手部动作。
                last_grasp_actions = torch.zeros(
                    (len(names), env.num_actions),
                    dtype=robot.data.joint_pos.dtype,
                    device=env.device,
                )

                # 策略抓取阶段：执行指定步数，并记录每个环境的最后一组抓取动作。

                for _ in range(grasp_steps):
                    policy_actions = policy(obs)

                    last_grasp_actions[active] = policy_actions[active]

                    actions = policy_actions.clone()
                    actions[~active] = 0.0

                    obs, _, dones, _ = env.step(actions)

                    done_mask = dones.to(dtype=torch.bool)
                    active &= ~done_mask
                    policy_nn.reset(dones)
                    if diagnostic is not None:
                        diagnostic.sample('grasp', actions, active)

                lift_start = (
                    robot.data.joint_pos[:, arm_joint_ids].clone()
                )
                if diagnostic is not None:
                    diagnostic.end_grasp()

                # clipping比较只改变策略动作；固定UR5抬升轨迹继续使用原控制。
                grasp_action_clip = env.clip_actions
                if grasp_action_clip is not None:
                    last_grasp_actions.clamp_(-grasp_action_clip, grasp_action_clip)
                env.clip_actions = None

                # B模式在每个物理子步保持抓取结束的实际限位目标。
                # 保持原arm相对动作与轨迹，A模式仍重复最后的hand delta。
                original_apply = action_term.apply_actions
                if args_cli.lift_hand_mode == 'hold-grasp-posture':
                    held_hand_target = action_term.target[:, 6:].clone()

                    def apply_with_hand_hold():
                        original_apply()
                        action_term.target[active, 6:] = held_hand_target[active]
                        robot.set_joint_position_target(action_term.target, joint_ids=action_term._joint_ids)

                    action_term.apply_actions = apply_with_hand_hold

                # 记录抓取结束时 UR5 关节角，作为后续插值抬升动作的起点。

                # 评估专用抬升阶段：手指沿用抓取动作，UR5 逐步移动到固定抬升姿态。

                for step_idx in range(lift_steps):
                    fraction = min(
                        (step_idx + 1) / float(lift_ramp_steps),
                        1.0,
                    )

                    arm_target = (
                        lift_start
                        + fraction * (lift_target - lift_start)
                    )

                    actions = last_grasp_actions.clone()
                    actions[:, : len(arm_joint_ids)] = (
                        arm_target
                        - robot.data.joint_pos[:, arm_joint_ids]
                    ) / ARM_ACTION_SCALE

                    actions[~active] = 0.0

                    obs, _, dones, _ = env.step(actions)

                    done_mask = dones.to(dtype=torch.bool)
                    active &= ~done_mask
                    policy_nn.reset(dones)
                    if diagnostic is not None:
                        diagnostic.sample('lift', actions, active)

                action_term.apply_actions = original_apply
                env.clip_actions = grasp_action_clip

                # 源指标只看抬升高度；严格指标同时要求环境未提前终止。

                height_gain = (
                    object_asset.data.root_pos_w[:, 2]
                    - initial_z
                )

                source_lifted = height_gain > args_cli.success_height
                lifted = active & source_lifted
                if diagnostic is not None:
                    diagnostic.finish_round(height_gain, source_lifted, lifted, lift_target, arm_joint_ids)

                round_successes = int(lifted.sum().item())

                for object_name, success, source_success in zip(
                    names,
                    lifted.tolist(),
                    source_lifted.tolist(),
                    strict=True,
                ):
                    attempts[object_name] += 1
                    successes[object_name] += int(success)
                    source_successes[object_name] += int(source_success)

                print(
                    f"[INFO] Round {round_idx + 1}/{args_cli.rounds}: "
                    f"Source lift success: {int(source_lifted.sum())} / {len(names)}; "
                    f"Strict lift success: {round_successes} / {len(names)}"
                )

        # 汇总每个物体和全部试验的成功数、失败数及成功率。

        rows: list[dict] = []

        print(
            "\n"
            f"{'Object':<32}"
            f"{'Successes':>11}"
            f"{'Attempts':>10}"
            f"{'Failures':>10}"
            f"{'Success rate':>15}"
        )
        print("-" * 78)

        # 使用去重后的物体名称逐项汇总，避免为每个并行复制打印重复行。
        for name in dict.fromkeys(names):
            object_attempts = attempts[name]
            object_successes = successes[name]
            object_failures = object_attempts - object_successes
            success_rate = (
                object_successes / object_attempts
                if object_attempts > 0
                else 0.0
            )

            print(
                f"{name:<32}"
                f"{object_successes:>11d}"
                f"{object_attempts:>10d}"
                f"{object_failures:>10d}"
                f"{100.0 * success_rate:>14.2f}%"
            )

            rows.append(
                {
                    "object": name,
                    "attempts": object_attempts,
                    "successes": object_successes,
                    "failures": object_failures,
                    "success_rate": success_rate,
                    "source_successes": source_successes[name],
                    "strict_successes": object_successes,
                    "source_success_rate": source_successes[name] / object_attempts,
                }
            )

        # 计算所有物体和并行副本合并后的总体成功率。
        total_successes = sum(successes.values())
        total_source_successes = sum(source_successes.values())
        total_attempts = sum(attempts.values())
        overall_rate = (
            total_successes / total_attempts
            if total_attempts > 0
            else 0.0
        )

        print("-" * 78)
        print(
            f"Source lift success: {total_source_successes} / {total_attempts}\n"
            f"Strict lift success: {total_successes} / {total_attempts}"
        )

        # 优先使用用户指定的结果目录，否则写入检查点运行目录下的 evaluation 子目录。
        output_dir = (
            Path(args_cli.output_dir).expanduser().resolve()
            if args_cli.output_dir is not None
            else Path(log_dir).resolve() / "evaluation"
        )

        # JSON 汇总保存任务、检查点、评估参数和总体/逐物体结果。
        summary = {
            "task": args_cli.task,
            "checkpoint": str(checkpoint),
            "lift_hand_mode": args_cli.lift_hand_mode,
            "diagnostics": args_cli.diagnostics,
            "dataset": EVALUATION_DATASET,
            "rounds": args_cli.rounds,
            "repeat_per_object": args_cli.repeat_per_object,
            "num_envs": len(names),
            "grasp_duration_s": args_cli.grasp_duration_s,
            "lift_duration_s": args_cli.lift_duration_s,
            "lift_ramp_duration_s": args_cli.lift_ramp_duration_s,
            "step_dt": env.unwrapped.step_dt,
            "grasp_steps": grasp_steps,
            "lift_steps": lift_steps,
            "lift_ramp_steps": lift_ramp_steps,
            "success_height_m": args_cli.success_height,
            "total_attempts": total_attempts,
            "total_successes": total_successes,
            "total_failures": total_attempts - total_successes,
            "overall_success_rate": overall_rate,
            "source_lift_success_rate": total_source_successes / total_attempts,
            "strict_lift_success_rate": overall_rate,
            "total_source_successes": total_source_successes,
            "per_object": {
                row["object"]: {
                    key: value
                    for key, value in row.items()
                    if key != "object"
                }
                for row in rows
            },
        }

        _save_results(
            output_dir=output_dir,
            summary=summary,
            rows=rows,
        )
        if diagnostic is not None:
            diagnostic.save(output_dir)

    finally:
        env.close()


if __name__ == "__main__":
    # 无论评估正常结束或抛出异常，都关闭 Isaac Sim 应用。
    try:
        main()
    finally:
        simulation_app.close()

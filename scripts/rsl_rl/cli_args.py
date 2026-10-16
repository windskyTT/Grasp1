"""Teacher训练/播放共享参数和checkpoint infos契约。"""
from pathlib import Path

TASK = "Grasp1-UR5-Allegro-Teacher-v0"


def add_rsl_rl_args(parser):
    parser.add_argument("--resume", action="store_true", help="Resume a compatible Teacher checkpoint.")
    parser.add_argument("--checkpoint", type=str, help="Checkpoint path or filename expression.")
    parser.add_argument("--load_run", type=str, default=".*", help="Run folder expression.")
    parser.add_argument("--run_name", type=str, default="", help="Training run suffix.")
    parser.add_argument("--no_log", action="store_true", help="Run PPO in memory without logs or checkpoints.")


def checkpoint_path(agent_cfg, checkpoint=None, load_run=".*"):
    from isaaclab_tasks.utils import get_checkpoint_path
    if checkpoint and Path(checkpoint).expanduser().is_file():
        return str(Path(checkpoint).expanduser().resolve())
    root = str(Path("logs/rsl_rl", agent_cfg.experiment_name).resolve())
    return get_checkpoint_path(root, load_run, checkpoint or agent_cfg.load_checkpoint)


def _plain(value):
    """将配置中的callable转为名称，使infos只含可直接比较的基本值。"""
    if isinstance(value, dict):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    if callable(value):
        return f"{value.__module__}:{value.__qualname__}"
    return value


def teacher_contract(env_cfg, agent_cfg):
    from Grasp1.data.object_set import training_object_names
    from Grasp1.utils.paths import object_asset_dir
    pool = list(dict.fromkeys(training_object_names()))
    scene = env_cfg.scene.to_dict()
    scene.pop("num_envs")
    scene["object"]["spawn"]["usd_path"] = [str(object_asset_dir("new_training_set", n)/f"{n}.usd") for n in pool]
    events = env_cfg.events.to_dict()
    for name in ("initialize_teacher_data", "reset_teacher"):
        events[name]["params"]["object_names"] = pool
    sim = env_cfg.sim.to_dict()
    for key in ("device", "render_interval", "render", "use_fabric", "enable_scene_query_support"):
        sim.pop(key, None)
    task = dict(task=TASK, semantic_revision=5, dataset="new_training_set", objects=pool,
                observation_dim=153, action_dim=22,
                timing=[env_cfg.sim.dt, env_cfg.decimation, env_cfg.episode_length_s],
                actions=env_cfg.actions.to_dict(), observations=env_cfg.observations.to_dict(),
                rewards=env_cfg.rewards.to_dict(), terminations=env_cfg.terminations.to_dict(),
                events=events, scene=scene, sim=sim,
                safety={key: getattr(env_cfg, key) for key in (
                    "support_height", "success_lift_m", "success_hold_s", "tip_contact_threshold_N",
                    "arm_contact_limit_N", "hand_table_contact_limit_N", "joint_limit_tolerance",
                    "object_speed_limit", "object_angular_speed_limit")},
                obs_groups=agent_cfg.obs_groups, policy=agent_cfg.policy.to_dict(), clip_actions=agent_cfg.clip_actions)
    training = dict(algorithm=agent_cfg.algorithm.to_dict(), num_steps_per_env=agent_cfg.num_steps_per_env)
    return _plain(task), _plain(training)


def validate_teacher_checkpoint(path, task, training=None):
    import torch
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    infos = checkpoint.get("infos")
    if not isinstance(infos, dict) or "task_contract" not in infos:
        raise ValueError("Checkpoint has no Teacher-v0 infos contract; legacy weights cannot be loaded.")
    if infos["task_contract"] != task:
        differing = [key for key in task if infos["task_contract"].get(key) != task[key]]
        raise ValueError(f"Checkpoint Teacher-v0 MDP mismatch: {differing}")
    if training is not None and infos.get("training_contract") != training:
        raise ValueError("Checkpoint PPO training configuration differs; Resume refused.")
    if infos.get("completed_updates") != checkpoint["iter"]+1:
        raise ValueError("Checkpoint PPO update count is inconsistent.")
    return infos

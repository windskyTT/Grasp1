"""Teacher检查点的实际控制语义；训练恢复和推理入口共用，旧权重不被重写。"""
import importlib.metadata
import json
from pathlib import Path
import subprocess


SCHEMA_VERSION = 1
SEMANTIC_FIELDS = (
    'schema_version', 'task_id', 'physics_dt', 'decimation', 'control_dt',
    'action_semantics_version', 'action_scale_arm', 'action_scale_hand',
    'observation_semantics_version', 'reward_semantics_version',
    'physics_parameters', 'asset_versions', 'rsl_rl_version', 'clip_actions',
)
RESUME_FIELDS = SEMANTIC_FIELDS + (
    'episode_length_s', 'gamma', 'lam', 'minimum_action_std', 'ppo_parameters',
    'object_dataset', 'object_names',
)


def build_contract(task_id, env_cfg, agent_cfg):
    """记录训练时长；评估40秒协议扩展不被误当成训练控制律变化。"""
    root = Path(__file__).resolve().parents[2]
    def git(*args):
        return subprocess.check_output(['git', '-C', str(root), *args], text=True).strip()
    robot = env_cfg.scene.robot
    obj = env_cfg.scene.object
    scale = env_cfg.actions.teacher.scale
    arm_names = env_cfg.actions.teacher.joint_names[:6]
    hand_names = env_cfg.actions.teacher.joint_names[6:]
    return dict(
        schema_version=SCHEMA_VERSION, task_id=task_id, physics_dt=env_cfg.sim.dt,
        decimation=env_cfg.decimation, control_dt=env_cfg.sim.dt*env_cfg.decimation,
        episode_length_s=env_cfg.episode_length_s,
        action_semantics_version='current_q_residual_clamped_once_per_control_step_v2',
        action_scale_arm=[scale[n] for n in arm_names], action_scale_hand=[scale[n] for n in hand_names],
        observation_semantics_version='teacher153_final_cached_target_error_v2',
        reward_semantics_version='source_time_scaling_terminal_minus10_contact_local_v2',
        gamma=agent_cfg.algorithm.gamma, lam=agent_cfg.algorithm.lam,
        minimum_action_std=agent_cfg.minimum_action_std, clip_actions=agent_cfg.clip_actions,
        ppo_parameters=dict(rollout=agent_cfg.num_steps_per_env, algorithm=agent_cfg.algorithm.to_dict(),
                            policy=agent_cfg.policy.to_dict()),
        physics_parameters=dict(solver=env_cfg.sim.physx.solver_type,
            robot_articulation=robot.spawn.articulation_props.to_dict(),
            actuators={n:a.to_dict() for n,a in robot.actuators.items()},
            object_articulation=obj.spawn.articulation_props.to_dict(),
            object_rigid=obj.spawn.rigid_props.to_dict(),
            object_joint_drive=obj.spawn.joint_drive_props.to_dict(),
            object_armature=env_cfg.events.configure_object_joint.params['armature']),
        asset_versions=dict(robot=git('log','-1','--format=%H','--','assets/robots/ur5_allegro'),
                            objects=git('log','-1','--format=%H','--','assets/objects'),
                            table=git('log','-1','--format=%H','--','assets/table')),
        object_dataset=env_cfg.object_dataset, object_names=list(env_cfg.object_names),
        training_git_sha=git('rev-parse','HEAD'),
        training_working_tree=git('status','--short'),
        rsl_rl_version=importlib.metadata.version('rsl-rl-lib'),
    )


def metadata_path(checkpoint):
    checkpoint = Path(checkpoint)
    # IsaacLab get_checkpoint_path使用re.match；不能让JSON以model_开头。
    return checkpoint.with_name('env_contract_'+checkpoint.name+'.json')


def compare_contract(saved, current, *, resume):
    fields = RESUME_FIELDS if resume else SEMANTIC_FIELDS
    return {key:dict(checkpoint=saved.get(key), requested=current.get(key))
            for key in fields if key not in saved or saved[key] != current[key]}


def validate_checkpoint(checkpoint, current, *, resume=False, allow_mismatch=False):
    """元数据缺失或语义不同时默认拒绝；诊断覆盖始终返回字段级审计。"""
    path = metadata_path(checkpoint)
    saved = json.loads(path.read_text()) if path.exists() else {}
    # JSON规范化将配置中的tuple转换为list，和磁盘记录一致。
    current = json.loads(json.dumps(current))
    differences = compare_contract(saved, current, resume=resume)
    warning = ('Diagnostic checkpoint override: results are excluded from formal candidate comparisons.'
               if differences else None)
    audit = dict(checkpoint=str(checkpoint), metadata=str(path), metadata_present=path.exists(),
                 differences=differences, compatible=not differences,
                 comparability_warning=warning, training_episode_length_s=saved.get('episode_length_s'))
    if differences and (resume or not allow_mismatch):
        raise ValueError(f'Checkpoint {checkpoint} environment contract rejected before runner.load(): '
                         + json.dumps(differences, sort_keys=True)
                         + '. Start fresh training, or audit legacy source/configuration before writing metadata. '
                         + ('Use --allow_mismatched_checkpoint only for diagnostic inference.' if not resume else 'Training resume cannot override this check.'))
    if differences:
        print('[WARNING] '+warning+' '+json.dumps(differences,sort_keys=True), flush=True)
    return audit, saved


def save_metadata(checkpoint, contract, iteration):
    data = dict(contract, checkpoint_iteration=int(iteration), completed_updates=int(iteration)+1,
                next_update_index=int(iteration)+1)
    metadata_path(checkpoint).write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')

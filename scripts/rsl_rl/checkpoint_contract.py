"""Teacher检查点的实际控制语义；训练恢复和推理入口共用，旧权重不被重写。"""
import importlib.metadata
import hashlib
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
GRASP_TRAIN_FIELDS = ('reward_profile', 'reward_terms', 'termination_terms', 'asset_contents', 'scene_parameters',
                     'normalizer_configuration', 'policy_architecture')
GRASP_TRAIN_RESUME_FIELDS = ('curriculum_stage', 'reset_distribution', 'object_asset_contents')
FULL_GRASP_FIELDS = ('episode_length_s', 'full_grasp_parameters', 'tip_sensor_configuration',
                     'reward_terms', 'termination_terms', 'reset_distribution', 'curriculum_stage',
                     'normalizer_configuration', 'policy_architecture')
MOUNT_GRASP_FIELDS = ('mount_semantics_version', 'tcp_semantics_version', 'mount_tcp_parameters', 'scene_parameters',
                     'asset_contents', 'object_asset_contents', 'action_joint_names', 'observation_dim', 'action_dim')


def asset_contents(env_cfg, *, include_objects=True):
    """新Task契约消费真实USD及引用mesh/材质字节，检出未提交资产变动。"""
    from pxr import UsdUtils
    import omni.client
    import carb.tokens
    roots = [env_cfg.scene.robot.spawn.usd_path, env_cfg.scene.table.spawn.usd_path]
    if include_objects:
        objects = env_cfg.scene.object.spawn.usd_path
        roots.extend(objects if isinstance(objects, list) else [objects])
    paths, unresolved = set(), set()
    for root in sorted(set(roots)):
        layers, assets, missing = UsdUtils.ComputeAllDependencies(root)
        paths.update(str(layer.realPath or layer.identifier) for layer in layers)
        paths.update(str(path) for path in assets)
        unresolved.update(str(path) for path in missing)
    # USD依赖枚举不使用Kit的内置MDL搜索路径；本机实际材质为这个核心模块。
    if 'OmniPBR.mdl' in unresolved:
        builtin = Path(carb.tokens.get_tokens_interface().resolve('${kit}/mdl/core/Base/OmniPBR.mdl'))
        if builtin.is_file():
            paths.add(str(builtin))
            unresolved.remove('OmniPBR.mdl')
    if unresolved:
        raise FileNotFoundError(f'GraspTrain asset dependencies unresolved: {sorted(unresolved)}')
    values = {}
    for path in sorted(paths):
        if '://' in path:
            result, _, content = omni.client.read_file(path)
            if result != omni.client.Result.OK:
                raise OSError(f'Cannot read loaded asset dependency {path}: {result}')
            values[path] = hashlib.sha256(bytes(content)).hexdigest()
        else:
            with Path(path).open('rb') as handle:
                values[path] = hashlib.file_digest(handle, 'sha256').hexdigest()
    return values


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
    contract = dict(
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
    if task_id == 'Grasp1-UR5-Allegro-GraspTrain-v1':
        shared_assets = asset_contents(env_cfg, include_objects=False)
        all_assets = asset_contents(env_cfg)
        contract.update(schema_version=2, reward_profile=env_cfg.reward_profile,
            reward_semantics_version=f'{env_cfg.reward_profile}_latest_physics_frame_v1',
            reward_terms=env_cfg.rewards.to_dict(), termination_terms=env_cfg.terminations.to_dict(),
            asset_contents=shared_assets,
            object_asset_contents={path:sha for path,sha in all_assets.items() if path not in shared_assets},
            curriculum_stage=env_cfg.curriculum_stage,
            normalizer_configuration=dict(actor=agent_cfg.policy.actor_obs_normalization,
                critic=agent_cfg.policy.critic_obs_normalization),
            policy_architecture=dict(actor_hidden_dims=agent_cfg.policy.actor_hidden_dims,
                critic_hidden_dims=agent_cfg.policy.critic_hidden_dims, activation=agent_cfg.policy.activation),
            reset_distribution=env_cfg.events.reset_teacher.to_dict(),
            scene_parameters=dict(robot_initial_state=robot.init_state.to_dict(), table=env_cfg.scene.table.to_dict(),
                physics_material=env_cfg.sim.physics_material.to_dict()))
    if task_id in ('Grasp1-UR5-Allegro-GraspTrain-v2', 'Grasp1-UR5-Allegro-GraspTrain-v3'):
        contract.update(schema_version=3,
            action_semantics_version='previous_target_delta_clamped_once_per_control_step_full_grasp_v1',
            observation_semantics_version='teacher153_plus_full_grasp58_corotating_velocity_v4',
            reward_semantics_version=env_cfg.reward_profile,
            reward_terms=env_cfg.rewards.to_dict(), termination_terms=env_cfg.terminations.to_dict(),
            reset_distribution=env_cfg.events.reset_teacher.to_dict(), curriculum_stage=env_cfg.curriculum_stage,
            full_grasp_parameters={name:getattr(env_cfg, name) for name in (
                'legacy_joint_friction', 'hand_static_friction', 'reset_xy_center', 'reset_xy_range', 'reset_yaw_range',
                'reset_clearance', 'pregrasp_distance', 'reset_max_attempts', 'lift_height',
                'hold_duration_s', 'tip_contact_threshold_N', 'arm_action_speed', 'hand_action_speed')},
            tip_sensor_configuration={name:value.to_dict() for name,value in vars(env_cfg.scene).items()
                if name.startswith('full_tip_') or name.startswith('full_support_')},
            normalizer_configuration=dict(actor=agent_cfg.policy.actor_obs_normalization,
                critic=agent_cfg.policy.critic_obs_normalization),
            policy_architecture=dict(actor_hidden_dims=agent_cfg.policy.actor_hidden_dims,
                critic_hidden_dims=agent_cfg.policy.critic_hidden_dims, activation=agent_cfg.policy.activation))
    if task_id == 'Grasp1-UR5-Allegro-GraspTrain-v3':
        shared_assets = asset_contents(env_cfg, include_objects=False)
        all_assets = asset_contents(env_cfg)
        contract.update(schema_version=4,
            asset_contents=shared_assets,
            object_asset_contents={path:sha for path,sha in all_assets.items() if path not in shared_assets},
            action_joint_names=list(env_cfg.actions.teacher.joint_names), observation_dim=211, action_dim=22,
            action_semantics_version='previous_target_delta_COM_vertex_increment_projection_preserve_preload_v3',
            observation_semantics_version='teacher153_geometric_TCP93_96_plus_full_grasp58_corotating_velocity_v5',
            mount_semantics_version=env_cfg.mount_semantics_version, tcp_semantics_version=env_cfg.tcp_semantics_version,
            mount_tcp_parameters=dict(robot_usd_path=robot.spawn.usd_path,
                flange_to_hand_xyz=(0., 0., 0.), flange_to_hand_quaternion_wxyz=(1., 0., 0., 0.),
                palm_quaternion_w=env_cfg.palm_quaternion_w, tcp_offset_h=env_cfg.tcp_offset_h,
                tcp_initial_height=env_cfg.tcp_initial_height, action_table_clearance_m=env_cfg.action_table_clearance_m),
            scene_parameters=dict(robot_initial_state=robot.init_state.to_dict(), table=env_cfg.scene.table.to_dict()))
    # InteractiveScene实例化会原地展开配置路径；构建前后应记录同一真实prim/filter。
    # 避免同一MDP的训练后保存与创建前Resume/play校验出现占位符误差。
    def resolved_paths(value):
        if isinstance(value, str):
            return value.replace('{ENV_REGEX_NS}', '/World/envs/env_.*')
        if isinstance(value, dict):
            return {key:resolved_paths(item) for key,item in value.items()}
        if isinstance(value, (tuple,list)):
            return [resolved_paths(item) for item in value]
        return value
    return resolved_paths(contract)


def metadata_path(checkpoint):
    checkpoint = Path(checkpoint)
    # IsaacLab get_checkpoint_path使用re.match；不能让JSON以model_开头。
    return checkpoint.with_name('env_contract_'+checkpoint.name+'.json')


def compare_contract(saved, current, *, resume):
    if not resume and current.get('schema_version') == 4 and saved.get('schema_version') == 4:
        # 推理的20/1环境与训练128环境使用同一木块Reset分布；重复分配数量不改变语义。
        saved, current = dict(saved), dict(current)
        for contract in (saved,current):
            reset = json.loads(json.dumps(contract['reset_distribution']))
            reset['params']['object_names'] = sorted(set(reset['params']['object_names']))
            contract['reset_distribution'] = reset
    fields = RESUME_FIELDS if resume else SEMANTIC_FIELDS
    if saved.get('schema_version') == 2 or current.get('schema_version') == 2:
        fields += GRASP_TRAIN_FIELDS
        if resume:
            fields += GRASP_TRAIN_RESUME_FIELDS
    if saved.get('schema_version') in (3, 4) or current.get('schema_version') in (3, 4):
        fields += FULL_GRASP_FIELDS
    if saved.get('schema_version') == 4 or current.get('schema_version') == 4:
        fields += MOUNT_GRASP_FIELDS
    differences = {key:dict(checkpoint=saved.get(key), requested=current.get(key))
                   for key in fields if key not in saved or saved[key] != current[key]}
    if not resume and current.get('schema_version') in (3, 4):
        if not set(current['object_names']).issubset(saved.get('object_names', [])):
            differences['object_names'] = dict(checkpoint=saved.get('object_names'), requested=current['object_names'])
    if not resume and current.get('schema_version') == 2:
        # 允许heldout新对象；同一对象路径的字节改变仍是动力学不兼容。
        saved_objects = saved.get('object_asset_contents', {})
        current_objects = current.get('object_asset_contents', {})
        changed = {path:dict(checkpoint=saved_objects[path], requested=current_objects[path])
                   for path in saved_objects.keys() & current_objects.keys()
                   if saved_objects[path] != current_objects[path]}
        if changed:
            differences['object_asset_contents'] = changed
    return differences


def validate_checkpoint(checkpoint, current, *, resume=False, allow_mismatch=False):
    """元数据缺失或语义不同时默认拒绝；诊断覆盖始终返回字段级审计。"""
    path = metadata_path(checkpoint)
    saved = json.loads(path.read_text()) if path.exists() else {}
    # JSON规范化将配置中的tuple转换为list，和磁盘记录一致。
    current = json.loads(json.dumps(current))
    differences = compare_contract(saved, current, resume=resume)
    if saved.get('schema_version') == 2:
        with Path(checkpoint).open('rb') as handle:
            actual_hash = hashlib.file_digest(handle, 'sha256').hexdigest()
        if saved.get('checkpoint_sha256') != actual_hash:
            differences['checkpoint_sha256'] = dict(checkpoint=saved.get('checkpoint_sha256'), requested=actual_hash)
    warning = ('Diagnostic checkpoint override: results are excluded from formal candidate comparisons.'
               if differences else None)
    audit = dict(checkpoint=str(checkpoint), metadata=str(path), metadata_present=path.exists(),
                 differences=differences, compatible=not differences,
                 comparability_warning=warning, training_episode_length_s=saved.get('episode_length_s'))
    if differences and (resume or not allow_mismatch or current.get('schema_version') == 4):
        raise ValueError(f'Checkpoint {checkpoint} environment contract rejected before runner.load(): '
                         + json.dumps(differences, sort_keys=True)
                         + '. Start fresh training, or audit legacy source/configuration before writing metadata. '
                         + ('v3 requires matching mount, TCP and curriculum; overrides are disabled.' if current.get('schema_version') == 4
                            else 'Use --allow_mismatched_checkpoint only for diagnostic inference.' if not resume
                            else 'Training resume cannot override this check.'))
    if differences:
        print('[WARNING] '+warning+' '+json.dumps(differences,sort_keys=True), flush=True)
    return audit, saved


def validate_warm_start(checkpoint, current):
    """v3仅放行课程Reset/TCP起点变化；网络、归一化、物理及动作含义仍严格一致。"""
    saved = json.loads(metadata_path(checkpoint).read_text())
    current = json.loads(json.dumps(current))
    if saved.get('schema_version') != 4 or current.get('schema_version') != 4:
        raise ValueError('Warm Start only supports audited v3/schema4 policies.')
    comparable = dict(current)
    allowed = ('curriculum_stage', 'reset_distribution')
    comparable['curriculum_stage'] = saved['curriculum_stage']
    comparable['reset_distribution'] = json.loads(json.dumps(current['reset_distribution']))
    comparable['reset_distribution']['params']['tcp_height'] = saved['reset_distribution']['params']['tcp_height']
    comparable['mount_tcp_parameters'] = dict(current['mount_tcp_parameters'])
    comparable['mount_tcp_parameters']['tcp_initial_height'] = saved['mount_tcp_parameters']['tcp_initial_height']
    comparable['full_grasp_parameters'] = dict(current['full_grasp_parameters'])
    comparable['full_grasp_parameters']['reset_xy_center'] = saved['full_grasp_parameters']['reset_xy_center']
    differences = compare_contract(saved, comparable, resume=True)
    if differences:
        raise ValueError('Warm Start semantic mismatch: '+json.dumps(differences, sort_keys=True))
    return dict(source_checkpoint=str(Path(checkpoint).resolve()), source_curriculum=saved['curriculum_stage'],
        target_curriculum=current['curriculum_stage'], source_completed_updates=saved['completed_updates'],
        source_cumulative_updates=saved.get('cumulative_lineage_updates', saved['completed_updates']),
        target_completed_updates=0, optimizer='fresh', normalizers='included_in_model_state_dict',
        curriculum_changes={key:dict(source=saved[key], target=current[key]) for key in
            (*allowed, 'mount_tcp_parameters', 'full_grasp_parameters') if saved[key] != current[key]})


def save_metadata(checkpoint, contract, iteration):
    data = dict(contract, checkpoint_iteration=int(iteration), completed_updates=int(iteration)+1,
                next_update_index=int(iteration)+1)
    if contract['schema_version'] == 2:
        with Path(checkpoint).open('rb') as handle:
            data['checkpoint_sha256'] = hashlib.file_digest(handle, 'sha256').hexdigest()
    metadata_path(checkpoint).write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')

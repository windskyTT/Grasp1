"""UR5 + Allegro Teacher 的 Manager-Based 任务配置。

本文件只负责组合 Teacher 的 MDP terms 与旧 RobustDexGrasp 配置参数。
复杂计算分别放在 ``mdp/actions.py``、``mdp/observations.py``、
``mdp/rewards.py``、``mdp/events.py``、``mdp/terminations.py`` 中。
"""

from __future__ import annotations

from typing import Final

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import mdp as isaaclab_mdp
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass

from Grasp1.data.object_set import (
    TRAINING_DATASET,
    object_names as dataset_object_names,
    training_object_names,
)
from Grasp1.robots.robot_profile import (
    ARM_CONTACT_LINK_NAMES,
    ALLEGRO_JOINT_NAMES,
    HAND_CONTACT_LINK_NAMES,
    UR5_JOINT_NAMES,
)
from Grasp1.utils.paths import object_asset_dir

from ... import mdp
from ...grasp1_env_cfg import CONTROL_DT, TABLE_CENTER_XY, TABLE_HEIGHT, Grasp1EnvCfg


# -----------------------------------------------------------------------------
# Teacher 来源参数及任务常量。
# -----------------------------------------------------------------------------

# MDP 函数的模块路径前缀；下方 term 配置通过该路径引用实现。
_MDP: Final[str] = "Grasp1.tasks.manager_based.grasp1.mdp"
# 原始 Teacher 观测和动作向量的维度。
TEACHER_OBSERVATION_DIM: Final[int] = 153
TEACHER_ACTION_DIM: Final[int] = 22
# 将 Allegro 关节名转换为 USD 导入器使用的运行时名称。
_HAND_USD_JOINT_NAMES: Final[tuple[str, ...]] = tuple(name.replace(".", "_") for name in ALLEGRO_JOINT_NAMES)

# cfg_reg.yaml 中 UR5 和 Allegro 相对关节位置动作使用的缩放系数。
ARM_ACTION_SCALE: Final[float] = 0.005
HAND_ACTION_SCALE: Final[float] = 0.015
ACTION_DELAY_PROBABILITY: Final[float] = 0.5
ACTION_DELAY_SIM_STEPS: Final[int] = 1

# ContactSensor 每个 source body 只过滤 object top。较高的 contact-data
# 上限避免复杂 mesh 接触时丢失摩擦接触点。
CONTACT_SENSOR_MAX_CONTACTS_PER_PRIM: Final[int] = 16

# 抓取采样选项：是否偏置点云、是否限制为俯视抓取，以及是否非均匀采样。
BIASED_POINT_CLOUD: Final[bool] = False
TOP_DOWN_GRASP: Final[bool] = False
NON_UNIFORM_SAMPLING: Final[bool] = True
OBJECT_BIAS_DISTANCE_THRESHOLD: Final[float] = 0.07

# 预抓取候选生成使用的相机位置和候选数量。
CAMERA_POSITION: Final[tuple[float, float, float]] = (
    0.035,
    -0.58,
    1.531,
)

PREGRASP_SAMPLE_NUM: Final[int] = 10
# 预抓取候选评分中，长度差异和角度差异对应的权重系数。
PREGRASP_LENGTH_SCORE_COEFF: Final[float] = 5.0
PREGRASP_ANGLE_SCORE_COEFF: Final[float] = 1.0


def _isaaclab_reward_weight(source_coeff: float) -> float:
    """将旧 Teacher 每控制步的奖励系数换算为 Isaac Lab RewardManager 权重。

    RewardManager 会按 ``term × weight × env.step_dt`` 累加奖励；旧实现每个控制
    步直接乘 YAML 系数。除以控制周期后，可抵消 manager 再乘的 ``step_dt``，保持
    原来的每控制步奖励尺度。

    参数 ``source_coeff`` 是旧配置中的奖励系数；返回值用于 ``RewTerm.weight``。
    """
    return source_coeff / CONTROL_DT


def _build_finger_reward_weights() -> tuple[float, ...]:
    """按旧 Teacher 的 17 个手部关键点顺序构造奖励权重。"""
    weights = [1.0] * 17

    # 四个指尖关键点权重乘以 4。
    for index in (4, 8, 12, 16):
        weights[index] *= 4.0

    # 拇指指尖（索引 16）在此基础上再乘以 2。
    weights[16] *= 2.0

    # 先按全部权重总和归一化，与原实现顺序一致。
    normalizer = sum(weights)
    weights = [weight / normalizer for weight in weights]

    # 手掌/手腕基座关键点不参与该项奖励。
    weights[0] = 0.0

    # 最后将各关键点权重整体乘以 16。
    weights = [weight * 16.0 for weight in weights]

    return tuple(weights)


FINGER_REWARD_WEIGHTS: Final[tuple[float, ...]] = _build_finger_reward_weights()


def _build_contact_reward_weights() -> tuple[float, ...]:
    """复现 Environment.hpp 中 13 个接触刚体的 fingertip/thumb 权重。"""
    weights = [1.0] * 13
    for index in (3, 6, 9, 12):
        weights[index] *= 3.0
    weights[0] = 0.0
    for index in (10, 11, 12):
        weights[index] *= 2.0
    weights[12] *= 2.0
    normalizer = sum(weights)
    return tuple(weight * 13.0 / normalizer for weight in weights)


CONTACT_REWARD_WEIGHTS: Final[tuple[float, ...]] = _build_contact_reward_weights()

# 原 train.py 调用了 reward_r.clip(min=-2.0)，但没有接收返回值，也没有传 out=；
# NumPy 因而不会原地修改 reward_r。该值只记录源码参数，不应在 IsaacLab 中额外
# 实施总奖励截断，否则反而会改变原 Teacher 的实际训练语义。
TEACHER_REWARD_FLOOR: Final[float] = -2.0


# -----------------------------------------------------------------------------
# Teacher 物体池。
# -----------------------------------------------------------------------------


def _build_teacher_object_cfg(
    object_names: tuple[str, ...],
    *,
    dataset_name: str,
) -> ArticulationCfg:
    """按给定顺序构造 Teacher 的多 USD 物体 articulation 配置。"""
    return ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Object",
        spawn=sim_utils.MultiUsdFileCfg(
            # 按物体列表提供 USD 资产；random_choice=False 保留列表顺序与环境编号的对应关系。
            usd_path=[str(object_asset_dir(dataset_name, name) / f"{name}.usd") for name in object_names],
            random_choice=False,
            # 启用接触传感器，以供接触类奖励读取物体接触信息。
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                # 物体受重力影响，并限制碰撞分离时的最大速度。
                disable_gravity=False,
                max_depenetration_velocity=5.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                # 物体根部可运动，且关闭自身碰撞。
                fix_root_link=False,
                enabled_self_collisions=False,
            ),
        ),
        # 安全的暂存初始状态；reset_teacher() 随后根据采样的 XY/朝向和
        # lowest_point_new.txt 中的最低点设置物体根位姿。
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(
                TABLE_CENTER_XY[0],
                TABLE_CENTER_XY[1],
                TABLE_HEIGHT + 0.15,
            ),
            rot=(1.0, 0.0, 0.0, 0.0),
            joint_pos={".*": 0.0},
            joint_vel={".*": 0.0},
        ),
        # 原始配置不为物体设置执行器，物体由物理仿真自由运动。
        actuators={},
    )


# -----------------------------------------------------------------------------
# Actions
# -----------------------------------------------------------------------------


@configclass
class TeacherActionsCfg:
    """按 6 个 UR5、16 个 Allegro 关节的顺序输出原 Teacher 22 维动作。"""

    # 一个动作项共用原实现的延迟抽样：每个环境有 50% 概率延迟一个物理子步。
    teacher = mdp.TeacherJointPositionActionCfg(
        asset_name="robot",
        joint_names=list(UR5_JOINT_NAMES + _HAND_USD_JOINT_NAMES),
        scale={
            **{name: ARM_ACTION_SCALE for name in UR5_JOINT_NAMES},
            **{name: HAND_ACTION_SCALE for name in _HAND_USD_JOINT_NAMES},
        },
        preserve_order=True,
        delay_probability=ACTION_DELAY_PROBABILITY,
        delay_sim_steps=ACTION_DELAY_SIM_STEPS,
    )


# -----------------------------------------------------------------------------
# Observations
# -----------------------------------------------------------------------------


@configclass
class TeacherObservationsCfg:
    """定义 Teacher 的策略观测组：102 维基础观测加 51 维 affordance 信息，共 153 维。"""

    @configclass
    class PolicyCfg(ObsGroup):
        # 由 teacher_observation 统一构造策略所需的拼接观测向量。
        teacher = ObsTerm(
            func=f"{_MDP}.observations:teacher_observation",
            params={"table_height": TABLE_HEIGHT},
        )
        def __post_init__(self) -> None:
            """关闭观测噪声，并将该组各观测项拼接成单个向量。"""
            self.enable_corruption = False
            self.concatenate_terms = True

    # Actor 和 critic 均使用同一个 policy 观测组；对应 RSL-RL runner 的 obs_groups 设置。
    policy: PolicyCfg = PolicyCfg()


# -----------------------------------------------------------------------------
# Events
# -----------------------------------------------------------------------------


@configclass
class TeacherEventsCfg:
    """配置数据准备、预抓取重置与物理步后的碰撞/位置事件。"""

    # 启动时为物体池准备网格、最低点和元数据，避免每次 reset 重复读取 CPU 几何数据。
    initialize_teacher_data = EventTerm(
        func=f"{_MDP}.events:initialize_teacher_data",
        mode="startup",
        params={
            "dataset_name": TRAINING_DATASET,
            # 初始化完整环境配置时由物体池名称替换此占位空元组。
            "object_names": (),
        },
    )

    reset_teacher = EventTerm(
        # 每回合重置机器人和物体，并按给定抓取参数生成预抓取姿态。
        func=f"{_MDP}.events:reset_teacher",
        mode="reset",
        params={
            "dataset_name": TRAINING_DATASET,
            # 初始化完整环境配置时由物体池名称替换此占位空元组。
            "object_names": (),
            "biased": BIASED_POINT_CLOUD,
            "top": TOP_DOWN_GRASP,
            "non_uniform_sampling": NON_UNIFORM_SAMPLING,
            "camera_position": CAMERA_POSITION,
            "sample_num": PREGRASP_SAMPLE_NUM,
            "length_score_coeff": PREGRASP_LENGTH_SCORE_COEFF,
            "angle_score_coeff": PREGRASP_ANGLE_SCORE_COEFF,
        },
    )

    # sim.forward() 不计算新位姿的接触力；第一次物理步后再检查初始碰撞。
    check_initial_collision = EventTerm(
        func=f"{_MDP}.events:check_initial_collision",
        mode="interval",
        interval_range_s=(CONTROL_DT, CONTROL_DT),
        is_global_time=True,
    )

    # 原 train.py 的 biased curriculum 在 rollout 中检查最近 hand-affordance
    # 距离；首次小于 0.07 m 时只移动一次 object X/Y。
    apply_object_position_bias = EventTerm(
        func=f"{_MDP}.events:apply_object_position_bias",
        mode="interval",
        interval_range_s=(CONTROL_DT, CONTROL_DT),
        is_global_time=True,
        params={
            "distance_threshold": OBJECT_BIAS_DISTANCE_THRESHOLD,
        },
    )


# -----------------------------------------------------------------------------
# Rewards
# -----------------------------------------------------------------------------


@configclass
class TeacherRewardsCfg:
    """连接旧 cfg_reg.yaml 中的 Teacher 奖励项及其权重和参数。

    IsaacLab RewardManager 会再乘 ``env.step_dt``，因此这里使用
    ``source_coeff / CONTROL_DT`` 作为 manager weight，保证最终每个 control
    step 的系数与 RobustDexGrasp 相同。
    """

    # 手指关键点与物体 affordance 区域的接近/抓取奖励。
    affordance_reward = RewTerm(
        func=f"{_MDP}.rewards:affordance_reward",
        weight=_isaaclab_reward_weight(0.5),
        params={"finger_weights": FINGER_REWARD_WEIGHTS},
    )
    # 与 affordance 区域建立接触的奖励。
    affordance_contact_reward = RewTerm(
        func=f"{_MDP}.rewards:affordance_contact_reward",
        weight=_isaaclab_reward_weight(1.5),
        params={"contact_weights": CONTACT_REWARD_WEIGHTS},
    )
    # 根据 affordance 区域接触冲量计算的奖励。
    affordance_impulse_reward = RewTerm(
        func=f"{_MDP}.rewards:affordance_impulse_reward",
        weight=_isaaclab_reward_weight(1.0),
        params={"contact_weights": CONTACT_REWARD_WEIGHTS},
    )

    # 惩罚手指关键点进入桌面区域，按关键点权重计算。
    table_reward = RewTerm(
        func=f"{_MDP}.rewards:table_reward",
        weight=_isaaclab_reward_weight(-0.03),
        params={
            "finger_weights": FINGER_REWARD_WEIGHTS,
            "table_height": TABLE_HEIGHT,
        },
    )
    # 惩罚手部与桌面发生接触。
    table_contact_reward = RewTerm(
        func=f"{_MDP}.rewards:table_contact_reward",
        weight=_isaaclab_reward_weight(-1.0),
        params={"contact_weights": CONTACT_REWARD_WEIGHTS},
    )
    # 根据手部与桌面的接触冲量施加惩罚。
    table_impulse_reward = RewTerm(
        func=f"{_MDP}.rewards:table_impulse_reward",
        weight=_isaaclab_reward_weight(-0.5),
        params={"contact_weights": CONTACT_REWARD_WEIGHTS},
    )

    # 惩罚手臂关键点低于桌面高度。
    arm_height_reward = RewTerm(
        func=f"{_MDP}.rewards:arm_height_reward",
        weight=_isaaclab_reward_weight(-0.05),
        params={"table_height": TABLE_HEIGHT},
    )
    # 惩罚机械臂与场景物体发生接触。
    arm_contact_reward = RewTerm(
        func=f"{_MDP}.rewards:arm_contact_reward",
        weight=_isaaclab_reward_weight(-0.1),
    )
    # 根据机械臂与场景的接触冲量施加惩罚。
    arm_impulse_reward = RewTerm(
        func=f"{_MDP}.rewards:arm_impulse_reward",
        weight=_isaaclab_reward_weight(-0.1),
    )
    # 对机械臂碰撞施加惩罚。
    arm_collision_reward = RewTerm(
        func=f"{_MDP}.rewards:arm_collision_reward",
        weight=_isaaclab_reward_weight(-1.0),
    )

    # 原始奖励系数为 -0.0，因此保留该项但将权重设为 0。
    push_reward = RewTerm(
        func=f"{_MDP}.rewards:push_reward",
        weight=0.0,
    )

    # 惩罚手腕线速度。
    wrist_vel_reward_ = RewTerm(
        func=f"{_MDP}.rewards:wrist_vel_reward_",
        weight=_isaaclab_reward_weight(-1.0),
    )
    # 惩罚手腕角速度。
    wrist_qvel_reward_ = RewTerm(
        func=f"{_MDP}.rewards:wrist_qvel_reward_",
        weight=_isaaclab_reward_weight(-0.1),
    )
    # 惩罚物体线速度。
    obj_vel_reward_ = RewTerm(
        func=f"{_MDP}.rewards:obj_vel_reward_",
        weight=_isaaclab_reward_weight(-15.0),
    )
    # 惩罚物体角速度。
    obj_qvel_reward_ = RewTerm(
        func=f"{_MDP}.rewards:obj_qvel_reward_",
        weight=_isaaclab_reward_weight(-0.2),
    )
    # 惩罚物体相对初始状态的位移。
    obj_displacement_reward = RewTerm(
        func=f"{_MDP}.rewards:obj_displacement_reward",
        weight=_isaaclab_reward_weight(-5.0),
    )
    # 惩罚 UR5 关节速度。
    arm_joint_vel_reward_ = RewTerm(
        func=f"{_MDP}.rewards:arm_joint_vel_reward_",
        weight=_isaaclab_reward_weight(-1.0),
    )


# -----------------------------------------------------------------------------
# Terminations
# -----------------------------------------------------------------------------


@configclass
class TeacherTerminationsCfg:
    """定义 Teacher 的超时终止和非法观测/机器人状态终止项。"""

    # 达到 episode 时长时结束当前回合，并标记为 time-out。
    time_out = DoneTerm(
        func=isaaclab_mdp.time_out,
        time_out=True,
    )
    # 原 Teacher：17 个手部关键点中任一个低于桌面即终止。
    invalid_hand_height = DoneTerm(
        func=f"{_MDP}.terminations:invalid_hand_height",
        params={
            "hand_keypoints_w_fn": mdp.hand_keypoints_w,
            "table_height": TABLE_HEIGHT,
        },
    )

    # policy 观测中出现 NaN 或无穷值时结束对应环境的回合。
    invalid_observation = DoneTerm(
        func=f"{_MDP}.terminations:invalid_observation",
        params={
            "group_names": ("policy",),
        },
    )

    # mdp.invalid_asset_state 可作为额外调试/安全项按需启用；
    # 原 Teacher 没有独立的 asset-state termination，因此默认不加入。


# -----------------------------------------------------------------------------
# Complete Teacher environment
# -----------------------------------------------------------------------------


@configclass
class UR5AllegroTeacherEnvCfg(Grasp1EnvCfg):
    """组合 UR5 + Allegro Teacher 的场景、动作、观测、事件、奖励和终止配置。"""

    # 将 ManagerBasedRLEnvCfg 中的默认 MDP 配置替换为 Teacher 专用配置。
    actions: TeacherActionsCfg = TeacherActionsCfg()
    observations: TeacherObservationsCfg = TeacherObservationsCfg()
    events: TeacherEventsCfg = TeacherEventsCfg()
    rewards: TeacherRewardsCfg = TeacherRewardsCfg()
    terminations: TeacherTerminationsCfg = TeacherTerminationsCfg()

    # 当前环境的数据集与物体池；play/evaluate 可在创建环境前覆盖。
    object_dataset: str = TRAINING_DATASET
    object_names: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """构建训练物体池、绑定物体和事件参数，并设置原 Teacher 回合长度。"""
        super().__post_init__()

        # 训练集保留 RobustDexGrasp 的困难物体重复权重；play/evaluate
        # 切换到其他数据集时按数据集目录顺序读取 object，不额外施加训练权重。
        object_names = tuple(
            training_object_names()
            if self.object_dataset == TRAINING_DATASET
            else dataset_object_names(self.object_dataset)
        )
        if not object_names:
            raise RuntimeError(
                f"No Teacher objects found in dataset {self.object_dataset!r}."
            )
        self.object_names = object_names

        # 为每个物体配置一个并行环境；关闭物理复制，并按列表顺序分配 USD。
        self.scene.num_envs = len(object_names)
        self.scene.replicate_physics = False
        self.scene.object = _build_teacher_object_cfg(
            object_names,
            dataset_name=self.object_dataset,
        )

        # 原 Teacher 的 13 个 contacts_r_af / impulses_r_af 槽按
        # AllegroSim.cpp::contact_bodies_ 的顺序构造。IsaacLab 2.3.2 的
        # filtered ContactSensor 要求 sensor prim 在每个 env 中只匹配一个刚体，
        # 因此这里为每个 source body 建一个 sensor，并只过滤 object ``top``。
        self.scene.robot.spawn.activate_contact_sensors = True
        for index, source_body_name in enumerate(HAND_CONTACT_LINK_NAMES):
            runtime_body_name = source_body_name.replace(".", "_")
            setattr(
                self.scene,
                f"teacher_af_contact_{index}",
                ContactSensorCfg(
                    prim_path=(
                        f"{{ENV_REGEX_NS}}/Robot/.*{runtime_body_name}"
                    ),
                    update_period=0.0,
                    filter_prim_paths_expr=[
                        "{ENV_REGEX_NS}/Object/.*top"
                    ],
                    track_friction_forces=True,
                    max_contact_data_count_per_prim=(
                        CONTACT_SENSOR_MAX_CONTACTS_PER_PRIM
                    ),
                ),
            )

            # 原 C++ 对同一 13 个手部接触刚体分别统计桌面接触与冲量。
            setattr(
                self.scene,
                f"teacher_table_contact_{index}",
                ContactSensorCfg(
                    prim_path=f"{{ENV_REGEX_NS}}/Robot/.*{runtime_body_name}",
                    update_period=0.0,
                    filter_prim_paths_expr=[
                        "{ENV_REGEX_NS}/Table",
                    ],
                    track_friction_forces=True,
                    max_contact_data_count_per_prim=CONTACT_SENSOR_MAX_CONTACTS_PER_PRIM,
                ),
            )

        # 六个 UR5 刚体的物体/桌面接触与未过滤的全部接触分别供原奖励项使用。
        for index, body_name in enumerate(ARM_CONTACT_LINK_NAMES):
            setattr(
                self.scene,
                f"teacher_arm_contact_{index}",
                ContactSensorCfg(
                    prim_path=f"{{ENV_REGEX_NS}}/Robot/.*{body_name}",
                    update_period=0.0,
                    filter_prim_paths_expr=[
                        "{ENV_REGEX_NS}/Object/top",
                        "{ENV_REGEX_NS}/Object/bottom",
                        "{ENV_REGEX_NS}/Table",
                    ],
                    track_friction_forces=True,
                    track_air_time=True,
                    force_threshold=1.0e-6,
                    max_contact_data_count_per_prim=CONTACT_SENSOR_MAX_CONTACTS_PER_PRIM,
                ),
            )

        # 将同一数据集和物体池传给启动数据准备事件及每回合 reset 事件。
        self.events.initialize_teacher_data.params["dataset_name"] = self.object_dataset
        self.events.initialize_teacher_data.params["object_names"] = object_names

        self.events.reset_teacher.params["dataset_name"] = self.object_dataset
        self.events.reset_teacher.params["object_names"] = object_names

        # 原 train.py 在每个 PPO update 前显式 reset_state()，随后固定收集
        # grasp_steps=70 个 control steps。旧 VectorizedEnvironment 的 step/reset
        # 路径没有读取 cfg_reg.yaml 的 max_time=4.0 做时间终止，因此这里保留：
        #
        #   70 policy steps * 0.2 s = 14.0 s
        #
        # 作为标准 IsaacLab 的 rollout/reset 边界，并与 RSL-RL
        # num_steps_per_env=70 对齐。
        self.episode_length_s = 70 * CONTROL_DT

    def set_num_envs(self, num_envs: int) -> None:
        """按原物体池顺序循环分配环境，并同步 USD、观测及 reset 元数据。"""
        names = tuple(self.object_names[index % len(self.object_names)] for index in range(num_envs))
        self.scene.num_envs = num_envs
        self.object_names = names
        self.scene.object.spawn.usd_path = [
            str(object_asset_dir(self.object_dataset, name) / f"{name}.usd") for name in names
        ]
        self.events.initialize_teacher_data.params["object_names"] = names
        self.events.reset_teacher.params["object_names"] = names


# 对外公开的 Teacher 任务参数、MDP 配置类和完整环境配置类。
__all__ = [
    "TEACHER_OBSERVATION_DIM",
    "TEACHER_ACTION_DIM",
    "ARM_ACTION_SCALE",
    "HAND_ACTION_SCALE",
    "ACTION_DELAY_PROBABILITY",
    "ACTION_DELAY_SIM_STEPS",
    "CONTACT_SENSOR_MAX_CONTACTS_PER_PRIM",
    "TEACHER_REWARD_FLOOR",
    "BIASED_POINT_CLOUD",
    "TOP_DOWN_GRASP",
    "NON_UNIFORM_SAMPLING",
    "OBJECT_BIAS_DISTANCE_THRESHOLD",
    "CAMERA_POSITION",
    "PREGRASP_SAMPLE_NUM",
    "PREGRASP_LENGTH_SCORE_COEFF",
    "PREGRASP_ANGLE_SCORE_COEFF",
    "FINGER_REWARD_WEIGHTS",
    "CONTACT_REWARD_WEIGHTS",
    "TeacherActionsCfg",
    "TeacherObservationsCfg",
    "TeacherEventsCfg",
    "TeacherRewardsCfg",
    "TeacherTerminationsCfg",
    "UR5AllegroTeacherEnvCfg",
]

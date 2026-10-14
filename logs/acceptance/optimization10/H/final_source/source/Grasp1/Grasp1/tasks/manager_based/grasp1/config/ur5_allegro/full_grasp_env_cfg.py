"""独立完整抓取v2：22D策略全控制，211D状态，真实碰撞和10秒Lift/Hold。"""
from isaaclab.utils import configclass
from isaaclab.managers import EventTermCfg, ObservationGroupCfg, ObservationTermCfg, RewardTermCfg, TerminationTermCfg
from isaaclab.envs import mdp as lab_mdp
from isaaclab.sensors import ContactSensorCfg

from .teacher_env_cfg import UR5AllegroTeacherEnvCfg, TeacherEventsCfg
from ...grasp1_env_cfg import SUPPORT_HEIGHT
from ...mdp import full_grasp_events as events, full_grasp_rewards as rewards, full_grasp_terminations as terms
from ...mdp.full_grasp_observations import FullGraspObservation
from ...mdp.full_grasp_metrics import TIP_NAMES
from ...mdp.actions import FullGraspJointDeltaAction
from Grasp1.utils.paths import object_asset_dir


@configclass
class FullGraspEventsCfg(TeacherEventsCfg):
    configure_hand_friction = EventTermCfg(func=events.configure_hand_friction, mode='prestartup')
    configure_hand_model = EventTermCfg(func=events.configure_hand_model, mode='startup')
    reset_teacher = EventTermCfg(func=events.reset_full_grasp, mode='reset',
        params={'dataset_name': 'new_training_set', 'object_names': ()})
    check_initial_collision = None
    apply_object_position_bias = None


@configclass
class FullGraspObservationsCfg:
    @configclass
    class PolicyCfg(ObservationGroupCfg):
        full_grasp = ObservationTermCfg(func=FullGraspObservation, params={'support_height': SUPPORT_HEIGHT})
        enable_corruption = False
        concatenate_terms = True
    policy = PolicyCfg()


@configclass
class FullGraspRewardsCfg:
    approach = RewardTermCfg(func=rewards.approach, weight=2.5)
    alignment = RewardTermCfg(func=rewards.alignment, weight=0.5)
    grasp = RewardTermCfg(func=rewards.stable_grasp, weight=3.0)
    lift = RewardTermCfg(func=rewards.lift, weight=8.0)
    hold = RewardTermCfg(func=rewards.hold, weight=10.0)
    success = RewardTermCfg(func=rewards.success_event, weight=20.0)
    failure = RewardTermCfg(func=rewards.safety_event, weight=-10.0)
    table_contact = RewardTermCfg(func=rewards.table_contact, weight=-5.0)
    speed = RewardTermCfg(func=rewards.speed, weight=-0.05)
    action_rate = RewardTermCfg(func=rewards.action_rate, weight=-0.01)
    joint_limit = RewardTermCfg(func=rewards.limit, weight=-100.0)


@configclass
class FullGraspTerminationsCfg:
    time_out = TerminationTermCfg(func=lab_mdp.time_out, time_out=True)
    unsafe = TerminationTermCfg(func=terms.unsafe)
    reset_failed = TerminationTermCfg(func=terms.reset_failed)
    dropped = TerminationTermCfg(func=terms.dropped)
    success = TerminationTermCfg(func=terms.success)


@configclass
class UR5AllegroFullGraspEnvCfg(UR5AllegroTeacherEnvCfg):
    events = FullGraspEventsCfg()
    observations = FullGraspObservationsCfg()
    rewards = FullGraspRewardsCfg()
    terminations = FullGraspTerminationsCfg()
    decimation = 4
    episode_length_s = 10.0  # 脚本实测约9.5–10秒完成.10m抬升及1秒保持。
    training_objects: tuple[str, ...] = ('wood_block_oriented',)
    curriculum_stage: str = 'C1_wood_block_narrow_P64_T10_v2'
    reward_profile: str = 'full_grasp_phase_orientation_v5'
    legacy_joint_friction: float | None = None  # 默认保留旧物理；未通过的候选不采用。
    hand_static_friction: float | None = None
    reset_xy_center: tuple[float, float] = (0.35, -0.75152)
    reset_xy_range: float = 0.015
    reset_yaw_range: float = 0.15
    reset_clearance: float = 0.02
    pregrasp_distance: float = 0.13  # .08m实测拇指碰撞顶点穿桌.02005m，增加.05m真实余量。
    reset_max_attempts: int = 8
    lift_height: float = 0.10
    hold_duration_s: float = 1.0
    tip_contact_threshold_N: float = 0.05
    arm_action_speed: float = 0.15  # rad/s，显式新MDP；D4每步0.005 rad。
    hand_action_speed: float = 0.45  # rad/s，D4每步0.015 rad。

    def __post_init__(self):
        super().__post_init__()
        self.actions.teacher.class_type = FullGraspJointDeltaAction
        self.viewer.eye = (1.5, -2.0, 1.6)
        self.viewer.lookat = (0.2, -0.65, 0.9)
        # 新安全域的真实手物接触对照：32P p99.99=.00160，64P=.00059 rad。
        # 历史危险轨迹仍独立复验；此参数不改旧Teacher/v1。
        self.scene.robot.spawn.articulation_props.solver_position_iteration_count = 64
        self.set_num_envs(1)
        for index, name in enumerate(TIP_NAMES):
            setattr(self.scene, f'full_tip_{index}', ContactSensorCfg(
                prim_path='{ENV_REGEX_NS}/Robot/'+name, update_period=0,
                filter_prim_paths_expr=['{ENV_REGEX_NS}/Object/top', '{ENV_REGEX_NS}/Object/bottom', '{ENV_REGEX_NS}/Table'],
                track_friction_forces=True, max_contact_data_count_per_prim=32))
        for name in ('top', 'bottom'):
            setattr(self.scene, f'full_support_{name}', ContactSensorCfg(
                prim_path='{ENV_REGEX_NS}/Object/'+name, update_period=0,
                filter_prim_paths_expr=['{ENV_REGEX_NS}/Table'], max_contact_data_count_per_prim=32))

    def synchronize_control_timing(self):
        assert self.sim.dt == 1/120 and self.decimation in (2, 4)
        self.sim.render_interval = self.decimation
        names = self.actions.teacher.joint_names
        self.actions.teacher.scale = {name: (self.arm_action_speed if i < 6 else self.hand_action_speed)*self.control_dt()
                                      for i, name in enumerate(names)}

    def set_num_envs(self, num_envs):
        assert self.object_dataset == 'new_training_set'
        names = tuple(self.training_objects[i % len(self.training_objects)] for i in range(num_envs))
        self.scene.num_envs, self.object_names = num_envs, names
        self.scene.object.spawn.usd_path = [str(object_asset_dir(self.object_dataset, name)/f'{name}.usd') for name in names]
        self.events.initialize_teacher_data.params.update(dataset_name=self.object_dataset, object_names=names)
        self.events.reset_teacher.params.update(dataset_name=self.object_dataset, object_names=names)

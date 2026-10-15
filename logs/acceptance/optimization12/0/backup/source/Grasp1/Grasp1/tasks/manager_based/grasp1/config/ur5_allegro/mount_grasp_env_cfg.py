"""v3隔离课程：源模型安装不变，固定优化10已验证掌轴，显式几何TCP。"""
import math
from isaaclab.utils import configclass
from .full_grasp_env_cfg import UR5AllegroFullGraspEnvCfg
from Grasp1.robots.robot_profile import GRASP_TCP_OFFSET
from ...mdp.actions import TableClearanceJointDeltaAction


@configclass
class UR5AllegroMountGraspEnvCfg(UR5AllegroFullGraspEnvCfg):
    mount_semantics_version: str = 'source_urdf_identity_flange_hand_v1'
    tcp_semantics_version: str = 'geometric_center_negative_legacy_reset_offset_v1'
    # 数值来自源候选和优化10 seed21真实掌轴：+X朝下、+Y朝-Y、+Z朝-X。
    # 右图缺失，不能将该可复现方向称为用户已确认的标准方向。
    palm_quaternion_w: tuple[float, ...] = (0., -math.sqrt(.5), 0., math.sqrt(.5))
    tcp_offset_h: tuple[float, ...] = GRASP_TCP_OFFSET
    tcp_initial_height: float | None = None
    action_table_clearance_m: float = .008  # 旧脚本最小真实间距.01518m；不阻止正常闭合路径。
    curriculum_stage: str = 'C1_fixed_palm_wood_v3'

    def __post_init__(self):
        super().__post_init__()
        self.actions.teacher.class_type = TableClearanceJointDeltaAction
        self.observations.policy.full_grasp.params['tcp_offset_h'] = self.tcp_offset_h
        self.set_curriculum('C1')

    def set_curriculum(self, stage):
        """课程与契约一起更新；C1保留近预抓取，C2/C3使用桌心绝对TCP。"""
        self.curriculum_stage = stage+'_fixed_palm_wood_v3'
        self.reset_xy_center = (.35, -.75152) if stage == 'C1' else (.2, -.75152)
        self.tcp_initial_height = {'C1': None, 'C2': .30, 'C3': .50}[stage]
        self.events.reset_teacher.params.update(palm_quaternion_w=self.palm_quaternion_w,
            tcp_offset_h=self.tcp_offset_h, tcp_height=self.tcp_initial_height)

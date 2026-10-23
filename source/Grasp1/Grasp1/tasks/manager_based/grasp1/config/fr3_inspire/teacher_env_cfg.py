"""FR3-Inspire Teacher：共享原17项奖励/终止，仅适配真实机器人结构。

ADAPTED：19轴、13个真实手部link原点、7个臂部link原点；非Allegro 17点。
FR3预抓取使用用户批准的官方Lula IK与原投影宽度排序。
"""
from isaaclab.utils import configclass
from Grasp1.data.object_set import TRAINING_DATASET, training_object_names
from Grasp1.robots.fr3_inspire_cfg import FR3_INSPIRE_CFG, FR3_PALM_CENTER
from Grasp1.robots.robot_profile import (
    FR3_JOINT_NAMES, FR3_LINK_NAMES, INSPIRE_JOINT_NAMES,
    INSPIRE_HAND_KEYPOINT_NAMES, INSPIRE_DISTAL_LINK_NAMES,
    INSPIRE_POINT_WEIGHTS, INSPIRE_CONTACT_WEIGHTS,
)
from ...grasp1_env_cfg import Grasp1EnvCfg, PHYSICS_DT, ROBOT_ROOT_POSITION
from ..ur5_allegro.teacher_env_cfg import UR5AllegroTeacherEnvCfg, _build_teacher_object_cfg


@configclass
class FR3InspireTeacherEnvCfg(UR5AllegroTeacherEnvCfg):
    task_id = "Grasp1-FR3-Inspire-Teacher-v0"
    semantic_revision = 1
    robot_type = "fr3_inspire"
    action_dim = 19
    # 2*19关节字段 + 2*13接触字段 + 13手点 + 7臂点 + 9腕字段 + 3*13距离向量。
    observation_dim = 132
    teacher_point_weights = INSPIRE_POINT_WEIGHTS
    teacher_contact_weights = INSPIRE_CONTACT_WEIGHTS
    teacher_impulse_caps = (0.1,) * 9 + (0.2,) * 4
    palm_center = FR3_PALM_CENTER

    def __post_init__(self):
        # 不运行UR5专属传感器/动作配置；桌子、数据集和物理参数沿用共享场景。
        Grasp1EnvCfg.__post_init__(self)
        self.scene.replicate_physics = False
        self.scene.robot = FR3_INSPIRE_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.robot.init_state.pos = ROBOT_ROOT_POSITION
        self.scene.robot.init_state.rot = (1., 0., 0., 0.)
        self.scene.object = _build_teacher_object_cfg(tuple(training_object_names()), dataset_name=TRAINING_DATASET)
        self.actions.teacher.joint_names = list(FR3_JOINT_NAMES + INSPIRE_JOINT_NAMES)
        self.set_num_envs(128)
        for index, body in enumerate(INSPIRE_HAND_KEYPOINT_NAMES):
            self._contact(f"teacher_af_contact_{index}", f"Robot/{body}", ["Object/top"])
            self._contact(f"teacher_table_contact_{index}", f"Robot/{body}", ["Table"])
        for index, body in enumerate(FR3_LINK_NAMES):
            self._contact(f"teacher_arm_contact_{index}", f"Robot/{body}", ["Object/top", "Object/bottom", "Table"])
        for index, body in enumerate(INSPIRE_DISTAL_LINK_NAMES):
            self._contact(f"teacher_tip_object_{index}", f"Robot/{body}", ["Object/top"])
            self._contact(f"teacher_tip_table_{index}", f"Robot/{body}", ["Table"])
        self.synchronize_control_timing()

    def synchronize_control_timing(self):
        if self.sim.dt != PHYSICS_DT or self.decimation != 2:
            raise ValueError("Teacher requires physics 120Hz and policy 60Hz.")
        # 沿用Teacher相对关节动作尺度；不增加笛卡尔/姿态控制器。
        self.actions.teacher.scale = {**dict.fromkeys(FR3_JOINT_NAMES, 0.00125),
                                      **dict.fromkeys(INSPIRE_JOINT_NAMES, 0.1)}
        self.sim.render_interval = self.decimation

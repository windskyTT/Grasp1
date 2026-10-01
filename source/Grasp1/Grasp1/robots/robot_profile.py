"""定义 Grasp1 Teacher 中 UR5 + Allegro 机器人的语义名称和分组。

本模块只保存机器人关节、连杆、关键点、接触部位等名称与静态描述，不创建
Isaac Lab articulation、执行器、初始关节状态、运行时索引、Torch 张量或场景。
实际关节和刚体索引应从已加载的 Isaac Lab articulation 中按名称解析；
关节和关键点查询需要 preserve_order=True 才能保持此处列出的顺序。
URDF 中由固定关节连接的连杆在 USD 转换时可能合并，因此这里的
URDF 连杆名不保证一定对应一个独立的 articulation 刚体。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final


# -----------------------------------------------------------------------------
# 数量配置：用于描述机器人关节、手部关键点和接触连杆的数量。
# -----------------------------------------------------------------------------

UR5_NUM_JOINTS: Final[int] = 6
ALLEGRO_NUM_JOINTS: Final[int] = 16
ALLEGRO_NUM_FINGERS: Final[int] = 4

NUM_CONTROLLED_JOINTS: Final[int] = UR5_NUM_JOINTS + ALLEGRO_NUM_JOINTS

ARM_KEYPOINT_COUNT: Final[int] = 6
HAND_KEYPOINT_COUNT: Final[int] = 17

ARM_CONTACT_LINK_COUNT: Final[int] = 6
HAND_CONTACT_LINK_COUNT: Final[int] = 13


# -----------------------------------------------------------------------------
# UR5 机械臂的关节和连杆名称，顺序与控制向量及关键点定义一致。
# -----------------------------------------------------------------------------

UR5_JOINT_NAMES: Final[tuple[str, ...]] = (
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
)

UR5_LINK_NAMES: Final[tuple[str, ...]] = (
    "shoulder_link",
    "upper_arm_link",
    "forearm_link",
    "wrist_1_link",
    "wrist_2_link",
    "wrist_3_link",
)


# 原始 RobustDexGrasp 使用六个 UR5 关节坐标系作为手臂关键点。
ARM_KEYPOINT_FRAME_NAMES: Final[tuple[str, ...]] = UR5_JOINT_NAMES

# 上述关键点坐标系在源 URDF 中对应的子连杆名称。
ARM_KEYPOINT_LINK_NAMES: Final[tuple[str, ...]] = UR5_LINK_NAMES


# -----------------------------------------------------------------------------
# UR5 与 Allegro 手部连接处的坐标系和 URDF 连杆名称。
# -----------------------------------------------------------------------------

# 原始 RaiSim 中作为手腕/末端执行器使用的坐标系名称。
END_EFFECTOR_FRAME_NAME: Final[str] = "Flange2hand_fixed_joint"

# 原始 URDF 中 Flange2hand_fixed_joint 的父连杆和子连杆。
END_EFFECTOR_LINK_NAME: Final[str] = "Flange_base_link"
HAND_BASE_LINK_NAME: Final[str] = "Allegro_base_link"


# -----------------------------------------------------------------------------
# Allegro 手部的 16 个关节和 16 个连杆名称。
# -----------------------------------------------------------------------------

ALLEGRO_JOINT_NAMES: Final[tuple[str, ...]] = (
    "joint_0.0",
    "joint_1.0",
    "joint_2.0",
    "joint_3.0",
    "joint_4.0",
    "joint_5.0",
    "joint_6.0",
    "joint_7.0",
    "joint_8.0",
    "joint_9.0",
    "joint_10.0",
    "joint_11.0",
    "joint_12.0",
    "joint_13.0",
    "joint_14.0",
    "joint_15.0",
)

ALLEGRO_LINK_NAMES: Final[tuple[str, ...]] = (
    "link_0.0",
    "link_1.0",
    "link_2.0",
    "link_3.0",
    "link_4.0",
    "link_5.0",
    "link_6.0",
    "link_7.0",
    "link_8.0",
    "link_9.0",
    "link_10.0",
    "link_11.0",
    "link_12.0",
    "link_13.0",
    "link_14.0",
    "link_15.0",
)


# 按手指分成四组；每组包含连续的 4 个自由度关节。
ALLEGRO_FINGER_JOINT_GROUPS: Final[tuple[tuple[str, ...], ...]] = (
    ALLEGRO_JOINT_NAMES[0:4],
    ALLEGRO_JOINT_NAMES[4:8],
    ALLEGRO_JOINT_NAMES[8:12],
    ALLEGRO_JOINT_NAMES[12:16],
)


# -----------------------------------------------------------------------------
# 四个指尖对应的坐标系名称及源 URDF 连杆名称。
# -----------------------------------------------------------------------------

ALLEGRO_FINGERTIP_FRAME_NAMES: Final[tuple[str, ...]] = (
    "joint_3.0_tip",
    "joint_7.0_tip",
    "joint_11.0_tip",
    "joint_15.0_tip",
)

ALLEGRO_FINGERTIP_LINK_NAMES: Final[tuple[str, ...]] = (
    "link_3.0_tip",
    "link_7.0_tip",
    "link_11.0_tip",
    "link_15.0_tip",
)


# -----------------------------------------------------------------------------
# 17 个手部关键点的顺序和对应名称。
# -----------------------------------------------------------------------------
#
# 原始 RobustDexGrasp 的关键点顺序：
#
#   0       hand / wrist base
#
#   1-4     finger 0
#   5-8     finger 1
#   9-12    finger 2
#   13-16   finger 3
#
# 每根手指依次贡献以下四个关键点：
#   proximal frame 近端框架
#   middle frame 中间框架
#   distal frame 远端框架
#   fingertip frame 指尖框架
#
# 观测和奖励计算依赖此顺序，因此保持不变。
# -----------------------------------------------------------------------------

HAND_KEYPOINT_FRAME_NAMES: Final[tuple[str, ...]] = (
    "Flange2hand_fixed_joint",
    "joint_1.0",
    "joint_2.0",
    "joint_3.0",
    "joint_3.0_tip",
    "joint_5.0",
    "joint_6.0",
    "joint_7.0",
    "joint_7.0_tip",
    "joint_9.0",
    "joint_10.0",
    "joint_11.0",
    "joint_11.0_tip",
    "joint_13.0",
    "joint_14.0",
    "joint_15.0",
    "joint_15.0_tip",
)

# 源 URDF 中与关键点坐标系对应的连杆。固定指尖连杆可能在 USD 中合并；
# 关键点逻辑应使用加载后 articulation 的刚体及坐标系偏移。
HAND_KEYPOINT_LINK_NAMES: Final[tuple[str, ...]] = (
    "Flange_base_link",
    "link_1.0",
    "link_2.0",
    "link_3.0",
    "link_3.0_tip",
    "link_5.0",
    "link_6.0",
    "link_7.0",
    "link_7.0_tip",
    "link_9.0",
    "link_10.0",
    "link_11.0",
    "link_11.0_tip",
    "link_13.0",
    "link_14.0",
    "link_15.0",
    "link_15.0_tip",
)

# 每个切片提取对应手指的四个关键点，不包含索引 0 的手腕基座点。
HAND_FINGER_KEYPOINT_SLICES: Final[tuple[slice, ...]] = (
    slice(1, 5),
    slice(5, 9),
    slice(9, 13),
    slice(13, 17),
)

# 四个指尖在 17 个手部关键点序列中的索引。
HAND_FINGERTIP_KEYPOINT_INDICES: Final[tuple[int, ...]] = (
    4,
    8,
    12,
    16,
)


# -----------------------------------------------------------------------------
# 接触检测使用的 UR5 和 Allegro 连杆名称。
# -----------------------------------------------------------------------------

ARM_CONTACT_LINK_NAMES: Final[tuple[str, ...]] = UR5_LINK_NAMES

# RobustDexGrasp AllegroSim 使用的接触刚体列表。
#
# 虽然 wrist_3_link 属于 UR5，这里仍按原始列表将它纳入手部接触部位。
HAND_CONTACT_LINK_NAMES: Final[tuple[str, ...]] = (
    "wrist_3_link",
    "link_1.0",
    "link_2.0",
    "link_3.0",
    "link_5.0",
    "link_6.0",
    "link_7.0",
    "link_9.0",
    "link_10.0",
    "link_11.0",
    "link_13.0",
    "link_14.0",
    "link_15.0",
)


# -----------------------------------------------------------------------------
# 抓取中心相对于手部连接坐标系的偏移。
# -----------------------------------------------------------------------------

# 抓取坐标系在 Flange2hand_fixed_joint 坐标系中的位置。
# 数值迁移自 cfg_reg.yaml：
#
#     hand_center: [-0.0091, 0.0, -0.095]
#
HAND_CENTER: Final[tuple[float, float, float]] = (
    -0.0091,
    0.0,
    -0.095,
)


# -----------------------------------------------------------------------------
# 将 UR5 和 Allegro 关节合并为策略控制顺序，并定义各自的切片范围。
# -----------------------------------------------------------------------------

# 策略动作的关节顺序：先 UR5 六轴，再 Allegro 十六个手指关节。
CONTROLLED_JOINT_NAMES: Final[tuple[str, ...]] = (
    UR5_JOINT_NAMES + ALLEGRO_JOINT_NAMES
)

# 从合并后的受控关节序列中选取 UR5 机械臂关节。
ARM_JOINT_SLICE: Final[slice] = slice(
    0,
    UR5_NUM_JOINTS,
)

# 从合并后的受控关节序列中选取 Allegro 手部关节。
HAND_JOINT_SLICE: Final[slice] = slice(
    UR5_NUM_JOINTS,
    NUM_CONTROLLED_JOINTS,
)


# -----------------------------------------------------------------------------
# 汇总机器人语义配置的数据类及本任务采用的具体配置实例。
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class RobotProfile:
    """以名称和有序分组描述 UR5 + Allegro 的源资产语义。"""

    # 控制用关节名称；顺序决定策略动作向量中的排列。
    arm_joint_names: tuple[str, ...]
    hand_joint_names: tuple[str, ...]

    # 机械臂和手部的 URDF 连杆名称列表。
    arm_link_names: tuple[str, ...]
    hand_link_names: tuple[str, ...]

    # 手腕末端坐标系、其 URDF 连杆以及 Allegro 手部基座连杆。
    end_effector_frame_name: str
    end_effector_link_name: str
    hand_base_link_name: str

    # 手臂关键点的坐标系名称及对应 URDF 连杆名称。
    arm_keypoint_frame_names: tuple[str, ...]
    arm_keypoint_link_names: tuple[str, ...]

    # 手部关键点的坐标系名称及对应 URDF 连杆名称，顺序需保持一致。
    hand_keypoint_frame_names: tuple[str, ...]
    hand_keypoint_link_names: tuple[str, ...]

    # 四个指尖的坐标系名称及对应 URDF 连杆名称。
    fingertip_frame_names: tuple[str, ...]
    fingertip_link_names: tuple[str, ...]

    # 用于接触检测的手臂和手部连杆名称。
    arm_contact_link_names: tuple[str, ...]
    hand_contact_link_names: tuple[str, ...]

    # 抓取中心在末端执行器坐标系中的位置偏移。
    hand_center: tuple[float, float, float]

    @property # 把一个“方法”包装成一个可以像“属性”一样访问的接口
    def controlled_joint_names(self) -> tuple[str, ...]:
        """返回策略动作顺序排列的全部受控关节名（手臂后接手部）。"""
        return self.arm_joint_names + self.hand_joint_names

    @property
    def num_arm_joints(self) -> int:
        """返回机械臂关节数量。"""
        return len(self.arm_joint_names)

    @property
    def num_hand_joints(self) -> int:
        """返回手部关节数量。"""
        return len(self.hand_joint_names)

    @property
    def num_controlled_joints(self) -> int:
        """返回机械臂和手部的受控关节总数。"""
        return len(self.controlled_joint_names)

    @property
    def num_hand_keypoints(self) -> int:
        """返回手部关键点数量。"""
        return len(self.hand_keypoint_link_names)

    @property
    def num_arm_keypoints(self) -> int:
        """返回手臂关键点数量。"""
        return len(self.arm_keypoint_link_names)

    @property
    def num_fingertips(self) -> int:
        """返回指尖数量。"""
        return len(self.fingertip_link_names)


# 创建供 UR5 + Allegro Teacher 任务使用的机器人语义配置实例。
UR5_ALLEGRO_PROFILE: Final[RobotProfile] = RobotProfile(
    arm_joint_names=UR5_JOINT_NAMES,
    hand_joint_names=ALLEGRO_JOINT_NAMES,

    arm_link_names=UR5_LINK_NAMES,
    hand_link_names=ALLEGRO_LINK_NAMES,

    end_effector_frame_name=END_EFFECTOR_FRAME_NAME,
    end_effector_link_name=END_EFFECTOR_LINK_NAME,
    hand_base_link_name=HAND_BASE_LINK_NAME,

    arm_keypoint_frame_names=ARM_KEYPOINT_FRAME_NAMES,
    arm_keypoint_link_names=ARM_KEYPOINT_LINK_NAMES,

    hand_keypoint_frame_names=HAND_KEYPOINT_FRAME_NAMES,
    hand_keypoint_link_names=HAND_KEYPOINT_LINK_NAMES,

    fingertip_frame_names=ALLEGRO_FINGERTIP_FRAME_NAMES,
    fingertip_link_names=ALLEGRO_FINGERTIP_LINK_NAMES,

    arm_contact_link_names=ARM_CONTACT_LINK_NAMES,
    hand_contact_link_names=HAND_CONTACT_LINK_NAMES,

    hand_center=HAND_CENTER,
)


# 对外公开的机器人语义配置、名称列表和静态分组。
__all__ = [
    # 机器人配置类型及 UR5 + Allegro 配置实例。
    "RobotProfile",
    "UR5_ALLEGRO_PROFILE",

    # 关节、关键点和接触连杆数量。
    "UR5_NUM_JOINTS",
    "ALLEGRO_NUM_JOINTS",
    "ALLEGRO_NUM_FINGERS",
    "NUM_CONTROLLED_JOINTS",
    "ARM_KEYPOINT_COUNT",
    "HAND_KEYPOINT_COUNT",
    "ARM_CONTACT_LINK_COUNT",
    "HAND_CONTACT_LINK_COUNT",

    # UR5 关节和连杆名称。
    "UR5_JOINT_NAMES",
    "UR5_LINK_NAMES",

    # UR5 与 Allegro 连接处的名称。
    "END_EFFECTOR_FRAME_NAME",
    "END_EFFECTOR_LINK_NAME",
    "HAND_BASE_LINK_NAME",

    # Allegro 关节、连杆和手指分组。
    "ALLEGRO_JOINT_NAMES",
    "ALLEGRO_LINK_NAMES",
    "ALLEGRO_FINGER_JOINT_GROUPS",

    # 指尖坐标系和连杆名称。
    "ALLEGRO_FINGERTIP_FRAME_NAMES",
    "ALLEGRO_FINGERTIP_LINK_NAMES",

    # 手臂/手部关键点及指尖索引。
    "ARM_KEYPOINT_FRAME_NAMES",
    "ARM_KEYPOINT_LINK_NAMES",
    "HAND_KEYPOINT_FRAME_NAMES",
    "HAND_KEYPOINT_LINK_NAMES",
    "HAND_FINGER_KEYPOINT_SLICES",
    "HAND_FINGERTIP_KEYPOINT_INDICES",

    # 接触检测连杆名称。
    "ARM_CONTACT_LINK_NAMES",
    "HAND_CONTACT_LINK_NAMES",

    # 抓取中心偏移。
    "HAND_CENTER",

    # 合并后的关节名称顺序和手臂/手部切片。
    "CONTROLLED_JOINT_NAMES",
    "ARM_JOINT_SLICE",
    "HAND_JOINT_SLICE",
]

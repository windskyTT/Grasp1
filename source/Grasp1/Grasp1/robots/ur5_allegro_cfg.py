"""配置 Teacher 任务使用的 UR5 + Allegro articulation、初始状态和执行器参数。

机器人关节的语义名称与关节顺序由 ``robot_profile.py`` 定义；本模块根据 USD
资产和原始 RobustDexGrasp 参数构建 Isaac Lab 的 articulation 配置。
"""

from __future__ import annotations

from typing import Final

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg

from Grasp1.robots.robot_profile import ALLEGRO_JOINT_NAMES, UR5_JOINT_NAMES
from Grasp1.utils.paths import robot_asset_dir


# -----------------------------------------------------------------------------
# USD 中的关节名称转换。
# -----------------------------------------------------------------------------

# Isaac Sim 的 URDF 导入器会清理 USD prim/关节名称中的特殊字符。特别是 Allegro
# 关节名会按如下方式转换：
#
#     joint_0.0 -> joint_0_0
#
# robot_profile.py 保留源 URDF 名称；这里生成与运行时 USD 关节名对应的名称。
_HAND_USD_JOINT_NAMES: Final[tuple[str, ...]] = tuple(
    name.replace(".", "_") for name in ALLEGRO_JOINT_NAMES
)


# -----------------------------------------------------------------------------
# 迁移自 RobustDexGrasp 的 UR5 PD 参数。
# -----------------------------------------------------------------------------

# 来源于 UR5Identification_id5hz.txt；原文件前六项是刚度（P），后六项是阻尼（D）。
# 这里按 UR5 六个关节的顺序分别保存两组参数。
_ARM_STIFFNESS: Final[tuple[float, ...]] = (
    15845.1220703125,
    16202.208984375,
    15775.421875,
    16162.2900390625,
    16039.8310546875,
    16183.51953125,
)

_ARM_DAMPING: Final[tuple[float, ...]] = (
    478.6603698730469,
    512.3048706054688,
    281.6572265625,
    577.3147583007812,
    576.9467163085938,
    443.34552001953125,
)


# -----------------------------------------------------------------------------
# 手部初始关节状态。
# -----------------------------------------------------------------------------

# 迁移自 allegro_teacher/cfgs/cfg_reg.yaml 的 init_finger_pose，顺序与 Allegro
# 原始关节顺序一致；配置时会与转换后的 USD 关节名配对。
_HAND_INIT_POS: Final[tuple[float, ...]] = (
    0.2,
    0.6,
    0.2,
    0.5,
    0.2,
    0.6,
    0.2,
    0.5,
    0.2,
    0.6,
    0.2,
    0.5,
    1.3,
    0.0,
    -0.1,
    0.2,
)


# -----------------------------------------------------------------------------
# UR5 和 Allegro 执行器使用的力矩/速度上限及 PD 参数。
# -----------------------------------------------------------------------------

# 保留自原始 UR5 + Allegro URDF 的 UR5 关节力矩上限和速度上限。
_ARM_EFFORT_LIMIT: Final[tuple[float, ...]] = (
    150.0,
    150.0,
    150.0,
    28.0,
    28.0,
    28.0,
)

_ARM_VELOCITY_LIMIT: Final[tuple[float, ...]] = (
    3.15,
    3.15,
    3.15,
    3.2,
    3.2,
    3.2,
)

# Allegro 手部所有关节共用的隐式执行器刚度、阻尼、力矩上限和速度上限。
_HAND_STIFFNESS: Final[float] = 600.0
_HAND_DAMPING: Final[float] = 20.0
_HAND_EFFORT_LIMIT: Final[float] = 0.7
_HAND_VELOCITY_LIMIT: Final[float] = 7.0


# -----------------------------------------------------------------------------
# Isaac Lab articulation 配置：指定 USD 资产、初始关节位置和分组执行器。
# -----------------------------------------------------------------------------

UR5_ALLEGRO_CFG: Final[ArticulationCfg] = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        # 加载 UR5 + Allegro 组合机器人的 USD 资产。
        usd_path=str(robot_asset_dir("ur5_allegro") / "ur5_allegro.usd"),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            # 原始 UR5 + Allegro 系统为固定底座 articulation。
            fix_root_link=True,
            solver_position_iteration_count=32,
            solver_velocity_iteration_count=1,
            # 开启自碰撞的专项验收出现巨大手部接触力，保持原关闭配置。
            enabled_self_collisions=False,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        # 此处将 UR5 关节设为零位，手部使用上方配置的初始姿态。
        # Teacher 的 reset/event 后续会用选定的预抓取 IK 解更新 UR5 状态。
        joint_pos={
            **dict.fromkeys(UR5_JOINT_NAMES, 0.0),
            **dict(zip(_HAND_USD_JOINT_NAMES, _HAND_INIT_POS)),
        },
    ), # **：把一个字典中的所有 key: value 键值对展开，合并到当前字典中
    actuators={
        # UR5 机械臂：每个关节分别设置 PD 增益、力矩上限和速度上限。
        "arm": ImplicitActuatorCfg(
            joint_names_expr=list(UR5_JOINT_NAMES),
            stiffness=dict(zip(UR5_JOINT_NAMES, _ARM_STIFFNESS)),
            damping=dict(zip(UR5_JOINT_NAMES, _ARM_DAMPING)),
            effort_limit_sim=dict(zip(UR5_JOINT_NAMES, _ARM_EFFORT_LIMIT)),
            velocity_limit_sim=dict(zip(UR5_JOINT_NAMES, _ARM_VELOCITY_LIMIT)),
        ),
        # Allegro 手部：所有手指关节共用同一组 PD 增益和运动上限。
        "hand": ImplicitActuatorCfg(
            joint_names_expr=list(_HAND_USD_JOINT_NAMES),
            stiffness=_HAND_STIFFNESS,
            damping=_HAND_DAMPING,
            effort_limit_sim=_HAND_EFFORT_LIMIT,
            velocity_limit_sim=_HAND_VELOCITY_LIMIT,
        ),
    },
)


# 对外公开的 UR5 + Allegro articulation 配置。
__all__ = ["UR5_ALLEGRO_CFG"]

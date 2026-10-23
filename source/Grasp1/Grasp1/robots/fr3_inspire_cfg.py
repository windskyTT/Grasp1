"""FR3-Inspire现有URDF的执行器/限位；不修改URDF、Mesh或导入惯量。"""
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
import yaml
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.actuators import ImplicitActuatorCfg
from Grasp1.utils.paths import robot_asset_dir
from .robot_profile import FR3_JOINT_NAMES, INSPIRE_JOINT_NAMES

FR3_URDF_PATH = robot_asset_dir("fr3_inspire") / "fr3_inspire_tac_L_right_safety_visual_realistic.urdf"
FR3_USD_PATH = FR3_URDF_PATH.with_suffix(".usd")
_urdf = ET.parse(FR3_URDF_PATH).getroot()
_joints = {j.get("name"): j for j in _urdf.findall("joint")}
# 读取安装版本附带的官方FR3默认姿态；仅按本URDF更紧的硬限位裁剪。
from isaacsim.core.utils.extensions import get_extension_path_from_name
import omni.kit.app
omni.kit.app.get_app().get_extension_manager().set_extension_enabled_immediate(
    "isaacsim.robot_motion.motion_generation", True)
_motion_dir = Path(get_extension_path_from_name("isaacsim.robot_motion.motion_generation"))
FR3_LULA_DESCRIPTION = _motion_dir / "motion_policy_configs/FR3/rmpflow/fr3_robot_description.yaml"
_lula_cfg = yaml.safe_load(FR3_LULA_DESCRIPTION.read_text())
FR3_DEFAULT_ARM = tuple(max(float(_joints[n].find("limit").get("lower")),
                            min(float(_joints[n].find("limit").get("upper")), q))
                        for n, q in zip(FR3_JOINT_NAMES, _lula_cfg["default_q"]))
# URDF弧度→USD度数→PhysX float32有舍入；源默认q4越限后裁剪到边界时，
# 向区间内取一个float32相邻值，避免Isaac Lab把-2.84误判为越界。限位不变。
FR3_DEFAULT_ARM = tuple(float(np.nextafter(np.float32(q), np.float32(0.)))
                        if q == float(_joints[n].find("limit").get("lower"))
                        or q == float(_joints[n].find("limit").get("upper")) else q
                        for n, q in zip(FR3_JOINT_NAMES, FR3_DEFAULT_ARM))
FR3_PALM_CENTER = tuple(float(v) for v in _urdf.find("link[@name='base_link']/inertial/origin").get("xyz").split())


def _limits(names, key):
    return {name: float(_joints[name].find("limit").get(key)) for name in names}


FR3_INSPIRE_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(usd_path=str(FR3_USD_PATH), activate_contact_sensors=True),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={**dict(zip(FR3_JOINT_NAMES, FR3_DEFAULT_ARM)), **dict.fromkeys(INSPIRE_JOINT_NAMES, 0.)},
        joint_vel={".*": 0.},
    ),
    actuators={
        "arm": ImplicitActuatorCfg(
            joint_names_expr=list(FR3_JOINT_NAMES),
            # K/D来自该URDF的dynamics属性；没有自行设计增益。
            stiffness={n: float(_joints[n].find("dynamics").get("K")) for n in FR3_JOINT_NAMES},
            damping={n: float(_joints[n].find("dynamics").get("D")) for n in FR3_JOINT_NAMES},
            effort_limit_sim=_limits(FR3_JOINT_NAMES, "effort"),
            velocity_limit_sim=_limits(FR3_JOINT_NAMES, "velocity"),
        ),
        "hand": ImplicitActuatorCfg(
            joint_names_expr=list(INSPIRE_JOINT_NAMES),
            # 用户本次明确授权沿用Teacher手部增益；URDF本身未定义手部PD。
            stiffness=3.0, damping=0.1,
            effort_limit_sim=_limits(INSPIRE_JOINT_NAMES, "effort"),
            velocity_limit_sim=_limits(INSPIRE_JOINT_NAMES, "velocity"),
        ),
    },
)

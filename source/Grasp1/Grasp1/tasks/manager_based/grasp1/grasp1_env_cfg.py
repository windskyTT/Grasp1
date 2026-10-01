"""UR5 + Allegro Teacher 的共享场景与仿真基础配置。"""

from __future__ import annotations

from dataclasses import MISSING
from typing import Final

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from Grasp1.utils.paths import ASSETS_DIR

from Grasp1.robots.ur5_allegro_cfg import UR5_ALLEGRO_CFG


CONTROL_DT: Final[float] = 0.2
TABLE_CENTER_XY: Final[tuple[float, float]] = (0.2, -0.75152)
TABLE_HEIGHT: Final[float] = 0.771


@configclass
class Grasp1SceneCfg(InteractiveSceneCfg):
    """每个环境包含机器人、物体槽位和桌子；地面与灯光由场景共享。"""

    # 地面平面及其静态摩擦、动态摩擦和恢复系数。
    ground = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        spawn=sim_utils.GroundPlaneCfg(
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=0.8,
                dynamic_friction=0.8,
                restitution=0.0,
            ),
        ),
    )

    # 将机器人 articulation 配置放入每个并行环境各自的 Robot prim 下。
    robot: ArticulationCfg = UR5_ALLEGRO_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # 具体物体 USD 和初始姿态由任务配置；原抓取物体具有可动关节。
    # MISSING 表示此共享场景类要求具体任务配置提供物体 articulation。
    object: ArticulationCfg = MISSING

    # 桌面碰撞体去掉 USD 根姿态后，Z 范围为 [-1.003, -0.003] m，
    # XY 中心为 (0, 0.1561)。旋转 90° 后平移，使顶面高度为 0.771 m、
    # 中心为原任务的 (0.2, -0.75152)，桌脚落在地面。
    # 本地 USD 引用原桌子并添加 kinematic 刚体，使 GPU 接触过滤可读取桌面接触。
    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(ASSETS_DIR / "table" / "table.usda"),
            scale=(1.0, 1.0, 0.771),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(0.3561, -0.75152, 0.773313),
            rot=(0.70710678, 0.0, 0.0, 0.70710678),
        ),
    )

    # 场景穹顶灯，为所有环境提供统一的环境光照。
    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(
            color=(0.75, 0.75, 0.75),
            intensity=2500.0,
        ),
    )


@configclass
class Grasp1EnvCfg(ManagerBasedRLEnvCfg):
    """共享仿真参数；动作、观测和奖励等由具体任务配置。"""

    # 随机数种子，供环境和训练组件初始化随机过程。
    seed = 1

    # 并行场景配置；默认创建 1 个环境，环境之间间隔 3 米。
    scene: Grasp1SceneCfg = Grasp1SceneCfg(num_envs=1, env_spacing=3.0)

    def __post_init__(self) -> None:
        """在配置对象初始化后设置仿真频率、渲染频率和物理材质。"""

        # 物理仿真每步 0.01 秒；每 20 个仿真步执行一次控制动作。
        self.sim.dt = 0.01
        self.decimation = 20  # control_dt = 0.01 × 20 = 0.2 s

        # 每个回合持续 4 秒；渲染间隔与控制间隔一致。
        self.episode_length_s = 4.0
        self.sim.render_interval = self.decimation

        # 对应旧 world 默认材质；远程桌面 USD 的独立材质不受此值覆盖。
        self.sim.physics_material = sim_utils.RigidBodyMaterialCfg(
            static_friction=0.8,
            dynamic_friction=0.8,
            restitution=0.0,
        )

# 更换桌子任务验收报告（2026-10-01）

## 修改文件

- `assets/robots/ur5_allegro/ur5_allegro.urdf`：移除 world、platform_base_link、Link_1 以及连接它们的三个固定关节。保留机械臂、Allegro 手和必要的固定坐标系。
- `assets/robots/ur5_allegro/configuration/ur5_allegro_base.usd`、`ur5_allegro_physics.usd`、`ur5_allegro_robot.usd`：使用 Isaac Sim 5.1 的 URDF importer 重新生成。入口仍为 `assets/robots/ur5_allegro/ur5_allegro.usd`，入口引用结构保持不变，实际模型内容在组成层中更新。
- `source/Grasp1/Grasp1/tasks/manager_based/grasp1/grasp1_env_cfg.py`：集中定义实际桌子尺寸、边界、support height 和底座测量值，计算桌边安装 root pose。
- `config/ur5_allegro/teacher_env_cfg.py`（同一 grasp1 目录）：统一传入 support_height，相机位置改为相对真实底座。
- `mdp/events.py`：物体采样和碰撞回退以真实 robot root XY 为参考；相机跟随底座平移；新 USD 的 IK base 不再加旧平台高度，按真实 root quaternion 转换 DH/base 坐标系。
- `mdp/observations.py`：手中心 XY 去除真实 robot root 平移，Z 保留源 Teacher 的地面高度语义；关键点高度统一使用 support_height。
- `mdp/rewards.py`、`mdp/terminations.py`：高度参数统一命名为 support_height，保持原公式与权重。

## Robot

- asset：UR5-Allegro，仅保留机械臂与灵巧手。
- root position（相对各自 env origin）：`(0.2, -0.421641, 0.7710023993)` m。
- root quaternion（wxyz）：`(0.70738827, 0, 0, -0.70682518)`。
- root yaw：`-1.57` rad，即约 `-89.954°`。这是原 platform2ur5_joint 的朝向移到 root，机械臂的原工作方向保持为 −Y。
- base_radius：visual/collision 实测最大半径 `0.1101204912` m，配置向上取整为 `0.110121` m。
- base mounting offset：`-0.00000239932` m；root Z = support_height − offset。
- base footprint（世界轴、减去 env origin）：X `[0.0899959372, 0.2735506090]` m；Y `[-0.4952742669, -0.3480623256]` m。
- base collision lowest Z：`0.771000012522` m；与桌面误差约 `0.00001252` mm。
- 结构检查：22 个 revolute joints、32 个 rigid bodies、25 个 collision prims、28 个 visual meshes。
- `robot-contract.log` 验证保留 link/joint 定义、质量、质心、惯量、关节限位与原资产一致。执行器、22 维动作、153 维观测、DH/IK 模型、PPO 和 reward weights 保持原配置。

## Table

当前实际 Table 是已有的本地 USD 引用桌子，继续原样使用；没有采用文档中的 Cuboid 示例尺寸。

- 桌子碰撞几何中心：`(0.2, -0.75152, 0.3855)` m。
- size：`(1.28, 0.91, 0.771)` m。
- bounds：X `[-0.44, 0.84]`；Y `[-1.20652, -0.29652]`；Z `[0, 0.771]` m。
- top Z / support_height：`0.771` m。
- 原 USD、几何、旋转、缩放、材质及接触过滤用的 kinematic 刚体保留。

## Robot-Table relation

源采样角度为 `[-0.7π, -0.3π]`，工作区在 −Y 方向，因此安装在 max-Y 近侧边缘。

- root 到近侧 max-Y 边：`0.125121` m。
- root 到远侧 min-Y 边：`0.784879` m。
- root 到 min-X / max-X 边：均为 `0.64` m。
- base footprint 到近侧边的最小间隙：`0.0515423256` m；按最大半径计算的安装安全余量为 `0.015` m。
- base fully supported：`true`。四个环境的底座边界逐一通过检查。

## 删除内容

- 修改前场景已经没有独立 Mat、WoodenTable；本次确认 source 和 Stage 均不存在它们。
- 本次实际删除的是机器人资产内部的旧白色 platform_base_link 几何及其支撑链，并重新转换 USD。
- 没有添加新的支撑物。清除了新导入 USD 中四个空 visual 的无效引用。
- 删除转换器自动产生的 `.asset_hash`、`config.yaml`，不增加新的运行流程。

## 坐标逻辑修改

- `_sample_object_pose()`：原 `xy + env_origins` 改为 `xy + robot.data.root_pos_w[..., :2]`。物体 Z 独立使用桌面高度和最低点数据；env origin Z 仅用于场景克隆的高度平移。
- `_visible_points()`、`_solve_pregrasp()`：相机原先加 env origin，改为加真实 robot root position；相机相对底座的 Z 去除源平台高度，保持原相机相对工作区的高度。
- `_solver_base_pose()`：此前已经读真实 root pose；本次因模型 root 从旧支撑链变为 base_link，删除 `+0.771`，使用 root position 和 `root_quaternion × Rz(π)` 构造 DH/base 位姿。
- `_apply_collision_fallback()`：安全同物体姿态按真实底座间平移复制；固定回退 XY 也加真实底座 XY。
- TeacherObservation 的 `93:96`：XY 相对真实机器人底座；Z 保留源地面高度语义。观测维度和排列不变。
- stable reset：稳定状态中的源桌面基准转换为当前 support_height，保留原有可配置的 stable_state_height_offset。

## Contact filters

现有实现为每个 source body 分别配置传感器，并不存在旧的统一 support_contact_filter_slice，因此不新增 slice 接口。

- Allegro affordance：`/World/envs/env_.*/Object/.*top`，filter count = 1，force_matrix_w 为 `[N, 1, 1, 3]`。
- Allegro support：`/World/envs/env_.*/Table`，filter count = 1，support index = 0，force_matrix_w 为 `[N, 1, 1, 3]`。
- UR5 arm：`Object/top`、`Object/bottom`、`Table`，filter count = 3，Table index = 2，force_matrix_w 为 `[N, 1, 3, 3]`。
- 实际手部触桌验证：support_contact_raw = `[0.3870967924594879]`；support_impulse_raw = `[0.09190444648265839]`，均读取真实接触传感器，未构造模拟力。
- 四环境传感器形状与实际过滤路径见 `four-env.json`。

## Reset

| 测试 | IK success | 初始碰撞回退 | 回退检查后的碰撞 | 物体支撑误差 |
|---|---:|---:|---:|---:|
| 修改前，单环境 100 次 | 100/100 | 0 | 0 | +2 mm |
| 修改后，单环境 100 次 | 100/100 | 0 | 0 | 0 mm |
| 修改后，四环境各 100 次 | 400/400 | 0 | 0 | 0 mm |

- 相同 100 个随机种子，修改前后 robot-object 相对 XY 的最大差异：`5.96046447754e-08` m。
- 实测 planar distance：`[0.450005272, 0.749424724]` m；保持源 `0.45–0.75` m 距离分布。
- 示例 reset：robot = `[0.20000000298023224, -0.4216409921646118, 0.7710024118423462]`，object relative vector = `[-0.17021550238132477, -0.7248533964157104, 0.015988588333129883]`。
- 四个环境的安装相对位姿一致；100 次正常 reset 未观察到跨环境碰撞或初始碰撞。

## Smoke test

- `git diff --check`：通过，exit 0。
- conda grasp 下 `python -m compileall -q source`：通过，exit 0。
- single env：通过，exit 0，见 `single-env-pass.log`、`single-env.json`。
- 4 env：通过，exit 0，见 `four-env.log`、`four-env.json`。
- 观测 `[N,153]`、动作 `[N,22]`；reset 后控制步的观测和奖励均 finite。
- 项目原 RSL-RL 训练入口完成一个 iteration，280 timesteps，exit 0。
- mean value_function loss = `162.4433`；mean surrogate loss = `-0.0303`；mean entropy loss = `31.2047`；mean reward = `-46.70`，均为有限值。
- 训练日志无 PhysX error、invalid prim、ContactSensor filter error 或 CUDA error。
- checkpoint：`logs/rsl_rl/grasp1_ur5_allegro_teacher/2026-10-01_14-46-17/model_0.pt`。

实际训练命令（项目目录下运行）：

```bash
env -u DISPLAY \
  PYTHONPATH=/home/windsky/project/Grasp1/source/Grasp1 \
  PYTHONUNBUFFERED=1 TERM=xterm \
  /home/windsky/miniconda3/bin/conda run --no-capture-output -n grasp \
  /home/windsky/IsaacLab/isaaclab.sh -p scripts/rsl_rl/train.py \
  --task Grasp1-UR5-Allegro-Teacher-v0 \
  --num_envs 4 --max_iterations 1 --headless
```

## Screenshot

`/home/windsky/project/Grasp1/logs/acceptance/2026-10-01/scene-close.png`

实际 Isaac Sim RGB 渲染图，已查看，清楚显示机械臂与灵巧手、黑色 Table、Object 和底座在近侧桌边的安装关系，无旧白色平台。

## 尚存问题

本次安装与训练验收未发现阻塞问题。截图日志仍有 Isaac Sim Fabric 对实例内部 mesh 路径的查询警告；截图已正常生成，物理、接触及训练验收通过。未修改该引擎行为。

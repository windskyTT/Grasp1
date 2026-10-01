# Codex 背景 Prompt：RobustDexGrasp Teacher → Grasp1 / IsaacLab 迁移

你正在维护一个 IsaacLab 外部项目 `Grasp1`。本文件是后续 Codex 任务的**长期背景与迁移约束**。

> 重要：本文件本身只提供背景。除非后续任务明确要求，否则不要主动创建、修改或删除任何文件。

---

# 1. 项目环境

项目根目录：

```text
/home/windsky/project/Grasp1
```

项目创建方式：

```bash
./isaaclab.sh --new
```

环境：

```text
IsaacLab: 2.3.2
Isaac Sim: 5.1.0
安装方式: Isaac Sim Pre-built Binaries
GPU: NVIDIA RTX 3070 Ti
NVIDIA Driver: 580
Python 环境: conda
RL 任务类型: Manager-Based
RL 库: RSL-RL
```

默认在代码目录下运行命令，并使用现有 conda 环境。

---

# 2. 迁移目标

原项目：

```text
路径: /home/windsky/project/RobustDexGrasp
```

当前只迁移：

```text
raisimGymTorch/raisimGymTorch/env/envs/allegro_teacher
```

目标：

```text
RobustDexGrasp Teacher
        ↓
IsaacLab Manager-Based Teacher
        ↓
UR5 + Allegro
```

第一阶段必须先完成：

```text
RobustDexGrasp Teacher
→ UR5 + Allegro IsaacLab Teacher
```

并验证：

```text
scene
action
observation
reward
reset
pre-grasp
IK
termination
success criterion
PPO training
```

全部行为基本等价后，才考虑：

```text
FR3 + Inspire
Student
Teacher-Student distillation
```

当前阶段**不要把 FR3 + Inspire 或 Student 一起迁移**。

---

# 3. Grasp1 目标项目结构

预期结构：

```text
Grasp1/
├── assets/
│   ├── robots/
│   │   ├── ur5_allegro/
│   │   └── fr3_inspire/
│   └── objects/
│       ├── new_training_set/
│       ├── shapenet-30obj/
│       └── dummy/
│
├── scripts/
│   ├── rsl_rl/
│   │   ├── train.py
│   │   ├── play.py
│   │   └── evaluate.py
│   └── tools/
│       └── get_lowest_point.py
│
├── source/
│   └── Grasp1/
│       └── Grasp1/
│           ├── robots/
│           │   ├── __init__.py
│           │   ├── robot_profile.py
│           │   ├── ur5_allegro_cfg.py
│           │   └── fr3_inspire_cfg.py
│           │
│           ├── data/
│           │   ├── __init__.py
│           │   └── object_set.py
│           │
│           ├── utils/
│           │   ├── __init__.py
│           │   └── paths.py
│           │
│           └── tasks/
│               └── manager_based/
│                   └── grasp1/
│                       ├── grasp1_env_cfg.py
│                       │
│                       ├── mdp/
│                       │   ├── __init__.py
│                       │   ├── actions.py
│                       │   ├── observations.py
│                       │   ├── rewards.py
│                       │   ├── events.py
│                       │   ├── terminations.py
│                       │   ├── geometry.py
│                       │   └── keypoints.py
│                       │
│                       └── config/
│                           ├── __init__.py
│                           │
│                           ├── ur5_allegro/
│                           │   ├── __init__.py
│                           │   ├── teacher_env_cfg.py
│                           │   ├── student_env_cfg.py
│                           │   └── agents/
│                           │       ├── __init__.py
│                           │       └── rsl_rl_ppo_cfg.py
│                           │
│                           └── fr3_inspire/
│                               ├── __init__.py
│                               ├── teacher_env_cfg.py
│                               ├── student_env_cfg.py
│                               └── agents/
│                                   ├── __init__.py
│                                   └── rsl_rl_ppo_cfg.py
│
└── logs/
```

当前 Teacher 迁移只关心：

```text
UR5 + Allegro
```

不要因为目录中存在 FR3 + Inspire 而提前实现它。

---

# 4. 核心迁移原则

## 4.1 不做逐文件机械翻译

RobustDexGrasp 是 RaiSim 架构。

IsaacLab 是 Manager-Based 架构。

因此：

```text
Environment.hpp
train.py
RaisimGymVecEnvOther.py
```

不能简单翻译成几个同名 Python 文件。

必须按照 IsaacLab 职责重新拆分。

---

## 4.2 `train.py` 使用 IsaacLab 项目自带版本

不要重写 RobustDexGrasp 的 PPO 训练循环。

目标：

```text
scripts/rsl_rl/train.py
```

直接使用 `./isaaclab.sh --new` 创建的标准版本。

RobustDexGrasp 原 `train.py` 中：

```text
PPO rollout loop
PPO update
checkpoint saver
NormalSampler
custom PPO class
```

不迁移。

只迁移其中的：

```text
object dataset logic
reset logic
pre-grasp logic
IK
reward shaping
evaluation semantics
PPO hyperparameters
```

对应拆到：

```text
data/object_set.py
mdp/events.py
mdp/geometry.py
mdp/rewards.py
agents/rsl_rl_ppo_cfg.py
```

---

# 5. MDP 文件职责

```text
actions.py
    Policy action 如何变成机器人关节 target

observations.py
    Policy / Critic 能看到什么

rewards.py
    强化学习 reward 如何计算

events.py
    reset、随机化、pre-grasp、IK 初始化

terminations.py
    episode 何时结束

geometry.py
    坐标变换、姿态、visible points、grasp geometry、IK helper

keypoints.py
    17 个 hand/body keypoints 和 arm keypoints 的批量计算

__init__.py
    对外导出 MDP term
```

其中：

```text
actions.py
observations.py
rewards.py
events.py
terminations.py
```

是 Manager-Based task 的直接 MDP 层。

```text
geometry.py
keypoints.py
```

是任务内部辅助模块。

---

# 6. RobustDexGrasp Teacher 的关键语义

## 6.1 Action

Teacher：

```text
action dimension = 22
```

含义：

```text
前 6 维
→ UR5

后 16 维
→ Allegro Hand
```

原 action 不是简单 absolute target，而是 residual-style：

```text
target = action * action_std + current/action_mean
```

并且：

```text
UR5 action std = rot_action_std = 0.005
Allegro action std = finger_action_std = 0.015
```

还存在：

```text
joint limit clipping
上一时刻 target
随机 1-step control delay
```

迁移时必须明确决定哪些行为需要保持。

---

# 7. Observation

RobustDexGrasp Teacher 最终：

```text
observation dimension = 153
```

这不是一个文件直接生成的。

结构：

```text
Environment.hpp
→ 102D base observation

RaisimGymVecEnvOther.observe_vision_new()
→ 额外加入 17 × 3 = 51D affordance vectors

102 + 51
= 153
```

`observe_vision_new()` 的核心逻辑：

```text
17 hand keypoints
        ↓
torch.cdist()
        ↓
每个 keypoint 找最近 affordance point
        ↓
nearest point - keypoint
        ↓
得到 17 个 3D vectors
        ↓
51D
```

因此迁移 observation 时：

> 不允许只保证维度等于 153，必须保证每一段 observation 的语义和坐标系尽量一致。

---

# 8. Reward

原 Reward 分布在两个地方：

```text
Environment.hpp
+
allegro_teacher/train.py
```

因此迁移时必须合并到：

```text
mdp/rewards.py
```

主要 reward 包括：

```text
affordance_reward
affordance_contact_reward
affordance_impulse_reward

table_reward
table_contact_reward
table_impulse_reward

arm_height_reward
arm_contact_reward
arm_impulse_reward
arm_collision_reward

push_reward

wrist_vel_reward_
wrist_qvel_reward_

obj_vel_reward_
obj_qvel_reward_
obj_displacement_reward

arm_joint_vel_reward_
```

其中尤其容易漏掉的是 `train.py` 中额外增加的：

```text
affordance_reward
table_reward
arm_height_reward
arm_collision_reward
```

原则：

```text
reward 数学公式
→ mdp/rewards.py

reward weight / coeff
→ teacher_env_cfg.py
```

不要在 `teacher_env_cfg.py` 中写复杂 Tensor 运算。

---

# 9. Reset / Events / Pre-grasp

Teacher reset 非常复杂，是迁移难度最高部分之一。

流程大致：

```text
选择 object
        ↓
读取 lowest_point_new.txt
        ↓
随机 object XY
        ↓
随机 yaw
        ↓
根据 lowest point 设置 Z
        ↓
获取 affordance point cloud
        ↓
计算 visible points
        ↓
计算 affordance center
        ↓
选择 approach direction
        ↓
sample_rot_mats()
        ↓
生成多个 wrist orientation candidates
        ↓
UR5 IK
        ↓
projection + wrist angle scoring
        ↓
选 best candidate
        ↓
设置 UR5 + Allegro 初始姿态
        ↓
reset simulation
        ↓
检查 collision
        ↓
fallback
```

主控制逻辑：

```text
mdp/events.py
```

数学和坐标变换：

```text
mdp/geometry.py
```

---

# 10. Object sampling

训练 object XY 的原始采样包括：

```text
uniform
```

以及：

```text
50% uniform
+
50% Beta(0.5, 0.5) edge-biased
```

采样区域基于：

```text
angle ∈ [-0.7π, -0.3π]
distance ∈ [0.45, 0.75]
x ∈ (-0.25, 0.25)
```

evaluation 使用 uniform sampling。

该逻辑属于：

```text
mdp/events.py
```

而不是 `train.py`。

---

# 11. Object dataset

训练数据集：

```text
new_training_set
```

评估还涉及：

```text
shapenet-30obj
```

以及：

```text
dummy
```

Teacher 训练脚本还会对部分困难物体增加采样次数，例如：

```text
037_scissors
off_water_body
019_pitcher_base
011_banana
mouse
hammer
small_block
```

对象目录可能包含：

```text
<object>.urdf
<object>_fixed_base.urdf

top_watertight_tiny.obj
top_watertight_tiny.stl

bottom_watertight_tiny.obj
bottom_watertight_tiny.stl

lowest_point_new.txt

可选：
<object>.npy
```

其中 `.npy` 主要与某些 stable-state evaluation 模式相关，不一定是标准训练必需。

---

# 12. Point cloud / affordance

旧 `RaisimGymVecEnvOther.py` 会为 object：

```text
load top_watertight_tiny.obj
load bottom_watertight_tiny.obj

sample 200 surface points
sample corresponding normals

compute affordance center
compute non-affordance center
```

原实现使用：

```text
trimesh
numpy
torch
```

最终 point cloud 搬到 CUDA。

IsaacLab 迁移时尽量避免 CPU/GPU 来回搬运。

能批量 Torch 化的部分应优先 Torch 化。

---

# 13. Keypoints

Allegro Teacher 使用：

```text
17 hand/body keypoints
```

其中包含：

```text
wrist / hand base
finger joints
finger tips
```

这些 keypoints 同时被：

```text
observation
affordance distance
reward
```

使用。

因此必须集中到：

```text
mdp/keypoints.py
```

不要在 `observations.py` 和 `rewards.py` 分别复制一套 link 查询逻辑。

---

# 14. Robot abstraction

`robot_profile.py` 的职责：

```text
joint names
link names
EEF frame
contact links
17 hand keypoints
arm links
fingertip links
hand center
```

目的：

```text
Teacher task logic
尽量不直接硬编码 UR5 / Allegro link name
```

这样后续才能把相同 task logic 替换到：

```text
FR3 + Inspire
```

但当前第一阶段只保证 UR5 + Allegro 正确。

---

# 15. Robot asset

原：

```text
rsc/ur5_allegro/ur5_allegro.urdf
```

以及 mesh：

```text
Flange_meshes/
allegro_meshes/
meshes/
platform_meshes/
ur5_meshes/
```

目标：

```text
assets/robots/ur5_allegro/
```

运行时推荐：

```text
USD
```

同时保留：

```text
URDF
```

即：

```text
assets/robots/ur5_allegro/
├── ur5_allegro.urdf
├── ur5_allegro.usd
└── meshes...
```

URDF → USD 时必须验证：

```text
joint names
joint limits
fixed joints
mass
inertia
collision
visual mesh
link names
```

不要假设转换后所有语义自动完全一致。

---

# 16. Object asset

每个 object 也建议：

```text
URDF 和 USD 放在同一个 object 文件夹
```

例如：

```text
assets/objects/new_training_set/011_banana/
├── 011_banana.urdf
├── 011_banana.usd
├── top_watertight_tiny.obj
├── top_watertight_tiny.stl
├── bottom_watertight_tiny.obj
├── bottom_watertight_tiny.stl
└── lowest_point_new.txt
```

---

# 17. Table

不要重新实现 RobustDexGrasp 中 RaiSim：

```cpp
world_->addBox(...)
```

作为最终桌子资产。

Grasp1 使用：

```text
NVIDIA Isaac Sim 远程资产
table_instanceable.usd
```

但迁移时必须保持任务所依赖的：

```text
tabletop height
robot/table/object relative coordinates
```

原 Teacher 逻辑大量使用：

```text
0.771
0.773
```

这与 tabletop 高度和 object 初始 Z 紧密相关。

不要机械保留常数而不检查新 table 的实际坐标。

---

# 18. `paths.py`

目标文件：

```text
source/Grasp1/Grasp1/utils/paths.py
```

作用：

```text
统一项目路径
避免 ../../../../..
避免硬编码 /home/windsky/project/Grasp1
```

设计原则：

```text
PROJECT_ROOT
通过 __file__ 推导

支持：
GRASP1_ROOT
环境变量覆盖

只使用 Python 标准库
```

预期至少提供：

```text
PROJECT_ROOT
PACKAGE_ROOT
SOURCE_DIR
ASSETS_DIR
ROBOTS_ASSETS_DIR
OBJECTS_ASSETS_DIR
SCRIPTS_DIR
LOGS_DIR
PACKAGE_DATA_DIR
TASKS_DIR

project_path()
robot_asset_dir()
object_dataset_dir()
object_asset_dir()
require_path()
```

后续文件优先使用该模块，不要重新写路径推导逻辑。

---

# 19. 全部迁移文件总表

下面按**迁移实现难度从低到高**排序，并使用固定编号 `01`–`31`。后续 Codex 任务可以直接用“迁移文件 07”“迁移文件 21”等方式引用表中项目；除非本背景 Prompt 被显式更新，否则不要自行改变这些编号。

|编号| Grasp1 中需要生成/修改的文件 | RobustDexGrasp 对应原文件 | 难度与迁移内容|
|--| ---------------- | ------------------------- | ----------- |
|1| `source/Grasp1/Grasp1/robots/__init__.py`| **无直接对应原文件**| ★ 极低。IsaacLab Python 包导出文件，导出 `robot_profile.py`、`ur5_allegro_cfg.py`。|
|2| `source/Grasp1/Grasp1/data/__init__.py`| **无直接对应原文件**| ★ 极低。导出 object dataset 配置。|
|3| `source/Grasp1/Grasp1/utils/__init__.py`| **无直接对应原文件**| ★ 极低。导出路径工具。|
|4| `source/Grasp1/Grasp1/tasks/manager_based/grasp1/mdp/__init__.py`| **无直接对应原文件**| ★ 极低。统一导出 `actions / observations / rewards / events / terminations / geometry / keypoints`。|
|5| `source/Grasp1/Grasp1/tasks/manager_based/grasp1/config/__init__.py`| **无直接对应原文件** | ★ 极低。IsaacLab config package 文件。|
|6| `source/Grasp1/Grasp1/tasks/manager_based/grasp1/config/ur5_allegro/agents/__init__.py`| **无直接对应原文件**| ★ 极低。导出 RSL-RL runner 配置。|
|7| `source/Grasp1/Grasp1/utils/paths.py`| `allegro_teacher/train.py` **60–75 行**；`visual_eval.py`、`quantitative_eval.py` 中对应的 `task_path/home_path` 路径部分| ★ 极低。把旧代码到处使用的 `../../../../..` 相对路径统一成 Grasp1 路径解析。|
|8| `scripts/tools/get_lowest_point.py`| `rsc/get_lowest_point.py` **整文件**| ★ 极低。基本可直接迁移；只改 dataset 根路径。|
|9| `assets/robots/ur5_allegro/ur5_allegro.urdf`| `rsc/ur5_allegro/ur5_allegro.urdf` **整文件**| ★ 极低。原始机器人模型直接保留。|
|10| `assets/robots/ur5_allegro/Flange_meshes/*`、`allegro_meshes/*`、`ur5_meshes/*`、`platform_meshes/*` | `rsc/ur5_allegro/` 下对应 mesh 目录 **整目录**| ★ 极低。模型资源直接迁移，主要检查 URDF 中 mesh 相对路径。|
|11| `assets/objects/new_training_set/<object>/*`| `rsc/new_training_set/<object>/*` **整文件/整目录**| ★ 极低。包括 `<object>.urdf`、`top_watertight_tiny.obj/.stl`、`bottom_watertight_tiny.obj/.stl`、`lowest_point_new.txt`。|
|12| `assets/objects/shapenet-30obj/<object>/*`| `rsc/shapenet-30obj/<object>/*` **整文件/整目录**| ★ 极低。用于 Teacher quantitative evaluation。|
|13| `assets/objects/dummy/*`| `rsc/dummy/*` **整目录**| ★ 极低。保留测试资源。|
|14| `source/Grasp1/Grasp1/data/object_set.py`| `allegro_teacher/train.py` **85–125 行**（训练物体集合、重复采样、困难物体加权）；**240–250 行**（`lowest_point_new.txt`）；`RaisimGymVecEnvOther.py` **45–97 行**（top/bottom mesh、200 点采样、affordance center）| ★★ 低。把“数据集是什么、每个物体有哪些 metadata”从旧 `train.py` 和 VecEnv 中独立出来。|
|15| `source/Grasp1/Grasp1/robots/robot_profile.py`| `hardware/arm/UR5Sim.cpp` **63–110 行**、**145–157 行**；`hardware/hand/AllegroSim.cpp` **109–166 行**；`cfg_reg.yaml` **31–32 行**| ★★ 低。提取 UR5/Allegro 的 joint、link、EE、fingertip、contact link、17 keypoint、`hand_center` 等语义名称。**不迁移 C++ 类结构。**|
|16| `source/Grasp1/Grasp1/tasks/manager_based/grasp1/config/ur5_allegro/agents/rsl_rl_ppo_cfg.py`| `cfgs/cfg_reg.yaml` **121–124 行**（`[128,128]` Actor/Critic）；`allegro_teacher/train.py` **177–208 行**（网络、PPO epochs、gamma、lambda、mini-batches 等）| ★★ 低。只迁移超参数，不迁移 RobustDexGrasp 的 `ppo.py/module.py`。|
|17| `source/Grasp1/Grasp1/tasks/manager_based/grasp1/mdp/terminations.py`| `Environment.hpp` **1065–1089 行**；`cfg_reg.yaml` **62–63 行**| ★★ 低。迁移非法 hand height、NaN/仿真异常检测；正常 episode 长度交给 IsaacLab `time_out`。第一版不建议额外新增 drop termination。|
|18| `source/Grasp1/Grasp1/tasks/manager_based/grasp1/config/ur5_allegro/__init__.py`| **无直接对应原文件**；任务参数来源于 `cfg_reg.yaml`| ★★ 低。进行 Gym/IsaacLab task registration，把 Teacher EnvCfg 和 RSL-RL runner cfg 注册给标准 `train.py`。|
|19| `assets/robots/ur5_allegro/ur5_allegro.usd`| `rsc/ur5_allegro/ur5_allegro.urdf` + 对应全部 meshes| ★★ 低～中。使用 Isaac Sim 导入/转换；需要验证 joint、fixed joint、collision、mass/inertia。最终训练直接加载 USD。|
|20| `assets/objects/<dataset>/<object>/<object>.usd`| 对应 `<object>.urdf` + OBJ/STL mesh **整文件**| ★★ 低～中。建议批量转换。URDF 与 USD 保持在同一个 object 文件夹中。|
|21| `source/Grasp1/Grasp1/robots/ur5_allegro_cfg.py`| `cfg_reg.yaml` **5–32 行**；`hardware/arm/UR5Sim.cpp` **整文件**；`hardware/hand/AllegroSim.cpp` **整文件**；`hardware/arm/UR5Identification_id5hz.txt` **整文件**；`hardware/hand/Allegrotemp.txt` **整文件**；`rsc/ur5_allegro/ur5_allegro.urdf` **整文件**| ★★★ 中。定义 `ArticulationCfg`、初始 joint pose、UR5/Allegro actuator、stiffness/damping、USD 路径。两个 `.txt` 的 PD 参数应转换为 IsaacLab actuator 参数，**不再运行时读取 txt**。|
|22| `source/Grasp1/Grasp1/tasks/manager_based/grasp1/grasp1_env_cfg.py`| `Environment.hpp` **27–91 行**（world、ground、materials、robot、table、base pose）；**297–329 行**（object load 基本属性）；`cfg_reg.yaml` **52–72 行**（dt、control dt、episode、dataset、action std 等）| ★★★ 中。定义共享 scene/simulation 基础结构。原 RaiSim box table 不照搬，替换为你确定的 NVIDIA `table_instanceable.usd`；ground 使用 IsaacLab ground plane。|
|23| `source/Grasp1/Grasp1/tasks/manager_based/grasp1/config/ur5_allegro/teacher_env_cfg.py`| `cfg_reg.yaml` **34–119 行**；`train.py` **164–175 行**；`Environment.hpp` **218–247 行**| ★★★ 中。负责把 Teacher 的 Actions/Observations/Rewards/Events/Terminations 组合起来。主要是**配置和 wiring**，不要在这里写复杂 Tensor 算法。|
|24| `scripts/rsl_rl/play.py`| `allegro_teacher/visual_eval.py` **整文件作为行为参考**| ★★★ 中。保留“加载 Teacher checkpoint、选物体、reset、rollout、可视化”的语义；RaiSim VecEnv、旧 PPO loader、Unity 接口全部换成 IsaacLab/RSL-RL 标准接口。|
|25| `scripts/rsl_rl/evaluate.py` **建议新增**| `allegro_teacher/quantitative_eval.py` **整文件作为行为参考**| ★★★ 中。迁移 ShapeNet-30obj、repeat evaluation、lift success、per-object success rate 和总体 success rate。比 `play.py` 更复杂，但不属于 MDP 核心。                                               |
|26| `source/Grasp1/Grasp1/tasks/manager_based/grasp1/mdp/actions.py`| `Environment.hpp` **223–246 行**（joint limits、action std）；**600–640 行**（residual action → target、clip、control delay、PD target）；`cfg_reg.yaml` **69–72 行**| ★★★ 中。必须保持原来的 **22D = 6 UR5 + 16 Allegro** 语义，并决定是否完全复刻 residual target、上一时刻状态作为 action mean、1-step delay。|
|27| `source/Grasp1/Grasp1/tasks/manager_based/grasp1/mdp/keypoints.py`| `hardware/hand/AllegroSim.cpp` **149–166 行**；`hardware/arm/UR5Sim.cpp` **156–157 行**；`Environment.hpp` **938–971 行**；`RaisimGymVecEnvOther.py` **192–210 行**| ★★★★ 中～高。统一计算 `[num_envs,17,3]` hand keypoints、arm keypoints、world/object frame 坐标，供 observation/reward 共用。|
|28| `source/Grasp1/Grasp1/tasks/manager_based/grasp1/mdp/geometry.py`| `helper/initial_pose_final.py` **整文件**；`helper/inverseKinematicsUR5.py` **整文件作为 IK 语义参考**；`allegro_teacher/train.py` **305–493 行**；`Environment.hpp` **900–1030 行** 中坐标变换部分| ★★★★ 高。负责 visible points、object/world/wrist/UR5-base 变换、approach direction、`sample_rot_mats()`、projection score、candidate pose 和 IK 输入。应尽可能改成批量 Torch/IsaacLab math。|
|29| `source/Grasp1/Grasp1/tasks/manager_based/grasp1/mdp/observations.py`| `Environment.hpp` **745–1030 行**；`RaisimGymVecEnvOther.py` **183–212 行**；`train.py` **164–165 行**| ★★★★★ 高。必须恢复原 Teacher **153D observation 的语义**。其中 `Environment.hpp` 本身是 **102D base observation**，`observe_vision_new()` 再加入 **17×3=51D affordance vector**，最终才是 153D。|
|30| `source/Grasp1/Grasp1/tasks/manager_based/grasp1/mdp/rewards.py`| `Environment.hpp` **642–739 行**；`allegro_teacher/train.py` **214–223 行**（finger weights）；**608–629 行**（额外 affordance/table/arm-height/arm-collision reward）；`cfg_reg.yaml` **75–119 行**| ★★★★★ 高。必须把原来分散在 C++ environment 和 Python `train.py` 的 reward **合并**。这是最容易漏迁的地方之一。reward 公式放这里，weight 放 `teacher_env_cfg.py`。|
|31| `source/Grasp1/Grasp1/tasks/manager_based/grasp1/mdp/events.py`| `Environment.hpp` **381–550 行**（reset/reset_state）；`allegro_teacher/train.py` **297–548 行**（finger init、object pose、visible point、pre-grasp IK、best IK、collision fallback）；**567–603 行**（biased object position）；`cfg_reg.yaml` **35–50 行**；并调用 `initial_pose_final.py` 与 `inverseKinematicsUR5.py` 的迁移结果 | ★★★★★ **最高**。这是整个迁移最复杂的文件。包含 object XY/yaw sampling、lowest-point Z、uniform/edge-biased sampling、pre-grasp generation、IK、多 candidate scoring、机器人/物体 reset、初始碰撞检查及 fallback。 |

---

# 20. 原文件与目标文件的关键映射

```text
Environment.hpp
    ↓
grasp1_env_cfg.py
actions.py
observations.py
rewards.py
events.py
terminations.py
geometry.py
keypoints.py
ur5_allegro_cfg.py

cfg_reg.yaml
    ↓
grasp1_env_cfg.py
teacher_env_cfg.py
ur5_allegro_cfg.py
rsl_rl_ppo_cfg.py

train.py
    ↓
data/object_set.py
events.py
geometry.py
rewards.py
rsl_rl_ppo_cfg.py
evaluate.py

RaisimGymVecEnvOther.py
    ↓
object_set.py
observations.py
keypoints.py
geometry.py

initial_pose_final.py
    ↓
geometry.py

inverseKinematicsUR5.py
    ↓
geometry.py
events.py

UR5Sim.cpp
AllegroSim.cpp
hardware.hpp
    ↓
robot_profile.py
ur5_allegro_cfg.py

UR5Identification_id5hz.txt
Allegrotemp.txt
    ↓
ur5_allegro_cfg.py
```

---

# 21. 不直接迁移的 RaiSim 框架文件

以下属于 RaiSim runtime / framework abstraction：

```text
RaisimGymEnv.hpp
VectorizedEnvironment.hpp
Reward.hpp
raisim_gym.cpp
Yaml.hpp
Yaml.cpp
activation.raisim
```

不要在 Grasp1 中创建对应文件。

替代关系：

```text
RaisimGymEnv / VectorizedEnvironment
→ IsaacLab ManagerBasedRLEnv / vectorized simulation

Reward.hpp
→ IsaacLab RewardManager

YAML runtime config
→ IsaacLab configclass

activation.raisim
→ 完全不需要
```

---

# 22. 不迁移旧 PPO 实现

不要复制：

```text
raisimGymTorch/algo/ppo/ppo.py
raisimGymTorch/algo/ppo/module.py
```

使用：

```text
RSL-RL
```

只从旧代码提取：

```text
policy network shape
value network shape
gamma
lambda
num learning epochs
mini batch count
rollout length
minimum action std 等必要训练参数
```

如果 RSL-RL 与旧实现的参数含义不同，应按 RSL-RL API 正确映射，不要机械抄值。

---

# 23. 原 Teacher 关键训练参数

已知关键参数：

```text
policy_net = [128, 128]
value_net = [128, 128]

observation = 153
action = 22

grasp_steps = 70

num_learning_epochs = 4
gamma = 0.996
lambda = 0.95
num_mini_batches = 4
shuffle_batch = False
```

环境：

```text
simulation_dt = 0.01
control_dt = 0.2
max_time = 4.0
```

注意：

```text
grasp_steps = 70
```

和：

```text
max_time / control_dt
```

并不简单对应。

迁移时不要直接假设 episode length 等价，需要按照原训练循环的实际 rollout 语义检查。

---

# 24. 原 Reward 系数

Teacher 原配置包含：

```text
affordance_reward            0.5
affordance_contact_reward    1.5
affordance_impulse_reward    1.0

table_reward                -0.03
table_contact_reward        -1.0
table_impulse_reward        -0.5

arm_height_reward           -0.05
arm_contact_reward          -0.1
arm_impulse_reward          -0.1
arm_collision_reward        -1.0

push_reward                 -0.0

wrist_vel_reward_           -1.0
wrist_qvel_reward_          -0.1

obj_vel_reward_             -15.0
obj_qvel_reward_            -0.2
obj_displacement_reward     -5.0

arm_joint_vel_reward_       -1.0
```

必须区分：

```text
原始 reward value
×
cfg coeff
=
最终 contribution
```

---

# 25. 原 Hardware 参数

Teacher hardware：

```text
arm_type = ur5
hand_type = allegro
kinematic_type = simfk

init_finger_pose =
[
  0.2, 0.6, 0.2, 0.5,
  0.2, 0.6, 0.2, 0.5,
  0.2, 0.6, 0.2, 0.5,
  1.3, 0.0, -0.1, 0.2
]

hand_center =
[-0.0091, 0.0, -0.095]

table_friction = 0.2
```

PD 参数来源：

```text
hardware/arm/UR5Identification_id5hz.txt
hardware/hand/Allegrotemp.txt
```

IsaacLab 中：

```text
不要运行时读取旧 txt
```

应提取数值并写入：

```text
ur5_allegro_cfg.py
```

作为 actuator stiffness/damping 等配置。

---

# 26. 性能原则

IsaacLab 迁移后尽量：

```text
Torch tensor
GPU batch processing
```

而不是：

```text
Python for-loop over every env
numpy
cpu()
numpy()
再搬回 cuda
```

尤其：

```text
keypoint transforms
distance calculations
reward
observation
object reset
```

尽可能 GPU vectorized。

但第一目标是：

```text
正确复现语义
```

然后再优化性能。

不要为了 vectorization 改变 task behavior。

---

# 27. `teacher_env_cfg.py` 与 `grasp1_env_cfg.py`

分工：

```text
grasp1_env_cfg.py
    通用 Scene / Simulation / 基础 ManagerBased Env 配置

teacher_env_cfg.py
    Teacher 专属：
    actions
    observations
    rewards
    events
    terminations
    task-specific parameters
```

不要把所有内容堆进一个文件。

---

# 28. `robot_profile.py` 与 `ur5_allegro_cfg.py`

分工：

```text
robot_profile.py
    语义：
    哪些 joint
    哪些 body
    哪些 fingertip
    哪些 contact link
    EEF 名称
    keypoint 名称

ur5_allegro_cfg.py
    物理与 IsaacLab asset config：
    USD
    initial state
    actuators
    stiffness
    damping
    effort
    velocity
```

---

# 29. `object_set.py` 与 `paths.py`

分工：

```text
paths.py
    项目 filesystem 路径

object_set.py
    数据集逻辑：
    object names
    difficult-object weighting
    object metadata
    lowest point
    top/bottom mesh
    USD/URDF path
```

不要把 object dataset 业务逻辑塞到 `paths.py`。

---

# 30. `geometry.py` 与 `events.py`

分工：

```text
geometry.py
    无状态数学函数
    transform
    quaternion
    rotation matrix
    projection
    visible point helper
    grasp candidate
    IK helper

events.py
    管理 reset 流程
    调用 geometry
    写入 simulation state
```

不要让 `geometry.py` 直接承担完整 episode reset orchestration。

---

# 31. `keypoints.py` 与 `observations.py`

分工：

```text
keypoints.py
    从 articulation / scene 中取得 keypoint positions

observations.py
    根据 keypoints + object state 组成 policy observation
```

尤其：

```text
17 keypoints × 3D affordance vector
```

应通过 `keypoints.py` 的统一结果计算。

---

# 32. Termination 原则

第一阶段以原 Teacher 行为为准。

不要为了“更合理”随意增加：

```text
object dropped
workspace violation
early success
joint near limit
```

等新的 early termination。

否则会改变：

```text
rollout distribution
reward distribution
training dynamics
```

先做行为等价，再考虑改进。

---

# 33. Evaluation

原 Teacher 有：

```text
visual_eval.py
quantitative_eval.py
```

IsaacLab 目标：

```text
scripts/rsl_rl/play.py
scripts/rsl_rl/evaluate.py
```

其中 `evaluate.py` 负责：

```text
multiple objects
multiple trials
lift test
success threshold
per-object success rate
overall success rate
```

原代码 lift success 核心语义是：

```text
object z - initial object z > 0.1
```

如果迁移时坐标基准变化，应保持“抬高 0.1 m”的语义，而不是死用旧 global-state index。

---

# 34. Checkpoint

旧 checkpoint，例如：

```text
teacher_ckpt/full_12500_r.pt
```

不保证能直接加载到 RSL-RL。

原因：

```text
旧 Actor/Critic class
旧 distribution class
state_dict key
observation normalization
action distribution
```

均可能不同。

第一阶段默认：

```text
重新训练 IsaacLab Teacher
```

不要为了直接加载旧 checkpoint 而破坏架构。

如未来需要 checkpoint converter，必须先保证：

```text
observation order
action semantics
network architecture
distribution semantics
```

完全对应。

---

# 35. 迁移实施顺序建议

推荐按以下顺序工作：

```text
1. Python package skeleton
2. paths.py
3. asset directories / URDF / meshes
4. object_set.py
5. robot_profile.py
6. rsl_rl_ppo_cfg.py
7. terminations.py
8. robot USD / object USD
9. ur5_allegro_cfg.py
10. grasp1_env_cfg.py
11. actions.py
12. keypoints.py
13. geometry.py
14. observations.py
15. rewards.py
16. events.py
17. teacher_env_cfg.py
18. task registration
19. play.py
20. evaluate.py
21. standard train.py integration test
```

虽然 `teacher_env_cfg.py` 文件本身不复杂，但最好等 MDP term 基本存在后再完成 wiring。

---

# 36. Codex 修改文件时的默认规则

每次收到后续具体任务时：

1. 先检查目标文件是否已经存在。
2. 如果已有内容，优先做最小修改。
3. 不因为当前任务而提前创建后续阶段文件。
4. 不修改与任务无关的文件。
5. 不随意重命名已有目录。
6. 不随意改变资产路径。
7. 不把 RaiSim API 带入 IsaacLab。
8. 不加入未请求的 ROS、相机、传感器或 domain randomization。
9. 不为了“优化”改变原 Teacher reward/reset/observation 语义。
10. 代码优先使用 IsaacLab 2.3.2 API。
11. 使用 Torch 完成批量 tensor 运算。
12. 避免不必要的 NumPy/GPU 数据来回转换。
13. 保持类型注解和 docstring 简洁。
14. 如果某个原始行为无法在 IsaacLab 中一一对应，明确注释差异，不要静默改变。
15. 后续如果需要查 RobustDexGrasp 原代码，优先确认原始函数的真实行为，而不是凭名称猜测。

---

# 37. 当前绝对禁止事项

除非后续明确要求，不要：

```text
重写 scripts/rsl_rl/train.py

复制 RaiSim C++ framework

复制旧 PPO implementation

创建 activation.raisim

实现 Student

实现 FR3 + Inspire Teacher

实现 teacher-student distillation

修改 IsaacLab 源码

修改 Isaac Sim 安装目录

安装新依赖

删除旧 URDF

只保留 USD 而删除原始 mesh

用 hard-coded /home/windsky/... 路径

使用 Path.cwd() 作为项目根目录依据
```

---

# 38. 后续任务执行格式

当后续 prompt 要求创建一个具体文件时：

```text
只完成该文件及其明确要求的必要改动。
```

完成后报告：

```text
修改了哪些文件
核心实现了什么
使用了哪些已有模块
是否存在需要后续文件配合的部分
是否通过基本语法检查
```

不要主动扩展任务范围。

---

# 39. 最终迁移目标

最终期望：

```text
python scripts/rsl_rl/train.py \
    --task <Grasp1 UR5 Allegro Teacher Task ID> \
    --headless
```

即可通过 IsaacLab 标准 RSL-RL pipeline 训练 Teacher。

最终架构应满足：

```text
标准 train.py
      ↓
task registration
      ↓
teacher_env_cfg.py
      ↓
ManagerBasedRLEnv
      ↓
actions / observations / rewards / events / terminations
      ↓
UR5 + Allegro + Object + Table
      ↓
RSL-RL PPO
```

而不是重新复制一套 RobustDexGrasp/RaiSim runtime。

---

# 40. 最重要的验收原则

迁移完成的判断标准不是：

```text
“代码能跑”
```

而是至少同时满足：

```text
1. UR5 + Allegro asset 语义正确
2. 22D action 语义正确
3. 153D Teacher observation 语义正确
4. 17 keypoints 定义正确
5. affordance point cloud 语义正确
6. reward 公式和权重基本等价
7. reset sampling 基本等价
8. pre-grasp candidate generation 基本等价
9. IK / best candidate selection 基本等价
10. collision fallback 基本等价
11. episode / rollout 长度语义明确
12. lift success 判定等价
13. RSL-RL 能正常训练和保存 checkpoint
14. 多环境 GPU simulation 正常运行
```

任何一个关键语义被悄悄改掉，都不能视为完整迁移。


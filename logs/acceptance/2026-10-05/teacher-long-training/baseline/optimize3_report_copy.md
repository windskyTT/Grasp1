# 优化3执行报告（2026-10-05）

已完成 Phase 0–7、Phase 8 条件判断与失败诊断。100 iteration 后 ShapeNet source / strict 均为 **0/30**，按文档要求未执行500 iteration。当前策略已经接近并接触部分物体，尚未形成可通过抬升测试的抓取。

验收证据：`logs/acceptance/2026-10-05/teacher-convergence-fix/`。全程使用已有 Conda `grasp`，原 Task `Grasp1-UR5-Allegro-Teacher-v0`。120/60 Hz、decimation=2、4秒回合、TGS、robot 32/1、object 16/0、self collision=False、22动作/153观测、rollout=32及 runtime cache 保留。

## 1. 修改文件列表

| 文件（相对仓库根目录） | 当前修改 |
|---|---|
| `source/Grasp1/Grasp1/tasks/manager_based/grasp1/config/ur5_allegro/teacher_env_cfg.py` | 物体零PD、armature startup、动作类与scale、奖励时间权重 |
| `source/Grasp1/Grasp1/tasks/manager_based/grasp1/mdp/events.py` | startup写入物体关节armature |
| `source/Grasp1/Grasp1/tasks/manager_based/grasp1/mdp/actions.py` | 轻量Relative Action子类，仅clamp最终target |
| `source/Grasp1/Grasp1/tasks/manager_based/grasp1/mdp/__init__.py` | 导出动作类 |
| `source/Grasp1/Grasp1/tasks/manager_based/grasp1/grasp1_env_cfg.py` | 共享源控制周期0.2秒 |
| `source/Grasp1/Grasp1/tasks/manager_based/grasp1/config/ur5_allegro/agents/rsl_rl_ppo_cfg.py` | gamma物理时间等效表达式 |
| `source/Grasp1/Grasp1/tasks/manager_based/grasp1/mdp/rewards.py` | 更新时间语义说明，奖励函数未修改 |
| `scripts/rsl_rl/evaluate.py` | source / strict双成功指标 |

另新增本报告及验收目录中的测量脚本、baseline、阶段diff、JSON/CSV与日志。训练脚本本体未修改；验收包装器执行现有train/evaluate，并记录本轮要求的数据。

Phase 0 保存当前HEAD `7b0a2ebdfbb4ee6dcbe14d457d89056f8d013655`、git status、空tracked worktree diff、环境/机器人/PPO配置、旧资产检查与最终evaluation。原有未跟踪的 `docs/优化3.md` 保留。

## 2. Object joint root cause

66个资产（35训练、30 ShapeNet、1 dummy）的source URDF限位均为0–0.001 rad；USD为0–0.0572957806 degree；runtime为0–0.00100000005 rad。单位转换正确。USD角限位单位见 [OpenUSD定义](https://openusd.org/release/api/class_usd_physics_revolute_joint.html)。

拓扑为单一bottom articulation root、bottom → rotation → top，两个rigid bodies、一个revolute joint；负Z轴由USD关节局部旋转表示，没有发现双根或自由脱离刚体。完整URDF字段、USD属性/关系与刚体质量惯量见 `phase1-free-baseline.assets.json`。

历史配置复现得到关节范围 **-0.271607～+0.180905 rad**，与优化2记录一致。自由落体无此严重越限，接触条件下才出现。从这些对照判断，主要问题是接触条件下近固定关节的求解稳定性。典型dummy bottom质量0.00098 kg、惯量约1.63e-8 kg·m²；还发现USD导入器生成了非零acceleration drive，runtime stiffness约35809.86、damping约57.30，而源 `Environment.hpp` 显式使用零PD增益。

## 3. Object joint fix

物体仍为原Articulation，限位仍为0–0.001 rad。生成场景时清除导入器drive的PD增益，startup为rotation写入 **armature=0.1 kg·m²**。armature增加关节空间有效惯量，是本轮近固定关节的数值稳定化表示；两刚体自身质量、惯量、collision、接触过滤和affordance frame保持原值。原URDF/USD资产文件未改。

隔离对照结果：

| 对照 | 最小q rad | 最大q rad | 判定 |
|---|---:|---:|---|
| 原配置，16 env | -0.288097 | 0.160547 | 严重越限 |
| 仅清除drive | -0.542630 | 0.145449 | 无效 |
| object 16/1 | -0.288097 | 0.160547 | 无改善，保留16/0 |
| 全局contact-last | -0.000136 | 0.001142 | 手指速度约2742 rad/s，未采用 |
| armature=0.0001 | -0.128610 | 0.062763 | 不足 |
| 零PD + armature=0.01，全35训练物体 | -0.000681 | 0.001022 | gun/scissors仍有越限 |
| 零PD + armature=0.1，全35训练物体 | -0.00000971 | 0.00100323 | 专项通过 |

训练、ShapeNet、dummy全物体池300步检查通过；三个典型物体分别完成自由落体、无机器人接触的桌面场景与人工接触，逐物理步记录至1/4/10秒。典型物体专项最大越限1.01e-5 rad。cracker人工接触峰值11.71 N；Blue_camera几何重叠压力测试峰值约1899 N，该峰值属于人工fixture，不是训练场景。

reset、partial reset、top pose/velocity奖励和153维观测均可用，相关数据有限。`phase1/gate.json`记录专项2e-5 rad容差；100 iteration扩展样本范围为 **-2.6023832e-05～0.0010466449 rad**，最大剩余越限 **4.6645e-5 rad**。严重越限已消除，仍存在小量数值误差；100 iteration样本不满足专项2e-5的更严容差，未将其写成严格零越限。

## 4. Action target fix

`TeacherRelativeJointPositionAction`继承官方Relative Action，每个物理子步执行 `target = clamp(q_current + processed_action, lower, upper)`。不引入delay、previous-target或子步切换。

22关节按源顺序保持映射。current / near upper / near lower × zero / +1 / -1 ×22，共 **198组**通过。16 env/300 steps同seed对照：越限target比例 **1.17661% → 0%**，contact峰值两者均373.869 N；观测153维、partial reset残差清零与有限性检查通过。

## 5. Robot velocity分析

精确复现优化2流程，**29.9802704 rad/s来自Allegro `joint_15_0`**，不是UR5。逐22关节结果见 `historical-velocity-by-joint.json` 与 `velocity-by-joint.csv`。

PhysX Tensor API读取到UR5前三关节3.15、后三关节3.2、全部Allegro 7 rad/s，配置已加载。`velocity_limit_sim`是求解器中的速度约束/制动参数，不是每个输出状态的最终硬截断；耦合驱动、位置限位和接触的迭代求解会产生残余越限。依据：[PhysX约束求解顺序](https://nvidia-omniverse.github.io/PhysX/physx/5.6.1/docs/Articulations.html)及本机IsaacLab `actuator_base_cfg.py`。

恢复target clamp后，1.0×scale的物理子步峰值24.5395 rad/s仍在joint_15_0；zero-action峰值7.30276 rad/s。未通过直接改写qvel来掩盖物理状态。后续采用的0.25×样本UR5峰值2.40931、Allegro峰值6.02800 rad/s。

## 6. Action scale选择

16 env、300 steps、seed=42、独立action seed=20261005、uniform[-1,1]；只改变scale。1.0×复用Phase2同配置/动作序列的运行证据。

| factor | UR5 / Allegro scale | 物理子步max qvel rad/s | mean qvel rad/s | max contact N | max impulse N·s | target clipping | max target delta rad |
|---|---|---:|---:|---:|---:|---:|---:|
| 0.25× | 0.00125 / 0.00375 | 6.0280 | 0.05112 | 718.787 | 5.98989 | 1.30966% | 0.0234713 |
| 0.5× | 0.0025 / 0.0075 | 21.2019 | 0.07067 | 360.274 | 3.00228 | 1.11695% | 0.0514936 |
| 1.0× | 0.005 / 0.015 | 24.5395 | 0.10896 | 373.869 | 3.11558 | 1.14157% | 0.129148 |

三档target outside均0；最大effort150 N·m；invalid_hand_height/invalid_observation均0，奖励项已记录。最终选择 **0.25×**，依据是逐物理步速度峰值最低；其contact峰值较高是明确取舍。clamp修正已越界实际q时，target delta可能大于action scale。

该选择是运行数据支持的带宽调整，并非严格恢复旧5 Hz动作时间等效；官方Relative Action仍每个物理子步重算目标。后续学习结果单独验证。

## 7. Reward时间语义选择

选择 **per-simulated-time**：`weight = source_coeff / SOURCE_CONTROL_DT`，SOURCE_CONTROL_DT=0.2秒；RewardManager再乘1/60秒，每步系数成为源每步的1/12。奖励公式、符号、接触定义及各项相对权重保持。

同seed、相同action sequence的A/B物体q极值、逐关节速度极值与终止统计完全一致；全部17项的step/second/completed-episode统计均按1/12缩放。

| 总奖励统计 | per-step实验 | per-time实验（最终） |
|---|---:|---:|
| mean / policy step | -0.8698223 | -0.0724852 |
| mean / simulated second | -52.1893362 | -4.3491115 |
| completed 4s episode mean | -195.7668013 | -16.3138993 |

重点六项和其余奖励详见 `phase4-comparison.json`，包含17项三种时间单位。实验脚本当前可用 `--reward_semantics step` / `--reward_semantics time` 显式选择。

## 8. Gamma/Lambda选择

`gamma = 0.996 ** ((1/60) / 0.2)`，实际 **0.9996660539892578**，保留旧约49.8999秒的指数折扣时间常数；直接沿用0.996时仅约4.1583秒。

lambda时间候选为 **0.9957346812224394**，本轮仍保留 **0.95**，隔离gamma。PPO-A gamma=0.996 / PPO-B时间等效gamma均使用lambda=0.95，各16 env/5 iteration，checkpoint与17项TensorBoard奖励均finite。选择B依据是时间等效；5 iteration是短训运行对照。

lambda仍对应约0.32493秒的独立衰减时间，选定gamma×lambda的GAE trace约 **0.322827秒**；rollout物理时长0.53333秒。GAE时间尺度未完全恢复源值，保留为下一阶段待验证项。网络、LR配置、entropy、clip、epochs、mini-batches与desired KL均保持。

## 9. Evaluate双指标

source：`height_gain > success_height`；strict：`active AND source_success`。console打印两种成功数；JSON新增 `source_lift_success_rate` / `strict_lift_success_rate`；CSV新增source/strict次数及source rate，既有successes/overall_success_rate仍表示strict。

5 iteration检查点完成30物体1 round的真实评估，JSON/CSV一致；100 iteration最终评估及接触诊断复评亦一致。

## 10. 100 iteration结果

训练通过验收包装器执行现有train.py，参数为task原ID、num_envs=2048、max_iterations=100、run_name=teacher_stage100、headless。

Run：`logs/rsl_rl/grasp1_ur5_allegro_teacher/2026-10-05_13-23-01_teacher_stage100`。最终checkpoint：`model_99.pt`（0起始编号，共100 iteration）。共6,553,600 env steps；scene creation227.857秒；训练阶段1523.82秒。

| 训练指标 | 均值 / 最终值 |
|---|---|
| Collection time | 平均14.9313秒/iteration |
| Learning time | 平均0.2684秒/iteration |
| FPS | 平均4318；最终4354 |
| Mean reward | 最终-18.011684 |
| Mean episode length | 最终240步（4秒） |
| Value loss | 最终1.205972；峰值141.428619（iteration40） |
| Surrogate loss | 最终0.023896 |
| Entropy | 最终28.880312 |
| Policy std | 最终0.909667 |
| Adaptive LR | 最终3.4254874e-05；曾有74次记录在0.01上限 |
| Terminations | timeout=27324；invalid_hand_height=0；invalid_observation=0 |
| NVML VRAM | 峰值5638 MiB；后半训练均为5638 MiB，无持续增长 |
| Finite | checkpoint、所有TensorBoard scalars、actions/obs/reward/关节状态通过 |

计时包含验收包装器的物理子步关节有限性与限位采集，不作为性能优化结果。LR配置仍为5e-4，表中运行值变化来自既有adaptive schedule。

迭代10–29平均奖励-17.666228，70–99为-17.001753，存在波动；最初回合长度被随机初始化，不能用iteration0的短回合奖励直接比较成熟回合。

17项奖励的TensorBoard统计（Episode_Reward按回合时长归一化，区别于上面的单步统计）：

| Term | 最终 | 100次记录均值 |
|---|---:|---:|
| affordance_reward | -3.1479 | -3.19698 |
| affordance_contact_reward | 0.108204 | 0.119066 |
| affordance_impulse_reward | 0.0580447 | 0.0627569 |
| table_reward | -0.139667 | -0.146207 |
| table_contact_reward | -0.0149207 | -0.0255587 |
| table_impulse_reward | -0.018807 | -0.0312307 |
| arm_height_reward | 0 | -1.31772e-09 |
| arm_contact_reward | -0.0292169 | -0.0271421 |
| arm_impulse_reward | -0.0248805 | -0.0241433 |
| arm_collision_reward | -0.292647 | -0.275923 |
| push_reward | 0 | 0 |
| wrist_vel_reward_ | -0.00708271 | -0.00782672 |
| wrist_qvel_reward_ | -0.00940063 | -0.0093129 |
| obj_vel_reward_ | -0.0417978 | -0.0546889 |
| obj_qvel_reward_ | -0.234292 | -0.258674 |
| obj_displacement_reward | -0.109084 | -0.134274 |
| arm_joint_vel_reward_ | -0.175797 | -0.193405 |

## 11. 500 iteration结果

**未执行**。100 iteration source success=0/30，文档Phase7明确要求先检查而不是直接运行500。条件决定见 `phase8-decision.json`。没有修改额外PPO超参或启动更长训练。

## 12. ShapeNet success变化

| 版本 | Source lift | Strict lift |
|---|---|---|
| 原baseline | 未单独输出 | 0/30 |
| 本轮5 iteration接口验证 | 0/30 | 0/30 |
| 本轮100 iteration | **0/30** | **0/30** |

最终评估30物体、1 round；grasp4秒、lift2秒、ramp1.5秒。逐物体结果见 `phase7-evaluation/per_object_success.csv`。

## 13. 剩余问题与100 iteration失败诊断

Pregrasp：初始最近关键点距离范围8.4146–13.8583 cm，中位12.3580 cm。源0.25 m预抓取偏置与HAND_CENTER保持一致；首个物理步arm最大变化0.001579 rad，未见明显fallback跳变。

Action：推理平均绝对raw action，arm=0.244241、hand=0.187654。手部关节抓取阶段平均绝对净变化0.17663 rad，最大1.55041 rad；动作确实驱动了机器人。抓取结束最近距离中位数降至 **1.8548 cm**，13/30物体至少一个策略步存在>0.1 N的手部–top过滤接触，峰值41.250 N。这说明出现接近和触碰，但接触本身未构成有效抓取。

Lift：UR5最终目标误差最大0.002011 rad；全部物体height gain≤0，最佳约-2.38e-7 m。抬升指令执行正常，物体未随手升起。各物体接触次数、距离与高度见 `phase7-grasp-diagnostics.csv`；0.1 N仅为该诊断统计阈值，奖励接触定义未改。

Reward/PPO：affordance距离项仍主导负奖励；标准差0.9097未坍缩；value loss出现141.43的有限峰值，adaptive LR曾到0.01，GAE真时间较短。这些是下一轮需隔离验证的因素，尚不能凭本轮数据认定单一RL根因。物体剩余数值越限4.6645e-5 rad也已如实保留。

## 14. 下一阶段建议

优先复查13个已触碰物体的接触位置、对向多指闭合和抬升时的保持情况，并结合17个未产生过滤接触的物体检查接近轨迹。随后按本轮数据决定是否单独验证lambda时间候选或adaptive LR行为。继续采用单变量对照；达到source lift success>0且趋势改善后，再进入文档的500 iteration阶段。

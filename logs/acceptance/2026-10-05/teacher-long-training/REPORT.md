# 优化4执行报告

2026-10-05启动，2026-10-06完成本轮基线。任务仍为 `Grasp1-UR5-Allegro-Teacher-v0`，使用 Conda `grasp`、RTX 3070Ti、2048训练环境与35训练物体。

已完成Phase 0–6：保存baseline、修正Observation、correctness smoke、行为诊断、500/1000/2000训练及每个节点30个ShapeNet物体1 round的A/B评估。Phase 7结论为不继续5000：行为指标持续退化，未达到文档规定的继续门槛。

## A. 已证明

### Observation与冻结配置

- 原HEAD `a79fd6d82f3ed72427314f3c38e7aba0b2125e44`及初始status/diff、指定配置与优化3报告已存于 `baseline/`。原未跟踪的 `docs/优化4.md` 保留。
- action保存实际发送给articulation的clamped target；Observation 0:22仍为当前joint position，22:44为该target减当前测量位置。153维观测、22维动作及逐物理子步相对控制保留。观测未复制clamp逻辑。
- full/partial reset清除raw/processed action并把stored target同步为当前测量姿态；reset事件和碰撞fallback写姿态时也同步。非reset环境的joint/object/target/action状态保持。
- 16 env zero/random各300步，每组current/upper/lower × zero/positive/negative ×22关节共198组检查通过；包含episode timeout及full/partial reset。无NaN/Inf，target outside fraction=0。
- zero/random的物体q范围、逐关节qvel峰值、接触峰值、target clipping与终止统计，均与优化3同seed对应数据完全一致，详见 `obs-target-fix/optimization3-smoke-comparison.json`。
- random smoke：UR5/Allegro物理子步qvel峰值2.40931/6.02800 rad/s，contact峰值718.787 N，target clipping=1.30966%；物体q范围−1.02439e-6～0.0010000742 rad，invalid terminations=0。
- Physics 120Hz、policy 60Hz、decimation=2、episode=4s、TGS、robot 32/1、object 16/0、self collision=False、object articulation与armature=0.1、action scale 0.00125/0.00375、per-time reward、gamma时间等效、lambda=0.95和PPO配置保持。

### 修改范围

| 文件 | 修改 |
|---|---|
| `source/Grasp1/Grasp1/tasks/manager_based/grasp1/mdp/actions.py` | 保存实际target、提供property、同步reset |
| `source/Grasp1/Grasp1/tasks/manager_based/grasp1/mdp/observations.py` | 22:44读取actual target error |
| `source/Grasp1/Grasp1/tasks/manager_based/grasp1/mdp/events.py` | reset/fallback同步action target |
| `scripts/rsl_rl/evaluate.py` | 默认relative-repeat及可选hold-posture、诊断入口 |
| `scripts/rsl_rl/grasp_diagnostics.py` | 评估专用bottom sensor与接触过程统计 |

另有本验收目录中的baseline、smoke、训练测量/阶段脚本、数据与报告；未修改正式train.py、环境/PPO配置、资产或reward函数。

### 诊断与A/B语义

原reward top/table sensors保持原过滤与friction语义；只在 `--diagnostics` 的评估场景增加13个bottom过滤传感器。诊断以0.1N法向过滤接触、1/60秒采样，记录top/bottom/table、同时接触body/fingertip数量、最长body/any连续接触、2+/3+、thumb+index/middle/ring/multiple、动作、关节位移、lift保持、距离、高度及arm残差。四个指尖按USD合并后的distal body统计。

Mode A为原current-relative-repeat。Mode B保存grasp-end实际16维手部target，在每个物理子步保持该target，arm沿用原lift trajectory。所有B评估的hand target drift=0。100/500点A的最大drift分别0.19077/0.59982 rad，证明两种模式确实执行了不同手部控制。旧100 checkpoint只用于最初接口验证，未用于新训练resume或新学习曲线。

### 训练与checkpoint

本轮500从头训练；1000从本轮model_499.pt resume，新增500次；2000从model_999.pt resume，新增1000次。包装器将保存的0起始iter加1，并从optimizer同步adaptive LR字段。实际更新区间为1–500、501–1000、1001–2000，无重复最后一次更新。resume重新创建环境，属于checkpoint续训，不声称保留不中断仿真轨迹。

| 节点 | transitions | run / checkpoint |
|---|---:|---|
| 100 | 6,553,600 | `2026-10-05_23-12-19_teacher_obsfix_stage500/model_99.pt` |
| 500 | 32,768,000 | `2026-10-05_23-12-19_teacher_obsfix_stage500/model_499.pt` |
| 1000 | 65,536,000 | `2026-10-05_23-48-02_teacher_obsfix_stage1000/model_999.pt` |
| 2000 | 131,072,000 | `2026-10-06_00-22-35_teacher_obsfix_stage2000/model_1999.pt` |

Run根目录为 `logs/rsl_rl/grasp1_ur5_allegro_teacher/`；完整绝对路径见 `training_progress.json`。本轮100点由500 run中额外保存的model_99.pt获得，使用修正后的Observation。历史优化3的100点单独标注，不与本轮100点混同。

| iteration | reward最终 / last50 | episode steps | value loss最终 / 阶段峰值 | surrogate | entropy | std | LR | KL |
|---|---|---:|---|---|---|---|---|---|
| 100 | -17.769 / -16.775 | 240 | 1.35727 / 见TensorBoard | -0.00286 | 29.256 | 0.92263 | 0.01 | 0.0079503 |
| 500 | -14.779 / -14.874 | 240 | 1.49789 / 194.68451118469238 | 0.0030501 | 23.981 | 0.73751 | 0.00087791 | 0.014068 |
| 1000 | -19.071 / -18.1 | 240 | 1.73385 / 514.3917179107666 | 0.0038727 | 17.803 | 0.58949 | 0.00017086 | 0.016632 |
| 2000 | -24.381 / -24.402 | 240 | 3429.76 / 189208.4931640625 | 0.0015358 | 14.791 | 0.52732 | 7.5938e-05 | 0.0092427 |

| stage | collection / learning s均值 | FPS均值 | 训练wall s | NVML记录峰值MiB | timeout / invalid height / invalid obs | object q min/max rad | target outside |
|---|---|---|---|---|---|---|---|
| 500 | 3.4950 / 0.0517 | 18535 | 1778.39 | 5989.125 | 136543 / 1 / 0 | -3.478839e-05 / 0.001016429 | 0.0 |
| 1000 | 3.4732 / 0.0514 | 18603 | 1767.26 | 5989.125 | 136543 / 1 / 0 | -2.317947e-05 / 0.00101124 | 0.0 |
| 2000 | 3.1988 / 0.0513 | 20184 | 3259.93 | 5989.125 | 273083 / 5 / 0 | -2.053802e-05 / 0.001019381 | 0.0 |

wall time为训练循环，未包含scene创建和独立评估；VRAM逐iteration采样。三个阶段最后50次VRAM均约5989.125 MiB，没有后期持续增长。训练过程中qvel有限但存在较大峰值，500节点UR5/Allegro峰值23.4305/55.8812 rad/s，不能将16-env random smoke的低峰值推广为所有训练状态的硬上限；逐关节值保存在各stage/runtime.json。物体仍有小量残余越限，未宣称严格0。

所有本轮已保存checkpoint的参数/optimizer tensor及所有TensorBoard scalars有限，17 reward terms完整；运行时action/obs/reward/robot joint/object state有限。`finite`通过只证明未出现NaN/Inf，不能证明学习稳定性或任务成功。

### 17项reward的最终TensorBoard值

| term | 500 | 1000 | 2000 |
|---|---:|---:|---:|
| affordance_reward | -3.38351 | -4.49631 | -5.71693 |
| affordance_contact_reward | 0.124951 | 0.0391799 | 0 |
| affordance_impulse_reward | 0.0560192 | 0.0216255 | 0 |
| table_reward | -0.033184 | -0.00020247 | 0 |
| table_contact_reward | 0 | 0 | 0 |
| table_impulse_reward | 0 | 0 | 0 |
| arm_height_reward | 0 | 0 | 0 |
| arm_contact_reward | 0 | 0 | -0.000195313 |
| arm_impulse_reward | 0 | 0 | -3.42537e-05 |
| arm_collision_reward | 0 | 0 | -0.00209263 |
| push_reward | 0 | 0 | 0 |
| wrist_vel_reward_ | -0.0052822 | -0.00664456 | -0.0120027 |
| wrist_qvel_reward_ | -0.0101207 | -0.00948383 | -0.0280727 |
| obj_vel_reward_ | -0.0342707 | -0.0230203 | -0.00787687 |
| obj_qvel_reward_ | -0.120476 | -0.0563661 | -0.0341724 |
| obj_displacement_reward | -0.129787 | -0.115678 | -0.0785339 |
| arm_joint_vel_reward_ | -0.224078 | -0.162429 | -0.211449 |

这是TensorBoard `Episode_Reward` 的回合归一化统计，与 `Train/mean_reward` 区分；reward公式、权重比例与时间换算未修改。

### ShapeNet行为结果

| iteration | mode | source / strict | grasp-end median cm | object / top / bottom / table coverage | 2+ / 3+ / thumb-other | final median / max height gain m | lift any-contact retention |
|---|---|---|---|---|---|---|---|
| 100 | A | 0/30 / 0/30 | 1.7662 | 9 / 9 / 0 / 0 | 7 / 2 / 1 | -0.006041139 / -1.192093e-07 | 0.016666666666666666 |
| 100 | B | 0/30 / 0/30 | 1.7662 | 9 / 9 / 0 / 0 | 7 / 2 / 1 | -0.005843014 / -1.192093e-07 | 0.013541666666666667 |
| 500 | A | 0/30 / 0/30 | 1.8004 | 15 / 15 / 0 / 0 | 9 / 6 / 0 | -0.003370136 / -1.192093e-07 | 0.0020833333333333333 |
| 500 | B | 0/30 / 0/30 | 1.8004 | 15 / 15 / 0 / 0 | 9 / 6 / 0 | -0.003908008 / -5.960464e-08 | 0.0006944444444444445 |
| 1000 | A | 0/30 / 0/30 | 6.7256 | 1 / 1 / 0 / 0 | 0 / 0 / 0 | -0.003321201 / -5.960464e-08 | 0.0 |
| 1000 | B | 0/30 / 0/30 | 6.7256 | 1 / 1 / 0 / 0 | 0 / 0 / 0 | -0.003321201 / -5.960464e-08 | 0.0 |
| 2000 | A | 0/30 / 0/30 | 14.1633 | 0 / 0 / 0 / 0 | 0 / 0 / 0 | -0.002298266 / 7.152557e-07 | N/A（grasp-end无接触） |
| 2000 | B | 0/30 / 0/30 | 14.1633 | 0 / 0 / 0 / 0 | 0 / 0 / 0 | -0.002298266 / 7.152557e-07 | N/A（grasp-end无接触） |

Coverage为30物体中至少一次超过诊断阈值的数量；2+/3+为同时接触不同手部body，thumb-other统计拇指与其它finger同时接触。过程最长duration、指尖数、各拇指配对、force peak、hand movement、raw actions、height peak与lift arm residual见 `grasp_diagnostics.csv` 和各mode逐物体JSON。没有grasp-end contact时retention分母不存在，记录N/A而非伪造0%。

## B. 观察趋势

- 本轮top coverage为9→15→1→0，2+ coverage为7→9→0→0；500后接触与多指质量回退。
- 距离中位数1.766→1.800→6.726→14.163 cm；2000点手部已明显远离affordance。
- 500/1000/2000最近50次reward均值−14.8743→−18.0996→−24.4017；value loss阶段峰值194.685→514.392→189208.493，最终值1.498→1.734→3429.763。这条基线表现为退化和较大loss尖峰，不能称为稳定学习。
- std均值0.738→0.589→0.527；2000各动作std范围0.08735–0.80622，有限但均值不能代表每个动作维度。
- A/B均无source或strict成功；B保持target已验证，但未改善抬升。本轮没有证据将relative-repeat确定为主要失败原因。

### Phase 7决定及启动纠正

不继续5000/10000/20000。依据是接触、多指、距离和reward连续退化，并非单个0/30。2000最大lift过程height gain约9.54e-7 m，仅约1微米，且手部object contact为0；负height gain变小不构成抓取学习改善。

阶段脚本最初把负height gain变小误判为继续条件，曾启动5000 scene创建。已在PPO更新前终止，实际新增updates=0，无5000 checkpoint或evaluation结果。`stage5000/`只保存该启动日志及 `startup-aborted.json`，不得作为5000结果。继续条件已修正；`stage2000/decision.json`记录最终false决定。

本轮保留原PPO/reward/lambda配置，未执行可选单变量实验。当前已有2000更新的可信基线与明显退化证据，可供下一轮单独研究adaptive LR或lambda；不能同时修改这些变量，更不能为改善报表改物理或reward定义。

## C. 尚不能证明

- 尚不能证明策略最终收敛、算法永远无法抓取、reward公式错误或某个单一PPO参数是根因。
- 0/30只说明对应checkpoint、ShapeNet 30物体、1 round未满足本次lift criterion；不能外推最终训练结论。
- Force-closure proxy不等于完整grasp wrench space，诊断未测所有可能的接触刚体或亚策略步连续接触。
- 无NaN/Inf及VRAM稳定不等于reward/value稳定；2000基线的loss退化已经实测。
- 5000/10000/20000训练与结果不存在，未填造数据。

## 验收文件

`baseline/`、`obs-target-fix/`、`stage100/`、`stage500/`、`stage1000/`、`stage2000/`、`training_progress.json`、`training_progress.csv`、`evaluation_progress.csv`、`grasp_diagnostics.csv`、`final_validation.json`、`REPORT.md`。`stage5000/`是未开始更新的中止启动记录。
compileall与git diff --check通过。源代码改动仅限上述5个文件；未提交git。

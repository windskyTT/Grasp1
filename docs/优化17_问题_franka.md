# 优化17：FR3-Inspire checkpoint、GUI 与 TensorBoard 问题报告

检查日期：2026-10-26。项目：`/home/windsky/project/Grasp1`；Conda：`grasp`；Isaac Lab 2.3.2 / Isaac Sim 5.1.0；RTX 3070 Ti Laptop 8 GB。

**本次未找到达到当前成功条件的抓取 checkpoint。** 7 个候选覆盖全部 35 个训练物体、每个物体 3 次尝试，成功 **0/735**；另在实际 GUI 中完成两组评估，成功 **0/4**。

已保存 checkpoint 中，训练 `Train/mean_reward` 最高的是 **`model_7070.pt`，−289.054871**。本次候选中结束时高度判据值最大的是 **`model_7580.pt`，4.8483 cm**，仍未超过 10 cm，而且该样本在脚本抬臂之前已经有较高的判据值，不能称为已验证的最佳抓取。

## 1. 本次训练与候选

训练目录：

```text
/home/windsky/project/Grasp1/logs/rsl_rl/grasp1_fr3_inspire_teacher/2026-10-25_21-51-12_133017
```

- 2001 个 checkpoint：`model_0.pt` 至 `model_20000.pt`，每 10 次迭代保存。
- 最终文件 `iter=20000`、`completed_updates=20001`。
- 任务：`Grasp1-FR3-Inspire-Teacher-v0`，semantic revision **1**，**132D 观测 / 19D 动作**，即 7 个 FR3 关节与 12 个 Inspire 关节。
- 数据集：固定 `new_training_set`，35 种不同物体。
- 物理周期 `1/120 s`，decimation 2，控制周期 `1/60 s`；训练回合 14 秒 / 840 控制步，只训练抓取。
- 7 个候选均通过现有完整任务契约检查，模型状态张量均有限；训练标量全量检查非有限点为 **0**。这证明可以加载运行，不证明已学会抓取。

全程 `Train/mean_reward` 原始峰值为 **-286.028503**，位于迭代 **6405**，该迭代没有保存 checkpoint。对实际保存点排序，最高的是 `model_7070.pt`。

| 候选 | 筛选依据 | 保存迭代处 Train/mean_reward |
| --- | --- | ---: |
| `model_7070.pt` | 已保存迭代处 Train/mean_reward 最高 | -289.054871 |
| `model_6410.pt` | 全程平均回合 Reward 峰值附近 | -290.797150 |
| `model_7580.pt` | 已保存更新处 Teacher/reward 最高 | -379.702972 |
| `model_10780.pt` | Teacher/opposing 全程峰值对应保存点 | -476.823425 |
| `model_10090.pt` | 另一处较高对向接触诊断窗口 | -560.486084 |
| `model_780.pt` | 早期对向接触诊断峰值 | -526.202271 |
| `model_20000.pt` | 最终 checkpoint | -480.396698 |

`Train/*` 的迭代编号从 0 开始，`Teacher/*` 的已完成更新数从 1 开始。例如 `Teacher/reward` 更新 7581 对应 `model_7580.pt`，对向接触诊断峰值更新 10781 对应 `model_10780.pt`。曲线峰值用于筛选候选，不直接决定最佳抓取。

## 2. 可视化入口的必要适配

检查开始时，`scripts/rsl_rl/visual_eval.py` 只接受 UR5-Allegro 任务，并固定 153D/22D、6 个机械臂关节。本次仅在这个现有脚本补齐 FR3 分支：

- 使用已注册的 FR3 任务与现有 FR3 环境/PPO 配置，保持 checkpoint 严格校验。
- 从当前任务契约读取动作与观测维度，按 7 个机械臂关节、12 个手指关节拆分控制。
- FR3 脚本抬臂的绝对参考使用现有 `robots/fr3_inspire_cfg.py` 的 `FR3_DEFAULT_ARM`，值为：

```text
[0.0, -1.3, 0.0, -2.8399996757507324, 0.0, 2.0, 0.75]
```

**ADAPTED：** 原始例子的六轴 UR5 `theta0` 没有直接对应的七轴 FR3 向量。本次使用 FR3 既有默认姿态作绝对参考，保留切换时实测关节角到参考角的插值方式。这是 FR3 评估迁移，不宣称与原 UR5 抬臂轨迹完全等价，也不是学习到的自主抬升。

PPO 抓取执行 840 步 / 14 秒，脚本抬臂执行 360 步 / 6 秒，完整参考插值长度 960 步 / 16 秒，实际执行其中 360/960。抬臂阶段手指保留末次策略动作，按已有源 5 Hz、`q_measured + 0.015*a` 公式更新绝对目标，每 12 个环境步更新一次，中间保持目标。

UR5 分支保留原六轴绝对参考。本次未修改训练代码、奖励、终止、动作尺度、PD、物理参数、资产或 checkpoint，也没有继续训练。FR3 revision 1 与当前 UR5 revision 8 均为各自已有契约，本次没有更改它们。

## 3. 评估方法与成功判据

GUI 直接运行适配后的 `scripts/rsl_rl/visual_eval.py`，实际日志为 `GUI=True`，窗口显示 FR3、Inspire、物体和桌面。

批量比较在终端 Python 内存中执行该入口的同一控制实现，增加候选循环、物体分配及诊断采样；未新增 Python 源码或测试脚本。35 个并行环境各放一个不同训练物体，每个模型分别以 seed 42、43、44 显式重置后运行。GUI 的单物体初态与批量多物体初态不同。

模型先按 14 秒训练契约校验，再将独立评估时长设为 20 秒。成功条件保持：**连续完成同一次抓取和脚本抬臂，评估结束时 `top_z_current - root_z_reset > 0.10 m`**。最终正常 20 秒超时允许完成；任何中途终止均按失败处理。接触诊断不额外参与成功条件。

高度与接触诊断按控制步末采样。过程峰值截止首次失败，不统计自动重置后新回合的状态；结束时最大值仅统计连续完成的尝试。未声称覆盖每个 PhysX 子步的瞬时峰值。

## 4. 实测结果

| checkpoint | 成功/尝试 | 连续完成 | 中途终止 | 结束时最大高度差 cm | 过程最大高度差 cm |
| --- | ---: | ---: | ---: | ---: | ---: |
| `model_7070.pt` | 0/105 | 105 | 0 | 1.8018 | 3.0887 |
| `model_6410.pt` | 0/105 | 105 | 0 | 1.8018 | 4.3636 |
| `model_7580.pt` | 0/105 | 105 | 0 | 4.8483 | 5.1087 |
| `model_10780.pt` | 0/105 | 105 | 0 | 2.0077 | 4.8202 |
| `model_10090.pt` | 0/105 | 103 | 2 | 1.9994 | 12.9003 |
| `model_780.pt` | 0/105 | 104 | 1 | 2.2561 | 3.2957 |
| `model_20000.pt` | 0/105 | 105 | 0 | 0.7896 | 4.5899 |
| 合计 | **0/735** | **732** | **3** | **4.8483** | **12.9003** |

### 高度峰值不等于成功抓取

结束时最大判据样本为 `model_7580.pt`、seed 44、trial 3、env 8、`big_tape`：

- 过程峰值出现在控制步 **47**，即约 0.7833 秒，尚未开始脚本抬臂。
- 抓取结束时高度差 **0.044467390 m**，评估结束时 **0.048482776 m**；抬臂阶段 top 实际仅再上升约 **4.02 mm**。
- 抓取结束时末节接触计数为 0，对向接触为 False。

另有 `model_10090.pt`、seed 44、env 25、`051_large_clamp`：第 **28** 步出现 **12.9003 cm** 的早期瞬态，但抓取结束和评估结束时均约 **-0.003003 m**，仍为失败。不能把过程最大值替代脚本的结束时判据。

### GUI 结果

| checkpoint | 物体 | seed | 各次结束时高度差 m | 成功/尝试 |
| --- | --- | ---: | --- | ---: |
| `model_7070.pt` | `wood_block_oriented` | 42 | -0.003260, -0.003259, -0.003110 | 0/3 |
| `model_7580.pt` | `big_tape` | 1 | -0.002004 | 0/1 |

两组均完成整个流程，无中途重置，进程退出码均为 0。第二组为同类物体的独立尝试，不是对批量 env 8、seed 44 样本的精确重放。两组都没有通过成功条件。

## 5. 已确认的问题与证据

### 5.1 有效末节接触在抓取结束前已经缺失

在 732 次完成抓取阶段的尝试中：

- **722/732（98.63%）** 没有末节有效接触，另 **10** 次只有一个；没有两个及以上。
- 抓取结束时对向接触为 **0/732**；本批尝试的抓取段在控制步末采样中，对向接触计数也均为 0。
- 抬臂结束前一控制步，**727/732** 没有末节有效接触，对向接触仍为 0。
- 有 **115/732** 次在抓取结束时，13 个手部接触槽的法向力幅值之和大于 0。因此不能将“末节无有效接触”写成“整只手完全没有碰到物体”。

这里使用的是现有五个真实 Inspire 末节接触诊断及大于 1 N 的原诊断阈值，不是 Allegro 四个 fixed tip，也不是新增奖励或成功门槛。

在 732 次连续完成的尝试中，**731/732** 的腕部在抬臂阶段向上运动，Z 变化均值约 **22.26 cm**，范围 **-0.53 至 46.77 cm**；数据取自动重置前一控制步。抬臂阶段物体 top 的最大实际 Z 增量约 **5.2253 cm**。结合抓取末接触，当前证据指向抓取阶段尚未形成可随腕部抬起的夹持，不能仅归因于 GUI 没有执行抬臂。

### 5.2 回报主要受机械臂运动惩罚影响，接触正奖励很小

最后 1000 次更新，每控制步的平均加权分项为：

| 指标 | 均值 |
| --- | ---: |
| 总 Teacher/reward | -0.684995 |
| arm_joint_vel_reward_ | -0.531917 |
| affordance_reward | -0.069664 |
| affordance_contact_reward | 0.000109476 |
| affordance_impulse_reward | 0.000036091 |

机械臂关节速度惩罚约占同期净负回报幅值的 **77.65%**。训练全程仅 **50 / 20001** 个更新窗口出现非零对向接触诊断，最后 1000 次更新均为 0。该诊断不是抓取成功率，不能据此填报训练抬升成功率。

这些分项与接触结果提示需要检查 FR3 控制响应及奖励适配，但尚不能认定应直接调整某一项权重。

### 5.3 后期训练表现没有稳定保持

- 最终 `Train/mean_reward=-480.396698`，最后 1000 个点均值 **-580.891196**，低于中期峰值。
- `Policy/mean_noise_std` 从约 **0.500054** 增至 **7.207360**。这是训练探索分布的参数；GUI 使用确定性推理，不直接采样该噪声。
- `Loss/value_function` 在迭代 **2390** 达到 **5624302.000000**；最终 **19011.775391**，最后 1000 个点均值 **2274.767219**。
- 有限权重、无 NaN 和完成 20001 次更新，不足以证明策略收敛或学会抓取。

### 5.4 少数抓取阶段越过桌面边界，大部分失败仍发生在完整回合中

训练累计结束回合 **68222**：正常超时 **66714（97.7896%）**，手部高度终止 **1508**，非有限状态终止 **0**。最终平均回合长度 **840 步 / 14 秒**，最后 1000 个点均值约 838.23 步。

批量评估中的 3 次中途失败均为 `hand_height=True`、`invalid=False`、`timeout=False`，均发生在第 840 步抬臂切换之前：

| checkpoint | seed | env / 物体 | 首次失败控制步 | 时间 s | 当步物体高度差 m |
| --- | ---: | --- | ---: | ---: | ---: |
| `model_10090.pt` | 42 | 5 / `019_pitcher_base` | 65 | 1.0833 | -0.085666 |
| `model_10090.pt` | 44 | 9 / `036_wood_block` | 72 | 1.2000 | -0.066238 |
| `model_780.pt` | 44 | 9 / `036_wood_block` | 173 | 2.8833 | -0.079973 |

其余 732 次连续完成仍未成功，因此不能把全部失败解释为提前重置。

## 6. 仍待定位的原因与评估边界

**UNVERIFIED：** 尚未确定接触不足是由手部几何/关节方向、策略动作、PD 跟踪、接触适配还是奖励与探索共同导致。下一步应固定实际 Reset 状态，检查 FR3 零动作保持、关节目标/实测误差、手指与物体接触，以及奖励正反轨迹排序，再决定单变量修改。

当前代码中的重力平衡位置偏置 `arm_gravity_compensation=True` 只在 UR5 专属配置中启用；FR3 独立配置与本次 revision 1 checkpoint 没有启用该字段。其对 FR3 跟踪与关节速度惩罚的影响仍为 **UNVERIFIED**。本次没有直接把 UR5 补偿移植进 FR3，以免改变已训练策略的控制契约。

**ADAPTED：** FR3 默认姿态抬臂参考是本次必要的机器人适配；原有 Lula Reset、真实 Inspire 关键点/接触槽，以及 RaiSim 逐接触冲量与 PhysX 合冲量的差异，不能当作跨引擎严格等价。现有合冲量适配是否导致失败，本次没有证明。

**N/A：** 训练只学习抓取，没有本次独立脚本抬臂的训练成功率曲线。`Teacher/lift` 为训练抓取期间的平均高度差，不能代替独立抬臂成功率；不能将未记录的训练抬升成功率填成 0%。

**NOT RUN：** 未逐一评估全部 2001 个 checkpoint，未穷尽所有物体初态，未执行控制/奖励/PPO 的新 A/B 或重训。0/735 是上述候选与协议的实测结果，不代表所有权重在所有状态下均不可能成功。

## 7. 修改与验证范围

项目内只修改现有 `scripts/rsl_rl/visual_eval.py`，并新增本报告 `docs/优化17_问题_franka.md`。

- **PASS：** 当前脚本语法编译。
- **PASS：** FR3 132D/19D、7 个 checkpoint 严格任务契约与有限权重检查、真实 GUI 和多物体完整评估流程。
- **FAIL：** 本次抓取和 FR3 适配脚本抬臂成功条件，批量 0/735，GUI 0/4。
- **PASS：** UR5 当前 revision 8 兼容权重 `2026-10-25_18-24-42_655801_opt17_2_gravity_pilot/model_199.pt`，单环境 headless 完成 1200 控制步，正常退出。该次抓取成功 0/1，只作为原入口运行回归。
- 首次 UR5 回归使用的历史 revision 7 权重被正确拒绝，差异为 `semantic_revision` 和 `actions`；后续改用当前兼容权重完成回归，没有绕过或放宽契约校验。

## 8. TensorBoard 与 GUI 复现命令

TensorBoard 已启动并在浏览器查看本次完整 Reward 曲线：<http://127.0.0.1:6006/>。服务只加载本次 FR3 运行，标量保留上限 25000 点，实际 `Train/mean_reward` 为 19975 点、`Teacher/reward` 为 20001 点；已在图中关闭异常值隐藏、平滑设为 0，候选排序使用完整原始数据。

重点观察 `Train/mean_reward`、`Teacher/reward`、`Teacher/opposing`、`Train/mean_episode_length`、`Policy/mean_noise_std`、`Loss/value_function`。前两项分别为平均回合累计 Reward 和每控制步平均 Reward，单位不同。

需要重启 TensorBoard 时执行：

```bash
cd /home/windsky/project/Grasp1
conda activate grasp
python -m tensorboard.main \
  --logdir logs/rsl_rl/grasp1_fr3_inspire_teacher/2026-10-25_21-51-12_133017 \
  --host 127.0.0.1 --port 6006 \
  --samples_per_plugin scalars=25000 --load_fast=false
```

回放 Reward 最高候选；默认打开 GUI，不使用 `--mode visual`：

```bash
cd /home/windsky/project/Grasp1
conda activate grasp
python scripts/rsl_rl/visual_eval.py \
  --task Grasp1-FR3-Inspire-Teacher-v0 \
  --checkpoint logs/rsl_rl/grasp1_fr3_inspire_teacher/2026-10-25_21-51-12_133017/model_7070.pt \
  --num_envs 1 --episodes 3 --seed 42 --device cuda:0 --real-time
```

回放另一组已测 GUI：将 checkpoint 改为同目录 `model_7580.pt`，使用 `--episodes 1 --seed 1`，物体为 `big_tape`。这些命令用于查看失败候选，不是成功抓取演示。

## 9. 原始证据

主要结果已保存在本报告。以下是临时运行文件，可能被系统清理：

- `/tmp/grasp1_franka_opt17_metrics.json`、`/tmp/grasp1_franka_opt17_selection.json`：完整训练标量汇总、候选分数、Reward 分项。
- `/tmp/grasp1_franka_opt17_eval_results.json`：735 次逐尝试记录、首次失败、接触、腕部高度与奖励分项。
- `/tmp/grasp1_franka_opt17_eval.log`：内存批量评估日志，退出码 0；没有新增批量 Python 文件。
- `/tmp/grasp1_franka_opt17_gui_7070.log`、`/tmp/grasp1_franka_opt17_gui_7580.log`：两组实际 GUI 日志，退出码均为 0。
- `/tmp/grasp1_franka_opt17_ur5_regression.log`：历史不兼容权重的拒绝记录；`/tmp/grasp1_franka_opt17_ur5_regression_rev8.log`：兼容权重回归日志，退出码 0。
- `/tmp/grasp1_franka_opt17_reward_curve.png`、`/tmp/grasp1_franka_opt17_gui_start.png`：TensorBoard 曲线及首组 GUI 桌面截图。

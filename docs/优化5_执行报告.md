# 优化5执行报告

**Phase 0–9已完成，2000节点未通过进入5000的gate；5000/10000/20000未执行。PPO稳定性目标尚未达成。** 已完成baseline、源语义核实、四组单变量A/B、最终候选重新从头训练的100/500/1000/2000节点及每个节点ShapeNet正式评估。

2000首次有真实抬升成功：Mode A的source/strict均为1/90，Mode B均为2/90。接触与多接触改善、距离保持约1.9cm，但critic/reward/饱和/手部速度明显恶化，不能声称稳定训练或稳定抓取。

HEAD=5919609fe3eb5091677fb415ca0476a83ab1a9fe（main，V1.4—优化4）。任务仍为Grasp1-UR5-Allegro-Teacher-v0；Conda grasp、RTX3070Ti、IsaacLab2.3.2/IsaacSim5.1、2048训练环境、seed=1。

## 最终候选与改动

```text
clip_actions = None
schedule = fixed
learning_rate = 5e-4
lambda = 0.95 ** (CONTROL_DT / SOURCE_CONTROL_DT) = 0.9957346812224394
minimum_action_std = 0.2
```

该候选保留在当前代码，完整失败证据保留，没有另开超参数搜索。

| 正式源文件 | 改动 |
|---|---|
| source/Grasp1/Grasp1/tasks/manager_based/grasp1/config/ur5_allegro/agents/rsl_rl_ppo_cfg.py | fixed schedule、时间等效lambda、被train.py消费的minimum_action_std字段 |
| scripts/rsl_rl/train.py | 每次完整PPO update后仅投影std下界；None关闭；其余网络参数和optimizer更新保持 |
| scripts/rsl_rl/evaluate.py | 直接CLI覆盖候选clipping；策略抓取/手部repeat应用同一范围，固定UR5脚本抬升轨迹保持 |

本目录[train_measure.py](train_measure.py)、[run_stage.py](run_stage.py)为实际运行的测量/阶段脚本。未新增Task ID或Variant；保持观测22:44=actual clamped target−current q、full/partial reset、Teacher relative action、scale0.00125/0.00375、120Hz physics/60Hz policy、decimation2、episode4秒、TGS、robot32/1、object16/0、self-collision=False、object articulation/zero-PD/armature0.1、per-time reward、gamma0.9996660539892578、network、rollout32及其余PPO参数。

## 最终分阶段验收与gate

500重新从头训练，100是该运行额外保存的model_99.pt快照、无新增训练。1000从本轮model_499.pt恢复新增500更新，2000从model_999.pt恢复新增1000更新。实际区间1–500、501–1000、1001–2000，没有重复最后一次更新；续训重新创建环境，属于checkpoint续训。


| 节点 | reward last50 | value final / 当前阶段peak | value last50 mean / median / peak | hand qvel rad/s | target clip |
|---|---:|---|---|---:|---:|
| 100 | -17.5873 | 2.5302 / 1025.6345 | 2.1775 / 1.9955 / 5.9228 | 64.3125 | 1.602% |
| 500 | -14.2151 | 0.7614 / 1025.6345 | 0.9153 / 0.8897 / 1.4149 | 273.1656 | 7.905% |
| 1000 | -14.9354 | 4.0489 / 660.2166 | 2.8810 / 2.8215 / 4.8260 | 36.8854 | 15.706% |
| 2000 | -54.5291 | 1420.0894 / 4984.8348 | 899.6340 / 786.1421 / 4342.6013 | 406.5440 | 26.578% |

500末critic近期均值0.915、reward改善、接触保持，允许1000；1000末critic均值2.881、接触63/90、距离约2cm，没有旧基线接触/距离崩溃，允许2000验证。原判断见[stage500/gate.json](selected-config/stage500/gate.json)、[stage1000/gate.json](selected-config/stage1000/gate.json)，不能把当时的继续判断当作长期稳定证明。

**2000→5000不通过。** 文档第19节要求critic稳定且至少一类任务行为改善。本次行为改善条件成立，critic稳定条件不成立：最近50轮loss均值899.634、final1420.089、stage peak4984.835，最后300更新持续数百级loss；reward last50降至−54.529、target clipping26.578%、hand qvel406.544 rad/s。按文档停止后续长训练。见[stage2000/gate.json](selected-config/stage2000/gate.json)。

| 节点 | Mode | source / strict | distance median cm | object/top/bottom/table trials | 2+/3+/thumb-other trials | final height median / max m | lift retention |
|---|---|---|---:|---|---|---|---:|
| 100 | A | 0/90 / 0/90 | 1.8171 | 47/47/0/0 | 21/8/1 | -0.003493 / 0.000247 | 1.238% |
| 100 | B | 0/90 / 0/90 | 1.8171 | 47/47/0/0 | 21/8/1 | -0.003565 / 0.000177 | 1.119% |
| 500 | A | 0/90 / 0/90 | 1.9507 | 57/57/0/3 | 18/2/0 | -0.004187 / -0.000000 | 0.709% |
| 500 | B | 0/90 / 0/90 | 1.9660 | 58/58/0/3 | 19/2/0 | -0.004187 / -0.000000 | 0.799% |
| 1000 | A | 0/90 / 0/90 | 1.9818 | 63/63/0/1 | 11/4/7 | -0.005532 / 0.003989 | 0.417% |
| 1000 | B | 0/90 / 0/90 | 1.9502 | 63/63/0/1 | 10/3/7 | -0.005532 / 0.004231 | 0.395% |
| 2000 | A | 1/90 / 1/90 | 1.8658 | 81/81/0/30 | 52/31/37 | -0.008247 / 0.281730 | 4.760% |
| 2000 | B | 2/90 / 2/90 | 1.8582 | 81/81/0/30 | 52/32/37 | -0.007546 / 0.290189 | 5.429% |

正式比较均为每物体3 rounds、90 trials/Mode、seed=1。A=current-relative-repeat，B=hold-grasp-posture；所有B的hand target drift=0。覆盖计数按trial，不是30个不同物体数。2000成功trial：

- Mode A：Purse_brown_d58，round=3，height gain=0.281730m，source=True，strict=True。
- Mode B：Hammer_40，round=2，height gain=0.290189m，source=True，strict=True。
- Mode B：Purse_brown_d58，round=3，height gain=0.280921m，source=True，strict=True。

2000最大过程高度增量A/B为0.282612/0.291218m，末端最大0.281730/0.290189m，超过原0.1m阈值；但末端高度中位数为−8.247/−7.546mm，retention均值只有4.760%/5.429%。不能用少量最大值描述多数trial。A/B只相差一个成功，不能确定抬升模式优势。

std下界首次在第1538次更新达到0.2；1001–2000共69次更新位于下界（容差1e−6），最低std0.20000000298。没有无下界的2000配对实验，不能把成功或critic恶化归因于std投影。

| 节点 | transitions | learn wall s | FPS均值 | NVML peak / last50 MiB | object q min/max rad | timeout / invalid-height / invalid-obs |
|---|---:|---:|---:|---|---|---|
| 100 | 6,553,600 | 384.62 | 17140 | 5989.125 / 5989.125 | -2.3473376e-05 / 0.0010107592 | 27324 / 0 / 0 |
| 500 | 32,768,000 | 1887.27 | 17438 | 5989.125 / 5989.125 | -2.3473376e-05 / 0.0010174224 | 136543 / 0 / 0 |
| 1000 | 65,536,000 | 1800.53 | 18276 | 5989.125 / 5989.125 | -1.9940037e-05 / 0.0010153665 | 136542 / 1 / 0 |
| 2000 | 131,072,000 | 3601.00 | 18365 | 5989.125 / 5989.125 | -4.3369277e-05 / 0.0010337357 | 273083 / 5 / 0 |

100 wall已包含在500中，不重复相加。最终候选三段learn wall合计7288.79秒，不含scene启动及独立评估。

| 节点 | checkpoint（相对logs/rsl_rl/grasp1_ur5_allegro_teacher/） |
|---|---|
| 100 | `2026-10-06_19-55-10_teacher_stable_stage500/model_99.pt` |
| 500 | `2026-10-06_19-55-10_teacher_stable_stage500/model_499.pt` |
| 1000 | `2026-10-06_20-39-27_teacher_stable_stage1000/model_999.pt` |
| 2000 | `2026-10-06_21-21-18_teacher_stable_stage2000/model_1999.pt` |

## 2000动作与动力学

完整动作链见[action_diagnostics.csv](action_diagnostics.csv)。2000最近50轮raw mean abs4.080534、p95=9.286226、p99=10.824892；|a|>1为82.048%，|a|>2为71.110%。None时raw与wrapper相同。当前1001–2000窗口raw最大83.719986；processed delta mean abs0.0107485/max0.285337；final target mean abs0.941048/max3.141593。outside=0仍伴随26.578% saturation。

UR5 qvel峰值25.227695、手部406.543976 rad/s，contact force采样峰值87394.546875N。物体q−4.33693e−5～0.001033736rad仍有小量残余越限，物理/限位保持。每关节120Hz速度峰值：

| joint | qvel peak rad/s |
|---|---:|
| shoulder_pan_joint | 2.433517 |
| shoulder_lift_joint | 3.590515 |
| elbow_joint | 7.508719 |
| wrist_1_joint | 8.092596 |
| wrist_2_joint | 3.627389 |
| wrist_3_joint | 25.227695 |
| joint_0_0 | 67.384544 |
| joint_12_0 | 31.487505 |
| joint_4_0 | 58.121414 |
| joint_8_0 | 17.885239 |
| joint_1_0 | 146.659149 |
| joint_13_0 | 89.448883 |
| joint_5_0 | 92.903664 |
| joint_9_0 | 27.409912 |
| joint_2_0 | 130.827881 |
| joint_14_0 | 85.056961 |
| joint_6_0 | 47.733887 |
| joint_10_0 | 39.482265 |
| joint_3_0 | 242.049698 |
| joint_15_0 | 406.543976 |
| joint_7_0 | 9.005825 |
| joint_11_0 | 56.924820 |

## A/B选择与全部完成节点

## A. 已证明

V1.4 baseline位于baseline/，HEAD=5919609fe3eb5091677fb415ca0476a83ab1a9fe。
Source action range semantics：高斯采样不做[-1,1] clipping或tanh，仅最终joint target限位。clip1是工程A/B；源minimum std=0.2在完整PPO更新之后执行。详见source-action-semantics.md。
PPO诊断在adaptive/fixed下与原更新的参数、optimizer及原损失完全相等，见smoke/ppo-instrumentation-equivalence.json。
本轮None候选的100节点模型参数及optimizer与V1.4同节点完全一致，见smoke/v14-node100-equivalence.json。该结论限定于已直接对照的前100次更新。
minimum std扩展在完整PPO更新后执行；16-env激活下界测试中首次更新的actor/critic参数及optimizer完全一致，仅std为精确clamp(min=0.2)，见smoke/minimum-std-projection-check.json。该smoke使用init std=0.1，正式实验保持1.0。
训练保持2048环境、seed=1、32 rollout、原physics/reward/action scale/network/gamma；单变量candidate配置逐节点保存。
正式评估每物体3 rounds，90 trials，Mode A/B；policy clipping作用于抓取策略及重复的手部动作，固定UR5抬升轨迹保持。

| candidate | iteration | reward last50 | value final / peak | std min / median / max | target clip | UR5 / hand qvel |
|---|---:|---:|---|---|---|---|
| action-clipping/clip1 | 1000 | -20.141 | 79.691 / 338.85 | 0.07828 / 0.3948 / 0.6781 | 4.812% | 19.42 / 217.8 |
| action-clipping/clip1 | 500 | -19.043 | 13.58 / 205.22 | 0.1856 / 0.5844 / 0.8717 | 2.427% | 19.46 / 69.41 |
| action-clipping/none | 1000 | -18.1 | 1.7339 / 514.39 | 0.1079 / 0.6328 / 0.8555 | 6.293% | 14.66 / 151.7 |
| action-clipping/none | 500 | -14.874 | 1.4979 / 194.68 | 0.3484 / 0.7362 / 0.9038 | 7.271% | 23.43 / 55.88 |
| lambda/095 (复用) | 1000 | -18.868 | 1.5143 / 123.8 | 0.2611 / 0.8125 / 1.002 | 18.169% | 18.11 / 214.4 |
| lambda/095 (复用) | 500 | -17.124 | 0.41743 / 56.641 | 0.3374 / 0.8877 / 1.034 | 10.129% | 17.28 / 319 |
| lambda/time-equivalent | 1000 | -14.935 | 4.0489 / 660.22 | 0.3197 / 0.8083 / 1.035 | 15.706% | 22.3 / 36.89 |
| lambda/time-equivalent | 500 | -14.215 | 0.76142 / 1025.6 | 0.5604 / 0.8753 / 0.9766 | 7.905% | 17.99 / 273.2 |
| learning-rate/adaptive (复用) | 1000 | -18.1 | 1.7339 / 514.39 | 0.1079 / 0.6328 / 0.8555 | 6.293% | 14.66 / 151.7 |
| learning-rate/adaptive (复用) | 500 | -14.874 | 1.4979 / 194.68 | 0.3484 / 0.7362 / 0.9038 | 7.271% | 23.43 / 55.88 |
| learning-rate/fixed | 1000 | -18.868 | 1.5143 / 123.8 | 0.2611 / 0.8125 / 1.002 | 18.169% | 18.11 / 214.4 |
| learning-rate/fixed | 500 | -17.124 | 0.41743 / 56.641 | 0.3374 / 0.8877 / 1.034 | 10.129% | 17.28 / 319 |
| min-std/min020 | 1000 | -14.935 | 4.0489 / 660.22 | 0.3197 / 0.8083 / 1.035 | 15.706% | 22.3 / 36.89 |
| min-std/min020 | 500 | -14.215 | 0.76142 / 1025.6 | 0.5604 / 0.8753 / 0.9766 | 7.905% | 17.99 / 273.2 |
| min-std/none (复用) | 1000 | -14.935 | 4.0489 / 660.22 | 0.3197 / 0.8083 / 1.035 | 15.706% | 22.3 / 36.89 |
| min-std/none (复用) | 500 | -14.215 | 0.76142 / 1025.6 | 0.5604 / 0.8753 / 0.9766 | 7.905% | 17.99 / 273.2 |
| selected-config | 100 | -17.587 | 2.5302 / 1025.6 | 0.904 / 0.9877 / 1.026 | 1.602% | 11.85 / 64.31 |
| selected-config | 1000 | -14.935 | 4.0489 / 660.22 | 0.3197 / 0.8083 / 1.035 | 15.706% | 22.3 / 36.89 |
| selected-config | 2000 | -54.529 | 1420.1 / 4984.8 | 0.2292 / 0.6915 / 0.9461 | 26.578% | 25.23 / 406.5 |
| selected-config | 500 | -14.215 | 0.76142 / 1025.6 | 0.5604 / 0.8753 / 0.9766 | 7.905% | 17.99 / 273.2 |

| candidate | iteration | mode | source / strict | distance median cm | contact / 2+ / 3+ / thumb-other trials | height gain median / max m |
|---|---:|---|---|---:|---|---|
| action-clipping/clip1 | 1000 | A | 0/90 / 0/90 | 4.7469 | 3 / 0 / 0 / 0 | -0.003014 / 0.00161 |
| action-clipping/clip1 | 1000 | B | 0/90 / 0/90 | 4.7469 | 3 / 0 / 0 / 0 | -0.003014 / 0.002253 |
| action-clipping/clip1 | 500 | A | 0/90 / 0/90 | 5.0163 | 3 / 1 / 0 / 0 | -0.003014 / 2.384e-07 |
| action-clipping/clip1 | 500 | B | 0/90 / 0/90 | 5.0163 | 3 / 1 / 0 / 0 | -0.003014 / 2.384e-07 |
| action-clipping/none | 1000 | A | 0/90 / 0/90 | 6.1175 | 5 / 2 / 0 / 0 | -0.003014 / 0.0004517 |
| action-clipping/none | 1000 | B | 0/90 / 0/90 | 6.1175 | 5 / 2 / 0 / 0 | -0.003014 / 0.0004516 |
| action-clipping/none | 500 | A | 0/90 / 0/90 | 1.7362 | 53 / 33 / 17 / 0 | -0.003063 / 0.0002408 |
| action-clipping/none | 500 | B | 0/90 / 0/90 | 1.7408 | 53 / 33 / 18 / 0 | -0.003321 / 0.0001668 |
| lambda/095 | 1000 | A | 0/90 / 0/90 | 4.2215 | 8 / 1 / 0 / 0 | -0.003014 / 0 |
| lambda/095 | 1000 | B | 0/90 / 0/90 | 4.2215 | 8 / 1 / 0 / 0 | -0.003014 / -1.192e-07 |
| lambda/095 | 500 | A | 0/90 / 0/90 | 2.3002 | 14 / 3 / 1 / 0 | -0.004187 / 7.153e-07 |
| lambda/095 | 500 | B | 0/90 / 0/90 | 2.3002 | 14 / 3 / 1 / 0 | -0.004187 / 0.0003568 |
| lambda/time-equivalent | 1000 | A | 0/90 / 0/90 | 1.9818 | 63 / 11 / 4 / 7 | -0.005532 / 0.003989 |
| lambda/time-equivalent | 1000 | B | 0/90 / 0/90 | 1.9502 | 63 / 10 / 3 / 7 | -0.005532 / 0.004231 |
| lambda/time-equivalent | 500 | A | 0/90 / 0/90 | 1.9507 | 57 / 18 / 2 / 0 | -0.004187 / -1.192e-07 |
| lambda/time-equivalent | 500 | B | 0/90 / 0/90 | 1.966 | 58 / 19 / 2 / 0 | -0.004187 / -1.192e-07 |
| learning-rate/adaptive | 1000 | A | 0/90 / 0/90 | 6.1175 | 5 / 2 / 0 / 0 | -0.003014 / 0.0004517 |
| learning-rate/adaptive | 1000 | B | 0/90 / 0/90 | 6.1175 | 5 / 2 / 0 / 0 | -0.003014 / 0.0004516 |
| learning-rate/adaptive | 500 | A | 0/90 / 0/90 | 1.7362 | 53 / 33 / 17 / 0 | -0.003063 / 0.0002408 |
| learning-rate/adaptive | 500 | B | 0/90 / 0/90 | 1.7408 | 53 / 33 / 18 / 0 | -0.003321 / 0.0001668 |
| learning-rate/fixed | 1000 | A | 0/90 / 0/90 | 4.2215 | 8 / 1 / 0 / 0 | -0.003014 / 0 |
| learning-rate/fixed | 1000 | B | 0/90 / 0/90 | 4.2215 | 8 / 1 / 0 / 0 | -0.003014 / -1.192e-07 |
| learning-rate/fixed | 500 | A | 0/90 / 0/90 | 2.3002 | 14 / 3 / 1 / 0 | -0.004187 / 7.153e-07 |
| learning-rate/fixed | 500 | B | 0/90 / 0/90 | 2.3002 | 14 / 3 / 1 / 0 | -0.004187 / 0.0003568 |
| min-std/min020 | 1000 | A | 0/90 / 0/90 | 1.9818 | 63 / 11 / 4 / 7 | -0.005532 / 0.003989 |
| min-std/min020 | 1000 | B | 0/90 / 0/90 | 1.9502 | 63 / 10 / 3 / 7 | -0.005532 / 0.004231 |
| min-std/min020 | 500 | A | 0/90 / 0/90 | 1.9507 | 57 / 18 / 2 / 0 | -0.004187 / -1.192e-07 |
| min-std/min020 | 500 | B | 0/90 / 0/90 | 1.966 | 58 / 19 / 2 / 0 | -0.004187 / -1.192e-07 |
| min-std/none | 1000 | A | 0/90 / 0/90 | 1.9818 | 63 / 11 / 4 / 7 | -0.005532 / 0.003989 |
| min-std/none | 1000 | B | 0/90 / 0/90 | 1.9502 | 63 / 10 / 3 / 7 | -0.005532 / 0.004231 |
| min-std/none | 500 | A | 0/90 / 0/90 | 1.9507 | 57 / 18 / 2 / 0 | -0.004187 / -1.192e-07 |
| min-std/none | 500 | B | 0/90 / 0/90 | 1.966 | 58 / 19 / 2 / 0 | -0.004187 / -1.192e-07 |
| selected-config | 100 | A | 0/90 / 0/90 | 1.8171 | 47 / 21 / 8 / 1 | -0.003493 / 0.0002469 |
| selected-config | 100 | B | 0/90 / 0/90 | 1.8171 | 47 / 21 / 8 / 1 | -0.003565 / 0.0001773 |
| selected-config | 1000 | A | 0/90 / 0/90 | 1.9818 | 63 / 11 / 4 / 7 | -0.005532 / 0.003989 |
| selected-config | 1000 | B | 0/90 / 0/90 | 1.9502 | 63 / 10 / 3 / 7 | -0.005532 / 0.004231 |
| selected-config | 2000 | A | 1/90 / 1/90 | 1.8658 | 81 / 52 / 31 / 37 | -0.008247 / 0.2817 |
| selected-config | 2000 | B | 2/90 / 2/90 | 1.8582 | 81 / 52 / 32 / 37 | -0.007546 / 0.2902 |
| selected-config | 500 | A | 0/90 / 0/90 | 1.9507 | 57 / 18 / 2 / 0 | -0.004187 / -1.192e-07 |
| selected-config | 500 | B | 0/90 / 0/90 | 1.966 | 58 / 19 / 2 / 0 | -0.004187 / -1.192e-07 |

选择 clip_actions = None：

- 源实现没有[-1,1]裁剪；None保留源动作范围。
- clip1降低target clipping，但500节点contact 53/90→3/90，distance 1.736→5.016 cm，value final 1.498→13.580。
- 1000节点clip1 value final79.691 vs None1.734；Allegro qvel217.766 vs151.661 rad/s；contact3/90 vs5/90，2+0/90 vs2/90。
- clip1在1000距离4.747 cm优于None6.117 cm，但未抵消critic、速度和接触指标退化。
- None也不是稳定配置；其500→1000 contact53→5，distance1.736→6.117 cm，后续仍需LR/lambda/std实验。

选择 lambda = 0.9957346812224394：

- 500节点time lambda相对0.95：reward last50 -14.215 > -17.124，距离约1.95 cm < 2.30 cm，contact 57/90 > 14/90，2+ contact 18/90 > 3/90（Mode A）。
- 1000节点time lambda相对0.95：reward last50 -14.935 > -18.868；距离A/B 1.982/1.950 cm < 4.221 cm；contact 63/90 > 8/90，thumb-other 7/90 > 0/90。
- time lambda 500→1000的object contact 57/58→63/63，距离约1.95/1.97→1.98/1.95 cm，没有V1.4式接触崩溃与距离扩大；但2+ contact从18/19下降到11/10。
- 1000阶段hand qvel peak 36.89 < 214.40 rad/s，target clipping 15.71% < 18.17%；UR5 peak 22.30 > 18.11 rad/s，仍有动作/dynamics代价。
- critic代价明确：time lambda的500 peak 1025.63 > 56.64，1000 peak 660.22 > 123.80，1000 final 4.049 > 1.514。峰值有波动，但最终没有V1.4的持续大幅爆炸；后续必须继续通过gate。
- 选择基于本次接触保持、距离及reward改善；source/strict仍均0/90，抬升保持率不足1%，不能称已学会稳定抓取。

选择 schedule = fixed：

- 1000节点value loss阶段峰值：fixed 123.799 < adaptive 514.392；final 1.514 < 1.734。500阶段峰值也为56.641 < 194.685。
- 1000节点A/B距离中位数：fixed 4.221 cm < adaptive 6.117 cm；有object contact的trial：8/90 > 5/90。
- 500→1000仍然退化，但fixed接触14→8、距离2.300→4.221 cm、reward last50 -17.124→-18.868；adaptive接触53→5、距离约1.74→6.117 cm、reward -14.874→-18.100，fixed回退较小。
- 代价：fixed的500节点行为更差；1000阶段target clipping 18.17% > 6.29%，hand qvel peak 214.40 > 151.66 rad/s，2+ contact 1/90 < 2/90。不能声称解决了动作饱和或高速问题。
- 依据本阶段critic峰值和1000节点距离/接触选择fixed作为lambda实验基线；尚未达到稳定长训练门槛，两者source/strict均0/90。

选择 minimum_action_std = 0.2：

- 500和1000节点None/0.2的全部model参数、optimizer state以及逐轮PPO/Action/Dynamics记录严格相等，详见ab-equivalence.json；两种A/B抬升评估也一致。
- 所选fixed LR + time lambda下，1000次完整更新的最低std为0.319182 > 0.2，因此本次正式对照没有触发下界，不能声称0.2改善了contact、critic或success。
- 16-env激活下界测试证明扩展只在完整PPO update后精确投影std，actor/critic其他参数和optimizer与None完全一致。
- 选择0.2恢复已核实的源Teacher std参数下界；当前短程没有代价，也没有观测到任务收益。该选择是在1000轮对照后作出；后续2000观察到下界激活，但无None的2000配对实验，不能确定其收益。

选择 final PPO candidate = {'clip_actions': None, 'schedule': 'fixed', 'lam': 0.9957346812224394, 'minimum_std': 0.2, 'learning_rate': 0.0005}：

- clip_actions=None：源Gaussian无[-1,1]裁剪，clip1工程对照虽降低target clipping，却恶化critic和接触，详见action-clipping/decision.json。
- fixed LR=5e-4：降低critic峰值，1000节点比adaptive的距离与接触更好，代价是500节点较差及部分qvel/saturation偏高，详见learning-rate/decision.json。
- lambda=0.95**(CONTROL_DT/SOURCE_CONTROL_DT)：500→1000接触保持在57/58→63/63、距离约2 cm，没有旧基线的接触崩溃；仍有critic尖峰与2+ contact回退，详见lambda/decision.json。
- minimum_action_std=0.2：恢复源下界；前1000轮未激活且参数/optimizer严格等价，不宣称其任务收益，详见min-std/decision.json。
- Phase6选择的是用于最终from-scratch验收的候选；现已完成100/500/1000/2000节点，2000的critic稳定性gate未通过，未进入5000。

复用行指配置、seed和评估协议完全一致的已完成节点引用，不计为新增训练或独立重复；原始文件路径见各reference.json及training_progress.json的reused_from。

## B. 观察趋势

候选选择及节点继续决定必须同时看critic、reward、接触、距离、qvel和动作分布；单个0/90不是选择依据。选择理由另存各phase/decision.json。

## C. 尚不能证明

finite checkpoint及TensorBoard不等于稳定学习或成功抓取。2000节点实测source/strict A=1/90、B=2/90；没有证明稳定抓取，也不能预言20000失败。

原始逐iteration动作、KL、std、VRAM记录在对应stage/iterations.jsonl；qvel为物理子步测量，contact peak为奖励读取时的过滤法向+切向总力，不能称为所有物理子步的contact上限。2+/3+来自现有诊断的同时接触body数，不能等同于已验证的不同手指闭合或force closure。
raw与wrapper动作p95/p99按完整rollout精确计算；根action_diagnostics.csv的max_abs是阶段峰值，其余动作分布字段为最近50次rollout统计的均值，不是全历史样本的全局分位数。std median为torch.median的下中位数。wall time为learn循环，不含scene创建及独立评估。


## 最终结论的证据边界

**已证明：** 上述源语义与等价检查、四组单变量结果、最终2000的少量source/strict成功均为直接实测。当前未重现V1.4接触归零和距离约14cm，但实测critic/reward恶化。checkpoint/optimizer、TensorBoard标量及运行时finite，17项reward记录完整、outside=0和late VRAM平台通过，不能勾选稳定学习。

**观察趋势：** 500→1000→2000接触57/58→63/63→81/81，距离约2cm保持；2+先下降后增加，3+/thumb-other在2000增加。table-contact同时增加至30/90，多数trial未保留抬升。1700之后critic持续数百级loss，伴随reward变差和动作饱和增加。真实任务进展与优化退化并存。

与V1.4 2000相比，value final1420.09低于3429.76、stage peak4984.83低于189208.49，但reward last50 −54.53差于−24.40，clipping26.58%高于约12.14%。旧评估30 trials/1 round，本轮90 trials/3 rounds，不能直接作配对成功率推断；旧证据见[baseline/optimize4_report.md](baseline/optimize4_report.md)。

**尚不能证明：** 该候选稳定收敛、20000必失败、某一个PPO参数是唯一根因、reward定义错误或action scale必错。没有None的2000对照，std下界收益未确定。单训练seed、3-round证据不能支持稳健成功率或统计显著性结论。未执行reward/action scale/object physics搜索。

## 测量范围与验收文件

- stage500聚合覆盖1–500，stage1000覆盖501–1000，stage2000覆盖1001–2000；stage100是同fresh run前100次快照。value/raw峰值、qvel、force、termination、target clip及processed/final target统计均是当前窗口，不是所有历史累计。total_env_transitions为候选累计更新×2048×32；复用行不计新增训练。
- qvel每个物理子步120Hz测量；contact force在reward读取60Hz测量过滤法向+摩擦冲量/physics_dt，不是全部物理子步的峰值。eval接触诊断为60Hz、0.1N阈值。2+/3+统计同时接触body数，不证明不同手指闭合或force closure。
- retention为grasp-end有接触trial的抬升any-contact时间比例均值；grasp-end无接触时None。raw/wrapper分位数每rollout精确计算，汇总为最近50轮统计均值；std median为下中位数。FPS/learn wall包含本轮测量开销，不能作为优化benchmark。
- from_scratch诊断最初在train.py消费CLI后读取sys.argv导致resume误标；已依据命令及更新区间修正，不改变训练/checkpoint。见[metadata-corrections.json](metadata-corrections.json)。
- [training_progress.json](training_progress.json)、[training_progress.csv](training_progress.csv)、[evaluation_progress.csv](evaluation_progress.csv)、[action_diagnostics.csv](action_diagnostics.csv)是全部完成节点；逐轮记录在stage/iterations.jsonl，逐trial诊断在mode/grasp_diagnostics.json/.csv；checkpoint与TensorBoard在对应training run路径。
- clip1首次eval的Hydra None→float错误保留在action-clipping/clip1/stage500/modeA-hydra-type-error.log；直接CLI修复后使用同一checkpoint完成评估，未重训。

## 最终验收清单

- [x] baseline保存、source动作/std语义确认。
- [x] 四组单变量A/B、raw/wrapper/processed/target、动力学和正式90-trial A/B评估完成。
- [x] 选出最终验收候选，重新从头500、记录100，恢复至1000/2000。
- [x] checkpoint/optimizer/TensorBoard/runtime finite、17 reward记录、outside=0、VRAM无持续增长。
- [ ] 获得稳定PPO配置；500→1000→2000消除系统性优化退化。2000的critic/reward/saturation/qvel恶化，未通过。
- [ ] 5000；文档第19节条件未满足，因此未执行。
- [ ] 10000/20000；前置条件未满足，因此未执行。

语法、git diff与完整记录核查见[final_validation.json](final_validation.json)。未提交git。

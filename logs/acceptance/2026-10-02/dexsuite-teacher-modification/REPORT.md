# 优化2执行与验收记录（2026-10-02）

已按 Phase 0–7 在 `grasp` 环境中修改和验证现有 `Grasp1-UR5-Allegro-Teacher-v0`。
本轮完成配置迁移、运行稳定性、profile 和 PPO smoke；抓取效果单独记录在评估结果中。
环境：Conda grasp；IsaacLab 2.3.2；Isaac Sim 5.1.0 prebuilt；RTX 3070 Ti Laptop，Driver 580。

## 最终配置

| 项目 | 实际配置 |
|---|---|
| Task / Robot | 原 Task ID；UR5 + Allegro 单一 22-DOF Articulation |
| Physics / Policy | 120 Hz / 60 Hz，dt=1/120，decimation=2 |
| Episode | 4 秒，240 policy steps |
| Robot / Object solver | 32/1；16/0，加载后 USD 属性已核验 |
| PhysX | 显式 TGS，solver_type=1 |
| Self collision | False；开启试验出现巨大接触力 |
| Object | 保留源 Articulation，技术原因见下文 |
| Action | 官方 RelativeJointPositionAction，22 关节保持源顺序 |
| Scale | UR5 0.005，Allegro 0.015 |
| Observation | 153 维；22:44 为 processed_actions，即 Δq |
| PPO | rollout=32，默认 max_iterations=20001 |
| Evaluate | grasp=4s，lift=2s，ramp=1.5s，按实际 step_dt 换算 |

Teacher 的 17 项奖励公式、数据集、200 点 affordance、几何、ContactSensor 过滤和 PPO 网络/其它超参保持现有配置。
`runtime.py`、`geometry.py`、`keypoints.py`、`rewards.py` 中的 Optimization-1 实现保留。
删除旧延迟动作类及其导出；重置状态时清零相对动作残差，包含 partial reset 和 collision fallback。

## 按阶段的运行证据

Phase 0：`baseline/` 保存 git status、HEAD、diff、环境/PPO 配置；保存 256/1024 benchmark 和 256 CPU/CUDA trace。
baseline-1024 测量 JSON 和正常关闭日志已保存；会话中断导致 shell exit 文件未写出。
Phase 1：compile；16 env zero/random 各 300 步；300 步数值/partial reset/cache 验证；真实接触探针和事件缓存测试。
Phase 2：compile；self collision off/on 各 300 步对照；zero/random 各 300 步；256 env benchmark/profile。
Phase 3：全量源 URDF/USD 检查，保留 Articulation；compile 和 16 env 300 步/cache 验证。
Phase 4：compile；22 个单关节及全零动作、target 与 observation 语义验证；zero/random 各 300 步；scale=0.1 对照；partial reset 残差清零及事件缓存测试。
Phase 5：compile；16 env PPO 单 iteration，checkpoint 和 TensorBoard 17 项奖励检查。
Phase 6：compile；30 ShapeNet 物体执行 duration-based evaluate；play 完成 1 个回合。
Phase 7：256/1024/2048 benchmark；最终 256/2048 CPU/CUDA profile；2048×32 PPO 单 iteration 与 20 iteration smoke；最终 checkpoint evaluate/play。
各阶段已完成的 compile/runtime 命令退出码为 0，详见对应 `.exit`、`.log`、`.json`。

初始碰撞：真实重叠探针在两个 physics substeps（1/60 秒）后检测到碰撞；原首次 control-step 等待保留。
partial reset 验证环境 [3,7,11]，其它行状态及缓存保持；同一步缓存复用、新控制步失效，运行步内名称解析次数为 0。
stable-state、bias、collision fallback 缓存边界测试通过；reset 文件读取次数为 0。

## Object representation 与 self collision

检查 35 个训练物体、30 个 ShapeNet 物体和 dummy，共 66 个资产，全部为两个 rigid bodies 和一个启用的 revolute joint。
源 URDF 已含该关节，范围 0–0.001 rad，USD 范围为 0–0.05729578 degree；该关节未锁定且运行中真实运动。
Teacher 以 top 位姿构造 affordance/关键点，并读取 top 线速度和角速度奖励；top/bottom 保留各自质量、惯量和接触过滤。
因此按本任务保持这些物理语义的要求，保留 Articulation 及 16/0 solver。RigidObject 对照项不适用，未虚构转换或对照 profile。
详见 `phase2-256.assets.json` 与各运行 JSON 中的 object_joint_limits/min/max。
同时观察到物体关节有超出窄限位的状态，旧配置基线已有此现象；本轮数值有限性通过，不将其表述为严格限位精度已通过。

| 16 env / 300 steps | 最大接触力 N | 最大关节速度 rad/s | invalid_hand_height |
|---|---:|---:|---:|
| phase2-self-off | 259.132 | 16.977 | 0 |
| phase2-self-on | 59726.762 | 16.977 | 0 |
| phase4-validation | 299.911 | 29.980 | 0 |
| phase4-scale01 | 2046.784 | 317.411 | 5 |

开启 self collision 出现约 59.7kN 峰值，保持关闭。scale=0.1 出现过高关节速度、接触冲量及高度终止，保留分关节 scale。
Relative Action 使用官方目标实现；无显式目标 clipping。限位外目标比例在小 scale / 0.1 试验中分别为 1.1506% / 4.4735%，该比例是越界目标统计。

## Benchmark

计时包括 env.step、数值检查、奖励/终止累计；GPU 每 1 秒采样。
未开启 validate/scale/self_collision 的 benchmark 不采集 max_joint/effort/contact/target 字段，其 JSON 初始值 0 不代表测得的峰值。
simulated seconds/s 按每个环境的模拟时长计算；physics substeps/s 按 N×steps×decimation 计算。
旧 70 steps=14s；新 70 steps=1.166667s。固定 4s：旧 20 steps、新 240 steps。
固定 4s 测试中旧配置无回合重置，新配置会整批重置，耗时包含实际 reset 工作。

| 配置 | envs | steps | 模拟 s | wall s | env steps/s | physics env-substeps/s | sim s/wall s | wall s/sim s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline-256 | 256 | 70 | 14.000000 | 63.690 | 281 | 5627 | 0.2198 | 4.5493 |
| baseline-256 | 256 | 20 | 4.000000 | 14.169 | 361 | 7227 | 0.2823 | 3.5422 |
| baseline-1024 | 1024 | 70 | 14.000000 | 134.271 | 534 | 10677 | 0.1043 | 9.5908 |
| baseline-1024 | 1024 | 20 | 4.000000 | 32.440 | 631 | 12626 | 0.1233 | 8.1100 |
| phase7-256 | 256 | 70 | 1.166667 | 2.693 | 6654 | 13308 | 0.4332 | 2.3084 |
| phase7-256 | 256 | 240 | 4.000000 | 12.386 | 4960 | 9921 | 0.3229 | 3.0966 |
| phase7-1024 | 1024 | 70 | 1.166667 | 3.293 | 21767 | 43535 | 0.3543 | 2.8226 |
| phase7-1024 | 1024 | 240 | 4.000000 | 15.300 | 16062 | 32125 | 0.2614 | 3.8251 |
| phase7-2048 | 2048 | 70 | 1.166667 | 4.462 | 32129 | 64257 | 0.2615 | 3.8246 |
| phase7-2048 | 2048 | 240 | 4.000000 | 19.677 | 24980 | 49960 | 0.2033 | 4.9191 |

最终 256/1024/2048 固定 4s 测试分别完成 256/1024/2048 次 timeout，invalid_observation 和 invalid_hand_height 均为 0。

| 运行 | NVML peak MiB | GPU 利用率全程均值 % | 高显存样本均值 % | GPU 利用率峰值 % |
|---|---:|---:|---:|---:|
| baseline-256 | 2844 | 55.67 | 72.33 | 99 |
| baseline-1024 | 4034 | 45.57 | 96.94 | 100 |
| phase7-256 | 2844 | 20.11 | 46.62 | 96 |
| phase7-1024 | 4096 | 9.49 | 69.30 | 80 |
| phase7-2048 | 5648 | 6.92 | 55.71 | 99 |
| phase7-ppo-single | 5568 | 0.71 | 31.25 | 62 |
| phase7-ppo-counted-smoke | 5572 | 14.01 | 63.84 | 74 |

GPU 全程均值包含 CPU 场景创建；高显存样本指 memory.used≥本次 peak 的 90%。原始逐秒数据保留在 `.gpu.csv`。

counted smoke 高显存阶段为 5552–5572 MiB，20 iterations 内未见持续累积增长。

## GPU / CPU Profile

每份 profile 记录 10 个 policy steps。旧对应 200 physics steps/2s，新对应 20 physics steps/1/6s。
百分比的分母是该 trace 全部 CUDA kernel duration 总和。

| 运行 | Kernel | absolute ms | count | kernel time % |
|---|---|---:|---:|---:|
| baseline-256 | artiSolveInternalConstraintsTGS1T | 3999.696 | 6600 | 53.098 |
| baseline-256 | stepArticulation1TTGS | 1641.544 | 6400 | 21.793 |
| baseline-256 | artiPropagateRigidImpulsesAndSolveSelfConstraintsTGS1T | 364.194 | 6600 | 4.835 |
| phase7-256 | artiSolveInternalConstraintsTGS1T | 99.680 | 660 | 49.734 |
| phase7-256 | stepArticulation1TTGS | 42.501 | 640 | 21.205 |
| phase7-256 | artiPropagateRigidImpulsesAndSolveSelfConstraintsTGS1T | 2.071 | 660 | 1.033 |
| phase7-2048 | artiSolveInternalConstraintsTGS1T | 168.446 | 660 | 43.789 |
| phase7-2048 | stepArticulation1TTGS | 107.613 | 640 | 27.975 |
| phase7-2048 | artiPropagateRigidImpulsesAndSolveSelfConstraintsTGS1T | 5.361 | 660 | 1.394 |

| 运行 | 阶段 | ms/control step | 调用数/10 policy steps |
|---|---|---:|---:|
| baseline-256 | physics | 770.062 | 200 |
| baseline-256 | scene_update | 28.970 | 200 |
| baseline-256 | reward | 13.350 | 10 |
| baseline-256 | observation | 1.176 | 10 |
| phase7-256 | physics | 19.795 | 20 |
| phase7-256 | scene_update | 2.381 | 20 |
| phase7-256 | reward | 11.694 | 10 |
| phase7-256 | observation | 1.168 | 10 |
| phase7-2048 | physics | 37.052 | 20 |
| phase7-2048 | scene_update | 2.460 | 20 |
| phase7-2048 | reward | 15.984 | 10 |
| phase7-2048 | observation | 1.543 | 10 |

基线 CPU 采用 trace 的 user_annotation duration；最终采用 perf_counter_ns 包围实际调用。Torch key_averages 部分 user scopes 返回 0，因此保留原 trace 并采用上述有效计时。

## Reward 时间尺度

所有 source_coeff/CONTROL_DT 随 CONTROL_DT=1/60 生效；每步系数保留，单位模拟秒累计尺度随频率变化。
下表来自 Relative Action 16 env / 300 steps，episode 项为实际完成的 4 秒回合平均值。

| Reward | mean/step | mean/simulated second | completed episode mean |
|---|---:|---:|---:|
| affordance_reward | -0.722952 | -43.377094 | -156.997925 |
| affordance_contact_reward | 0.006300 | 0.378024 | 1.890120 |
| affordance_impulse_reward | 0.003130 | 0.187814 | 0.939069 |
| table_reward | -0.024550 | -1.472975 | -7.364877 |
| table_contact_reward | -0.011956 | -0.717338 | -3.586692 |
| table_impulse_reward | -0.013055 | -0.783286 | -3.916428 |
| arm_height_reward | 0.000000 | 0.000000 | 0.000000 |
| arm_contact_reward | -0.005791 | -0.347463 | -1.737317 |
| arm_impulse_reward | -0.005106 | -0.306364 | -1.531821 |
| arm_collision_reward | -0.065000 | -3.900000 | -19.500000 |
| push_reward | 0.000000 | 0.000000 | 0.000000 |
| wrist_vel_reward_ | -0.006100 | -0.366003 | -1.374142 |
| wrist_qvel_reward_ | -0.006239 | -0.374314 | -1.543699 |
| obj_vel_reward_ | -0.017569 | -1.054138 | -4.908110 |
| obj_qvel_reward_ | -0.093301 | -5.598081 | -26.616585 |
| obj_displacement_reward | -0.020741 | -1.244465 | -5.374759 |
| arm_joint_vel_reward_ | -0.225192 | -13.511518 | -64.596413 |

各规模 benchmark JSON 也保留全部 17 项 mean、RMS 和 episode 累计；reward/observation 均有限。

## PPO

2048×32 正式单 iteration 共 65536 env steps；20 iteration smoke 共 1310720 env steps。
单 iteration 使用现有 train.py；计数 smoke 在验收进程中调用同一 train.py，仅累计 termination masks。
首次 smoke 已通过 finite 验收；结束应用前未写出额外计数，修正为在 learn 返回时写出后补跑 counted smoke。

| Metric | 单 iteration | smoke 均值 | smoke 最后值 | smoke 范围 |
|---|---:|---:|---:|---|
| Perf/collection time | 2.81102 | 3.36286 | 3.20569 | 2.81706–3.61563 |
| Perf/learning_time | 0.125922 | 0.0521462 | 0.0497699 | 0.0473344–0.0990169 |
| Perf/total_fps | 22314 | 19264.5 | 20131 | 17881–22474 |
| Train/mean_reward | -43.0211 | -242.835 | -236.764 | -384.691–-43.0211 |
| Train/mean_episode_length | 27.67 | 198.891 | 240 | 27.67–240 |
| Loss/value_function | 256.663 | 239.532 | 85.8167 | 85.8167–538.121 |
| Loss/surrogate | -0.00203575 | -0.00177565 | -0.000898566 | -0.00393419–0.00103086 |
| Loss/entropy | 31.1476 | 30.3591 | 29.7339 | 29.714–31.1476 |
| Policy/mean_noise_std | 0.993667 | 0.963634 | 0.93862 | 0.93862–0.993667 |
| Loss/learning_rate | 0.00379688 | 0.00860465 | 0.01 | 0.00197531–0.01 |

Runner 保留 init_at_random_ep_len=True；早期 mean episode length 包含随机起点的短片段，实际 episode 上限为 240。
learning_rate 按原有 adaptive schedule 变化。全部 scalar、model/optimizer checkpoint 张量有限；17 项奖励均有记录。
逐步终止 term 次数：`{"time_out": 5471, "invalid_hand_height": 0, "invalid_observation": 0}`。

| Reward log | smoke mean | smoke last |
|---|---:|---:|
| Episode_Reward/affordance_reward | -36.3744 | -46.3052 |
| Episode_Reward/affordance_contact_reward | 0.222782 | 0.0927716 |
| Episode_Reward/affordance_impulse_reward | 0.111157 | 0.0443958 |
| Episode_Reward/table_reward | -0.423459 | -0.0784419 |
| Episode_Reward/table_contact_reward | -0.039261 | -0.0262673 |
| Episode_Reward/table_impulse_reward | -0.0462489 | -0.0278993 |
| Episode_Reward/arm_height_reward | 0 | 0 |
| Episode_Reward/arm_contact_reward | -0.104516 | -0.061211 |
| Episode_Reward/arm_impulse_reward | -0.112256 | -0.0777988 |
| Episode_Reward/arm_collision_reward | -0.988822 | -0.608774 |
| Episode_Reward/push_reward | 0 | 0 |
| Episode_Reward/wrist_vel_reward_ | -1.10279 | -0.838479 |
| Episode_Reward/wrist_qvel_reward_ | -0.546374 | -0.539942 |
| Episode_Reward/obj_vel_reward_ | -1.1625 | -0.66884 |
| Episode_Reward/obj_qvel_reward_ | -5.20832 | -1.40332 |
| Episode_Reward/obj_displacement_reward | -1.25579 | -1.10551 |
| Episode_Reward/arm_joint_vel_reward_ | -12.5882 | -6.70651 |

Checkpoint：

- 单 iteration：`logs/rsl_rl/grasp1_ur5_allegro_teacher/2026-10-02_20-22-17_teacher_dexsuite_validation/model_0.pt`
- 最终 counted smoke：`logs/rsl_rl/grasp1_ur5_allegro_teacher/2026-10-02_20-34-07_teacher_dexsuite_counted_smoke/model_19.pt`

## Evaluate / Play 与训练边界

抓取默认 4 秒，与训练回合一致；抬升 2 秒，其中 1.5 秒平滑过渡、0.5 秒保持。
实际 env.step_dt=1/60，对应 240 / 120 / 90 steps，评估 episode 额外留一控制步。
Play 实时 pacing 原已使用 env.step_dt；视频长度仍是 policy iteration 数，保留原接口。
最终 counted-smoke checkpoint 的 ShapeNet 单轮评估：0/30，成功率 0.00%。
结果见 `final-evaluation/quantitative_eval.json` 与 `per_object_success.csv`；最终 Play 完成 1 个回合。
本轮不直接运行完整 20001 iterations。100/500 及更长训练按文档的学习趋势、抬升行为和成功率条件另行推进。

所有配置改动和源/运行证据保存在本目录；`summary.json` 汇总机器可读结果，完整 benchmark 数据分别保存在各运行 JSON 中。

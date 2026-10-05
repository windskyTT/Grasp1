"""汇总优化2本轮已经落盘的测量与验收证据。"""
import csv
import json
from pathlib import Path

root = Path(__file__).parent
def read(label):
    return json.loads((root / (label+'.json')).read_text())
def gpu(label):
    rows = list(csv.reader((root/(label+'.gpu.csv')).open()))[1:]
    samples = [(float(row[1].split()[0]), float(row[2].split()[0])) for row in rows]
    peak = max(memory for memory,util in samples)
    high = [(memory,util) for memory,util in samples if memory >= peak*.9]
    return dict(peak_mib=peak, utilization_mean_all=sum(util for memory,util in samples)/len(samples),
                utilization_mean_high_memory=sum(util for memory,util in high)/len(high),
                utilization_max=max(util for memory,util in samples),
                high_memory_range_mib=[min(memory for memory,util in high),max(memory for memory,util in high)])

benchmarks = {label:read(label) for label in ['baseline-256','baseline-1024','phase7-256','phase7-1024','phase7-2048']}
single, smoke = read('phase7-single-check'), read('phase7-counted-smoke-check')
counts = read('ppo-termination-counts')
evaluation = json.loads((root/'final-evaluation/quantitative_eval.json').read_text())
gpu_data = {label:gpu(label) for label in list(benchmarks)+['phase7-ppo-single','phase7-ppo-counted-smoke']}
summary = dict(benchmarks=benchmarks, gpu=gpu_data,
               ppo_single=single, ppo_smoke=smoke, ppo_termination_counts=counts,
               evaluation=evaluation)
(root/'summary.json').write_text(json.dumps(summary,indent=2))

lines = ['# 优化2执行与验收记录（2026-10-02）', '',
         '已按 Phase 0–7 在 `grasp` 环境中修改和验证现有 `Grasp1-UR5-Allegro-Teacher-v0`。',
         '本轮完成配置迁移、运行稳定性、profile 和 PPO smoke；抓取效果单独记录在评估结果中。',
         '环境：Conda grasp；IsaacLab 2.3.2；Isaac Sim 5.1.0 prebuilt；RTX 3070 Ti Laptop，Driver 580。', '',
         '## 最终配置', '',
         '| 项目 | 实际配置 |', '|---|---|',
         '| Task / Robot | 原 Task ID；UR5 + Allegro 单一 22-DOF Articulation |',
         '| Physics / Policy | 120 Hz / 60 Hz，dt=1/120，decimation=2 |',
         '| Episode | 4 秒，240 policy steps |',
         '| Robot / Object solver | 32/1；16/0，加载后 USD 属性已核验 |',
         '| PhysX | 显式 TGS，solver_type=1 |',
         '| Self collision | False；开启试验出现巨大接触力 |',
         '| Object | 保留源 Articulation，技术原因见下文 |',
         '| Action | 官方 RelativeJointPositionAction，22 关节保持源顺序 |',
         '| Scale | UR5 0.005，Allegro 0.015 |',
         '| Observation | 153 维；22:44 为 processed_actions，即 Δq |',
         '| PPO | rollout=32，默认 max_iterations=20001 |',
         '| Evaluate | grasp=4s，lift=2s，ramp=1.5s，按实际 step_dt 换算 |', '',
         'Teacher 的 17 项奖励公式、数据集、200 点 affordance、几何、ContactSensor 过滤和 PPO 网络/其它超参保持现有配置。',
         '`runtime.py`、`geometry.py`、`keypoints.py`、`rewards.py` 中的 Optimization-1 实现保留。',
         '删除旧延迟动作类及其导出；重置状态时清零相对动作残差，包含 partial reset 和 collision fallback。', '',
         '## 按阶段的运行证据', '',
         'Phase 0：`baseline/` 保存 git status、HEAD、diff、环境/PPO 配置；保存 256/1024 benchmark 和 256 CPU/CUDA trace。',
         'baseline-1024 测量 JSON 和正常关闭日志已保存；会话中断导致 shell exit 文件未写出。',
         'Phase 1：compile；16 env zero/random 各 300 步；300 步数值/partial reset/cache 验证；真实接触探针和事件缓存测试。',
         'Phase 2：compile；self collision off/on 各 300 步对照；zero/random 各 300 步；256 env benchmark/profile。',
         'Phase 3：全量源 URDF/USD 检查，保留 Articulation；compile 和 16 env 300 步/cache 验证。',
         'Phase 4：compile；22 个单关节及全零动作、target 与 observation 语义验证；zero/random 各 300 步；scale=0.1 对照；partial reset 残差清零及事件缓存测试。',
         'Phase 5：compile；16 env PPO 单 iteration，checkpoint 和 TensorBoard 17 项奖励检查。',
         'Phase 6：compile；30 ShapeNet 物体执行 duration-based evaluate；play 完成 1 个回合。',
         'Phase 7：256/1024/2048 benchmark；最终 256/2048 CPU/CUDA profile；2048×32 PPO 单 iteration 与 20 iteration smoke；最终 checkpoint evaluate/play。',
         '各阶段已完成的 compile/runtime 命令退出码为 0，详见对应 `.exit`、`.log`、`.json`。', '',
         '初始碰撞：真实重叠探针在两个 physics substeps（1/60 秒）后检测到碰撞；原首次 control-step 等待保留。',
         'partial reset 验证环境 [3,7,11]，其它行状态及缓存保持；同一步缓存复用、新控制步失效，运行步内名称解析次数为 0。',
         'stable-state、bias、collision fallback 缓存边界测试通过；reset 文件读取次数为 0。', '',
         '## Object representation 与 self collision', '',
         '检查 35 个训练物体、30 个 ShapeNet 物体和 dummy，共 66 个资产，全部为两个 rigid bodies 和一个启用的 revolute joint。',
         '源 URDF 已含该关节，范围 0–0.001 rad，USD 范围为 0–0.05729578 degree；该关节未锁定且运行中真实运动。',
         'Teacher 以 top 位姿构造 affordance/关键点，并读取 top 线速度和角速度奖励；top/bottom 保留各自质量、惯量和接触过滤。',
         '因此按本任务保持这些物理语义的要求，保留 Articulation 及 16/0 solver。RigidObject 对照项不适用，未虚构转换或对照 profile。',
         '详见 `phase2-256.assets.json` 与各运行 JSON 中的 object_joint_limits/min/max。',
         '同时观察到物体关节有超出窄限位的状态，旧配置基线已有此现象；本轮数值有限性通过，不将其表述为严格限位精度已通过。', '',
         '| 16 env / 300 steps | 最大接触力 N | 最大关节速度 rad/s | invalid_hand_height |',
         '|---|---:|---:|---:|']
for label in ['phase2-self-off','phase2-self-on','phase4-validation','phase4-scale01']:
    m=read(label)['measurements'][0]
    lines.append(f"| {label} | {m['max_contact_force']:.3f} | {m['max_joint_velocity']:.3f} | {m['done_counts']['invalid_hand_height']} |")
lines += ['', '开启 self collision 出现约 59.7kN 峰值，保持关闭。scale=0.1 出现过高关节速度、接触冲量及高度终止，保留分关节 scale。',
          'Relative Action 使用官方目标实现；无显式目标 clipping。限位外目标比例在小 scale / 0.1 试验中分别为 '
          f"{read('phase4-validation')['measurements'][0]['target_outside_limits_fraction']:.4%} / "
          f"{read('phase4-scale01')['measurements'][0]['target_outside_limits_fraction']:.4%}，该比例是越界目标统计。", '',
          '## Benchmark', '',
          '计时包括 env.step、数值检查、奖励/终止累计；GPU 每 1 秒采样。',
          '未开启 validate/scale/self_collision 的 benchmark 不采集 max_joint/effort/contact/target 字段，其 JSON 初始值 0 不代表测得的峰值。',
          'simulated seconds/s 按每个环境的模拟时长计算；physics substeps/s 按 N×steps×decimation 计算。',
          '旧 70 steps=14s；新 70 steps=1.166667s。固定 4s：旧 20 steps、新 240 steps。',
          '固定 4s 测试中旧配置无回合重置，新配置会整批重置，耗时包含实际 reset 工作。', '',
          '| 配置 | envs | steps | 模拟 s | wall s | env steps/s | physics env-substeps/s | sim s/wall s | wall s/sim s |',
          '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
for label,d in benchmarks.items():
    for m in d['measurements']:
        lines.append(f"| {label} | {d['num_envs']} | {m['steps']} | {m['simulated_seconds']:.6f} | {m['wall_seconds']:.3f} | {m['env_steps_per_second']:.0f} | {m['physics_env_substeps_per_second']:.0f} | {m['simulated_seconds_per_wall_second']:.4f} | {m['wall_seconds_per_simulated_second']:.4f} |")
lines += ['', '最终 256/1024/2048 固定 4s 测试分别完成 256/1024/2048 次 timeout，invalid_observation 和 invalid_hand_height 均为 0。', '',
          '| 运行 | NVML peak MiB | GPU 利用率全程均值 % | 高显存样本均值 % | GPU 利用率峰值 % |',
          '|---|---:|---:|---:|---:|']
for label,d in gpu_data.items():
    lines.append(f"| {label} | {d['peak_mib']:.0f} | {d['utilization_mean_all']:.2f} | {d['utilization_mean_high_memory']:.2f} | {d['utilization_max']:.0f} |")
lines += ['', 'GPU 全程均值包含 CPU 场景创建；高显存样本指 memory.used≥本次 peak 的 90%。原始逐秒数据保留在 `.gpu.csv`。', '',
          f"counted smoke 高显存阶段为 {gpu_data['phase7-ppo-counted-smoke']['high_memory_range_mib'][0]:.0f}–{gpu_data['phase7-ppo-counted-smoke']['high_memory_range_mib'][1]:.0f} MiB，20 iterations 内未见持续累积增长。", '',
          '## GPU / CPU Profile', '',
          '每份 profile 记录 10 个 policy steps。旧对应 200 physics steps/2s，新对应 20 physics steps/1/6s。',
          '百分比的分母是该 trace 全部 CUDA kernel duration 总和。', '',
          '| 运行 | Kernel | absolute ms | count | kernel time % |', '|---|---|---:|---:|---:|']
for label in ['baseline-256','phase7-256','phase7-2048']:
    for name,m in benchmarks[label]['gpu_profile'].items():
        lines.append(f"| {label} | {name} | {m['duration_us']/1000:.3f} | {m['count']} | {m['percent']:.3f} |")
lines += ['', '| 运行 | 阶段 | ms/control step | 调用数/10 policy steps |', '|---|---|---:|---:|']
for label in ['baseline-256','phase7-256','phase7-2048']:
    data=benchmarks[label]
    profile=data['cpu_trace_profile'] if label=='baseline-256' else data['cpu_wall_profile']
    for name,m in profile.items():
        lines.append(f"| {label} | {name} | {m['total_us']/10000:.3f} | {m['calls']} |")
lines += ['', '基线 CPU 采用 trace 的 user_annotation duration；最终采用 perf_counter_ns 包围实际调用。Torch key_averages 部分 user scopes 返回 0，因此保留原 trace 并采用上述有效计时。', '',
          '## Reward 时间尺度', '',
          '所有 source_coeff/CONTROL_DT 随 CONTROL_DT=1/60 生效；每步系数保留，单位模拟秒累计尺度随频率变化。',
          '下表来自 Relative Action 16 env / 300 steps，episode 项为实际完成的 4 秒回合平均值。', '',
          '| Reward | mean/step | mean/simulated second | completed episode mean |', '|---|---:|---:|---:|']
for name,m in read('phase4-validation')['measurements'][0]['reward_terms'].items():
    lines.append(f"| {name} | {m['mean_per_step']:.6f} | {m['mean_per_second']:.6f} | {m['completed_episode_mean']:.6f} |")
lines += ['', '各规模 benchmark JSON 也保留全部 17 项 mean、RMS 和 episode 累计；reward/observation 均有限。', '',
          '## PPO', '',
          '2048×32 正式单 iteration 共 65536 env steps；20 iteration smoke 共 1310720 env steps。',
          '单 iteration 使用现有 train.py；计数 smoke 在验收进程中调用同一 train.py，仅累计 termination masks。',
          '首次 smoke 已通过 finite 验收；结束应用前未写出额外计数，修正为在 learn 返回时写出后补跑 counted smoke。', '',
          '| Metric | 单 iteration | smoke 均值 | smoke 最后值 | smoke 范围 |', '|---|---:|---:|---:|---|']
for tag in ['Perf/collection time','Perf/learning_time','Perf/total_fps','Train/mean_reward','Train/mean_episode_length',
            'Loss/value_function','Loss/surrogate','Loss/entropy','Policy/mean_noise_std','Loss/learning_rate']:
    a,b=single['scalar_tags'][tag],smoke['scalar_tags'][tag]
    lines.append(f"| {tag} | {a['last']:.6g} | {b['mean']:.6g} | {b['last']:.6g} | {b['minimum']:.6g}–{b['maximum']:.6g} |")
lines += ['', 'Runner 保留 init_at_random_ep_len=True；早期 mean episode length 包含随机起点的短片段，实际 episode 上限为 240。',
          'learning_rate 按原有 adaptive schedule 变化。全部 scalar、model/optimizer checkpoint 张量有限；17 项奖励均有记录。',
          f"逐步终止 term 次数：`{json.dumps(counts)}`。", '',
          '| Reward log | smoke mean | smoke last |', '|---|---:|---:|']
for tag,m in smoke['scalar_tags'].items():
    if tag.startswith('Episode_Reward/'):
        lines.append(f"| {tag} | {m['mean']:.6g} | {m['last']:.6g} |")
lines += ['', 'Checkpoint：', '', f"- 单 iteration：`{single['run']}/model_0.pt`",
          f"- 最终 counted smoke：`{smoke['run']}/model_19.pt`", '',
          '## Evaluate / Play 与训练边界', '',
          '抓取默认 4 秒，与训练回合一致；抬升 2 秒，其中 1.5 秒平滑过渡、0.5 秒保持。',
          '实际 env.step_dt=1/60，对应 240 / 120 / 90 steps，评估 episode 额外留一控制步。',
          'Play 实时 pacing 原已使用 env.step_dt；视频长度仍是 policy iteration 数，保留原接口。',
          f"最终 counted-smoke checkpoint 的 ShapeNet 单轮评估：{evaluation['total_successes']}/{evaluation['total_attempts']}，成功率 {evaluation['overall_success_rate']:.2%}。",
          '结果见 `final-evaluation/quantitative_eval.json` 与 `per_object_success.csv`；最终 Play 完成 1 个回合。',
          '本轮不直接运行完整 20001 iterations。100/500 及更长训练按文档的学习趋势、抬升行为和成功率条件另行推进。', '',
          '所有配置改动和源/运行证据保存在本目录；`summary.json` 汇总机器可读结果，完整 benchmark 数据分别保存在各运行 JSON 中。']
(root/'REPORT.md').write_text('\n'.join(lines)+'\n')
print('REPORT', root/'REPORT.md')

# 优化6重现命令

运行目录与环境：

```bash
cd /home/windsky/project/Grasp1
source /home/windsky/miniconda3/etc/profile.d/conda.sh
conda activate grasp
export TERM=xterm
export PYTHONUNBUFFERED=1
```

以下验收参数与本轮实际执行一致，输出目录换成尚未使用的 `optimization6_replay/`，保留本轮证据。该目录已存在时选择新的目录。每条仿真命令单独执行，不在同一GPU并发。

## 已完成：四格正确性和10秒物理A/B

```bash
/home/windsky/IsaacLab/isaaclab.sh -p logs/acceptance/optimization6/check.py --decimation 2 --episode 4 --output_dir logs/acceptance/optimization6_replay/D2_T4 --headless
/home/windsky/IsaacLab/isaaclab.sh -p logs/acceptance/optimization6/check.py --decimation 2 --episode 10 --output_dir logs/acceptance/optimization6_replay/D2_T10 --headless
/home/windsky/IsaacLab/isaaclab.sh -p logs/acceptance/optimization6/check.py --decimation 4 --episode 4 --output_dir logs/acceptance/optimization6_replay/D4_T4 --headless
/home/windsky/IsaacLab/isaaclab.sh -p logs/acceptance/optimization6/check.py --decimation 4 --episode 10 --output_dir logs/acceptance/optimization6_replay/D4_T10 --headless
```

check.py每个mode默认10秒；当前脚本也包含后补的真实NaN观测恢复检查。初次D2/T4 10秒结果保存在D2_T4.initial.json；D2_T4.json是后续1秒接口复验。D2/T10、D4/T4、D4/T10保留对应初次完整10秒结果。额外1秒复验的原样参数：

```bash
/home/windsky/IsaacLab/isaaclab.sh -p logs/acceptance/optimization6/check.py --decimation 2 --episode 4 --duration 1 --output_dir logs/acceptance/optimization6_replay/short_D2_T4 --headless
/home/windsky/IsaacLab/isaaclab.sh -p logs/acceptance/optimization6/check.py --decimation 2 --episode 10 --duration 1 --output_dir logs/acceptance/optimization6_replay/invalid_D2 --headless
/home/windsky/IsaacLab/isaaclab.sh -p logs/acceptance/optimization6/check.py --decimation 4 --episode 10 --duration 1 --output_dir logs/acceptance/optimization6_replay/invalid_D4 --headless
```

已完成：物理异常复验、GPU完整120Hz轨迹：

```bash
/home/windsky/IsaacLab/isaaclab.sh -p logs/acceptance/optimization6/check.py --decimation 4 --episode 10 --trace_mode zero --output_dir logs/acceptance/optimization6_replay/diagnosis --headless
```

已完成：默认20/20/16参数、runtime timeout属性和独立5秒保持分类器函数测试。该脚本固定输出本轮根目录protocol_unit.json；其结果已保存，不必再次运行来重复覆盖。原样命令：

```bash
/home/windsky/IsaacLab/isaaclab.sh -p logs/acceptance/optimization6/protocol_unit.py --headless
```

## 已完成：2-update PPO及短评估接口smoke

```bash
/home/windsky/IsaacLab/isaaclab.sh -p logs/acceptance/optimization6/train_measure.py --measurement_dir logs/acceptance/optimization6_replay/ppo_interface --task Grasp1-UR5-Allegro-Teacher-v0 --num_envs 16 --max_iterations 2 --run_name opt6_interface_smoke_retry --seed 1 --decimation 2 --episode_length_s 10 --headless
```

本轮实际smoke checkpoint如下；复验新训练时改成新运行的model_1.pt。

```bash
OPT6_CHECKPOINT=logs/rsl_rl/grasp1_ur5_allegro_teacher/2026-10-09_17-25-57_opt6_interface_smoke_retry/model_1.pt
/home/windsky/IsaacLab/isaaclab.sh -p scripts/rsl_rl/evaluate.py --checkpoint "$OPT6_CHECKPOINT" --decimation 2 --rounds 1 --grasp_duration_s 0.2 --lift_duration_s 0.2 --lift_ramp_duration_s 0.2 --diagnostics --seed 1 --output_dir logs/acceptance/optimization6_replay/evaluation_entry_D2_modeA --headless
/home/windsky/IsaacLab/isaaclab.sh -p scripts/rsl_rl/evaluate.py --checkpoint "$OPT6_CHECKPOINT" --decimation 4 --rounds 1 --grasp_duration_s 0.2 --lift_duration_s 0.2 --lift_ramp_duration_s 0.2 --lift_hand_mode hold-grasp-posture --diagnostics --seed 1 --output_dir logs/acceptance/optimization6_replay/evaluation_entry_D4_modeB --headless
```

## 未执行：E1–E4正式从头训练

必须先通过物理Gate；本轮没有执行以下命令。四组统一2048环境、seed1、100更新，之后按阶段Gate再续训。

```bash
/home/windsky/IsaacLab/isaaclab.sh -p logs/acceptance/optimization6/train_measure.py --measurement_dir logs/acceptance/optimization6_replay/E1/stage100 --task Grasp1-UR5-Allegro-Teacher-v0 --num_envs 2048 --max_iterations 100 --run_name optimization6_E1_stage100 --seed 1 --decimation 2 --episode_length_s 4 --headless
/home/windsky/IsaacLab/isaaclab.sh -p logs/acceptance/optimization6/train_measure.py --measurement_dir logs/acceptance/optimization6_replay/E2/stage100 --task Grasp1-UR5-Allegro-Teacher-v0 --num_envs 2048 --max_iterations 100 --run_name optimization6_E2_stage100 --seed 1 --decimation 2 --episode_length_s 10 --headless
/home/windsky/IsaacLab/isaaclab.sh -p logs/acceptance/optimization6/train_measure.py --measurement_dir logs/acceptance/optimization6_replay/E3/stage100 --task Grasp1-UR5-Allegro-Teacher-v0 --num_envs 2048 --max_iterations 100 --run_name optimization6_E3_stage100 --seed 1 --decimation 4 --episode_length_s 4 --headless
/home/windsky/IsaacLab/isaaclab.sh -p logs/acceptance/optimization6/train_measure.py --measurement_dir logs/acceptance/optimization6_replay/E4/stage100 --task Grasp1-UR5-Allegro-Teacher-v0 --num_envs 2048 --max_iterations 100 --run_name optimization6_E4_stage100 --seed 1 --decimation 4 --episode_length_s 10 --headless
```

100→500新增400更新，500→1000新增500，1000→2000新增1000。对应`--resume --load_run`使用实际上阶段run目录名、`--load_checkpoint model_99.pt/model_499.pt/model_999.pt`；measure入口将checkpoint iter+1，避免重复最后一次更新。四组必须保持同一环境规模；若显存要求统一改规模，应全部组从头重跑。后续checkpoint尚不存在，不能给出伪造的实际路径或声称已完成续训。

## 未执行：20+20秒源时长和独立稳定保持

仅在Gate通过后使用正式候选checkpoint，30对象×3轮。下面使用已存在的接口smoke checkpoint仅提供可运行的协议重现入口，**不得把其结果当成正式候选评估**；本轮这些长评估没有运行。

```bash
/home/windsky/IsaacLab/isaaclab.sh -p scripts/rsl_rl/evaluate.py --checkpoint "$OPT6_CHECKPOINT" --decimation 2 --rounds 3 --grasp_duration_s 20 --lift_duration_s 20 --lift_ramp_duration_s 16 --success_height 0.1 --lift_hand_mode current-relative-repeat --diagnostics --seed 1 --output_dir logs/acceptance/optimization6_replay/source_20_20_D2 --headless
/home/windsky/IsaacLab/isaaclab.sh -p scripts/rsl_rl/evaluate.py --checkpoint "$OPT6_CHECKPOINT" --decimation 4 --rounds 3 --grasp_duration_s 20 --lift_duration_s 20 --lift_ramp_duration_s 16 --success_height 0.1 --lift_hand_mode current-relative-repeat --diagnostics --seed 1 --output_dir logs/acceptance/optimization6_replay/source_20_20_D4 --headless
/home/windsky/IsaacLab/isaaclab.sh -p scripts/rsl_rl/evaluate.py --checkpoint "$OPT6_CHECKPOINT" --decimation 2 --rounds 3 --grasp_duration_s 20 --lift_duration_s 20 --lift_ramp_duration_s 14 --paper_stability --success_height 0.1 --max_relative_translation_drift_m 0.02 --max_relative_rotation_drift_rad 0.2 --diagnostics --seed 1 --output_dir logs/acceptance/optimization6_replay/paper_20_20_ramp14 --headless
```

省略时长参数时默认就是20+20/16，省略lift_hand_mode时默认Mode A。paper_stability强制独立14秒Ramp，默认完整协议仍20+20/16。

## 静态检查

```bash
python -m compileall -q source/Grasp1/Grasp1/tasks/manager_based/grasp1 scripts/rsl_rl scripts/zero_agent.py scripts/random_agent.py logs/acceptance/optimization6

git diff --check
```

结果分析使用根目录JSON/CSV，异常分析用`diagnosis/zero_physics_trace.pt`的CPU张量；峰值位置及邻近帧已写入peak_case.json，三个图轴分别为速度、q/target和法向接触力。所有正式四格训练/正式20+20成功率仍N/A。

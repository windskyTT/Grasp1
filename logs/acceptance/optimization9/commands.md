# 优化9真实执行命令

所有命令在 `/home/windsky/project/Grasp1`，Isaac运行使用Conda grasp，串行执行。

## Phase 0 / A 已执行
```bash
source /home/windsky/miniconda3/etc/profile.d/conda.sh
conda activate grasp
export TERM=xterm PYTHONUNBUFFERED=1
/home/windsky/IsaacLab/isaaclab.sh -p scripts/rsl_rl/train.py --help > logs/acceptance/optimization9/0/train_help.txt 2>&1
/home/windsky/IsaacLab/isaaclab.sh -p scripts/rsl_rl/play.py --help > logs/acceptance/optimization9/0/play_help.txt 2>&1
/home/windsky/IsaacLab/isaaclab.sh -p logs/acceptance/optimization8/A/physics_probe.py --headless --num_envs 1 --decimation 4 --episode 10 --duration 10 --modes replay --initial_state logs/acceptance/optimization7/A/baseline_D4/initial_zero.pt --source_env 7 --isolation no_object --output_dir logs/acceptance/optimization9/A/original_replay
/home/windsky/IsaacLab/isaaclab.sh -p logs/acceptance/optimization9/A/contact_constraint_probe.py --headless --num_envs 1 --decimation 4 --episode 10 --duration 10 --modes replay --initial_state logs/acceptance/optimization7/A/baseline_D4/initial_zero.pt --source_env 7 --isolation no_object --output_dir logs/acceptance/optimization9/A/all_body_contact
```

## GUI截图基线：历史Teacher，诊断语义覆盖
```bash
/home/windsky/IsaacLab/isaaclab.sh -p scripts/rsl_rl/play.py --task Grasp1-UR5-Allegro-Teacher-v0 --checkpoint /home/windsky/project/Grasp1/logs/acceptance/optimization9/0/gui_capture_checkpoint/model_999.pt --dataset new_training_set --object wood_block_oriented --num_envs 1 --episodes 3 --episode_length_s 2 --seed 1 --real-time --allow_mismatched_checkpoint agent.clip_actions=null 'env.viewer.eye=[1.5,-2.0,1.6]' 'env.viewer.lookat=[0.2,-0.65,0.9]'
```

## 其余真实物理回放

### tip_table_pair_isolation
```bash
/home/windsky/IsaacLab/isaaclab.sh -p logs/acceptance/optimization9/A/contact_constraint_probe.py --headless --num_envs 1 --decimation 4 --episode 10 --duration 10 --modes replay --initial_state logs/acceptance/optimization7/A/baseline_D4/initial_zero.pt --source_env 7 --isolation no_object --diagnostic_filter_body link_15_0_tip --output_dir logs/acceptance/optimization9/A/tip_table_pair_isolation
```

### friction005_replay
```bash
/home/windsky/IsaacLab/isaaclab.sh -p logs/acceptance/optimization9/A/contact_constraint_probe.py --headless --num_envs 1 --decimation 4 --episode 10 --duration 10 --modes replay --initial_state logs/acceptance/optimization7/A/baseline_D4/initial_zero.pt --source_env 7 --isolation no_object --legacy_joint_friction 0.05 --output_dir logs/acceptance/optimization9/A/friction005_replay
```

### position128_replay
```bash
/home/windsky/IsaacLab/isaaclab.sh -p logs/acceptance/optimization9/A/contact_constraint_probe.py --headless --num_envs 1 --decimation 4 --episode 10 --duration 10 --modes replay --initial_state logs/acceptance/optimization7/A/baseline_D4/initial_zero.pt --source_env 7 --isolation no_object --diagnostic_position_iterations 128 --output_dir logs/acceptance/optimization9/A/position128_replay
```

### merged_thumb_retry
```bash
/home/windsky/IsaacLab/isaaclab.sh -p logs/acceptance/optimization9/A/contact_constraint_probe.py --headless --num_envs 1 --decimation 4 --episode 10 --duration 10 --modes replay --initial_state logs/acceptance/optimization7/A/baseline_D4/initial_zero.pt --source_env 7 --isolation no_object --diagnostic_merge_tip --output_dir logs/acceptance/optimization9/A/merged_thumb_retry
```

### position128_velocity8_replay
```bash
/home/windsky/IsaacLab/isaaclab.sh -p logs/acceptance/optimization9/A/contact_constraint_probe.py --headless --num_envs 1 --decimation 4 --episode 10 --duration 10 --modes replay --initial_state logs/acceptance/optimization7/A/baseline_D4/initial_zero.pt --source_env 7 --isolation no_object --diagnostic_position_iterations 128 --velocity_iterations 8 --output_dir logs/acceptance/optimization9/A/position128_velocity8_replay
```

### position128_velocity32_replay
```bash
/home/windsky/IsaacLab/isaaclab.sh -p logs/acceptance/optimization9/A/contact_constraint_probe.py --headless --num_envs 1 --decimation 4 --episode 10 --duration 10 --modes replay --initial_state logs/acceptance/optimization7/A/baseline_D4/initial_zero.pt --source_env 7 --isolation no_object --diagnostic_position_iterations 128 --velocity_iterations 32 --output_dir logs/acceptance/optimization9/A/position128_velocity32_replay
```

### external_each_iteration_replay
```bash
/home/windsky/IsaacLab/isaaclab.sh -p logs/acceptance/optimization9/A/contact_constraint_probe.py --headless --num_envs 1 --decimation 4 --episode 10 --duration 10 --modes replay --initial_state logs/acceptance/optimization7/A/baseline_D4/initial_zero.pt --source_env 7 --isolation no_object --diagnostic_external_forces_every_iteration --output_dir logs/acceptance/optimization9/A/external_each_iteration_replay
```

### position255_velocity32_replay
```bash
/home/windsky/IsaacLab/isaaclab.sh -p logs/acceptance/optimization9/A/contact_constraint_probe.py --headless --num_envs 1 --decimation 4 --episode 10 --duration 10 --modes replay --initial_state logs/acceptance/optimization7/A/baseline_D4/initial_zero.pt --source_env 7 --isolation no_object --diagnostic_position_iterations 255 --velocity_iterations 32 --output_dir logs/acceptance/optimization9/A/position255_velocity32_replay
```

### position255_velocity32_external_replay
```bash
/home/windsky/IsaacLab/isaaclab.sh -p logs/acceptance/optimization9/A/contact_constraint_probe.py --headless --num_envs 1 --decimation 4 --episode 10 --duration 10 --modes replay --initial_state logs/acceptance/optimization7/A/baseline_D4/initial_zero.pt --source_env 7 --isolation no_object --diagnostic_position_iterations 255 --velocity_iterations 32 --diagnostic_external_forces_every_iteration --output_dir logs/acceptance/optimization9/A/position255_velocity32_external_replay
```

### full_scene_replay
```bash
/home/windsky/IsaacLab/isaaclab.sh -p logs/acceptance/optimization9/A/contact_constraint_probe.py --headless --num_envs 1 --decimation 4 --episode 10 --duration 10 --modes replay --initial_state logs/acceptance/optimization7/A/baseline_D4/initial_zero.pt --source_env 7 --output_dir logs/acceptance/optimization9/A/full_scene_replay
```

### D2_historical_random_replay
```bash
/home/windsky/IsaacLab/isaaclab.sh -p logs/acceptance/optimization9/A/contact_constraint_probe.py --headless --num_envs 1 --decimation 2 --episode 10 --duration 10 --modes replay --initial_state logs/acceptance/optimization7/C/E2_screen/initial_random.pt --source_env 7 --replay logs/acceptance/optimization7/C/E2_screen/trace_random.pt --output_dir logs/acceptance/optimization9/A/D2_historical_random_replay
```

首次拇指合并在 `merged_thumb_replay` 启动失败（Gf.Rotation构造器），改用Matrix4d提取四元数后在独立 `merged_thumb_retry` 运行。首次汇总的D2轨迹长度比较1440/1200不一致，修正为实际1200帧后重算。

```bash
python logs/acceptance/optimization9/A/summarize_physics.py
python logs/acceptance/optimization9/0/capture_gui_baseline.py
```

最终检查：新诊断脚本py_compile通过；真实目标/分母/Gate状态一致；物理推进区间互不重叠；报告链接和JSON有效；git diff --check通过，tracked源码diff为空。

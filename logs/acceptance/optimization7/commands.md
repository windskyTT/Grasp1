# 优化7命令与实际状态

从 `/home/windsky/project/Grasp1` 运行；A的参数命令由实际loaded_parameters重建，显式写出的默认值与原运行一致。历史阶段诊断字段不同，以各原始文件为准。

```bash
source /home/windsky/miniconda3/etc/profile.d/conda.sh
conda activate grasp
export TERM=xterm PYTHONUNBUFFERED=1
```

## 已执行

状态：`executed`；证据：`A/legacy_repro.log`

```bash
/home/windsky/IsaacLab/isaaclab.sh -p logs/acceptance/optimization6/check.py --decimation 4 --episode 10 --trace_mode zero --output_dir logs/acceptance/optimization7/A/legacy_repro --headless
```

状态：`executed`；证据：`logs/acceptance/optimization7/A/baseline_D4/result.json`

```bash
/home/windsky/IsaacLab/isaaclab.sh -p logs/acceptance/optimization7/A/minimal_repro.py --decimation 4 --episode 10.0 --num_envs 16 --duration 10.0 --seed 1 --modes zero random fixed --source_env 7 --isolation baseline --output_dir logs/acceptance/optimization7/A/baseline_D4 --headless
```

状态：`executed`；证据：`logs/acceptance/optimization7/A/collider_audit_replay/result.json`

```bash
/home/windsky/IsaacLab/isaaclab.sh -p logs/acceptance/optimization7/A/minimal_repro.py --decimation 4 --episode 10.0 --num_envs 1 --duration 10.0 --seed 1 --modes replay --initial_state logs/acceptance/optimization7/A/baseline_D4/initial_zero.pt --source_env 7 --solver tgs --isolation no_object --contact_detail --output_dir logs/acceptance/optimization7/A/collider_audit_replay --headless
```

状态：`executed`；证据：`logs/acceptance/optimization7/A/hand_armature0012_D2/result.json`

```bash
/home/windsky/IsaacLab/isaaclab.sh -p logs/acceptance/optimization7/A/minimal_repro.py --decimation 2 --episode 10.0 --num_envs 35 --duration 12.0 --seed 1 --modes zero random fixed --source_env 7 --hand_armature 0.0012 --isolation baseline --output_dir logs/acceptance/optimization7/A/hand_armature0012_D2 --headless
```

状态：`executed`；证据：`logs/acceptance/optimization7/A/hand_armature0012_D4/result.json`

```bash
/home/windsky/IsaacLab/isaaclab.sh -p logs/acceptance/optimization7/A/minimal_repro.py --decimation 4 --episode 10.0 --num_envs 35 --duration 12.0 --seed 1 --modes zero random fixed --source_env 7 --hand_armature 0.0012 --isolation baseline --output_dir logs/acceptance/optimization7/A/hand_armature0012_D4 --headless
```

状态：`executed`；证据：`logs/acceptance/optimization7/A/hand_armature_D2/result.json`

```bash
/home/windsky/IsaacLab/isaaclab.sh -p logs/acceptance/optimization7/A/minimal_repro.py --decimation 2 --episode 10.0 --num_envs 16 --duration 10.0 --seed 1 --modes zero random fixed --source_env 7 --hand_armature 0.0006 --isolation baseline --output_dir logs/acceptance/optimization7/A/hand_armature_D2 --headless
```

状态：`executed`；证据：`logs/acceptance/optimization7/A/hand_armature_D4/result.json`

```bash
/home/windsky/IsaacLab/isaaclab.sh -p logs/acceptance/optimization7/A/minimal_repro.py --decimation 4 --episode 10.0 --num_envs 16 --duration 10.0 --seed 1 --modes zero random fixed --source_env 7 --hand_armature 0.0006 --isolation baseline --output_dir logs/acceptance/optimization7/A/hand_armature_D4 --headless
```

状态：`executed`；证据：`logs/acceptance/optimization7/A/legacy_friction0_D4/result.json`

```bash
/home/windsky/IsaacLab/isaaclab.sh -p logs/acceptance/optimization7/A/minimal_repro.py --decimation 4 --episode 10.0 --num_envs 16 --duration 10.0 --seed 1 --modes zero random fixed replay --source_env 7 --legacy_joint_friction 0.0 --isolation baseline --contact_detail --output_dir logs/acceptance/optimization7/A/legacy_friction0_D4 --headless
```

状态：`executed`；证据：`logs/acceptance/optimization7/A/limit_force/result.json`

```bash
/home/windsky/IsaacLab/isaaclab.sh -p logs/acceptance/optimization7/A/minimal_repro.py --decimation 4 --episode 10.0 --num_envs 1 --duration 10.0 --seed 1 --modes fixed --initial_state logs/acceptance/optimization7/A/baseline_D4/initial_zero.pt --source_env 7 --limit_case outward --solver tgs --isolation no_contacts --force_perturbation --output_dir logs/acceptance/optimization7/A/limit_force --headless
```

状态：`executed`；证据：`logs/acceptance/optimization7/A/limit_inward/result.json`

```bash
/home/windsky/IsaacLab/isaaclab.sh -p logs/acceptance/optimization7/A/minimal_repro.py --decimation 4 --episode 10.0 --num_envs 1 --duration 10.0 --seed 1 --modes fixed --initial_state logs/acceptance/optimization7/A/baseline_D4/initial_zero.pt --source_env 7 --limit_case inward --solver tgs --isolation no_contacts --output_dir logs/acceptance/optimization7/A/limit_inward --headless
```

状态：`executed`；证据：`logs/acceptance/optimization7/A/limit_lower/result.json`

```bash
/home/windsky/IsaacLab/isaaclab.sh -p logs/acceptance/optimization7/A/minimal_repro.py --decimation 4 --episode 10.0 --num_envs 1 --duration 10.0 --seed 1 --modes fixed --initial_state logs/acceptance/optimization7/A/baseline_D4/initial_zero.pt --source_env 7 --limit_case lower --solver tgs --isolation no_contacts --output_dir logs/acceptance/optimization7/A/limit_lower --headless
```

状态：`executed`；证据：`logs/acceptance/optimization7/A/limit_outward/result.json`

```bash
/home/windsky/IsaacLab/isaaclab.sh -p logs/acceptance/optimization7/A/minimal_repro.py --decimation 4 --episode 10.0 --num_envs 1 --duration 10.0 --seed 1 --modes fixed --initial_state logs/acceptance/optimization7/A/baseline_D4/initial_zero.pt --source_env 7 --limit_case outward --solver tgs --isolation no_contacts --output_dir logs/acceptance/optimization7/A/limit_outward --headless
```

状态：`executed`；证据：`logs/acceptance/optimization7/A/object_velocity4_D4/result.json`

```bash
/home/windsky/IsaacLab/isaaclab.sh -p logs/acceptance/optimization7/A/minimal_repro.py --decimation 4 --episode 10.0 --num_envs 16 --duration 10.0 --seed 1 --modes zero random fixed --source_env 7 --object_velocity_iterations 4 --isolation baseline --output_dir logs/acceptance/optimization7/A/object_velocity4_D4 --headless
```

状态：`executed`；证据：`logs/acceptance/optimization7/A/pgs_D2/result.json`

```bash
/home/windsky/IsaacLab/isaaclab.sh -p logs/acceptance/optimization7/A/minimal_repro.py --decimation 2 --episode 10.0 --num_envs 16 --duration 10.0 --seed 1 --modes zero random fixed --source_env 7 --solver pgs --isolation baseline --output_dir logs/acceptance/optimization7/A/pgs_D2 --headless
```

状态：`executed`；证据：`logs/acceptance/optimization7/A/single_no_contacts_D4/result.json`

```bash
/home/windsky/IsaacLab/isaaclab.sh -p logs/acceptance/optimization7/A/minimal_repro.py --decimation 4 --episode 10.0 --num_envs 1 --duration 10.0 --seed 1 --modes zero random fixed replay --initial_state logs/acceptance/optimization7/A/baseline_D4/initial_zero.pt --source_env 7 --isolation no_contacts --output_dir logs/acceptance/optimization7/A/single_no_contacts_D4 --headless
```

状态：`executed`；证据：`logs/acceptance/optimization7/A/single_no_object_D4/result.json`

```bash
/home/windsky/IsaacLab/isaaclab.sh -p logs/acceptance/optimization7/A/minimal_repro.py --decimation 4 --episode 10.0 --num_envs 1 --duration 10.0 --seed 1 --modes zero fixed replay --initial_state logs/acceptance/optimization7/A/baseline_D4/initial_zero.pt --source_env 7 --solver tgs --isolation no_object --output_dir logs/acceptance/optimization7/A/single_no_object_D4 --headless
```

状态：`executed`；证据：`logs/acceptance/optimization7/A/single_no_table_D4/result.json`

```bash
/home/windsky/IsaacLab/isaaclab.sh -p logs/acceptance/optimization7/A/minimal_repro.py --decimation 4 --episode 10.0 --num_envs 1 --duration 10.0 --seed 1 --modes zero fixed replay --initial_state logs/acceptance/optimization7/A/baseline_D4/initial_zero.pt --source_env 7 --solver tgs --isolation no_table --output_dir logs/acceptance/optimization7/A/single_no_table_D4 --headless
```

状态：`executed`；证据：`logs/acceptance/optimization7/A/single_wood_D4/result.json`

```bash
/home/windsky/IsaacLab/isaaclab.sh -p logs/acceptance/optimization7/A/minimal_repro.py --decimation 4 --episode 10.0 --num_envs 1 --duration 10.0 --seed 1 --modes zero random fixed replay --initial_state logs/acceptance/optimization7/A/baseline_D4/initial_zero.pt --source_env 7 --isolation baseline --contact_detail --output_dir logs/acceptance/optimization7/A/single_wood_D4 --headless
```

状态：`executed`；证据：`logs/acceptance/optimization7/A/source_arm_rotor_D2/result.json`

```bash
/home/windsky/IsaacLab/isaaclab.sh -p logs/acceptance/optimization7/A/minimal_repro.py --decimation 2 --episode 10.0 --num_envs 16 --duration 10.0 --seed 1 --modes zero random fixed --source_env 7 --arm_armature 1.0 --solver tgs --isolation baseline --output_dir logs/acceptance/optimization7/A/source_arm_rotor_D2 --headless
```

状态：`executed`；证据：`logs/acceptance/optimization7/A/source_arm_rotor_D4/result.json`

```bash
/home/windsky/IsaacLab/isaaclab.sh -p logs/acceptance/optimization7/A/minimal_repro.py --decimation 4 --episode 10.0 --num_envs 16 --duration 10.0 --seed 1 --modes zero random fixed --source_env 7 --arm_armature 1.0 --solver tgs --isolation baseline --output_dir logs/acceptance/optimization7/A/source_arm_rotor_D4 --headless
```

状态：`executed`；证据：`logs/acceptance/optimization7/A/table_filtered_unfiltered/result.json`

```bash
/home/windsky/IsaacLab/isaaclab.sh -p logs/acceptance/optimization7/A/minimal_repro.py --decimation 4 --episode 10.0 --num_envs 1 --duration 10.0 --seed 1 --modes replay --initial_state logs/acceptance/optimization7/A/baseline_D4/initial_zero.pt --source_env 7 --solver tgs --isolation no_object --contact_detail --output_dir logs/acceptance/optimization7/A/table_filtered_unfiltered --headless
```

状态：`executed`；证据：`logs/acceptance/optimization7/A/velocity4_D4/result.json`

```bash
/home/windsky/IsaacLab/isaaclab.sh -p logs/acceptance/optimization7/A/minimal_repro.py --decimation 4 --episode 10.0 --num_envs 16 --duration 10.0 --seed 1 --modes zero random fixed --source_env 7 --velocity_iterations 4 --isolation baseline --output_dir logs/acceptance/optimization7/A/velocity4_D4 --headless
```

状态：`executed`；证据：`logs/acceptance/optimization7/C/E1_screen/result.json`

```bash
/home/windsky/IsaacLab/isaaclab.sh -p logs/acceptance/optimization7/A/minimal_repro.py --decimation 2 --episode 4 --duration 12 --num_envs 16 --modes zero random fixed --output_dir logs/acceptance/optimization7/C/E1_screen --headless
```

状态：`executed`；证据：`logs/acceptance/optimization7/C/E2_screen/result.json`

```bash
/home/windsky/IsaacLab/isaaclab.sh -p logs/acceptance/optimization7/A/minimal_repro.py --decimation 2 --episode 10 --duration 12 --num_envs 16 --modes zero random fixed --output_dir logs/acceptance/optimization7/C/E2_screen --headless
```

状态：`executed`；证据：`logs/acceptance/optimization7/C/E3_screen/result.json`

```bash
/home/windsky/IsaacLab/isaaclab.sh -p logs/acceptance/optimization7/A/minimal_repro.py --decimation 4 --episode 4 --duration 12 --num_envs 16 --modes zero random fixed --output_dir logs/acceptance/optimization7/C/E3_screen --headless
```

状态：`executed`；证据：`logs/acceptance/optimization7/C/E4_screen/result.json`

```bash
/home/windsky/IsaacLab/isaaclab.sh -p logs/acceptance/optimization7/A/minimal_repro.py --decimation 4 --episode 10 --duration 12 --num_envs 16 --modes zero random fixed --output_dir logs/acceptance/optimization7/C/E4_screen --headless
```

状态：`executed`；证据：`B/regression_D2.log`

```bash
/home/windsky/IsaacLab/isaaclab.sh -p logs/acceptance/optimization6/check.py --decimation 2 --episode 10 --duration 0.2 --output_dir logs/acceptance/optimization7/B/regression_D2 --headless
```

状态：`executed`；证据：`B/regression_D4.log`

```bash
/home/windsky/IsaacLab/isaaclab.sh -p logs/acceptance/optimization6/check.py --decimation 4 --episode 10 --duration 0.2 --output_dir logs/acceptance/optimization7/B/regression_D4 --headless
```

状态：`executed`；证据：`B/D2.log`

```bash
/home/windsky/IsaacLab/isaaclab.sh -p logs/acceptance/optimization7/B/measurement_check.py --decimation 2 --num_envs 16 --output_dir logs/acceptance/optimization7/B/D2 --headless
```

状态：`executed`；证据：`B/D4_copies30.log`

```bash
/home/windsky/IsaacLab/isaaclab.sh -p logs/acceptance/optimization7/B/measurement_check.py --decimation 4 --num_envs 30 --copies --output_dir logs/acceptance/optimization7/B/D4_copies30 --headless
```

状态：`executed`；证据：`B/final_D2.log`

```bash
/home/windsky/IsaacLab/isaaclab.sh -p logs/acceptance/optimization7/B/measurement_check.py --decimation 2 --num_envs 16 --output_dir logs/acceptance/optimization7/B/final_D2 --headless
```

状态：`executed`；证据：`B/final_D4_copies30.log`

```bash
/home/windsky/IsaacLab/isaaclab.sh -p logs/acceptance/optimization7/B/measurement_check.py --decimation 4 --num_envs 30 --copies --output_dir logs/acceptance/optimization7/B/final_D4_copies30 --headless
```

状态：`executed`；证据：`B/protocol_tests.json`

```bash
python logs/acceptance/optimization7/B/protocol_tests.py
```

状态：`executed`；证据：`B/stability_tests.json`

```bash
python logs/acceptance/optimization7/B/stability_tests.py
```

状态：`executed`；证据：`B/contact_reward_audit.json`

```bash
python logs/acceptance/optimization7/B/contact_audit.py
```

状态：`executed`；证据：`B/ppo_regression.json`

```bash
python logs/acceptance/optimization7/B/ppo_regression.py
```

状态：`executed`；证据：`B/contract_train.log`

```bash
/home/windsky/IsaacLab/isaaclab.sh -p scripts/rsl_rl/train.py --task Grasp1-UR5-Allegro-Teacher-v0 --num_envs 16 --decimation 2 --episode_length_s 10 --seed 1 --headless --max_iterations 2 --run_name optimization7_contract_diagnostic
```

状态：`failed_metadata_discovery_then_fixed`；证据：`B/contract_resume.log`

```bash
/home/windsky/IsaacLab/isaaclab.sh -p scripts/rsl_rl/train.py --task Grasp1-UR5-Allegro-Teacher-v0 --num_envs 16 --decimation 2 --episode_length_s 10 --seed 1 --headless --max_iterations 1 --run_name optimization7_contract_resume_diagnostic --resume --load_run 2026-10-12_13-31-20_optimization7_contract_diagnostic agent.load_checkpoint=model_1.pt
```

状态：`executed`；证据：`B/contract_resume_retry.log`

```bash
/home/windsky/IsaacLab/isaaclab.sh -p scripts/rsl_rl/train.py --task Grasp1-UR5-Allegro-Teacher-v0 --num_envs 16 --decimation 2 --episode_length_s 10 --seed 1 --headless --max_iterations 1 --run_name optimization7_contract_resume_diagnostic_retry --resume --load_run 2026-10-12_13-31-20_optimization7_contract_diagnostic agent.load_checkpoint=model_1.pt
```

状态：`executed_nonformal_diagnostic`；证据：`logs/acceptance/optimization7/D/interface_20_20_ramp16/quantitative_eval.json`

```bash
/home/windsky/IsaacLab/isaaclab.sh -p scripts/rsl_rl/evaluate.py --checkpoint logs/rsl_rl/grasp1_ur5_allegro_teacher/2026-10-12_13-35-50_optimization7_contract_resume_diagnostic_retry/model_2.pt --decimation 2 --object Hammer_40 --rounds 1 --grasp_duration_s 20 --lift_duration_s 20 --lift_ramp_duration_s 16 --diagnostics --seed 1 --output_dir logs/acceptance/optimization7/D/interface_20_20_ramp16 --headless
```

状态：`executed_nonformal_diagnostic`；证据：`logs/acceptance/optimization7/D/interface_injected_20_20_ramp16/quantitative_eval.json`

```bash
/home/windsky/IsaacLab/isaaclab.sh -p logs/acceptance/optimization7/D/injected_40s.py --checkpoint logs/rsl_rl/grasp1_ur5_allegro_teacher/2026-10-12_13-35-50_optimization7_contract_resume_diagnostic_retry/model_2.pt --decimation 2 --object Hammer_40 --rounds 1 --grasp_duration_s 20 --lift_duration_s 20 --lift_ramp_duration_s 16 --diagnostics --seed 1 --output_dir logs/acceptance/optimization7/D/interface_injected_20_20_ramp16 --headless
```

状态：`executed_nonformal_diagnostic`；证据：`logs/acceptance/optimization7/D/interface_injected_engineering_ramp14/quantitative_eval.json`

```bash
/home/windsky/IsaacLab/isaaclab.sh -p logs/acceptance/optimization7/D/injected_40s.py --checkpoint logs/rsl_rl/grasp1_ur5_allegro_teacher/2026-10-12_13-35-50_optimization7_contract_resume_diagnostic_retry/model_2.pt --decimation 2 --object Hammer_40 --rounds 1 --paper_stability --diagnostics --seed 1 --output_dir logs/acceptance/optimization7/D/interface_injected_engineering_ramp14 --headless
```

```bash
python logs/acceptance/optimization7/plot_traces.py
python logs/acceptance/optimization7/summarize.py
python logs/acceptance/optimization7/final_check.py
python -m compileall -q source/Grasp1/Grasp1/tasks/manager_based/grasp1 scripts/rsl_rl logs/acceptance/optimization7
git diff --check
```

D注入程序首次运行因SimulationApp.close()的正常SystemExit(0)未到达外层事后断言；三个40秒循环已完整结束。随后对保存的真实轨迹运行同一段CPU断言并写入interface_injection_check.json，包装器已加入对正常退出的处理。

## 尚未执行

四格正式100/500/1000/2000训练、2048环境吞吐on/off重复测量、三seed、合格模型30×3源时长评估及工程保持评估均blocked。因没有合格checkpoint，不提供虚构路径的正式评估命令。原始完整模板见docs/优化7.md。

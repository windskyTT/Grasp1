# 优化11实际命令

项目根目录；Conda grasp；IsaacLab launcher；物理dt=1/120。

```bash
source /home/windsky/miniconda3/etc/profile.d/conda.sh
conda activate grasp
export TERM=xterm PYTHONUNBUFFERED=1 DISPLAY=:1
/home/windsky/IsaacLab/isaaclab.sh -p scripts/rsl_rl/train.py --help
/home/windsky/IsaacLab/isaaclab.sh -p scripts/rsl_rl/play.py --help
/home/windsky/IsaacLab/isaaclab.sh -p scripts/rsl_rl/scripted_grasp_probe.py --help
/home/windsky/IsaacLab/isaaclab.sh -p scripts/rsl_rl/scripted_grasp_probe.py --task Grasp1-UR5-Allegro-GraspTrain-v2 --dataset new_training_set --object wood_block_oriented --num_envs 1 --trials 10 --seed 21 --headless --output_dir logs/acceptance/optimization11/C/scripted_baseline
/home/windsky/IsaacLab/isaaclab.sh -p scripts/diagnostics/audit_ur5_allegro_mount.py --output_dir logs/acceptance/optimization11/A/mount
/home/windsky/IsaacLab/isaaclab.sh -p scripts/diagnostics/check_initial_tcp_pose.py --output_dir logs/acceptance/optimization11/B/tcp --heights .5 .45 .4 .35 .3
/home/windsky/IsaacLab/isaaclab.sh -p scripts/rsl_rl/scripted_grasp_probe.py --task Grasp1-UR5-Allegro-GraspTrain-v3 --curriculum C1 --mode interface --num_envs 16 --seed 21 --headless --output_dir logs/acceptance/optimization11/D/interface16
/home/windsky/IsaacLab/isaaclab.sh -p scripts/rsl_rl/scripted_grasp_probe.py --task Grasp1-UR5-Allegro-GraspTrain-v3 --curriculum C1 --mode random --num_envs 1 --trials 3 --seed 101 --headless --output_dir logs/acceptance/optimization11/C/random_fixed_palm
/home/windsky/IsaacLab/isaaclab.sh -p scripts/rsl_rl/scripted_grasp_probe.py --task Grasp1-UR5-Allegro-GraspTrain-v3 --curriculum C1 --mode scripted --num_envs 1 --trials 10 --seed 21 --headless --output_dir logs/acceptance/optimization11/C/scripted_fixed_palm
```

```bash
/home/windsky/IsaacLab/isaaclab.sh -p scripts/rsl_rl/scripted_grasp_probe.py --task Grasp1-UR5-Allegro-GraspTrain-v3 --curriculum C1 --mode interface --num_envs 16 --trials 10 --seed 21 --headless --output_dir logs/acceptance/optimization11/D/projected_interface16
```

```bash
/home/windsky/IsaacLab/isaaclab.sh -p scripts/rsl_rl/scripted_grasp_probe.py --task Grasp1-UR5-Allegro-GraspTrain-v3 --curriculum C1 --mode random --num_envs 1 --trials 3 --seed 101 --headless --output_dir logs/acceptance/optimization11/C/random_projected
```

```bash
/home/windsky/IsaacLab/isaaclab.sh -p scripts/rsl_rl/scripted_grasp_probe.py --task Grasp1-UR5-Allegro-GraspTrain-v3 --curriculum C1 --mode scripted --num_envs 1 --trials 3 --seed 21 --headless --output_dir logs/acceptance/optimization11/C/scripted_projected
```

```bash
/home/windsky/IsaacLab/isaaclab.sh -p scripts/rsl_rl/scripted_grasp_probe.py --task Grasp1-UR5-Allegro-GraspTrain-v3 --curriculum C1 --mode grasp_only --num_envs 1 --trials 3 --seed 21 --headless --output_dir logs/acceptance/optimization11/D/grasp_only
```

```bash
/home/windsky/IsaacLab/isaaclab.sh -p scripts/rsl_rl/scripted_grasp_probe.py --task Grasp1-UR5-Allegro-GraspTrain-v3 --curriculum C1 --mode interface --num_envs 16 --trials 10 --seed 21 --headless --output_dir logs/acceptance/optimization11/D/com_interface16
```

```bash
/home/windsky/IsaacLab/isaaclab.sh -p scripts/rsl_rl/scripted_grasp_probe.py --task Grasp1-UR5-Allegro-GraspTrain-v3 --curriculum C1 --mode random --num_envs 1 --trials 3 --seed 101 --headless --output_dir logs/acceptance/optimization11/C/random_com_projected
```

```bash
/home/windsky/IsaacLab/isaaclab.sh -p scripts/rsl_rl/scripted_grasp_probe.py --task Grasp1-UR5-Allegro-GraspTrain-v3 --curriculum C1 --mode interface --num_envs 16 --trials 10 --seed 21 --headless --output_dir logs/acceptance/optimization11/D/increment_interface16
```

```bash
/home/windsky/IsaacLab/isaaclab.sh -p scripts/rsl_rl/scripted_grasp_probe.py --task Grasp1-UR5-Allegro-GraspTrain-v3 --curriculum C1 --mode random --num_envs 1 --trials 3 --seed 101 --headless --output_dir logs/acceptance/optimization11/C/random_increment_projected
```

```bash
/home/windsky/IsaacLab/isaaclab.sh -p scripts/rsl_rl/scripted_grasp_probe.py --task Grasp1-UR5-Allegro-GraspTrain-v3 --curriculum C1 --mode scripted --num_envs 1 --trials 10 --seed 21 --headless --output_dir logs/acceptance/optimization11/C/final_scripted10
```

```bash
/home/windsky/IsaacLab/isaaclab.sh -p scripts/rsl_rl/scripted_grasp_probe.py --task Grasp1-UR5-Allegro-GraspTrain-v3 --curriculum C1 --mode random --num_envs 16 --trials 3 --seed 101 --headless --output_dir logs/acceptance/optimization11/C/final_random16
```

```bash
/home/windsky/IsaacLab/isaaclab.sh -p scripts/rsl_rl/scripted_grasp_probe.py --task Grasp1-UR5-Allegro-GraspTrain-v3 --curriculum C1 --mode scripted --num_envs 16 --trials 1 --seed 21 --headless --output_dir logs/acceptance/optimization11/C/final_scripted16
```

```bash
/home/windsky/IsaacLab/isaaclab.sh -p scripts/rsl_rl/scripted_grasp_probe.py --task Grasp1-UR5-Allegro-GraspTrain-v3 --curriculum C1 --mode random --num_envs 128 --trials 1 --seed 101 --headless --output_dir logs/acceptance/optimization11/C/final_random128
```

```bash
/home/windsky/IsaacLab/isaaclab.sh -p scripts/rsl_rl/scripted_grasp_probe.py --task Grasp1-UR5-Allegro-GraspTrain-v3 --curriculum C1 --mode scripted --num_envs 128 --trials 1 --seed 21 --headless --output_dir logs/acceptance/optimization11/C/final_scripted128
```

```bash
/home/windsky/IsaacLab/isaaclab.sh -p scripts/rsl_rl/scripted_grasp_probe.py --task Grasp1-UR5-Allegro-GraspTrain-v3 --curriculum C1 --mode interface --num_envs 16 --seed 21 --headless --output_dir logs/acceptance/optimization11/D/final_interface16
```

```bash
/home/windsky/IsaacLab/isaaclab.sh -p scripts/diagnostics/audit_ur5_allegro_mount.py --task Grasp1-UR5-Allegro-GraspTrain-v3 --output_dir logs/acceptance/optimization11/A/final_mount
```

```bash
/home/windsky/IsaacLab/isaaclab.sh -p scripts/diagnostics/check_initial_tcp_pose.py --heights .5 --approach_probe --output_dir logs/acceptance/optimization11/B/final_tcp
```

```bash
/home/windsky/IsaacLab/isaaclab.sh -p scripts/rsl_rl/play.py --task Grasp1-UR5-Allegro-GraspTrain-v3 --curriculum C1 --checkpoint logs/rsl_rl/grasp1_ur5_allegro_teacher/2026-10-12_13-35-50_optimization7_contract_resume_diagnostic_retry/model_2.pt --dataset new_training_set --object wood_block_oriented --num_envs 1 --episodes 1 --seed 21 --headless --allow_mismatched_checkpoint --output_dir logs/acceptance/optimization11/D/rejected_play
```

```bash
python scripts/rsl_rl/compare_training_checkpoints.py --checkpoints logs/rsl_rl/grasp1_ur5_allegro_teacher/2026-10-12_13-35-50_optimization7_contract_resume_diagnostic_retry/model_2.pt logs/rsl_rl/grasp1_ur5_allegro_teacher/2026-10-06_21-21-18_teacher_stable_stage2000/model_1999.pt --output_dir logs/acceptance/optimization11/D/old_checkpoint_rejection
```

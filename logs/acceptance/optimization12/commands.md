# 优化12实际命令

所有仿真从项目根目录使用Conda grasp、IsaacLab launcher串行执行。

```bash
cd /home/windsky/project/Grasp1
source /home/windsky/miniconda3/etc/profile.d/conda.sh
conda activate grasp
export TERM=xterm
/home/windsky/IsaacLab/isaaclab.sh -p scripts/rsl_rl/train.py --help
/home/windsky/IsaacLab/isaaclab.sh -p scripts/rsl_rl/play.py --help
/home/windsky/IsaacLab/isaaclab.sh -p scripts/rsl_rl/scripted_grasp_probe.py --help
/home/windsky/IsaacLab/isaaclab.sh -p scripts/diagnostics/audit_ur5_allegro_mount.py --task Grasp1-UR5-Allegro-GraspTrain-v3 --output_dir logs/acceptance/optimization12/A/mount
/home/windsky/IsaacLab/isaaclab.sh -p scripts/rsl_rl/scripted_grasp_probe.py --task Grasp1-UR5-Allegro-GraspTrain-v3 --curriculum C1 --mode scripted --num_envs 1 --trials 10 --seed 21 --headless --output_dir logs/acceptance/optimization12/B/scripted
/home/windsky/IsaacLab/isaaclab.sh -p scripts/rsl_rl/scripted_grasp_probe.py --task Grasp1-UR5-Allegro-GraspTrain-v3 --curriculum C1 --mode random --num_envs 128 --trials 1 --seed 101 --headless --output_dir logs/acceptance/optimization12/B/random128
```

首次help在TERM=dumb下由launcher的tabs退出；设置TERM=xterm后正常。未读取错误.md。

```bash
/home/windsky/IsaacLab/isaaclab.sh -p scripts/rsl_rl/train.py --task Grasp1-UR5-Allegro-GraspTrain-v3 --curriculum C1 --dataset new_training_set --object wood_block_oriented --num_envs 128 --seed 21 --headless --max_iterations 5 --run_name opt12_C1_smoke
```

严格Resume：5+95=累计100，环境/优化器配置未变。
```bash
/home/windsky/IsaacLab/isaaclab.sh -p scripts/rsl_rl/train.py --task Grasp1-UR5-Allegro-GraspTrain-v3 --curriculum C1 --dataset new_training_set --object wood_block_oriented --num_envs 128 --seed 21 --headless --resume --checkpoint /home/windsky/project/Grasp1/logs/rsl_rl/grasp1_ur5_allegro_mount_grasp_v3/2026-10-15_12-40-26_opt12_C1_smoke/model_4.pt --max_iterations 95 --run_name opt12_C1_to100
```

首次续训在加载前退出：原train.py将绝对路径当文件名正则。修复显式本地Checkpoint路径后重跑同命令；首次无新增更新。
```bash
/home/windsky/IsaacLab/isaaclab.sh -p scripts/rsl_rl/train.py --task Grasp1-UR5-Allegro-GraspTrain-v3 --curriculum C1 --dataset new_training_set --object wood_block_oriented --num_envs 128 --seed 21 --headless --resume --checkpoint /home/windsky/project/Grasp1/logs/rsl_rl/grasp1_ur5_allegro_mount_grasp_v3/2026-10-15_12-40-26_opt12_C1_smoke/model_4.pt --max_iterations 95 --run_name opt12_C1_to100
```

第二次续训在严格校验前拒绝：InteractiveScene原地展开prim/filter路径，保存后与创建前占位符文本不同。契约按真实展开路径统一，无放宽物理/课程/资产检查；拒绝阶段新增更新0。
```bash
/home/windsky/IsaacLab/isaaclab.sh -p scripts/rsl_rl/train.py --task Grasp1-UR5-Allegro-GraspTrain-v3 --curriculum C1 --dataset new_training_set --object wood_block_oriented --num_envs 128 --seed 21 --headless --resume --checkpoint /home/windsky/project/Grasp1/logs/rsl_rl/grasp1_ur5_allegro_mount_grasp_v3/2026-10-15_12-40-26_opt12_C1_smoke/model_4.pt --max_iterations 95 --run_name opt12_C1_to100
```

19次时观察到unsafe，SIGINT停止，实际结束更新见C/interrupted_training_state.json。补齐首个unsafe在自动reset之前的现场捕获/立即停训，从同一5更新模型重新运行定位；此分支独立计数，不叠加为一条训练链。
```bash
/home/windsky/IsaacLab/isaaclab.sh -p scripts/rsl_rl/train.py --task Grasp1-UR5-Allegro-GraspTrain-v3 --curriculum C1 --dataset new_training_set --object wood_block_oriented --num_envs 128 --seed 21 --headless --resume --checkpoint /home/windsky/project/Grasp1/logs/rsl_rl/grasp1_ur5_allegro_mount_grasp_v3/2026-10-15_12-40-26_opt12_C1_smoke/model_4.pt --max_iterations 95 --run_name opt12_C1_first_unsafe
```

固定实际Reset清单，真实完成1/5/12更新模型各20个初态；不是累计24更新分支的模型。
```bash
python scripts/rsl_rl/compare_training_checkpoints.py --checkpoints /home/windsky/project/Grasp1/logs/rsl_rl/grasp1_ur5_allegro_mount_grasp_v3/2026-10-15_12-40-26_opt12_C1_smoke/model_0.pt /home/windsky/project/Grasp1/logs/rsl_rl/grasp1_ur5_allegro_mount_grasp_v3/2026-10-15_12-40-26_opt12_C1_smoke/model_4.pt /home/windsky/project/Grasp1/logs/rsl_rl/grasp1_ur5_allegro_mount_grasp_v3/2026-10-15_12-46-52_opt12_C1_first_unsafe/model_11.pt --trials 20 --seed 21 --output_dir logs/acceptance/optimization12/D/comparison
```

首轮比较全部在加载前拒绝，尝试数0：Reset参数中相同木块的重复环境分配数128 vs20造成文本差异。仅推理比较按实际物体集合比较该参数，严格Resume不放宽。修复后重新批量比较，旧拒绝记录保留。
```bash
python scripts/rsl_rl/compare_training_checkpoints.py --checkpoints /home/windsky/project/Grasp1/logs/rsl_rl/grasp1_ur5_allegro_mount_grasp_v3/2026-10-15_12-40-26_opt12_C1_smoke/model_0.pt /home/windsky/project/Grasp1/logs/rsl_rl/grasp1_ur5_allegro_mount_grasp_v3/2026-10-15_12-40-26_opt12_C1_smoke/model_4.pt /home/windsky/project/Grasp1/logs/rsl_rl/grasp1_ur5_allegro_mount_grasp_v3/2026-10-15_12-46-52_opt12_C1_first_unsafe/model_11.pt --trials 20 --seed 21 --output_dir logs/acceptance/optimization12/D/comparison_batch
```

C3从0.5m初态完整脚本预算，原10s；仅脚本诊断。
```bash
/home/windsky/IsaacLab/isaaclab.sh -p scripts/rsl_rl/scripted_grasp_probe.py --task Grasp1-UR5-Allegro-GraspTrain-v3 --curriculum C3 --full_curriculum_approach --mode scripted --num_envs 1 --trials 1 --seed 21 --headless --output_dir logs/acceptance/optimization12/F/C3_T10
```

C3独立14s Episode候选；只变C3脚本诊断时限，C1训练及正式默认配置保持10s。
```bash
/home/windsky/IsaacLab/isaaclab.sh -p scripts/rsl_rl/scripted_grasp_probe.py --task Grasp1-UR5-Allegro-GraspTrain-v3 --curriculum C3 --full_curriculum_approach --mode scripted --num_envs 1 --trials 3 --seed 21 --episode_length_s 14 --headless --output_dir logs/acceptance/optimization12/F/C3_T14
```

修复play.py将long dones误作索引清零env0回合计时；保留旧输出，复制同一真实初态，重跑最终比较。相向夹持/抬升/保持判定未修改。
```bash
python scripts/rsl_rl/compare_training_checkpoints.py --checkpoints /home/windsky/project/Grasp1/logs/rsl_rl/grasp1_ur5_allegro_mount_grasp_v3/2026-10-15_12-40-26_opt12_C1_smoke/model_0.pt /home/windsky/project/Grasp1/logs/rsl_rl/grasp1_ur5_allegro_mount_grasp_v3/2026-10-15_12-40-26_opt12_C1_smoke/model_4.pt /home/windsky/project/Grasp1/logs/rsl_rl/grasp1_ur5_allegro_mount_grasp_v3/2026-10-15_12-46-52_opt12_C1_first_unsafe/model_11.pt --trials 20 --seed 21 --output_dir logs/acceptance/optimization12/D/comparison_final
```

真实GUI纯Policy，使用尚无抓取成功的best_progress.pt；不含headless，5回合，全部22D由policy生成。
```bash
/home/windsky/IsaacLab/isaaclab.sh -p scripts/rsl_rl/play.py --task Grasp1-UR5-Allegro-GraspTrain-v3 --curriculum C1 --dataset new_training_set --object wood_block_oriented --num_envs 1 --checkpoint /home/windsky/project/Grasp1/logs/acceptance/optimization12/D/comparison_final/best_progress.pt --episodes 5 --seed 21 --show_frames --capture_frames --real-time --output_dir logs/acceptance/optimization12/H/gui_policy
```

辅助检查均在现有grasp环境中执行：Python py_compile（本轮8个修改文件）；git diff --check；PIL解码两张参考PNG；torch.load核查实际RSL-RL保存结构；真实权重严格加载及8个归一化buffer逐张量一致检查；跨课程严格Resume、动作顺序/资产/奖励/TCP/归一化/Reset掌姿态错配拒绝检查；TensorBoard标量CSV导出与Matplotlib learning_curves.png；C3真实trace阶段时间/目标速率/间距统计。对应结果均在0/C/E/F目录。

5回合GUI真实完成。首图发现Fabric/render未同步，补齐截图前render刷新；保留原始错误首图，额外1回合验证正确首图（不改物理/策略）。
```bash
/home/windsky/IsaacLab/isaaclab.sh -p scripts/rsl_rl/play.py --task Grasp1-UR5-Allegro-GraspTrain-v3 --curriculum C1 --dataset new_training_set --object wood_block_oriented --num_envs 1 --checkpoint /home/windsky/project/Grasp1/logs/acceptance/optimization12/D/comparison_final/best_progress.pt --episodes 1 --seed 21 --show_frames --capture_frames --real-time --output_dir logs/acceptance/optimization12/H/gui_capture_fixed
```

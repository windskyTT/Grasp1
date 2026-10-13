# 优化8实际执行命令

工作目录：`/home/windsky/project/Grasp1`。所有仿真串行；GPU为3070Ti Laptop8GB，driver580.178.04。

```bash
source /home/windsky/miniconda3/etc/profile.d/conda.sh
conda activate grasp
export TERM=xterm PYTHONUNBUFFERED=1
```

## Phase 0

```bash
git status --short
git rev-parse HEAD
nvidia-smi
/home/windsky/IsaacLab/isaaclab.sh -p scripts/rsl_rl/train.py --help
/home/windsky/IsaacLab/isaaclab.sh -p logs/acceptance/optimization7/A/minimal_repro.py --headless --num_envs 1 --decimation 2 --episode 4 --duration 0.2 --modes zero fixed --output_dir logs/acceptance/optimization8/baseline/teacher_smoke
```

初次importlib.metadata查询isaacsim失败：预构建安装没有该分发元数据，改读实际VERSION。首次help因TERM=dumb导致tabs退出，按文档export TERM后重试通过。没有改依赖/官方安装。基线diff为空；初始仅用户优化8.md未跟踪。

## Phase A

每次保存1200个物理步，原始stdout在对应同名.log。前两个audit曾使用空BBoxCache范围，后改顶点AABB；前期使用callback，后期使用立即API，均未得到可用报告。这些是诊断采样变更，没有作为修复。

```bash
/home/windsky/IsaacLab/isaaclab.sh -p logs/acceptance/optimization8/A/physics_probe.py --headless --num_envs 1 --decimation 2 --episode 10 --duration 10 --modes replay --initial_state logs/acceptance/optimization7/C/E2_screen/initial_random.pt --replay logs/acceptance/optimization7/C/E2_screen/trace_random.pt --source_env 7 --output_dir logs/acceptance/optimization8/A/D2_historical_random_target_replay
```

```bash
/home/windsky/IsaacLab/isaaclab.sh -p logs/acceptance/optimization8/A/physics_probe.py --headless --num_envs 1 --decimation 4 --episode 10 --duration 10 --modes replay --initial_state logs/acceptance/optimization7/A/baseline_D4/initial_zero.pt --source_env 7 --isolation no_object --collision_variant base_filter --output_dir logs/acceptance/optimization8/A/base_filter_table_on_object_off
```

```bash
/home/windsky/IsaacLab/isaaclab.sh -p logs/acceptance/optimization8/A/physics_probe.py --headless --num_envs 1 --decimation 4 --episode 10 --duration 10 --modes replay --initial_state logs/acceptance/optimization7/A/baseline_D4/initial_zero.pt --source_env 7 --isolation no_object --collision_variant box --output_dir logs/acceptance/optimization8/A/box_table_on_object_off
```

```bash
/home/windsky/IsaacLab/isaaclab.sh -p logs/acceptance/optimization8/A/physics_probe.py --headless --num_envs 1 --decimation 4 --episode 10 --duration 10 --modes replay --initial_state logs/acceptance/optimization7/A/baseline_D4/initial_zero.pt --source_env 7 --isolation no_object --hand_stiffness 150 --output_dir logs/acceptance/optimization8/A/hand150_table_on_object_off
```

```bash
/home/windsky/IsaacLab/isaaclab.sh -p logs/acceptance/optimization8/A/physics_probe.py --headless --num_envs 1 --decimation 4 --episode 10 --duration 10 --modes replay --initial_state logs/acceptance/optimization7/A/baseline_D4/initial_zero.pt --source_env 7 --isolation no_table --output_dir logs/acceptance/optimization8/A/original_table_off_object_on
```

```bash
/home/windsky/IsaacLab/isaaclab.sh -p logs/acceptance/optimization8/A/physics_probe.py --headless --num_envs 1 --decimation 4 --episode 10 --duration 10 --modes replay --initial_state logs/acceptance/optimization7/A/baseline_D4/initial_zero.pt --source_env 7 --isolation no_object --output_dir logs/acceptance/optimization8/A/original_table_on_object_off
```

```bash
/home/windsky/IsaacLab/isaaclab.sh -p logs/acceptance/optimization8/A/physics_probe.py --headless --num_envs 1 --decimation 4 --episode 10 --duration 10 --modes replay --initial_state logs/acceptance/optimization7/A/baseline_D4/initial_zero.pt --source_env 7 --isolation no_object --output_dir logs/acceptance/optimization8/A/original_table_on_object_off_repeat
```

```bash
/home/windsky/IsaacLab/isaaclab.sh -p logs/acceptance/optimization8/A/physics_probe.py --headless --num_envs 1 --decimation 4 --episode 10 --duration 10 --modes replay --initial_state logs/acceptance/optimization7/A/baseline_D4/initial_zero.pt --source_env 7 --output_dir logs/acceptance/optimization8/A/original_table_on_object_on
```

```bash
python logs/acceptance/optimization8/A/summarize_physics.py
```

## B/C/E/H独立检查

```bash
python logs/acceptance/optimization8/E/source_forward_audit.py
/home/windsky/IsaacLab/isaaclab.sh -p logs/acceptance/optimization8/C/interface_check.py --headless --task GraspTrain-v1 --num_envs 1 --decimation 4 --static_checks --output_dir logs/acceptance/optimization8/H/new_task_final_object_contract
/home/windsky/IsaacLab/isaaclab.sh -p logs/acceptance/optimization8/C/interface_check.py --headless --task GraspTrain-v1 --num_envs 16 --decimation 4 --output_dir logs/acceptance/optimization8/C/new_D4_16
/home/windsky/IsaacLab/isaaclab.sh -p logs/acceptance/optimization8/C/interface_check.py --headless --task GraspTrain-v1 --num_envs 16 --decimation 2 --output_dir logs/acceptance/optimization8/C/new_D2_16
/home/windsky/IsaacLab/isaaclab.sh -p logs/acceptance/optimization8/C/interface_check.py --headless --task Teacher-v0 --num_envs 1 --decimation 2 --output_dir logs/acceptance/optimization8/H/old_teacher_final
python logs/acceptance/optimization8/H/legacy_protocol_tests.py
/home/windsky/IsaacLab/isaaclab.sh -p scripts/rsl_rl/train.py --headless --task Grasp1-UR5-Allegro-GraspTrain-v1 --num_envs 16 --decimation 2 --cfg job
python -m compileall -q source scripts logs/acceptance/optimization8
python logs/acceptance/optimization8/finalize_report.py
git diff --check
```

第一次C static检查在new_D4_1.log失败，通用USD依赖枚举无法解析Kit内置OmniPBR.mdl。契约补入真实`${kit}/mdl/core/Base/OmniPBR.mdl`字节后new_D4_1_retry通过，最终H/new_task_final_object_contract对新旧契约、heldout评估允许更换对象及normalizer/网络差异重新验证。每次失败/重试目录保留。

`--cfg job`只检查真实Hydra注册/配置解析，未运行train主循环或PPO。旧protocol脚本复制至优化8/H并改产出路径，未写优化7。

## 未执行

D正式短PPO、128/512/1024/2048吞吐比较、E物理迁移、F8–10小时训练、G抓取/MVP/90次ShapeNet和工程保持均因A失败未运行。不存在合格best/last；不提供指向虚构checkpoint的评估命令。

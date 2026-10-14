# 优化10实际命令

所有命令 cwd=/home/windsky/project/Grasp1，source /home/windsky/miniconda3/etc/profile.d/conda.sh && conda activate grasp；所有Isaac Sim串行。

- git rev-parse HEAD; git status --short; git diff --stat; nvidia-smi
- isaaclab.sh -p scripts/rsl_rl/train.py --help
- isaaclab.sh -p scripts/rsl_rl/play.py --help

## 真实诊断候选（依次串行）
所有命令前缀：/home/windsky/IsaacLab/isaaclab.sh -p；均 --headless，完整原始日志保留在对应目录。

```bash
logs/acceptance/optimization9/A/contact_constraint_probe.py --num_envs 1 --initial_state logs/acceptance/optimization9/A/original_replay/initial_replay.pt --source_env 0 --replay logs/acceptance/optimization9/A/original_replay/trace_replay.pt --modes replay --isolation no_object --solver pgs --duration 10 --output_dir logs/acceptance/optimization10/A/pgs_history_retry
logs/acceptance/optimization9/A/contact_constraint_probe.py --num_envs 1 --initial_state logs/acceptance/optimization9/A/original_replay/initial_replay.pt --source_env 0 --replay logs/acceptance/optimization9/A/original_replay/trace_replay.pt --modes replay --isolation no_object --legacy_joint_friction 0 --duration 10 --output_dir logs/acceptance/optimization10/A/zero_legacy_history
logs/acceptance/optimization9/A/contact_constraint_probe.py --num_envs 1 --initial_state logs/acceptance/optimization9/A/original_replay/initial_replay.pt --source_env 0 --replay logs/acceptance/optimization9/A/original_replay/trace_replay.pt --modes replay --isolation no_object --hand_friction 0.01 --duration 10 --output_dir logs/acceptance/optimization10/A/new_static_friction_history
```

## 软件失败与修正
- 首次PGS命令误用不存在的 optimization8/A/original_replay 路径，未推进物理；改为实际 optimization9 fixture 后重跑，旧失败日志 pgs_history.log 保留。
- scripted --help 首次未设置TERM，随后遇到AppLauncher提前解析required参数；改为在add_app_launcher_args后注册output_dir，真实--help成功。
- 首次Task构建尝试把无碰撞bottom当作碰撞Mesh，关闭时卡在Sim cleanup；保存日志并用SIGINT退出该进程，改为使用真实存在的碰撞形状。
- 首次P64控制分项日志字段lift重名，推进4个物理步后报TypeError；字段统一reward_前缀，失败trace和日志保留，后续P64_retry成功运行。

## 脚本、动作语义、接口与终版复验
以下同样使用isaaclab.sh -p前缀；除GUI外均--headless。每次启动实际参数另存output_dir/command.json，实际物理/Task配置存physics.json；中间阶段代码语义不同，其结果保留为历史对照。终版命令使用FullGraspJointDeltaAction。

```bash
scripts/rsl_rl/scripted_grasp_probe.py --mode scripted --num_envs 1 --trials 10 --seed 21 --output_dir logs/acceptance/optimization10/B/wood_scripted
scripts/rsl_rl/scripted_grasp_probe.py --mode scripted --num_envs 1 --trials 10 --seed 21 --approach_distance 0.04 --output_dir logs/acceptance/optimization10/B/wood_scripted_descent004
scripts/rsl_rl/scripted_grasp_probe.py --mode scripted --num_envs 1 --trials 3 --seed 21 --approach_distance 0.04 --position_iterations 64 --output_dir logs/acceptance/optimization10/B/wood_scripted_P64_retry
scripts/rsl_rl/scripted_grasp_probe.py --mode scripted --num_envs 1 --trials 3 --seed 21 --approach_distance 0.04 --position_iterations 64 --episode_length_s 10 --output_dir logs/acceptance/optimization10/B/wood_scripted_P64_T10
scripts/rsl_rl/scripted_grasp_probe.py --mode scripted --num_envs 1 --trials 10 --seed 21 --approach_distance 0.04 --position_iterations 64 --episode_length_s 10 --output_dir logs/acceptance/optimization10/B/wood_scripted_hold_v2
scripts/rsl_rl/scripted_grasp_probe.py --mode random --num_envs 1 --trials 3 --seed 101 --output_dir logs/acceptance/optimization10/A/random1
scripts/rsl_rl/scripted_grasp_probe.py --mode replay --num_envs 1 --trials 1 --seed 1 --output_dir logs/acceptance/optimization10/A/v2_tip_history
# 终版动作语义：
scripts/rsl_rl/scripted_grasp_probe.py --mode zero --num_envs 1 --trials 3 --seed 101 --output_dir logs/acceptance/optimization10/A/target_delta_zero1
scripts/rsl_rl/scripted_grasp_probe.py --mode random --num_envs 1 --trials 3 --seed 101 --output_dir logs/acceptance/optimization10/A/target_delta_random1
scripts/rsl_rl/scripted_grasp_probe.py --mode scripted --num_envs 1 --trials 10 --seed 21 --output_dir logs/acceptance/optimization10/B/final_target_delta_scripted
scripts/rsl_rl/scripted_grasp_probe.py --mode replay --num_envs 1 --trials 1 --seed 1 --output_dir logs/acceptance/optimization10/A/final_P64_history
scripts/rsl_rl/scripted_grasp_probe.py --mode scripted --num_envs 1 --trials 1 --seed 21 --decimation 2 --output_dir logs/acceptance/optimization10/C/scripted_D2
# 实际依次运行的接口批次：
scripts/rsl_rl/scripted_grasp_probe.py --mode interface --num_envs 1 --output_dir logs/acceptance/optimization10/C/target_delta_interface1
scripts/rsl_rl/scripted_grasp_probe.py --mode interface --num_envs 16 --output_dir logs/acceptance/optimization10/C/target_delta_interface16
scripts/rsl_rl/scripted_grasp_probe.py --mode interface --num_envs 128 --output_dir logs/acceptance/optimization10/C/target_delta_interface128
scripts/rsl_rl/scripted_grasp_probe.py --mode interface --num_envs 16 --decimation 2 --output_dir logs/acceptance/optimization10/C/target_delta_interface16_D2
scripts/rsl_rl/scripted_grasp_probe.py --mode interface --num_envs 1 --output_dir logs/acceptance/optimization10/C/final_contract_interface1
python logs/acceptance/optimization10/B/capture_scripted_gui.py
```

前两批final_interface1/16/128验证过旧相对q候选，最终target_delta_interface和final_contract_interface覆盖终版动作。所有帮助输出和编译均已真实运行；未运行train.py PPO，也没有运行play.py策略推理，因为没有合格新策略checkpoint。

## 最后独立一致性测试（真实执行）
```bash
scripts/rsl_rl/scripted_grasp_probe.py --mode scripted --num_envs 16 --trials 1 --seed 41 --output_dir logs/acceptance/optimization10/C/scripted16 --headless
scripts/rsl_rl/scripted_grasp_probe.py --mode scripted --num_envs 128 --trials 1 --seed 43 --output_dir logs/acceptance/optimization10/C/scripted128 --headless
scripts/rsl_rl/scripted_grasp_probe.py --mode interface --num_envs 1 --output_dir logs/acceptance/optimization10/C/phase_contract_final --headless
scripts/rsl_rl/scripted_grasp_probe.py --mode interface --num_envs 1 --output_dir logs/acceptance/optimization10/C/corotating_contract_final --headless
scripts/rsl_rl/scripted_grasp_probe.py --mode scripted --num_envs 1 --trials 1 --seed 21 --output_dir logs/acceptance/optimization10/C/corotating_scripted --headless
```
GUI第一次远景被浮窗挡住，实际近景截图见gui_scripted_visible_retry；中间xdotool未安装导致截图辅助失败，子进程物理试验仍成功结束，日志完整保留。最终通过已有X11/XTest库尝试浮窗关闭并抓取近景；浮窗仍在左侧，但木块、夹持手指和离桌间距在截图中可见。未更换Conda环境或安装依赖。

最后姿态奖励方向/契约检查：isaaclab.sh -p scripts/rsl_rl/scripted_grasp_probe.py --mode interface --num_envs 1 --output_dir logs/acceptance/optimization10/C/reward_contract_final --headless

终版真实Reward分项验收：isaaclab.sh -p scripts/rsl_rl/scripted_grasp_probe.py --mode scripted --num_envs 1 --trials 1 --seed 21 --output_dir logs/acceptance/optimization10/C/reward_scripted_final --headless

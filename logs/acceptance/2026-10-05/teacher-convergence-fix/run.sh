#!/bin/bash
# 从项目目录启动指定验收命令，并采样同一运行的 GPU 数据。
source /home/windsky/miniconda3/etc/profile.d/conda.sh
conda activate grasp
export PYTHONUNBUFFERED=1
cd /home/windsky/project/Grasp1
run_label=$1
shift
run_root=logs/acceptance/2026-10-05/teacher-convergence-fix
nvidia-smi --query-gpu=timestamp,memory.used,utilization.gpu --format=csv -l 1 > "$run_root/$run_label.gpu.csv" &
monitor_pid=$!
"$@" > "$run_root/$run_label.log" 2>&1
run_exit=$?
kill "$monitor_pid"
wait "$monitor_pid" 2>/dev/null
echo "$run_exit" > "$run_root/$run_label.exit"
tail -8 "$run_root/$run_label.log"
exit "$run_exit"

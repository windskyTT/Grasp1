#!/bin/bash
set -e
run_root=logs/acceptance/2026-10-05/teacher-convergence-fix
bash "$run_root/run.sh" phase1-all-shapenet python "$run_root/rollout_probe.py" --headless --num_envs 30 --dataset shapenet-30obj --output "$run_root/phase1-all-shapenet.json"
bash "$run_root/run.sh" phase1-dummy-rollout python "$run_root/rollout_probe.py" --headless --num_envs 16 --dataset dummy --output "$run_root/phase1-dummy-rollout.json"
for dataset in new_training_set shapenet-30obj dummy; do
    case "$dataset" in
        new_training_set) object=003_cracker_box ;;
        shapenet-30obj) object=Blue_camera ;;
        dummy) object=dummy ;;
    esac
    for condition in free table contact; do
        label="phase1-fixed-$dataset-$condition"
        bash "$run_root/run.sh" "$label" python "$run_root/object_probe.py" --headless --dataset "$dataset" --object "$object" --condition "$condition" --output "$run_root/$label.json"
    done
done

#!/usr/bin/env bash
set -u
taskroot=/mnt/c/Users/22174/Documents/Codex/2026-10-03/nin/scan200_expansion_work
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
/home/lewisliu/baselines/esam-env/bin/python "$taskroot/run_esam.py" > "$taskroot/esam_inference.log" 2>&1
status=$?
if [ "$status" -eq 0 ]; then
 /home/lewisliu/miniconda3/envs/scannet-sg/bin/python "$taskroot/evaluate_esam_only.py" > "$taskroot/esam_only_evaluation.log" 2>&1
 status=$?
fi
printf '%s\n' "$status" > "$taskroot/esam_only.exit"
exit "$status"

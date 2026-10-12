#!/usr/bin/env bash
set -u
taskroot=/mnt/c/Users/22174/Documents/Codex/2026-10-03/nin/v20_work
py=/home/lewisliu/miniconda3/envs/scannet-sg/bin/python
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
"$py" "$taskroot/prepare_four_domain_confirmation.py" > "$taskroot/image_confirmation_prepare.log" 2>&1
status=$?
if [ "$status" -eq 0 ]; then
  "$py" "$taskroot/build_image_confirmed_trials.py" > "$taskroot/image_confirmation_build.log" 2>&1
  status=$?
fi
if [ "$status" -eq 0 ]; then
  "$py" "$taskroot/evaluate_image_confirmed_trials.py" > "$taskroot/image_confirmation_eval.log" 2>&1
  status=$?
fi
printf '%s\n' "$status" > "$taskroot/image_confirmation.exit"
exit "$status"

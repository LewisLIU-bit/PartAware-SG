#!/usr/bin/env bash
set -u
taskroot=/mnt/c/Users/22174/Documents/Codex/2026-10-03/nin/v20_work/cue_candidate
py=/home/lewisliu/miniconda3/envs/scannet-sg/bin/python
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 CUDA_VISIBLE_DEVICES=''
"$py" "$taskroot/test_cue_ownership.py" > "$taskroot/tests.log" 2>&1
status=$?
if [ "$status" -eq 0 ]; then
  "$py" "$taskroot/build_cue_screen.py" > "$taskroot/build.log" 2>&1
  status=$?
fi
if [ "$status" -eq 0 ]; then
  "$py" "$taskroot/evaluate_cue_screen.py" > "$taskroot/evaluation.log" 2>&1
  status=$?
fi
printf '%s\n' "$status" > "$taskroot/service.exit"
exit "$status"

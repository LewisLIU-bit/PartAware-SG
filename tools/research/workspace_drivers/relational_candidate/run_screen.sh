#!/usr/bin/env bash
set -u
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2
work=/mnt/c/Users/22174/Documents/Codex/2026-10-03/nin/v20_work/relational_candidate
/home/lewisliu/miniconda3/envs/scannet-sg/bin/python "$work/register_hypothesis.py" > "$work/registration.log" 2>&1
registration_status=$?
if [ "$registration_status" -ne 0 ]; then
    printf '%s\n' "$registration_status" > "$work/screen.exit"
    exit "$registration_status"
fi
/home/lewisliu/miniconda3/envs/scannet-sg/bin/python "$work/build_screen.py" > "$work/build.log" 2>&1
build_status=$?
printf '%s\n' "$build_status" > "$work/build.exit"
if [ "$build_status" -ne 0 ]; then
    printf '%s\n' "$build_status" > "$work/screen.exit"
    exit "$build_status"
fi
/home/lewisliu/miniconda3/envs/scannet-sg/bin/python "$work/evaluate_screen.py" > "$work/evaluation.log" 2>&1
evaluation_status=$?
printf '%s\n' "$evaluation_status" > "$work/evaluation.exit"
printf '%s\n' "$evaluation_status" > "$work/screen.exit"
exit "$evaluation_status"

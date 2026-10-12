#!/usr/bin/env bash
set -u
root=/mnt/c/Users/22174/Documents/Codex/2026-10-03/nin
py=/home/lewisliu/miniconda3/envs/scannet-sg/bin/python
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
"$py" "$root/scan200_expansion_work/run_suite.py" > "$root/scan200_expansion_work/suite.log" 2>&1
status=$?
printf '%s\n' "$status" > "$root/scan200_expansion_work/suite.exit"
exit "$status"

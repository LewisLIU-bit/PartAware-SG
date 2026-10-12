#!/usr/bin/env bash
set -u
taskroot=/mnt/c/Users/22174/Documents/Codex/2026-10-03/nin/scan200_expansion_work
/home/lewisliu/miniconda3/envs/scannet-sg/bin/python "$taskroot/run_scan200.py" --scene scene0025_00 > "$taskroot/first_construction.log" 2>&1
status=$?
printf '%s\n' "$status" > "$taskroot/first_construction.exit"
exit "$status"

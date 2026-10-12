#!/usr/bin/env bash
set -u
taskroot=/mnt/c/Users/22174/Documents/Codex/2026-10-03/nin/v20_work
py=/home/lewisliu/miniconda3/envs/scannet-sg/bin/python
"$py" "$taskroot/replay_source_verified.py" > "$taskroot/sage_verified_driver.log" 2>&1
status=$?
if [ "$status" -eq 0 ]; then
  "$py" "$taskroot/evaluate_replay.py" --cycle sage_verified_core > "$taskroot/sage_verified_eval.log" 2>&1
  status=$?
fi
printf '%s\n' "$status" > "$taskroot/sage_verified.exit"
exit "$status"

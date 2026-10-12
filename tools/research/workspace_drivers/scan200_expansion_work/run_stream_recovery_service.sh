#!/usr/bin/env bash
set -u
taskroot=/mnt/c/Users/22174/Documents/Codex/2026-10-03/nin/scan200_expansion_work
exec 9> "$taskroot/recovery.lock"
flock -n 9 || exit 75
/home/lewisliu/miniconda3/envs/scannet-sg/bin/python "$taskroot/resume_after_transport.py" > "$taskroot/stream_recovery.log" 2>&1
status=$?
printf '%s\n' "$status" > "$taskroot/stream_recovery.exit"
exit "$status"

#!/usr/bin/env bash
set -u
taskroot=/mnt/c/Users/22174/Documents/Codex/2026-10-03/nin/scan200_expansion_work
/home/lewisliu/miniconda3/envs/scannet-sg/bin/python "$taskroot/resume_after_transport.py" > "$taskroot/resume.log" 2>&1
status=$?
printf '%s\n' "$status" > "$taskroot/resume.exit"
exit "$status"

#!/usr/bin/env bash
set -u
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
work=/mnt/c/Users/22174/Documents/Codex/2026-10-03/nin/v20_work/cross_domain_audit
/home/lewisliu/miniconda3/envs/scannet-sg/bin/python "$work/audit_010.py" > "$work/audit_010.log" 2>&1
audit_status=$?
printf '%s\n' "$audit_status" > "$work/audit_010.exit"
exit "$audit_status"

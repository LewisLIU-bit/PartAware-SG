#!/usr/bin/env bash
set -u
taskroot=/mnt/c/Users/22174/Documents/Codex/2026-10-03/nin/v20_work
py=/home/lewisliu/miniconda3/envs/scannet-sg/bin/python
work=/mnt/c/Users/22174/Documents/Codex/2026-10-03/nin/esam_hypersim_work
export OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2
"$py" "$taskroot/build_observed_domain_trials.py" \
  --source /home/lewisliu/datasets/scannet-sg-processed/partaware_v19/hypersim/ai_001_010 \
  --domain "$work/observed_domain.npz" --domain-receipt "$work/domain_receipt.json" \
  --peer "$work/ESAM_common_domain_masks.npz" \
  --manifest /home/lewisliu/datasets/scannet-sg-input/hypersim/ai_001_010_v3/manifest.json \
  --output /home/lewisliu/datasets/scannet-sg-processed/research_cycle/sage_domain_transfer/hypersim/ai_001_010 \
  > "$taskroot/010_sage_domain_build.log" 2>&1
status=$?
if [ "$status" -eq 0 ]; then
  "$py" "$taskroot/evaluate_observed_domain_trials.py" > "$taskroot/010_sage_domain_eval.log" 2>&1
  status=$?
fi
printf '%s\n' "$status" > "$taskroot/010_sage_domain.exit"
exit "$status"

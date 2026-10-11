# Local frozen ESAM baseline

The installed author repository is `/home/lewisliu/baselines/ESAM`, commit
188fc6de44f7577fecec2d69b75c7adbd9992251. The isolated interpreter is
`/home/lewisliu/baselines/esam-env/bin/python`. Native model source is unchanged;
the official SAM-H/ScanNet200 CA checkpoint loads all 553 keys strictly.

These two exact external adapters were used for this RTX5070-compatible CPU
sparse inference: TensorField decomposition follows input point order, and
the final PointOps 1NN is replaced by exact batched Euclidean KNN. They contain
independent numerical checks and no scene, node or GT instance conditions.
This does not establish bitwise equivalence to the original GPU environment.

Final data selection, hashes, point transfer and all per-GT diagnostics are in
`docs/esam_scan200_comparison.json`, `docs/esam_scan200_validation.json`,
`docs/esam_hypersim_010_comparison.json` and
`docs/esam_hypersim_010_validation.json`. The canonical human-readable report
is root `RESEARCH_LOG.html`, sections 26.8 and 26.9. Joint training is paused.

The exact local data-preparation, inference and independent-evaluation scripts
used in this experiment are preserved in the shared Codex workspace at
`C:/Users/22174/Documents/Codex/2026-10-03/nin/scan200_work` and
`C:/Users/22174/Documents/Codex/2026-10-03/nin/esam_hypersim_work`.
Their WSL paths begin `/mnt/c/Users/22174/Documents/Codex/2026-10-03/nin/`.
They expect the associated registration/input records in that directory;
they are experiment runners, not a new production algorithm.

Existing final inference can be independently rescored without API calls:

```bash
PY=/home/lewisliu/miniconda3/envs/scannet-sg/bin/python
WORK=/mnt/c/Users/22174/Documents/Codex/2026-10-03/nin
OMP_NUM_THREADS=2 "$PY" "$WORK/scan200_work/evaluate_scan200.py" \
  --construction-record "$WORK/scan200_work/construction_results.json" \
  --esam-record "$WORK/scan200_work/esam_results.json"
OMP_NUM_THREADS=2 "$PY" "$WORK/esam_hypersim_work/evaluate.py"
```

The three ScanNet scans are a fixed validation pilot, not the full 312 scans.
Hypersim 010 was previously used for development; it is not held out. Its
104-instance observed point AP and historical 109-instance box AP are separate.
No GT-assisted granularity union is used in either primary point comparison.
ESAM is a trained ScanNet200 baseline; this is not an ESAM++ reproduction.

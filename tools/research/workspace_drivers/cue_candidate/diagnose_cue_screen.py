"""Independent post-evaluation attribution; GT never reaches construction."""
import gc
import hashlib
import json
import os
import sys
from pathlib import Path
os.environ['OMP_NUM_THREADS'] = '2'
os.environ['OPENBLAS_NUM_THREADS'] = '2'
os.environ['MKL_NUM_THREADS'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
if hasattr(os, 'sched_getaffinity'):
    os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:2])
import numpy as np
WORK = Path(__file__).resolve().parent
ROOT = WORK.parents[1]
sys.path.insert(0, str(ROOT/'scan200_work'))
import evaluate_scan200 as ref


def owners(path):
    with np.load(path, allow_pickle=False) as data:
        masks, scores = data['masks'].copy(), data['scores'].copy()
    owner = np.full(masks.shape[1], -1, np.int32)
    sizes = masks.sum(1)
    for i, mask in enumerate(masks):
        owner[mask] = i
    del masks
    return owner, scores, sizes


def best_ious(gt, owner, sizes):
    rows = []
    for encoded in np.unique(gt):
        if encoded <= 0:
            continue
        points = gt == encoded
        count = int(points.sum())
        if count < 100:
            continue
        intersection = np.bincount(owner[points & (owner >= 0)], minlength=len(sizes))
        iou = intersection / np.maximum(count + sizes - intersection, 1)
        winner = int(np.argmax(iou))
        rows.append(dict(GT_encoded_id=int(encoded), GT_points=count,
            candidate=winner, best_IoU=float(iou[winner]), intersection=int(intersection[winner]),
            candidate_points=int(sizes[winner])))
    return {row['GT_encoded_id']: row for row in rows}


def main():
    receipt = json.loads((WORK/'construction.json').read_text())
    evaluation = json.loads((WORK/'evaluation.json').read_text())
    assert receipt['complete'] and evaluation['complete']
    result = dict(complete=True, GT_only_in_independent_diagnosis=True,
        builder_has_no_access_to_diagnosis=True, full_pipeline_replay=False, results=[])
    for item, metrics in zip(receipt['results'], evaluation['results']):
        if item['dataset'] == 'scannet':
            _, gt, gt_rows, _ = ref.ground_truth(item['scene'])
            labels = {row['encoded_id']: row['raw_label'] for row in gt_rows}
        else:
            with np.load(ROOT/'esam_hypersim_work/evaluation_GT.npz', allow_pickle=False) as data:
                gt = data['gt'].copy()
            labels = {}
        control, sa, control_sizes = owners(item['control_masks_file'])
        candidate, sb, cue_sizes = owners(item['masks_file'])
        np.testing.assert_array_equal(sa, sb)
        before, after = best_ious(gt, control, control_sizes), best_ious(gt, candidate, cue_sizes)
        changed = []
        for encoded, original in before.items():
            modified = after[encoded]
            if abs(original['best_IoU']-modified['best_IoU']) < 1e-9:
                continue
            changed.append(dict(GT_encoded_id=encoded, independent_GT_label=labels.get(encoded),
                before=original, after=modified, delta_best_IoU=modified['best_IoU']-original['best_IoU']))
        best_losses = sorted([row for row in changed if row['delta_best_IoU'] < 0],
                             key=lambda row: row['delta_best_IoU'])[:8]
        best_gains = sorted([row for row in changed if row['delta_best_IoU'] > 0],
                            key=lambda row: -row['delta_best_IoU'])[:8]
        point_change = control != candidate
        pairs, counts = np.unique(np.column_stack([control[point_change], candidate[point_change]]),
                                  axis=0, return_counts=True)
        order = np.argsort(-counts)[:10]
        transfers = [dict(from_candidate=int(pairs[i, 0]), to_candidate=int(pairs[i, 1]),
            points=int(counts[i]), destination_source_index=receipt['results'][len(result['results'])]
                ['proposal_selection']['kept'][int(pairs[i, 1])] if pairs[i, 1] >= 0 else None)
            for i in order]
        result['results'].append(dict(scene=item['scene'], scores_unchanged=True,
            delta_AP_percentage_points=metrics['delta_vs_same_geometry_core_percentage_points'],
            eligible_GT_best_IoU_improvements=sum(row['delta_best_IoU'] > 0 for row in changed),
            eligible_GT_best_IoU_regressions=sum(row['delta_best_IoU'] < 0 for row in changed),
            largest_best_IoU_gains=best_gains, largest_best_IoU_losses=best_losses,
            largest_published_point_transfers=transfers,
            distinction='Best-IoU diagnostics allow shared proposals and are not AP/FN counts'))
        del gt, control, candidate
        gc.collect()
    result['interpretation'] = [
        'The independent CORE controls exactly reproduce the earlier registered-domain screen/full replay mask AP, so gains are caused by ownership rather than score or NMS changes.',
        'Unique source-mask support can correct ownership, but actual cached source identity is not guaranteed correct; the small scene0015 regression falsifies uniform improvement.',
        'Hypersim010 gains over the weak same-domain CORE screen remain far below v19. Fixed proposal suppression/quality and source identity errors remain limitations; ownership alone does not establish an acceptable cross-domain replacement.',
        'Scene0019 AP25/AP50 do not change, so this method does not fix its missing-instance/source-undersegmentation bottleneck.',
        'No new strict-box scores or production graph claim follows from mask-space screening.'
    ]
    (WORK/'diagnosis.json').write_text(json.dumps(result, indent=2)+'\n')
    print('CUE_INDEPENDENT_DIAGNOSIS_COMPLETE', flush=True)


if __name__ == '__main__':
    main()

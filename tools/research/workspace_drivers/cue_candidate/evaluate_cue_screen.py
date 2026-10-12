"""Independent registered-domain evaluation. This module alone reads GT."""
import gc
import hashlib
import json
import os
import resource
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
from scannet_utils.evaluate_semantic_instance import scannet_eval
ref.verify_evaluator()


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as file:
        for block in iter(lambda: file.read(8*1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


def evaluate_masks(path, digest, gt):
    if sha(path) != digest:
        raise ValueError('Candidate changed after GT-free construction')
    with np.load(path, allow_pickle=False) as data:
        masks, scores = data['masks'].copy(), data['scores'].copy()
    if masks.shape != (len(scores), len(gt)) or np.any(masks.sum(0) > 1):
        raise ValueError('Published mask domain/ownership mismatch')
    predictions = {str(i): dict(mask=mask, label_id=3, conf=float(score))
        for i, (mask, score) in enumerate(zip(masks, scores))}
    metrics = scannet_eval([predictions], [gt], None, (3,), ('object',), {3:'object'})
    return metrics


def delta(after, before):
    return {key: 100*(after[key]-before[key])
        for key in ['all_ap_25%', 'all_ap_50%', 'all_ap']}


def main():
    construction = json.loads((WORK/'construction.json').read_text())
    assert construction['complete'] and not construction['GT_in_construction']
    assert sha(WORK/'hypothesis.json') == construction['hypothesis_sha256']
    baseline = json.loads((ROOT/'v20_work/010_sage_domain_screen.json').read_text())
    reference = json.loads((ROOT/'v20_work/sage_verified_core_evaluation.json').read_text())
    scannet_reference = {row['scene_id']: row['metrics'] for row in reference['results']}
    result = dict(complete=False, full_pipeline_replay=False, candidate_screen_only=True,
        GT_in_construction=False, hypersim_strict_box_pending=True,
        unchanged_evaluator=True, evaluator_sha256=sha(__file__),
        construction_sha256=sha(WORK/'construction.json'), results=[])
    for row in construction['results']:
        if row['dataset'] == 'scannet':
            _, gt, _, gt_hashes = ref.ground_truth(row['scene'])
            official_core_v_reference = scannet_reference[row['scene']]
            frozen_v19_reference = None
        else:
            gt_path = ROOT/'esam_hypersim_work/evaluation_GT.npz'
            with np.load(gt_path, allow_pickle=False) as data:
                gt = data['gt'].copy()
            gt_hashes = {'frozen_observed_GT_sha256': sha(gt_path)}
            official_core_v_reference = None
            frozen_v19_reference = baseline['baseline_point_metrics']
        control = evaluate_masks(row['control_masks_file'], row['control_masks_sha256'], gt)
        gc.collect()
        candidate = evaluate_masks(row['masks_file'], row['masks_sha256'], gt)
        gc.collect()
        record = dict(scene=row['scene'], dataset=row['dataset'], GT_hashes=gt_hashes,
            core_control_metrics=control, cue_metrics=candidate,
            delta_vs_same_geometry_core_percentage_points=delta(candidate, control),
            official_full_replay_core_v_reference=official_core_v_reference,
            frozen_v19_point_reference=frozen_v19_reference,
            delta_vs_frozen_v19_percentage_points=delta(candidate, frozen_v19_reference)
                if frozen_v19_reference else None,
            changed_conflict_owners=row['ownership_audit']['changed_conflict_owners'],
            changed_owners_after_publication=row['changed_owners_after_publication'],
            process_peak_RSS_GiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2)
        result['results'].append(record)
        (WORK/'evaluation.json').write_text(json.dumps(result, indent=2)+'\n')
        print('CUE_INDEPENDENT_AP', row['scene'], json.dumps(record), flush=True)
        del gt
    result['complete'] = True
    result['all_four_improve_all_three_metrics_vs_control'] = all(
        all(value > 0 for value in row['delta_vs_same_geometry_core_percentage_points'].values())
        for row in result['results'])
    result['production_upgrade_justified'] = False
    result['new_Hypersim_strict_box_result'] = None
    (WORK/'evaluation.json').write_text(json.dumps(result, indent=2)+'\n')
    print('CUE_FOUR_DOMAIN_EVALUATION_COMPLETE', flush=True)


if __name__ == '__main__':
    main()

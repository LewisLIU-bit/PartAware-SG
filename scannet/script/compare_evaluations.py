"""Accept a measured improvement only under a shared independent GT protocol."""
import argparse
import hashlib
import json
import math
from pathlib import Path


def compare(candidate, baseline):
    fields = ('protocol', 'input_manifest', 'frames', 'voxel_m', 'projection_stride',
              'minimum_observed_gt_voxels', 'excluded_nyu40_ids', 'gt_objects',
              'gt_labels_sha256', 'gt_bbox_source', 'prediction_bbox_source')
    if any(key not in report or report[key] is None
           for report in (candidate, baseline) for key in fields):
        raise ValueError('Acceptance requires complete evaluation scope metadata')
    if any(candidate.get(key) != baseline.get(key) for key in fields):
        raise ValueError('Candidate and baseline use different inputs or evaluation scope')
    values = []
    for name in ('geometry_only_box_AP25', 'geometry_only_box_AP50', 'geometry_only_box_AP75'):
        values.append((name, candidate[name], baseline[name], True))
    values.append(('absolute_log_count_error', candidate['object_count_consistency']['absolute_log_ratio'],
                   baseline['object_count_consistency']['absolute_log_ratio'], False))
    for threshold in ('0.25', '0.5', '0.75'):
        a, b = candidate['one_to_one_bbox_geometry'][threshold], baseline['one_to_one_bbox_geometry'][threshold]
        values.extend((f'{name}@{threshold}', a[name], b[name], name == 'TP') for name in ('TP','FP','FN'))
    changes = []
    for name, a, b, higher in values:
        if not math.isfinite(a) or not math.isfinite(b):
            raise ValueError('Acceptance requires finite independent metrics')
        delta = (a-b) if higher else (b-a)
        changes.append({'metric': name, 'candidate': a, 'baseline': b,
                        'direction': 'higher' if higher else 'lower',
                        'non_regressing': delta >= -1e-10, 'strictly_improved': delta > 1e-10})
    non_regressing = all(row['non_regressing'] for row in changes)
    improved = any(row['strictly_improved'] for row in changes)
    return {'accepted': non_regressing and improved, 'non_regressing': non_regressing,
            'strictly_improved': improved, 'checks': changes,
            'scope': 'these observed scenes only; no guarantee on unseen captures',
            'construction_uses_ground_truth': False}


def load_verified(path):
    path = Path(path).expanduser().resolve()
    report = json.loads(path.read_text())
    for field, digest in (('graph_path','graph_sha256'), ('prediction_ply','prediction_ply_sha256')):
        if hashlib.sha256(Path(report[field]).read_bytes()).hexdigest() != report[digest]:
            raise ValueError('Evaluation is stale or its measured artifact has changed')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate', required=True)
    parser.add_argument('--baseline', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    result = compare(load_verified(args.candidate), load_verified(args.baseline))
    result.update(candidate=str(Path(args.candidate).resolve()), baseline=str(Path(args.baseline).resolve()))
    Path(args.output).write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result), flush=True)

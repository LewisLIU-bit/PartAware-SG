"""Observe surface coverage and fragmentation separately from strict box AP.

Dominant ground-truth assignment is an evaluation-only diagnostic, never a
predicted hierarchy or an alternative average-precision matching protocol.
"""
import numpy as np
from scipy.spatial import cKDTree


def voxel_centers(keys, size=.01):
    keys = np.asarray(keys, dtype=np.int64)
    mask = (1 << 21)-1
    grid = np.column_stack([keys >> 42, (keys >> 21) & mask, keys & mask])
    return ((grid-(1 << 20)+.5)*size).astype(np.float32)


def surface_diagnostics(predictions, truth, tolerance=.02, purity_threshold=.8):
    gt_points = [voxel_centers(g['keys']) for g in truth]
    gt_sizes = np.array([len(p) for p in gt_points])
    if not len(truth) or not np.sum(gt_sizes):
        return {'status': 'unavailable_no_observed_ground_truth'}
    labels = np.repeat(np.arange(len(truth)), gt_sizes)
    tree = cKDTree(np.concatenate(gt_points))
    groups = [[] for _ in truth]
    per_prediction = []
    all_prediction_points = []
    near_count, prediction_count = 0, 0
    for prediction in predictions:
        points = voxel_centers(prediction['keys'])
        if len(points):
            all_prediction_points.append(points)
        prediction_count += len(points)
        distance, nearest = tree.query(points, workers=2)
        visible = distance <= tolerance
        near_count += int(visible.sum())
        counts = np.bincount(labels[nearest[visible]], minlength=len(truth))
        dominant = int(np.argmax(counts))
        purity = float(counts[dominant]/max(len(points), 1))
        eligible = purity >= purity_threshold
        if eligible:
            groups[dominant].append(points)
        per_prediction.append({'prediction_id': prediction['id'],
            'dominant_gt_id': truth[dominant]['id'] if counts[dominant] else None,
            'purity': purity, 'foreground_surface_precision': float(visible.mean()) if len(points) else 0.,
            'pure_fragment': eligible})
    per_gt = []
    coverages = []
    geometric_coverages = []
    geometric_tree = cKDTree(np.concatenate(all_prediction_points)) if all_prediction_points else None
    for j, gt in enumerate(truth):
        if groups[j]:
            distance, _ = cKDTree(np.concatenate(groups[j])).query(gt_points[j], workers=2)
            coverage = float(np.mean(distance <= tolerance))
        else:
            coverage = 0.
        coverages.append(coverage)
        geometric_coverage = float(np.mean(geometric_tree.query(gt_points[j],workers=2)[0] <= tolerance)) if geometric_tree else 0.
        geometric_coverages.append(geometric_coverage)
        per_gt.append({'gt_id': gt['id'], 'label': gt['label'], 'union_surface_coverage': coverage,
            'geometric_surface_coverage': geometric_coverage,
            'pure_fragment_count': len(groups[j])})
    precision = near_count/max(prediction_count, 1)
    weighted_coverage = float(np.average(coverages, weights=gt_sizes))
    geometric_weighted = float(np.average(geometric_coverages,weights=gt_sizes))
    return {'status': 'available', 'protocol': 'evaluation_only_pure_fragment_union_visible_surface',
        'distance_tolerance_m': tolerance, 'dominant_assignment_purity': purity_threshold,
        'surface_precision': precision, 'macro_surface_coverage': float(np.mean(coverages)),
        'weighted_surface_coverage': weighted_coverage,
        'surface_F1': 2*precision*weighted_coverage/max(precision+weighted_coverage, 1e-12),
        'macro_geometric_surface_coverage': float(np.mean(geometric_coverages)),
        'weighted_geometric_surface_coverage': geometric_weighted,
        'geometric_surface_F1': 2*precision*geometric_weighted/max(precision+geometric_weighted,1e-12),
        'geometric_gt_coverage_recall50': float(np.mean(np.asarray(geometric_coverages) >= .5)),
        'geometric_gt_coverage_recall75': float(np.mean(np.asarray(geometric_coverages) >= .75)),
        'gt_coverage_recall50': float(np.mean(np.asarray(coverages) >= .5)),
        'gt_coverage_recall75': float(np.mean(np.asarray(coverages) >= .75)),
        'pure_fragment_excess': sum(max(len(group)-1, 0) for group in groups),
        'mean_pure_fragments_per_observed_gt': float(np.mean([len(group) for group in groups])),
        'per_gt': per_gt, 'per_prediction': per_prediction,
        'part_PQ': None,
        'limitation': 'Geometric coverage ignores identity, with surface precision penalizing non-GT points. Other coverage fields require pure GT-assigned fragments. GT groups are evaluation only; this is not AP, official mask IoU, or proof of predicted parent ownership; Hypersim has no part GT.'}


def box_diagnostics(predictions, truth, spatial, surfaces):
    """Expose errors of individual objects without changing AP or GT identity."""
    def overlap(a, b):
        lo, hi = np.maximum(a[0], b[0]), np.minimum(a[1], b[1])
        intersection = float(np.prod(np.maximum(hi-lo, 0)))
        va = float(np.prod(np.maximum(np.asarray(a[1])-a[0], 0)))
        vb = float(np.prod(np.maximum(np.asarray(b[1])-b[0], 0)))
        return intersection/max(va+vb-intersection, 1e-12)
    matrix = np.array([[overlap(p['bounds'], g['bounds']) for g in truth] for p in predictions])
    gt_surface = {g['gt_id']: g for g in surfaces.get('per_gt', [])}
    pred_surface = {str(p['prediction_id']): p for p in surfaces.get('per_prediction', [])}
    per_gt = []
    for j, gt in enumerate(truth):
        order = np.argsort(-matrix[:, j], kind='stable')
        candidates = [{'prediction_id': predictions[i]['id'], 'label': predictions[i]['label'],
            'IoU': float(matrix[i, j]), 'confidence': float(predictions[i]['confidence'])} for i in order[:3]]
        matches = {str(t): next((p for p in spatial[str(t)]['matches'] if p['gt_id'] == gt['id']), None)
                   for t in [.25, .5, .75]}
        per_gt.append({'gt_id': gt['id'], 'label': gt['label'], 'observed_voxels': len(gt['keys']),
            'best_predictions': candidates, 'one_to_one_matches': matches,
            'surface': gt_surface.get(gt['id'])})
    rank = sorted(range(len(predictions)), key=lambda i: (-predictions[i]['confidence'], i))
    per_prediction = []
    ranking = {str(t): [] for t in [.25, .5, .75]}
    for t in [.25, .5, .75]:
        used = set()
        for r, i in enumerate(rank):
            available = [j for j in range(len(truth)) if matrix[i, j] >= t and j not in used]
            best = max(available, key=lambda j: matrix[i, j]) if available else None
            if best is not None:
                used.add(best)
            j = int(matrix[i].argmax())
            ranking[str(t)].append({'rank': r+1, 'prediction_id': predictions[i]['id'],
                'true_positive': best is not None, 'matched_gt_id': truth[best]['id'] if best is not None else None,
                'matched_IoU': float(matrix[i, best]) if best is not None else None,
                'best_gt_id': truth[j]['id'], 'best_IoU': float(matrix[i, j]),
                'reason': 'matched' if best is not None else 'duplicate_competition' if matrix[i, j] >= t else 'localization_below_threshold'})
    for i, prediction in enumerate(predictions):
        j = int(matrix[i].argmax())
        per_prediction.append({'prediction_id': prediction['id'], 'label': prediction['label'],
            'confidence': float(prediction['confidence']), 'best_gt_id': truth[j]['id'],
            'best_gt_label': truth[j]['label'], 'best_IoU': float(matrix[i, j]),
            'surface': pred_surface.get(str(prediction['id'])),
            'ranking': {t: next(r for r in rows if r['prediction_id'] == prediction['id']) for t, rows in ranking.items()}})
    return {'protocol': 'per_object_diagnostic_not_individual_AP', 'ground_truth_used_for_construction': False,
            'per_gt': per_gt, 'per_prediction': per_prediction, 'AP_ranking': ranking,
            'limitation': 'Best IoU can refer to a neighboring object. Surface dominant identity and one-to-one matching must also be checked. Diagnostic grouping never changes GT or predictions.'}

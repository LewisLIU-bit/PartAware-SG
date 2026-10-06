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

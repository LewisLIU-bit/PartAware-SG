"""ESAM peer proposals on a shared observed point domain, without GT inputs.

This research adapter emits masks, not a finished semantic scene graph.
Scores from the two systems are uncalibrated. Default v19 is never modified.
"""
import numpy as np
from scipy.optimize import linear_sum_assignment


def overlap(primary, peer):
    a, b = np.asarray(primary, bool), np.asarray(peer, bool)
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[1]:
        raise ValueError('Two mask sets must share the same point domain')
    intersection = a.astype(np.int64) @ b.astype(np.int64).T
    union = a.sum(1)[:, None] + b.sum(1)[None, :] - intersection
    return np.divide(intersection, union, out=np.zeros_like(union, dtype=float), where=union > 0)


def propose(primary, primary_scores, peer, peer_scores, mode='supplement', minimum_points=100):
    """Fixed .5 mask association; add peer candidates with >=.5 novel surface.

    Union repair only claims points not already owned by another primary.
    No source/instance IDs, names, GT, learned thresholds or target counts.
    """
    if mode not in ('supplement', 'union', 'intersection'):
        raise ValueError('Unknown peer proposal mode')
    a, b = np.asarray(primary, bool), np.asarray(peer, bool)
    sa, sb = np.asarray(primary_scores, float), np.asarray(peer_scores, float)
    if sa.shape != (len(a),) or sb.shape != (len(b),) or not np.isfinite(np.r_[sa, sb]).all():
        raise ValueError('Finite score per mask is required')
    iou = overlap(a, b)
    owned = np.any(a, axis=0)
    rows, cols = linear_sum_assignment(-iou) if iou.size else ([], [])
    matches = [(int(i), int(j)) for i, j in zip(rows, cols) if iou[i, j] >= .5]
    masks = [m.copy() for m in a]
    origins = [dict(source='v19', source_index=i, peer_index=None) for i in range(len(a))]
    if mode != 'supplement':
        ownership_count = a.sum(0)
        for i, j in matches:
            if mode == 'union':
                other_ownership = ownership_count > a[i].astype(int)
                masks[i] |= b[j] & ~other_ownership
            else:
                masks[i] &= b[j]
            origins[i]['peer_index'] = j
    added = []
    for j, mask in enumerate(b):
        count = int(mask.sum())
        novelty = float(np.count_nonzero(mask & ~owned) / count) if count else 0.
        if count < minimum_points or novelty < .5 or (len(a) and iou[:, j].max() >= .5):
            continue
        masks.append(mask.copy())
        added.append(j)
        origins.append(dict(source='ESAM_unverified_semantics', source_index=j, novel_fraction=novelty))
    result = np.asarray(masks, bool).reshape(-1, a.shape[1])
    return result, np.r_[sa, sb[added]], dict(mode=mode, matches=matches,
        new_peer_candidates=added, origins=origins, GT_used=False,
        semantic_validation_completed=False, scores_calibrated=False)

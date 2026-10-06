"""Reject a planar sink identity with repeated independent support-surface evidence.

This does not estimate material or replace depth. Brightness is not a veto;
unknown, occluded and depth-inconsistent projections cannot cast negative votes.
"""
import numpy as np


def rejection(evidence):
    return (evidence.get('applicable', False)
            and evidence['holdout_supported_point_fraction'] >= .5
            and evidence['holdout_background_point_fraction'] > .8
            and evidence['holdout_background_views'] >= 5)


def measure(name, points, views, observed_frames=(), upright=False):
    evidence = {'applicable': False, 'claim': name, 'material_classification': False,
        'holdout_supported_point_fraction': 0., 'holdout_background_point_fraction': 0.,
        'holdout_background_views': 0, 'frame_evidence': []}
    if not upright or 'sink' not in name.casefold().replace('_', ' ') or len(points) < 32:
        return evidence
    centered = points-points.mean(0)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    normal = vt[-1]
    evidence.update(horizontal_normal=float(abs(normal[2])),
        plane_fraction_1cm=float(np.mean(np.abs(centered @ normal) <= .01)))
    evidence['applicable'] = evidence['horizontal_normal'] >= .95 and evidence['plane_fraction_1cm'] >= .9
    if not evidence['applicable']:
        return evidence
    sample = points[::max(1, len(points)//2048)]
    count = np.zeros(len(sample), int)
    total, background = np.zeros(len(sample)), np.zeros(len(sample))
    source_frames = set(observed_frames)
    frames = sorted(views.jobs)
    selected = [frames[i] for i in np.linspace(0, len(frames)-1, min(len(frames),100)).astype(int)]
    for fid in selected:
        if fid in source_frames:
            continue
        indices, weights, labels = views.project(sample, fid)
        if len(indices) < 16 or weights.sum() <= 0:
            continue
        support_labels = [r['frame_instance_id'] for r in views.records[fid]
            if any(word in r['object_name'].casefold() for word in ['counter', 'table', 'desk'])]
        negative = (labels == 0) | np.isin(labels, support_labels)
        count[indices] += 1
        total[indices] += weights
        background[indices] += weights*negative
        fraction = float(weights[negative].sum()/weights.sum())
        evidence['holdout_background_views'] += int(fraction >= .8)
        evidence['frame_evidence'].append({'frame_id': fid, 'visible_fraction': len(indices)/len(sample),
            'weighted_support_surface_fraction': fraction})
    eligible = count >= 3
    evidence['holdout_supported_point_fraction'] = float(eligible.mean())
    ratios = background/np.maximum(total, 1e-8)
    evidence['holdout_background_point_fraction'] = float(np.mean(ratios[eligible] >= .8)) if eligible.any() else 0.
    return evidence

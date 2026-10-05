"""Reject a large false object identity using independent visible background.

Depth-consistent observed points remain in the original fusion/background exports.
This component changes object acceptance only; no depth, masks or names are guessed.
"""
import numpy as np


def rejection(evidence):
    return (evidence.get('applicable', False)
            and evidence.get('holdout_background_views', 0) >= 5
            and evidence.get('holdout_supported_point_fraction', 0) >= .5
            and evidence.get('holdout_background_point_fraction', 0) > .5)


def measure(points, views, observed_frames=(), maximum=100):
    span = float(np.max(np.ptp(points, axis=0)))
    evidence = {'applicable': span > 3., 'maximum_extent_m': span,
                'visible_views': 0, 'background_views': 0,
                'background_fraction': 0., 'frame_evidence': [],
                'holdout_background_views': 0, 'holdout_supported_point_fraction': 0.,
                'holdout_background_point_fraction': 0.}
    if not evidence['applicable']:
        return evidence
    sample = points[::max(1, len(points)//2048)]
    source_frames = set(observed_frames)
    point_views = np.zeros(len(sample), int)
    point_mass = np.zeros(len(sample))
    background_mass = np.zeros(len(sample))
    frames = sorted(views.jobs)
    selected = [frames[i] for i in np.linspace(0, len(frames)-1, min(len(frames), maximum)).astype(int)]
    for fid in selected:
        indices, weights, labels = views.project(sample, fid)
        fraction = len(indices)/max(len(sample), 1)
        if len(indices) >= 16 and weights.sum() > 0 and fid not in source_frames:
            point_views[indices] += 1
            point_mass[indices] += weights
            background_mass[indices] += weights * (labels == 0)
            evidence['holdout_background_views'] += int(weights[labels == 0].sum()/weights.sum() >= .8)
        if len(indices) < 16 or fraction < .3 or weights.sum() <= 0:
            continue
        evidence['visible_views'] += 1
        background = float(weights[labels == 0].sum()/weights.sum())
        negative = fraction >= .5 and background >= .8
        evidence['background_views'] += int(negative)
        evidence['frame_evidence'].append({'frame_id': fid, 'visible_fraction': fraction,
            'weighted_background_fraction': background, 'negative': negative})
    evidence['background_fraction'] = evidence['background_views']/max(evidence['visible_views'], 1)
    eligible = point_views >= 3
    evidence['holdout_supported_point_fraction'] = float(eligible.mean())
    ratios = background_mass/np.maximum(point_mass, 1e-8)
    evidence['holdout_background_point_fraction'] = float(np.mean(ratios[eligible] >= .8)) if eligible.any() else 0.
    return evidence

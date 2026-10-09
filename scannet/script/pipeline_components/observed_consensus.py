"""Verify object existence with joint registered depth/mask measurements.

Appearance scores are secondary to independent repeatable surface ownership.
This adapter uses no categories, annotation, language request or generated point.
"""
import numpy as np
from .thin_geometry import depth_mask_votes


def measure(points, gid, views, records, resolve, remap):
    sample = points[::max(1, len(points)//2048)]
    seen = np.zeros(len(sample), int)
    positive = seen.copy()
    votes, centers = [], []
    for fid in sorted(views.jobs):
        local_ids = [int(r['frame_instance_id']) for r in records[fid]
                     if resolve(str(r.get('instance_id', -1)), remap) == gid]
        pose, depth, mask = views.get(fid)
        visible, own = depth_mask_votes(sample, pose, depth/views.scale,
                                       np.isin(mask, local_ids), views.kd, views.kc)
        seen += visible
        positive += own
        fraction = float(own.sum()/max(visible.sum(), 1))
        if visible.sum() >= 32 and fraction >= .7:
            votes.append(fid)
            centers.append(pose[:3, 3])
    supported = (positive >= 3) & (positive/np.maximum(seen, 1) >= .65)
    baseline = float(np.max(np.linalg.norm(np.array(centers)-centers[0], axis=1))) if centers else 0.
    return {'joint_depth_mask_supported_fraction': float(supported.mean()) if len(sample) else 0.,
            'joint_depth_mask_support_views': len(votes), 'support_frame_ids': votes,
            'camera_baseline_m': baseline, 'sample_points': len(sample),
            'minimum_point_views': 3, 'minimum_point_ownership_ratio': .65,
            'minimum_frame_ownership_ratio': .7}


def confirms_existence(evidence, mixed_fraction):
    """A reliable surface may rescue semantics, never mixed-instance/background vetoes."""
    return (mixed_fraction <= .1 and evidence['joint_depth_mask_support_views'] >= 5
            and evidence['joint_depth_mask_supported_fraction'] >= .65
            and evidence['camera_baseline_m'] >= .08)


def verified_core(points, gid, views, records, resolve, remap):
    """Rescue a stable surface, not an entire weakly supported proposal."""
    seen = np.zeros(len(points), int); positive = seen.copy()
    for fid in sorted(views.jobs):
        local_ids = [int(r['frame_instance_id']) for r in records[fid]
                     if resolve(str(r.get('instance_id', -1)), remap) == gid]
        pose, depth, mask = views.get(fid)
        visible, own = depth_mask_votes(points, pose, depth/views.scale,
                                       np.isin(mask, local_ids), views.kd, views.kc)
        seen += visible; positive += own
    core = points[(positive >= 3) & (positive/np.maximum(seen, 1) >= .65)]
    retention = np.ptp(core, axis=0)/np.maximum(np.ptp(points, axis=0), .01) if len(core) else np.zeros(3)
    return core, {'verified_core_points': len(core), 'source_points': len(points),
                  'axis_extent_retention': retention.tolist()}


def confirms_core(evidence, core, mixed_fraction):
    return (mixed_fraction <= .1 and evidence['joint_depth_mask_support_views'] >= 5
            and evidence['camera_baseline_m'] >= .08 and core['verified_core_points'] >= 128
            and core['verified_core_points']/max(core['source_points'], 1) >= .3
            and sorted(core['axis_extent_retention'])[-1] >= .8
            and sorted(core['axis_extent_retention'])[-2] >= .6)


def refine_verified_surface(points, gid, views, records, resolve, remap):
    """Trim repeatedly contradicted fringes only around a strongly verified core."""
    seen = np.zeros(len(points), int)
    positive = seen.copy()
    for fid in sorted(views.jobs):
        local_ids = [int(r['frame_instance_id']) for r in records[fid]
                     if resolve(str(r.get('instance_id', -1)), remap) == gid]
        pose, depth, mask = views.get(fid)
        visible, own = depth_mask_votes(points, pose, depth/views.scale,
                                       np.isin(mask, local_ids), views.kd, views.kc)
        seen += visible
        positive += own
    ratio = positive/np.maximum(seen, 1)
    stable = (positive >= 3) & (ratio >= .65)
    rejected = (seen >= 5) & (seen-positive >= 3) & (ratio < .65)
    proposed = points[~rejected]
    retention = np.ptp(proposed, axis=0)/np.maximum(np.ptp(points, axis=0), .01) if len(proposed) else np.zeros(3)
    accepted = (stable.mean() >= .8 and len(proposed) >= 128
                and len(proposed)/len(points) >= .8 and np.min(retention) >= .6)
    return (proposed if accepted else points), {'accepted': bool(accepted),
        'stable_core_fraction': float(stable.mean()), 'source_points': len(points),
        'proposed_removed_points': int(rejected.sum()), 'removed_points': int(rejected.sum()) if accepted else 0,
        'axis_extent_retention': retention.tolist(), 'unobserved_points_preserved': True}

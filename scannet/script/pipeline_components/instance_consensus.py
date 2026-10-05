"""Observation-only instance consensus and depth-weighted surface refinement.

Inspired by MaskClustering and MV3DIS; this is a project-specific adapter,
not a reproduction of either paper's complete model. No GT is accessed.
"""
import json
import cv2
import numpy as np
from scipy.spatial import cKDTree
from partaware.geometry import load_capture, voxel_downsample


class Views:
    def __init__(self, context, jobs, kd, kc, scale, records):
        self.context, self.kd, self.kc, self.scale = context, kd, kc, scale
        self.jobs = {j['frame_id']: j for j in jobs}
        self.records, self.cache = records, {}

    def get(self, fid):
        if fid not in self.cache:
            job = self.jobs[fid]
            self.cache[fid] = (np.loadtxt(job['pose']),
                cv2.imread(str(job['depth']), cv2.IMREAD_UNCHANGED),
                cv2.imread(str(self.context.scene/'refined_instance'/f'{fid}.png'), cv2.IMREAD_UNCHANGED))
        return self.cache[fid]

    def project(self, points, fid):
        pose, depth, mask = self.get(fid)
        camera = (points-pose[:3, 3]) @ pose[:3, :3]
        z = camera[:, 2]
        safe = np.maximum(z, 1e-8)
        uv = np.rint(camera[:, :2]/safe[:, None] * [self.kc[0, 0], self.kc[1, 1]]
                     + [self.kc[0, 2], self.kc[1, 2]]).astype(int)
        duv = np.rint(camera[:, :2]/safe[:, None] * [self.kd[0, 0], self.kd[1, 1]]
                      + [self.kd[0, 2], self.kd[1, 2]]).astype(int)
        valid = (z > 0) & (uv[:, 0] >= 0) & (uv[:, 1] >= 0) & (uv[:, 0] < mask.shape[1]) & (uv[:, 1] < mask.shape[0])
        valid &= (duv[:, 0] >= 0) & (duv[:, 1] >= 0) & (duv[:, 0] < depth.shape[1]) & (duv[:, 1] < depth.shape[0])
        indices = np.flatnonzero(valid)
        measured = depth[duv[indices, 1], duv[indices, 0]]/self.scale
        tolerance = .025 + .01*measured
        weights = np.maximum(0, 1-np.abs(z[indices]-measured)/tolerance)
        visible = (measured > 0) & (weights > 0)
        indices, weights = indices[visible], weights[visible]
        labels = mask[uv[indices, 1], uv[indices, 0]].astype(int)
        return indices, weights, labels


def pair_evidence(left, right, views):
    """Require whole-object agreement and veto independent separation evidence."""
    fids = sorted(left.frames | right.frames)
    if len(fids) > 12:
        fids = [fids[i] for i in np.linspace(0, len(fids)-1, 12).astype(int)]
    a, b = left.points[::max(1, len(left.points)//1500)], right.points[::max(1, len(right.points)//1500)]
    support, conflicts, jointly_visible = 0, 0, 0
    for fid in fids:
        _, wa, la = views.project(a, fid)
        _, wb, lb = views.project(b, fid)
        if len(la) < 16 or len(lb) < 16:
            continue
        jointly_visible += 1
        ids = set(la) | set(lb)
        ca = {int(k): float(wa[la == k].sum()/wa.sum()) for k in ids if k != 0}
        cb = {int(k): float(wb[lb == k].sum()/wb.sum()) for k in ids if k != 0}
        same = max((min(ca.get(k, 0), cb.get(k, 0)) for k in ids if k != 0), default=0)
        if same >= .6:
            support += 1
        elif ca and cb:
            ka, kb = max(ca, key=ca.get), max(cb, key=cb.get)
            if ka != kb and ca[ka] >= .7 and cb[kb] >= .7:
                conflicts += 1
    return {'support_views': support, 'conflict_views': conflicts,
            'jointly_visible_views': jointly_visible,
            'consensus': support/max(jointly_visible, 1)}


def reconcile(tracks, views, audit):
    """Join physical duplicates without requiring identical text labels."""
    remap = {}
    for i, left in enumerate(tracks):
        if left.id in remap:
            continue
        for right in tracks[i+1:]:
            if right.id in remap or min(len(left.points), len(right.points)) < 32:
                continue
            if np.linalg.norm(left.points.mean(0)-right.points.mean(0)) > 1.5:
                continue
            a = float(np.mean(cKDTree(right.points).query(left.points)[0] < .03))
            b = float(np.mean(cKDTree(left.points).query(right.points)[0] < .03))
            overlapping_surface = min(a, b) >= .65 and max(a, b) >= .85
            if not overlapping_surface:
                if float(left.semantic @ right.semantic) < .5:
                    continue
                separation = np.maximum(np.maximum(left.points.min(0)-right.points.max(0),
                                                   right.points.min(0)-left.points.max(0)), 0)
                if np.linalg.norm(separation) > .25:
                    continue
            evidence = pair_evidence(left, right, views)
            minimum_views = 2 if overlapping_surface else 3
            minimum_consensus = .6 if overlapping_surface else .8
            if evidence['conflict_views'] or evidence['support_views'] < minimum_views or evidence['consensus'] < minimum_consensus:
                continue
            nl, nr = len(left.observations), len(right.observations)
            left.points = voxel_downsample(np.concatenate([left.points, right.points]), .01)
            left.semantic_sum += right.semantic_sum
            left.visual_sum += right.visual_sum
            left.color = (nl*left.color+nr*right.color)/(nl+nr)
            left.frames |= right.frames
            left.names.update(right.names)
            left.observations += right.observations
            remap[right.id] = left.id
            audit.append({'message': '三维表面与多视角共识合并重复实例',
                          'source_id': right.id, 'target_id': left.id,
                          'surface_coverage_3cm': [a, b], **evidence})
    return remap


def refine(tracks, views, remap, audit):
    """Reject repeatedly contradicted points, keeping unobserved surfaces unknown."""
    all_fids = list(views.jobs)
    for track in tracks:
        if track.id in remap:
            continue
        own = sorted(track.frames)
        extra = [f for f in all_fids if f not in track.frames]
        fids = own[::max(1, len(own)//12)][:12] + extra[::max(1, len(extra)//12)][:12]
        positive = np.zeros(len(track.points))
        total = np.zeros(len(track.points))
        positive_views = np.zeros(len(track.points), int)
        negative_views = np.zeros(len(track.points), int)
        for fid in fids:
            indices, weights, labels = views.project(track.points, fid)
            local_ids = {int(r['frame_instance_id']) for r in views.records[fid]
                         if remap.get(r['instance_id'], r['instance_id']) == track.id}
            member = np.isin(labels, list(local_ids))
            total[indices] += weights
            positive[indices[member]] += weights[member]
            positive_views[indices[member]] += 1
            negative_views[indices[~member]] += 1
        ratio = np.divide(positive, total, out=np.ones_like(total), where=total > 0)
        # Absence of a detection is weaker evidence than a positive measurement.
        rejected = (negative_views >= 3) & (ratio < .35) & (positive_views <= 1)
        before = len(track.points)
        track.points = track.points[~rejected]
        q = [o.get('mask_quality') for o in track.observations if o.get('mask_quality') is not None]
        mask_quality = float(np.mean(q)) if q else None
        supported = (positive_views >= 2) & ~rejected
        reliability = float(np.mean(ratio[~rejected])) if np.any(~rejected) else 0.
        per_frame = {}
        for observation in track.observations:
            per_frame.setdefault(observation['frame_id'], []).append(observation['confidence'])
        detector = float(np.mean([np.mean(x) for x in per_frame.values()]))
        quality_factor = .5+.5*mask_quality if mask_quality is not None else 1.
        track.quality = {'mask_quality': mask_quality, 'reprojection_support': reliability,
                         'multi_view_point_fraction': float(supported.sum()/max(before, 1)),
                         'confidence': detector*quality_factor*np.sqrt(max(reliability, 0)),
                         'confidence_type': 'uncalibrated_observation_quality_score',
                         'removed_points': int(rejected.sum())}
        audit.append({'message': '可见深度加权精修实例表面', 'global_id': track.id,
                      'before_points': before, 'after_points': len(track.points), **track.quality})


def process(context, tracks, records, data, jobs, kd, kc, scale, audit):
    views = Views(context, jobs, kd, kc, scale, records)
    remap = reconcile(tracks, views, audit)
    refine(tracks, views, remap, audit)
    context.event('物体共识精修完成', merged_duplicates=len(remap),
                  removed_points=sum(getattr(t, 'quality', {}).get('removed_points', 0) for t in tracks))
    return remap

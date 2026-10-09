"""Validate whole-object identity before exclusive ownership can delete a body.

This project adaptation uses native multiview masks as in 2D-guided 3D methods,
but preserves whole/part evidence instead of flattening it. No annotation,
category-specific prompt, target count or generated surface is used. Detach
HIERARCHY_VALIDATION from the v13 registry to remove the adapter.
"""
from collections import Counter
import hashlib
from pathlib import Path

import numpy as np

from .native_assembly import NativeMasks


def normalize(name):
    return name.replace('_', ' ').strip().lower()


def acceptance(evidence):
    """Require repeated identity, depth visibility and distinct camera positions."""
    return (evidence.get('applicable', False)
            and evidence.get('whole_identity_views', 0) >= 5
            and evidence.get('whole_identity_consensus', 0) >= .65
            and evidence.get('camera_baseline_m', 0) >= .1)


class Evidence:
    def __init__(self, views, tracks, nodes):
        self.views, self.tracks, self.nodes = views, tracks, nodes
        self.native = NativeMasks(views)
        self.decisions = {}

    def names(self, gid):
        votes = Counter(normalize(o['name']) for o in self.tracks[gid]['observations'])
        total = sum(votes.values())
        names = {name for name, count in votes.items() if count >= .25*max(total, 1)}
        names.add(normalize(self.nodes[gid]['name']))
        return names

    def best_mask(self, points, fid, names):
        mass = self.native.mass(points, fid)
        if mass is None:
            return None
        choices = [(float(mass[i]), float(r['confidence']), i)
                   for i, r in enumerate(self.native.records(fid))
                   if normalize(r['object_name']) in names]
        return max(choices) if choices else None

    def measure(self, points, gid, identity_gid=None):
        from .proposal_validation import sample_frames
        evidence = dict(applicable=False, whole_identity_views=0, identity_visible_views=0,
                        whole_identity_consensus=0., camera_baseline_m=0., frames=[],
                        generated_points=0, ground_truth_used=False)
        if (not self.native.available or len(points) < 128
                or np.max(np.ptp(points, axis=0)) < .5):
            self.decisions[gid] = evidence
            return evidence
        evidence['applicable'] = True
        names = self.names(identity_gid or gid)
        sample = points[::max(1, len(points)//2048)]
        own = sample_frames(self.tracks[gid]['observed_frames'], 24)
        frames = own+[fid for fid in sample_frames(self.views.jobs, 24) if fid not in own]
        centers = []
        for fid in frames:
            if fid not in self.native.expected:
                continue
            indices, weights, _ = self.views.project(sample, fid)
            if len(indices) < 32 or len(indices)/len(sample) < .3:
                continue
            evidence['identity_visible_views'] += 1
            best = self.best_mask(sample, fid, names)
            if best is None:
                continue
            fraction, confidence, index = best
            accepted = fraction >= .8 and confidence >= .5
            evidence['frames'].append(dict(frame_id=fid, native_mask_index=index,
                whole_identity_fraction=fraction, detector_confidence=confidence,
                accepted=accepted))
            if accepted:
                evidence['whole_identity_views'] += 1
                centers.append(self.views.get(fid)[0][:3, 3])
        evidence['whole_identity_consensus'] = (evidence['whole_identity_views']
                                               / max(evidence['identity_visible_views'], 1))
        if len(centers) >= 2:
            centers = np.asarray(centers)
            evidence['camera_baseline_m'] = float(np.max(np.linalg.norm(
                centers[:, None]-centers[None, :], axis=2)))
        evidence['confirmed'] = acceptance(evidence)
        self.decisions[gid] = evidence
        return evidence

    def mask(self, fid, index):
        width = self.views.get(fid)[2].shape[1]
        return np.unpackbits(self.native.get(fid)[index], axis=-1, count=width).astype(bool)

    def refine(self, points, gid):
        """Trim only repeatedly contradicted samples using the original identity mask."""
        positive = np.zeros(len(points), int)
        negative = positive.copy()
        total = np.zeros(len(points))
        own = total.copy()
        for frame in self.decisions[gid]['frames']:
            if not frame['accepted']:
                continue
            fid = frame['frame_id']
            indices, weights, _ = self.views.project(points, fid)
            pose = self.views.get(fid)[0]
            camera = (points[indices]-pose[:3, 3]) @ pose[:3, :3]
            uv = np.rint(camera[:, :2]/camera[:, 2, None]
                          * [self.views.kc[0, 0], self.views.kc[1, 1]]
                          + [self.views.kc[0, 2], self.views.kc[1, 2]]).astype(int)
            mask = self.mask(fid, frame['native_mask_index'])
            member = mask[uv[:, 1], uv[:, 0]]
            positive[indices[member]] += 1
            negative[indices[~member]] += 1
            total[indices] += weights
            own[indices] += weights*member
        ratio = own/np.maximum(total, 1e-8)
        rejected = (negative >= 3) & (positive <= 1) & (ratio < .35)
        return points[~rejected], dict(before_points=len(points), after_points=int((~rejected).sum()),
            removed_points=int(rejected.sum()), unknown_points_retained=int((total == 0).sum()),
            generated_points=0, point_coordinates_preserved=True)


def component_sha256():
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()

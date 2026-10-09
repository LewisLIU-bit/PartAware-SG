"""Assemble contacting measured fragments using original complete SAM3 masks.

No category-specific rule, annotation, target count or generated point is used.
Overlapping original masks are queried as packed bits, independently of the
exclusive public label image. Detach SURFACE_ASSEMBLY to remove this adapter.
"""
import hashlib
import json
from pathlib import Path
from collections import OrderedDict

import numpy as np
from scipy.spatial import cKDTree


class NativeMasks:
    def __init__(self, views):
        self.views = views
        self.source = Path(views.context.scene)
        visited = set()
        while not (self.source/'sam3_inference_audit.json').exists():
            receipt = self.source/'cache_reuse.json'
            if not receipt.exists() or self.source in visited:
                self.available = False
                return
            visited.add(self.source)
            self.source = Path(json.loads(receipt.read_text())['source'])
        receipt = json.loads((self.source/'sam3_inference_audit.json').read_text())
        self.expected = {r['frame_id']: r for r in receipt['frames']}
        self.available = bool(receipt.get('complete'))
        self.cache = OrderedDict()
        self.record_cache = {}

    def get(self, fid):
        if fid in self.cache:
            self.cache.move_to_end(fid)
            return self.cache[fid]
        with np.load(self.source/'sam3_cache'/f'{fid}.npz', allow_pickle=False) as saved:
            if str(saved['signature']) != self.expected[fid]['signature']:
                raise ValueError('Complete-mask cache signature changed')
            records = json.loads(str(saved['records']))
            full = [i for i, r in enumerate(records) if not r['is_zoom_proposal']]
            value = saved['packed_masks'][full].copy()
        rgb = Path(self.views.jobs[fid]['rgb'])
        if hashlib.sha256(rgb.read_bytes()).hexdigest() != self.expected[fid]['rgb_sha256']:
            raise ValueError('Complete-mask RGB provenance changed')
        self.cache[fid] = value
        self.record_cache[fid] = [records[i] for i in full]
        while len(self.cache) > 4:
            expired, _ = self.cache.popitem(last=False)
            self.record_cache.pop(expired, None)
        return value

    def records(self, fid):
        """Return metadata in exactly the packed-mask order, after provenance checks."""
        self.get(fid)
        return self.record_cache[fid]

    def mass(self, points, fid):
        indices, weights, _ = self.views.project(points, fid)
        if len(indices) < 16:
            return None
        pose = self.views.get(fid)[0]
        camera = (points[indices]-pose[:3, 3]) @ pose[:3, :3]
        uv = np.rint(camera[:, :2]/camera[:, 2, None]*[self.views.kc[0, 0], self.views.kc[1, 1]]
                     +[self.views.kc[0, 2], self.views.kc[1, 2]]).astype(int)
        packed = self.get(fid)
        hits = (packed[:, uv[:, 1], uv[:, 0]//8] >> (7-uv[:, 0]%8)) & 1
        return hits @ weights/max(weights.sum(), 1e-9)

    def measure(self, a, b, frames):
        from .proposal_validation import sample_frames
        support = separation = joint = 0
        for fid in sample_frames(frames):
            if fid not in self.expected:
                continue
            left, right = self.mass(a, fid), self.mass(b, fid)
            if left is None or right is None:
                continue
            joint += 1
            if np.any(np.minimum(left, right) >= .6):
                support += 1
            elif np.any(left >= .8) and np.any(right >= .8):
                separation += 1
        return {'native_whole_views': support, 'native_separation_views': separation,
                'native_joint_views': joint, 'native_consensus': support/max(joint, 1)}


def acceptance(evidence, semantic, contact, extent):
    return (extent >= .5 and semantic >= .8 and contact <= .03 and evidence['native_whole_views'] >= 3
            and evidence['native_consensus'] >= .6 and evidence['native_separation_views'] == 0)


def reconcile(survivors, geometry, tracks, metrics, views, records, remap, audit, nodes, aliases):
    from .whole_object_consensus import cosine
    native = NativeMasks(views)
    if not native.available:
        audit.append({'message': '原生完整掩码整物体关联', 'accepted': False,
                      'reason': '没有同场景完整原生掩码来源，保留原关联'})
        return survivors
    current = list(survivors)
    # Agglomerate the strongest observed relation, then remeasure the entire
    # proposed body. A chain of local contacts alone cannot establish an object.
    pair_cache = {}
    while True:
        eligible = []
        for i, left in enumerate(current):
            a = geometry[left]
            if len(a) < 128:
                continue
            for right in current[i+1:]:
                b = geometry[right]
                if len(b) < 128:
                    continue
                semantic = cosine(nodes[left]['text_embedding'], nodes[right]['text_embedding'])
                if semantic < .8:
                    continue
                extent = float(np.max(np.maximum(a.max(0), b.max(0))-np.minimum(a.min(0), b.min(0))))
                # Small touching items, including stacked dishes, must retain
                # independent identity rather than becoming a large-body assembly.
                if extent < .5:
                    continue
                gap = np.maximum(np.maximum(a.min(0)-b.max(0), b.min(0)-a.max(0)), 0)
                if np.linalg.norm(gap) > .03:
                    continue
                contact = float(cKDTree(a).query(b)[0].min())
                if contact > .03:
                    continue
                key = (left, right, len(a), len(b), len(tracks[left]['observations']), len(tracks[right]['observations']))
                if key not in pair_cache:
                    pair_cache[key] = native.measure(a[::max(1, len(a)//2048)], b[::max(1, len(b)//2048)],
                        tracks[left]['observed_frames']+tracks[right]['observed_frames'])
                evidence = pair_cache[key]
                if acceptance(evidence, semantic, contact, extent):
                    eligible.append((evidence['native_whole_views'], evidence['native_consensus'],
                                     left, right, semantic, contact, evidence))
        if not eligible:
            break
        _, _, left, right, semantic, contact, evidence = max(eligible)
        target, source = sorted([left, right], key=lambda gid: -len(geometry[gid]))
        remap[source] = target
        combined = np.concatenate([geometry[target], geometry[source]])
        _, indices = np.unique(np.floor(combined/.005).astype(np.int64), axis=0, return_index=True)
        geometry[target] = combined[np.sort(indices)]
        na, nb = len(tracks[target]['observations']), len(tracks[source]['observations'])
        field = 'visual_embedding'
        nodes[target][field] = ((na*np.asarray(nodes[target][field])+nb*np.asarray(nodes[source][field]))/(na+nb)).tolist()
        tracks[target]['observations'] += tracks[source]['observations']
        tracks[target]['observed_frames'] = sorted(set(tracks[target]['observed_frames']+tracks[source]['observed_frames']))
        tracks[target]['confidence'] = (na*tracks[target]['confidence']+nb*tracks[source]['confidence'])/(na+nb)
        tracks[target]['point_count'] = len(geometry[target])
        tracks[target].setdefault('native_assembly_source_ids', [target]).append(source)
        entry = {'message': '原生完整掩码与接触表面恢复整物体归属', 'accepted': True,
                 'source_id': source, 'target_id': target, 'semantic': semantic,
                 'measured_contact_m': contact, 'generated_points': 0, **evidence}
        aliases[source] = {'canonical_id': target, 'source_name': nodes[source]['name'],
                           'role': 'native_complete_mask_measured_fragment', 'evidence': entry}
        audit.append(entry)
        current.remove(source)
    return current


def component_sha256():
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()

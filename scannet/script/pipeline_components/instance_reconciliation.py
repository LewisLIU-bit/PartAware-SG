"""MIRA: Measured Instance Reconciliation and Association.

Project adaptation of mask-graph consensus and 3D-guided mask matching.
All coordinates are observed depth. No annotation, scene/category whitelist,
fixed multiplicity, object identifier, coordinate exception or new vision call.
"""
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
import open3d as o3d

from partaware.geometry import load_capture
from . import contact_instances as cmr
from .instance_consensus import Views
from .native_assembly import NativeMasks
from .part_geometry import replace_regions
from .visible_instances import features, simultaneous_separation
from .visual_part_anchoring import acceptance as part_acceptance


def admissible_pair(left, right):
    a, b = left['sample'], right['sample']
    gap = np.maximum(np.maximum(a.min(0)-b.max(0), b.min(0)-a.max(0)), 0)
    if np.linalg.norm(gap) > .006:
        return 0.
    if left['name'] != right['name']:
        x, y = features(left), features(right)
        if x is None or y is None or cmr.semantic_agreement(x, y) < .8:
            return 0.
    if simultaneous_separation(left, right):
        return 0.
    x, y = cmr.coverage(a, b), cmr.coverage(b, a)
    if min(x, y) < .15 or max(x, y) < .85:
        return 0.
    return min(x, y)


def associate(groups):
    result = []
    for group in sorted(groups, key=lambda g: (-len(g['frames']), -g['seed_size'])):
        candidates = [(admissible_pair(group, old), i) for i, old in enumerate(result)]
        score, index = max(candidates, default=(0., -1))
        if score == 0:
            result.append(group)
            continue
        old = result[index]
        old['points'].extend(group['points'])
        old['evidence'].extend(group['evidence'])
        old['frames'].update(group['frames'])
        union = cmr.sample_cells(np.concatenate([old['sample'], group['sample']]), .003)
        old['sample'] = union[::max(1, len(union)//2000)]
        old['seed_size'] = max(old['seed_size'], group['seed_size'])
    return result


def proposal(group, views, native):
    frames = sorted(group['frames'])
    if len(frames) < 4:
        return None
    local = [e['local_record'] for e in group['evidence'] if e['local_record'] is not None]
    quality = np.mean([max(e['confidence'] for e in group['evidence'] if e['frame_id'] == fid) for fid in frames])
    if len(local) < 3 or quality < (.55 if len(frames) >= 6 else .75):
        return None
    centers = np.array([e['camera_center'] for e in group['evidence']])
    baseline = float(np.linalg.norm(centers.max(0)-centers.min(0)))
    if baseline < .15:
        return None
    q, votes = cmr.projected_support(cmr.sample_cells(np.concatenate(group['points']), .002), group, views, native)
    if len(q) < 64 or np.ptp(q, axis=0).max() > 3:
        return None
    evidence = dict(frames=frames, native_mean_confidence=float(quality), camera_baseline_m=baseline,
        measured_points=len(q), minimum_point_views=3, generated_points=0,
        whole_image_frames=len({e['frame_id'] for e in group['evidence'] if not e['is_zoom_proposal']}))
    return q, evidence


def related_surface_duplicate(candidate, existing, candidate_semantic, existing_semantic):
    """A related name does not make an already owned surface a new object."""
    return (cmr.semantic_agreement(candidate_semantic, existing_semantic) >= .5
            and cmr.coverage(candidate, existing, .015) >= .8)


def construct(context):
    metadata, jobs, kd, kc, scale = load_capture(context.manifest, context.image_dir, context.scene/'refined_instance')
    audit = dict(algorithm='MIRA', ground_truth_used=False, vision_api_calls=0, generated_points=0,
        component_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), objects=[])
    if metadata.get('world_frame') != 'hypersim_world_z_up':
        return
    views = Views(context, jobs, kd, kc, scale, {})
    native = NativeMasks(views)
    if not native.available:
        return
    groups = getattr(context, 'mira_groups', None)
    if groups is None:
        groups = associate(cmr.observations(context, jobs, kd, kc, scale, native))
    proposals = [(g, *item) for g in groups if (item := proposal(g, views, native)) is not None]
    context.event('掩码图深度共识候选', groups=len(groups), proposals=len(proposals))
    graph_path = context.scene/'topology_map.json'
    graph = json.loads(graph_path.read_text())
    nodes = graph['object_nodes']['nodes']
    cloud_path = context.scene/'instance_cloud_cleaned.ply'
    cloud = o3d.io.read_point_cloud(str(cloud_path))
    points = np.asarray(cloud.points)
    rgb = np.rint(np.asarray(cloud.colors)*255).astype(int)
    ids = rgb[:, 0]+255*rgb[:, 1]+255**2*rgb[:, 2]
    regions = {gid: points[ids == int(gid)] for gid in nodes}
    tracks_path = context.scene/'validated_object_tracks.json'
    tracks = json.loads(tracks_path.read_text())
    next_id = max(map(int, nodes))+1
    replacements = {}
    for group, q, evidence in sorted(proposals, key=lambda item: -len(item[0]['frames'])):
        semantic = features(group)
        owners = []
        for gid, region in regions.items():
            if nodes[gid].get('atomic_instance_evidence') or nodes[gid].get('geometry_hypothesis') or len(region) < 48:
                continue
            agreement = cmr.semantic_agreement(semantic, nodes[gid]['text_embedding'])
            if agreement < .75:
                continue
            a, b = cmr.coverage(q, region, .015), cmr.coverage(region, q, .015)
            if (a >= .65 and b >= .3) or (a >= .4 and b >= .8):
                owners.append((a*b, gid, a, b))
        owners.sort(reverse=True)
        if owners:
            continue
        minimum_frames = 4 if evidence['native_mean_confidence'] >= .8 else 6
        if len(group['frames']) < minimum_frames or evidence['whole_image_frames'] < 3:
            continue
        if any(cmr.coverage(q, region, .01) >= .3 for gid, region in regions.items()
               if nodes[gid].get('atomic_instance_evidence') or nodes[gid].get('geometry_hypothesis')):
            continue
        part_owners = {}
        for part_id, part in graph.get('part_nodes', {}).items():
            parent = str(part.get('parent_id'))
            point_file = context.scene/'parts'/f'{part_id}.points.npy'
            if parent not in nodes or not point_file.is_file():
                continue
            source_features = part.get('source_object_features', {})
            if (part.get('status') == 'confirmed' and source_features.get('text_embedding') is not None
                    and cmr.semantic_agreement(semantic, source_features['text_embedding']) >= .8):
                part_points = np.load(point_file, allow_pickle=False)
                if cmr.coverage(q, part_points, .015) >= .6 and cmr.coverage(part_points, q, .015) >= .5:
                    part_owners.setdefault(parent, []).append((part_id, {'confirmed_visual_source_identity': True}))
                    continue
            proof = part_acceptance(part, q, regions[parent], group['name'], nodes[parent]['name'], group['frames'])
            if proof is None:
                continue
            part_points = np.load(point_file, allow_pickle=False)
            if cmr.coverage(q, part_points, .015) >= .8:
                part_owners.setdefault(parent, []).append((part_id, proof))
        if len(part_owners) == 1:
            parent, parts = next(iter(part_owners.items()))
            audit['objects'].append(dict(instance_id=parent, name=group['name'], action='defer_confirmed_part_duplicate',
                part_ids=[pid for pid, _ in parts], **evidence))
            continue
        if any(cmr.semantic_agreement(semantic, nodes[gid]['text_embedding']) >= .75
               and cmr.coverage(q, region, .015) >= .6 for gid, region in regions.items()):
            continue
        if any(related_surface_duplicate(q, region, semantic, nodes[gid]['text_embedding'])
               for gid, region in regions.items()):
            continue
        compact = [region for region in regions.values() if np.ptp(region, axis=0).max() < .7]
        if compact and cmr.coverage(q, np.concatenate(compact), .006) > .25:
            continue
        local = [e['local_record'] for e in group['evidence'] if e['local_record'] is not None]
        visual = np.mean([r['feature'] for r in local], axis=0)
        if semantic.shape != (384,) or visual.shape != (256,):
            raise ValueError('Incompatible real cached features')
        gid = str(next_id)
        next_id += 1
        nodes[gid] = dict(id=gid, name=group['name'], text_embedding=semantic.tolist(),
            visual_embedding=visual.tolist(), mask_graph_evidence=evidence)
        observations = [dict(frame_id=e['frame_id'], local_id=e['local_record']['frame_instance_id'],
            confidence=e['confidence'], name=group['name'], mask_quality=e['local_record'].get('sam_quality_score'))
            for e in group['evidence'] if e['local_record'] is not None]
        tracks[gid] = dict(observations=observations, observed_frames=evidence['frames'],
            name_votes=dict(Counter(o['name'] for o in observations)), point_count=len(q),
            confidence=evidence['native_mean_confidence']*.7, confidence_type='uncalibrated_native_multiview_quality')
        replacements[gid] = regions[gid] = q
        audit['objects'].append(dict(instance_id=gid, name=group['name'], action='new_measured_instance', **evidence))
    if replacements:
        replace_regions(cloud_path, replacements, nodes)
        prior = Path(context.graph_geometry)
        if prior != cloud_path:
            replace_regions(prior, replacements, nodes)
        graph_path.write_text(json.dumps(graph, indent=2)+'\n')
        tracks_path.write_text(json.dumps(tracks, indent=2)+'\n')
        context.canonical_geometry_input = graph_path
        from .canonical_geometry import publish
        publish(context)
    (context.scene/'mask_graph_audit.json').write_text(json.dumps(audit, indent=2)+'\n')
    context.event('MIRA独立实测小物体归属完成', new_instances=len(replacements))

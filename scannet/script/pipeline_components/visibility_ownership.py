"""Measure independent surface ownership before accepting object hypotheses.

Only registered visible RGB-D samples vote. Occlusion and missing depth do not
count as background. Semantic labels and annotation are not read by this module.
"""
import numpy as np


def diagnose(scene, manifest, output):
    """Export observation evidence without changing graph or measured geometry."""
    import json
    from pathlib import Path
    import open3d as o3d
    from scipy.spatial import cKDTree
    from partaware.geometry import load_capture
    from .instance_consensus import Views
    from .proposal_validation import resolve, inclusion, raw_pair_evidence
    scene = Path(scene)
    graph = json.loads((scene/'topology_map.json').read_text())
    nodes = graph['object_nodes']['nodes']
    tracks = json.loads((scene/'validated_object_tracks.json').read_text())
    data, jobs, kd, kc, scale = load_capture(manifest)
    mapping = {}
    for gid, track in tracks.items():
        for observation in track['observations']:
            mapping.setdefault(observation['frame_id'], {})[observation['local_id']] = int(gid)
    records = {}
    for job in jobs:
        fid = job['frame_id']
        values = json.loads((scene/'refined_instance'/f'{fid}_instance.json').read_text())
        for record in values:
            record['instance_id'] = mapping.get(fid, {}).get(record['frame_instance_id'], -1)
        records[fid] = values
    views = Views(type('Context', (), {'scene': scene})(), jobs, kd, kc, scale, records)
    cloud = o3d.io.read_point_cloud(str(scene/graph['geometry_provenance']['source']))
    points = np.asarray(cloud.points)
    colors = np.rint(np.asarray(cloud.colors)*255).astype(int)
    ids = colors[:, 0]+255*colors[:, 1]+255**2*colors[:, 2]
    geometry = {gid: points[ids == int(gid)] for gid in nodes}
    result = {'ground_truth_used': False, 'objects': {}, 'contained_pairs': []}
    for gid, local in geometry.items():
        evidence = measure(local, gid, views, records, resolve, {})
        evidence.pop('frame_evidence')
        result['objects'][gid] = {'name': nodes[gid]['name'], **evidence}
    for small, a in geometry.items():
        for big, b in geometry.items():
            if small == big or len(a) > len(b) or inclusion(a, b) < .9:
                continue
            coverage = float(np.mean(cKDTree(b).query(a)[0] <= .03))
            pair = raw_pair_evidence(a[::max(1, len(a)//2048)], b[::max(1, len(b)//2048)],
                views, tracks[small]['observed_frames']+tracks[big]['observed_frames'])
            result['contained_pairs'].append({'small': small, 'big': big, 'inclusion': inclusion(a, b),
                'coverage': coverage, **pair})
    Path(output).write_text(json.dumps(result, indent=2)+'\n')


def measure(points, gid, views, records, resolve, remap, maximum=2048):
    sample = points[::max(1, len(points)//maximum)]
    seen = np.zeros(len(sample), int)
    positive = seen.copy()
    total, own, background = [np.zeros(len(sample)) for _ in range(3)]
    prototypes = [np.asarray(r['bert_embedding'], float) for values in records.values()
                  for r in values if resolve(str(r['instance_id']), remap) == gid and 'bert_embedding' in r]
    semantic = np.mean(prototypes, axis=0) if prototypes else None
    if semantic is not None: semantic /= max(np.linalg.norm(semantic), 1e-8)
    compatible_mass = np.zeros(len(sample))
    compatible_views = np.zeros(len(sample), int)
    other = {}
    frame_evidence = []
    for fid in sorted(views.jobs):
        indices, weights, labels = views.project(sample, fid)
        if len(indices) < 16 or weights.sum() <= 0:
            continue
        mapping = {int(r['frame_instance_id']): resolve(str(r['instance_id']), remap) for r in records[fid]}
        globals = np.array([mapping.get(int(k), '-1') for k in labels])
        member, bg = globals == gid, labels == 0
        compatible_ids = []
        if semantic is not None:
            for record in records[fid]:
                if 'bert_embedding' not in record:
                    continue
                vector = np.asarray(record['bert_embedding'], float)
                if float(vector @ semantic/max(np.linalg.norm(vector), 1e-8)) >= .8:
                    compatible_ids.append(int(record['frame_instance_id']))
        compatible = member | np.isin(labels, compatible_ids)
        compatible_mass[indices] += weights*compatible
        compatible_views[indices[compatible]] += 1
        seen[indices] += 1
        total[indices] += weights
        own[indices] += weights*member
        positive[indices[member]] += 1
        background[indices] += weights*bg
        for target in set(globals)-{gid, '-1'}:
            mass = other.setdefault(target, np.zeros(len(sample)))
            mass[indices] += weights*(globals == target)
        frame_evidence.append({'frame_id': fid, 'visible_points': len(indices),
            'own_fraction': float(weights[member].sum()/weights.sum()),
            'background_fraction': float(weights[bg].sum()/weights.sum())})
    eligible = seen >= 3
    ratios = own/np.maximum(total, 1e-8)
    bg_ratios = background/np.maximum(total, 1e-8)
    return {'sample_points': len(sample), 'visible_point_fraction': float(eligible.mean()),
        'stable_own_point_fraction': float(np.mean(eligible & (positive >= 2) & (ratios >= .6))),
        'contradicted_point_fraction': float(np.mean(eligible & (ratios < .2))),
        'majority_background_point_fraction': float(np.mean(eligible & (bg_ratios >= .6))),
        'weighted_own_fraction': float(own.sum()/max(total.sum(), 1e-8)),
        'weighted_background_fraction': float(background.sum()/max(total.sum(), 1e-8)),
        'compatible_class_point_fraction': float(np.mean(eligible & (compatible_views >= 2)
            & (compatible_mass/np.maximum(total, 1e-8) >= .6))),
        'compatible_class_weighted_fraction': float(compatible_mass.sum()/max(total.sum(), 1e-8)),
        'other_owners': {target: {'majority_point_fraction': float(np.mean(eligible & (mass/np.maximum(total, 1e-8) >= .6))),
            'weighted_fraction': float(mass.sum()/max(total.sum(), 1e-8))} for target, mass in other.items()},
        'frame_evidence': frame_evidence}


def contradicted(evidence):
    """Require repeated visible contradiction; lack of visibility is unknown."""
    return (len(evidence['frame_evidence']) >= 5
            and evidence['visible_point_fraction'] >= .6
            and evidence['stable_own_point_fraction'] < .05
            and evidence['contradicted_point_fraction'] >= .8
            and evidence['weighted_own_fraction'] < .2)


def low_platforms(views):
    """Localize previously measured floor levels with independent background pixels."""
    import json
    from pathlib import Path
    from scipy.spatial import cKDTree
    from partaware.geometry import project_mask
    context = getattr(views, 'context', None)
    if context is None: return []
    path = Path(context.scene)/'floor_filter.json'
    if not path.is_file(): return []
    floor = json.loads(path.read_text()).get('estimate')
    if not floor: return []
    samples = {height: [] for height in floor.get('levels_m', [floor['height_m']])}
    for fid in sorted(views.jobs):
        pose, depth, mask = views.get(fid)
        points, _ = project_mask(mask == 0, np.zeros((*mask.shape, 3), np.uint8), depth, pose,
            views.kd, views.kc, views.scale, stride=16)
        for height, patches in samples.items():
            local = points[np.abs(points[:, 2]-height) <= .01, :2]
            if len(local) >= 30: patches.append(local)
    return [(height, cKDTree(np.concatenate(patches))) for height, patches in samples.items() if len(patches) >= 3]


def platform_fraction(points, platforms):
    sample = points[::max(1, len(points)//2048)]
    member = np.zeros(len(sample), bool)
    for height, tree in platforms:
        # A low platform's riser is part of the measured floor boundary.
        member |= ((sample[:, 2] <= height+.01) & (sample[:, 2] >= height-.35)
                   & (tree.query(sample[:, :2])[0] <= .35))
    return float(member.mean())


def maximal_containers(candidates, geometry):
    """Retain maximal measured extents; incomparable objects remain ambiguous."""
    from .proposal_validation import inclusion
    volumes = {key: np.prod(np.maximum(np.ptp(geometry[key], axis=0), .01))
               for key, *_ in candidates}
    return [candidate for candidate in candidates if not any(
        volumes[other[0]] >= 2*volumes[candidate[0]]
        and inclusion(geometry[candidate[0]], geometry[other[0]]) >= .95
        for other in candidates if other[0] != candidate[0])]


def prune(survivors, geometry, tracks, metrics, views, records, remap, audit, nodes=None):
    """Remove contradicted identities and aggregates of independently resolved objects."""
    from scipy.spatial import cKDTree
    from .proposal_validation import resolve, inclusion, raw_pair_evidence
    ownership = {gid: measure(geometry[gid], gid, views, records, resolve, remap) for gid in survivors}
    platforms = low_platforms(views)
    rejected = {}
    for gid in survivors:
        metrics[gid]['visibility_ownership'] = ownership[gid]
        ownership[gid]['measured_low_platform_fraction'] = platform_fraction(geometry[gid], platforms)
        if (len(ownership[gid]['frame_evidence']) >= 5
                and ownership[gid]['stable_own_point_fraction'] < .2
                and tracks[gid].get('reprojection_support', 0) < .6
                and ownership[gid]['compatible_class_point_fraction'] < .3
                and ownership[gid]['measured_low_platform_fraction'] >= .7):
            rejected[gid] = '候选位于实测地板平台边界且可见归属反复受到否定'
    seeds = [gid for gid in survivors if gid not in rejected
             and tracks[gid].get('reprojection_support', 0) >= .6]
    # A weak mask that spans measured floor and several foreground objects
    # cannot enlarge an independently measured, coherent whole-object extent.
    if nodes is not None:
        for gid in survivors:
            if (gid in rejected or tracks[gid].get('reprojection_support', 0) >= .6
                    or metrics[gid]['mixed_views'] < 3
                    or ownership[gid]['measured_low_platform_fraction'] < .15):
                continue
            prototype = np.asarray(nodes[gid]['text_embedding'])
            compatible = []
            for target in seeds:
                if target == gid: continue
                vector = np.asarray(nodes[target]['text_embedding'])
                similarity = float(prototype @ vector/max(np.linalg.norm(prototype)*np.linalg.norm(vector), 1e-8))
                volume_ratio = (np.prod(np.maximum(np.ptp(geometry[gid], axis=0), .01))
                    / np.prod(np.maximum(np.ptp(geometry[target], axis=0), .01)))
                coverage = float(np.mean(cKDTree(geometry[gid]).query(geometry[target])[0] <= .03))
                if (similarity >= .6 and volume_ratio >= 3
                        and inclusion(geometry[target], geometry[gid]) >= .95 and coverage >= .85):
                    compatible.append((target, similarity, volume_ratio))
            maximal = maximal_containers(compatible, geometry)
            metrics[gid]['bounded_owner_candidates'] = {'compatible': compatible, 'maximal': maximal}
            if len(maximal) != 1: continue
            target, similarity, volume_ratio = maximal[0]
            inside = np.all((geometry[gid] >= geometry[target].min(0)) & (geometry[gid] <= geometry[target].max(0)), axis=1)
            measured = geometry[gid][inside]
            for other in seeds:
                if other == target: continue
                if other in {item[0] for item in compatible}: continue
                measured = measured[cKDTree(geometry[other]).query(measured)[0] > .03]
            from partaware.geometry import voxel_downsample
            geometry[target] = voxel_downsample(np.concatenate([geometry[target], measured]), .01)
            remap[gid] = target
            tracks[target]['observations'] += tracks[gid]['observations']
            tracks[target]['observed_frames'] = sorted(set(tracks[target]['observed_frames']+tracks[gid]['observed_frames']))
            rejected[gid] = '混入地板的弱候选归并到唯一受独立实测支持的完整物体'
            metrics[gid]['bounded_observation_owner'] = {'id': target, 'semantic_cosine': similarity,
                'volume_ratio': volume_ratio, 'transferred_measured_points': len(measured),
                'bounds_expanded': False, 'independent_foreground_excluded': True}
    for gid in survivors:
        if gid in rejected:
            continue
        evidence = ownership[gid]
        if evidence['weighted_own_fraction'] >= .5 or evidence['visible_point_fraction'] < .6:
            continue
        tree = cKDTree(geometry[gid])
        children = []
        volume = np.prod(np.maximum(np.ptp(geometry[gid], axis=0), .01))
        for child in seeds:
            if child == gid or np.prod(np.maximum(np.ptp(geometry[child], axis=0), .01)) >= volume:
                continue
            contained = inclusion(geometry[child], geometry[gid])
            coverage = float(np.mean(tree.query(geometry[child])[0] <= .03))
            if contained >= .7 and coverage >= .3 and contained*coverage >= .2:
                children.append((child, contained, coverage))
        independent = []
        for i, (left, _, _) in enumerate(children):
            for right, _, _ in children[i+1:]:
                pair = raw_pair_evidence(geometry[left][::max(1, len(geometry[left])//2048)],
                    geometry[right][::max(1, len(geometry[right])//2048)], views, views.jobs, minimum=.6)
                # A table supporting an object, or parts sharing the same mask,
                # is insufficient: two distinct measured identities must recur.
                a, b = geometry[left], geometry[right]
                separation = np.linalg.norm(np.maximum(np.maximum(a.min(0)-b.max(0), b.min(0)-a.max(0)), 0))
                distinct_geometry = (separation > .03 and len(tracks[left]['observed_frames']) >= 2
                                     and len(tracks[right]['observed_frames']) >= 2)
                if pair['independent_separation_views'] >= 2 or distinct_geometry:
                    independent.append({'left': left, 'right': right, **pair})
        covered = (float(np.mean(cKDTree(np.concatenate([geometry[c] for c, _, _ in children])).query(geometry[gid])[0] <= .03))
                   if children else 0.)
        metrics[gid]['resolved_subobjects'] = {'children': children, 'independent_pairs': independent,
                                             'covered_measured_surface_fraction': covered}
        if independent and covered >= .6:
            rejected[gid] = '候选重复包裹多个已独立识别的实测物体'
    # A contradicted patch can be a duplicate measurement of a fuller object.
    # This rule cannot absorb a bottle resting on a table: near-total measured
    # surface inclusion and repeated whole masks are both required.
    for small in survivors:
        evidence = ownership[small]
        sparse_claim = (nodes is not None and not tracks[small].get('fine_scale_instance', False)
            and len(evidence['frame_evidence']) >= 5 and evidence['visible_point_fraction'] >= .6
            and evidence['stable_own_point_fraction'] < .05 and evidence['weighted_own_fraction'] < .2)
        strict_contradiction = contradicted(evidence)
        if small in rejected or not (strict_contradiction or sparse_claim): continue
        containers = sorted(survivors, key=lambda key: np.prod(np.maximum(np.ptp(geometry[key], axis=0), .01)))
        for big in containers:
            if small == big or big in rejected: continue
            if inclusion(geometry[small], geometry[big]) < .95: continue
            coverage = float(np.mean(cKDTree(geometry[big]).query(geometry[small])[0] <= .03))
            if coverage < .85: continue
            region = np.all((geometry[big] >= geometry[small].min(0)-.03)
                            & (geometry[big] <= geometry[small].max(0)+.03), axis=1)
            matching = geometry[big][region]
            if len(matching) < 32: continue
            pair = raw_pair_evidence(geometry[small][::max(1, len(geometry[small])//2048)],
                matching[::max(1, len(matching)//2048)], views, views.jobs)
            partial_identity = False
            if nodes is not None and not tracks[small].get('fine_scale_instance', False):
                a = np.asarray(nodes[small]['text_embedding'])
                b = np.asarray(nodes[big]['text_embedding'])
                similarity = float(a @ b/max(np.linalg.norm(a)*np.linalg.norm(b), 1e-8))
                larger = (np.prod(np.maximum(np.ptp(geometry[big], axis=0), .01))
                          >= 2*np.prod(np.maximum(np.ptp(geometry[small], axis=0), .01)))
                repeated = raw_pair_evidence(geometry[small][::max(1, len(geometry[small])//2048)],
                    matching[::max(1, len(matching)//2048)], views, views.jobs, minimum=.6)
                partial_identity = (similarity >= .6 and larger
                    and tracks[big].get('reprojection_support', 0) >= .6
                    and repeated['whole_mask_support_views'] >= 2
                    and repeated['independent_separation_views'] <= 1)
                metrics[small].setdefault('partial_identity_candidates', []).append(
                    {'id': big, 'semantic_cosine': similarity, 'larger_measured_extent': bool(larger),
                     'accepted': bool(partial_identity), **repeated})
            if partial_identity or (strict_contradiction and pair['whole_mask_support_views'] >= 3
                                    and pair['independent_separation_views'] == 0):
                rejected[small] = '同一实测表面已由完整物体表示，不重复计作独立物体'
                metrics[small]['duplicate_surface_owner'] = {'id': big, **pair}
                if big not in rejected:
                    from partaware.geometry import voxel_downsample
                    measured = geometry[small]
                    if partial_identity:
                        measured = measured[np.all((measured >= geometry[big].min(0))
                            & (measured <= geometry[big].max(0)), axis=1)]
                        for other in seeds:
                            if other in (small, big) or other in rejected: continue
                            measured = measured[cKDTree(geometry[other]).query(measured)[0] > .03]
                    geometry[big] = voxel_downsample(np.concatenate([geometry[big], measured]), .01)
                    remap[small] = big
                    tracks[big]['observations'] += tracks[small]['observations']
                    tracks[big]['observed_frames'] = sorted(set(tracks[big]['observed_frames']+tracks[small]['observed_frames']))
                break
    for gid, reason in rejected.items():
        audit.append({'message': '可见归属竞争验收', 'instance_id': gid, 'accepted': False,
            'reasons': [reason], 'visibility_ownership': ownership[gid],
            'resolved_subobjects': metrics[gid].get('resolved_subobjects')})
    return [gid for gid in survivors if gid not in rejected]


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--processed-scene', required=True)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    diagnose(args.processed_scene, args.manifest, args.output)

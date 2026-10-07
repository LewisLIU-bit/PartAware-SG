"""Resolve surface duplicates and misassigned connected regions from cached views.

This project adapter follows multi-view mask consensus. It preserves coordinates,
uses no annotation or language prompt, and never creates an object from proximity.
Remove its registry entry to detach it without changing the public graph schema.
"""
from pathlib import Path
import copy, hashlib, json
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree
from partaware.geometry import load_capture
from .instance_consensus import Views
from .proposal_validation import inclusion, raw_pair_evidence, resolve
from .part_geometry import replace_regions


def duplicate(evidence, containment, coverage, semantic):
    """A container and its contents must fail the shared-surface/semantic gates."""
    return (containment >= .95 and coverage >= .8 and semantic >= .6
            and evidence['whole_mask_support_views'] >= 5
            and evidence['whole_mask_consensus'] >= .5
            and evidence['independent_separation_views'] == 0)


def transfer(evidence, containment, coverage, semantic, gap):
    """Move only detached measured regions with a unique existing owner."""
    return (gap >= .03 and containment >= .95 and coverage >= .65
            and semantic >= .6 and evidence['support_views'] >= 8
            and evidence['source_support_views'] <= .25*evidence['support_views']
            and evidence['owner_consensus'] >= .7
            and evidence['camera_baseline_m'] >= .08)


def owner_evidence(points, source, views, records, remap, active):
    votes, source_votes, centers = {}, 0, {}
    sample = points[::max(1, len(points)//1024)]
    for fid in sorted(views.jobs):
        _, weights, labels = views.project(sample, fid)
        if len(labels) < 16:
            continue
        mapping = {int(r['frame_instance_id']): resolve(str(r.get('instance_id', -1)), remap)
                   for r in records[fid]}
        globals = np.array([mapping.get(int(k), '-1') for k in labels])
        mass = {gid: float(weights[globals == gid].sum()/max(weights.sum(), 1e-9))
                for gid in set(globals) if gid in active}
        if not mass:
            continue
        gid = max(mass, key=mass.get)
        if mass[gid] < .8:
            continue
        if gid == source:
            source_votes += 1
        else:
            votes.setdefault(gid, []).append(fid)
            centers.setdefault(gid, []).append(views.get(fid)[0][:3, 3])
    evidence = {}
    total = source_votes+sum(map(len, votes.values()))
    for gid, frames in votes.items():
        xyz = np.asarray(centers[gid])
        evidence[gid] = {'support_views': len(frames), 'support_frame_ids': frames,
            'source_support_views': source_votes, 'owner_consensus': len(frames)/max(total, 1),
            'camera_baseline_m': float(np.max(np.linalg.norm(xyz-xyz[0], axis=1)))}
    return evidence


def construct(context):
    path = context.scene/'topology_map_cleaned.json'
    graph = json.loads(path.read_text()); nodes = graph['object_nodes']['nodes']
    tracks_path = context.scene/'validated_object_tracks.json'
    if not tracks_path.exists():
        return
    tracks = json.loads(tracks_path.read_text())
    data, jobs, kd, kc, scale = load_capture(context.manifest, context.image_dir, context.scene/'refined_instance')
    records = {j['frame_id']: json.loads((context.scene/'refined_instance'/f"{j['frame_id']}_updated_instance.json").read_text()) for j in jobs}
    views = Views(context, jobs, kd, kc, scale, records)
    cloud_path = context.scene/'instance_cloud_cleaned.ply'
    cloud = o3d.io.read_point_cloud(str(cloud_path)); points = np.asarray(cloud.points)
    colors = np.rint(np.asarray(cloud.colors)*255).astype(int)
    ids = colors[:, 0]+255*colors[:, 1]+255**2*colors[:, 2]
    geometry = {gid: points[ids == int(gid)] for gid in nodes}
    report = {'algorithm': 'cached_mask_supported_instance_granularity_v11',
        'ground_truth_used': False, 'qwen_api_calls': 0, 'generated_points': 0,
        'point_coordinates_preserved': True, 'merges': [], 'transfers': [],
        'component_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    keys = sorted(nodes, key=lambda gid: -len(geometry[gid]))
    trees = {gid: cKDTree(geometry[gid]) for gid in keys}
    eligible = {}
    for i, big in enumerate(keys):
        for small in keys[i+1:]:
            a, b = geometry[big], geometry[small]
            if len(b) < 128 or inclusion(b, a, .025) < .95:
                continue
            coverage = float(np.mean(trees[big].query(b)[0] <= .015))
            x, y = np.asarray(nodes[big]['text_embedding']), np.asarray(nodes[small]['text_embedding'])
            semantic = float(x@y/max(np.linalg.norm(x)*np.linalg.norm(y), 1e-9))
            if coverage < .8 or semantic < .6:
                continue
            evidence = raw_pair_evidence(a[::max(1, len(a)//2048)], b[::max(1, len(b)//2048)], views, list(views.jobs))
            containment = inclusion(b, a, .025)
            if duplicate(evidence, containment, coverage, semantic):
                eligible.setdefault(small, []).append((big, {'source_id': small, 'target_id': big,
                    'containment': containment, 'surface_coverage_15mm': coverage, 'semantic': semantic, **evidence}))
    # Ambiguous ownership is left unchanged. Size order makes this graph acyclic.
    remap = {small: values[0][0] for small, values in eligible.items() if len(values) == 1}
    original = copy.deepcopy(tracks)
    for small in sorted(remap, key=lambda gid: -len(geometry[gid])):
        big = resolve(small, remap)
        geometry[big] = np.concatenate([geometry[big], geometry[small]])
        tracks[big]['observations'] += original[small]['observations']
        tracks[big]['observed_frames'] = sorted(set(tracks[big]['observed_frames']+original[small]['observed_frames']))
        tracks[big].setdefault('granularity_source_ids', [big]).append(small)
        graph.setdefault('object_identity_aliases', {})[small] = {'canonical_id': big,
            'source_name': nodes[small]['name'], 'role': 'shared_surface_duplicate', 'evidence': eligible[small][0][1]}
        report['merges'].append({**eligible[small][0][1], 'canonical_id': big})
        del nodes[small]; del tracks[small]
    for fid, values in records.items():
        for r in values:
            r['instance_id'] = int(resolve(str(r.get('instance_id', -1)), remap))
    # Update pre-existing aliases even when they pointed to a newly merged node.
    for alias in graph.get('object_identity_aliases', {}).values():
        alias['canonical_id'] = resolve(str(alias['canonical_id']), remap)
    for source in list(nodes):
        region = geometry[source]
        if len(region) < 256 or max(np.ptp(region, axis=0)) > 1.:
            continue
        pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(region))
        labels = np.asarray(pc.cluster_dbscan(.02, 5, print_progress=False))
        stable = [k for k in np.unique(labels) if k >= 0 and (labels == k).sum() >= 64]
        if len(stable) < 2:
            continue
        # Weak islands cannot veto separation between two stable surfaces. Attach
        # only nearby orphan points; remote measurements keep their source owner.
        stable_indices = np.flatnonzero(np.isin(labels, stable))
        orphan_indices = np.flatnonzero(~np.isin(labels, stable))
        if len(orphan_indices):
            distance, nearest = cKDTree(region[stable_indices]).query(region[orphan_indices])
            close = distance <= .03
            labels[orphan_indices[close]] = labels[stable_indices[nearest[close]]]
        candidates = []
        for k in stable:
            component = region[labels == k]
            other = region[np.isin(labels, [other for other in stable if other != k])]
            if len(other) < 128:
                continue
            gap = float(cKDTree(other).query(component)[0].min())
            if gap < .03:
                continue
            separation = raw_pair_evidence(component, other[::max(1, len(other)//2048)], views, list(views.jobs))
            if separation['independent_separation_views'] < 3 or separation['whole_mask_support_views']:
                continue
            owners = owner_evidence(component, source, views, records, remap, nodes)
            accepted = []
            for target, evidence in owners.items():
                a = geometry[target]
                containment = inclusion(component, a, .025)
                coverage = float(np.mean(cKDTree(a).query(component)[0] <= .015))
                x, y = np.asarray(nodes[source]['text_embedding']), np.asarray(nodes[target]['text_embedding'])
                semantic = float(x@y/max(np.linalg.norm(x)*np.linalg.norm(y), 1e-9))
                if transfer(evidence, containment, coverage, semantic, gap):
                    accepted.append((target, {'source_id': source, 'target_id': target,
                        'points': len(component), 'gap_m': gap, 'containment': containment,
                        'surface_coverage_15mm': coverage, 'semantic': semantic,
                        'separation_evidence': separation, **evidence}))
            if len(accepted) == 1:
                candidates.append((k, *accepted[0]))
        if candidates:
            moved = np.isin(labels, [k for k, _, _ in candidates])
            if (~moved).sum() < 128:
                continue
            geometry[source] = region[~moved]
            for k, target, evidence in candidates:
                geometry[target] = np.concatenate([geometry[target], region[labels == k]])
                report['transfers'].append(evidence)
    if report['merges'] or report['transfers']:
        replace_regions(cloud_path, geometry, nodes)
        for gid in nodes:
            tracks[gid]['point_count'] = len(geometry[gid])
        tracks_path.write_text(json.dumps(tracks, indent=2)+'\n')
        for fid, values in records.items():
            (context.scene/'refined_instance'/f'{fid}_updated_instance.json').write_text(json.dumps(values, indent=2)+'\n')
        graph['object_granularity_policy'] = {'policy': 'whole_mask_shared_surface_and_unique_region_ownership',
            'ground_truth_used': False, 'audit': 'instance_granularity_audit.json'}
        path.write_text(json.dumps(graph, indent=2)+'\n')
    (context.scene/'instance_granularity_audit.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
    context.event('完整掩码与共享表面修正实例粒度', merged_duplicates=len(report['merges']),
                  transferred_regions=len(report['transfers']), point_coordinates_preserved=True)

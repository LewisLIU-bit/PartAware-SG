"""VISTA: Visible Instance Surface Tracking and Assignment.

Reconcile signed, measured native surfaces with existing identities. Spatial
extent covariance is distinct from uncertainty in its estimated mean (PSG).
This is a project adaptation, not the PSG optimizer or SupeRGB-D network.
No annotation, scene name, category whitelist, new VLM call or synthetic point.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import open3d as o3d
from scipy.ndimage import label
from scipy.spatial import cKDTree
from scipy.stats import chi2

from partaware.geometry import load_capture
from . import contact_instances as cmr
from .instance_consensus import Views
from .native_assembly import NativeMasks
from .part_geometry import replace_regions


def features(group):
    local = [e['local_record'] for e in group['evidence'] if e['local_record'] is not None]
    return np.mean([r['bert_embedding'] for r in local], axis=0) if local else None


def simultaneous_separation(left,right):
    for a in left['evidence']:
        if a['is_zoom_proposal']:continue
        for b in right['evidence']:
            if b['is_zoom_proposal'] or a['frame_id']!=b['frame_id'] or a['native_index']==b['native_index']:continue
            x,y=a['segmentation_box'],b['segmentation_box']
            intersection=max(0,min(x[2],y[2])-max(x[0],y[0]))*max(0,min(x[3],y[3])-max(x[1],y[1]))
            smaller=min((x[2]-x[0])*(x[3]-x[1]),(y[2]-y[0])*(y[3]-y[1]))
            if intersection/max(smaller,1)<.5:return True
    return False


def partial_agreement(left, right):
    """Shared measured surface can link a partial view to a larger view."""
    a, b = left['sample'], right['sample']
    gap = np.maximum(np.maximum(a.min(0)-b.max(0), b.min(0)-a.max(0)), 0)
    if np.linalg.norm(gap) > .006 or cmr.independent_entities(left, right) or simultaneous_separation(left, right):
        return False
    if left['name'] != right['name']:
        x, y = features(left), features(right)
        if x is None or y is None or cmr.semantic_agreement(x, y) < .8:
            return False
    x, y = cmr.coverage(a, b), cmr.coverage(b, a)
    return max(x, y) >= .9 and min(x, y) >= .15


def reconcile_groups(groups):
    result = []
    for group in sorted(groups, key=lambda g: (-len(g['frames']), -g['seed_size'])):
        match = next((old for old in result if partial_agreement(group, old)), None)
        if match is None:
            result.append(group)
            continue
        match['points'].extend(group['points'])
        match['evidence'].extend(group['evidence'])
        match['frames'].update(group['frames'])
        if group['seed_size'] > match['seed_size']:
            match['sample'], match['seed_size'] = group['sample'], group['seed_size']
    return result


def measured_proposal(group, views, native):
    """Test all measured coordinates, even if voxel consensus already passes."""
    frames = sorted(group['frames'])
    if len(frames) < 4:
        return None
    q, _ = cmr.projected_support(cmr.sample_cells(np.concatenate(group['points']), .002), group, views, native)
    centers = np.array([e['camera_center'] for e in group['evidence']])
    confidence = float(np.mean([e['confidence'] for e in group['evidence']]))
    local = [e['local_record'] for e in group['evidence'] if e['local_record'] is not None]
    baseline = float(np.linalg.norm(centers.max(0)-centers.min(0)))
    if len(q) < 64 or np.ptp(q, axis=0).max() > .65 or confidence < .75 or len(local) < 3 or baseline < .1:
        return None
    per_view = [np.concatenate([p for p, e in zip(group['points'], group['evidence'])
                               if e['frame_id'] == fid]).mean(0) for fid in frames]
    uncertainty = np.cov(np.array(per_view).T)/len(frames)
    extent = np.cov(q.T)
    return q, dict(frames=frames, measured_points=len(q), generated_points=0,
        independent_point_views=3, native_mean_confidence=confidence, camera_baseline_m=baseline,
        position_mean=q.mean(0).tolist(), physical_extent_covariance=extent.tolist(),
        mean_estimation_covariance=uncertainty.tolist(),
        uncertainty_model='empirical_independent_view_centroids_not_NIW_reproduction')


def owner_candidates(q, semantic, nodes, regions):
    candidates = []
    for gid, region in regions.items():
        if (nodes[gid].get('atomic_instance_evidence') or nodes[gid].get('geometry_hypothesis')
                or len(region) < 48 or np.ptp(region, axis=0).max() > .7):
            continue
        agreement = cmr.semantic_agreement(semantic, nodes[gid]['text_embedding'])
        if agreement < .4:
            continue
        a, b = cmr.coverage(q, region, .015), cmr.coverage(region, q, .015)
        if a >= .8 and b >= .3:
            candidates.append((a*b, gid, a, b, agreement))
    candidates.sort(reverse=True)
    if candidates and (len(candidates) == 1 or candidates[0][0] >= 1.5*candidates[1][0]):
        return candidates[0]
    return None


def recover_surface(prior, measured, explained, uncertainty):
    """Preserve broad partial bodies; reject only supported compact tails."""
    eigen = np.linalg.eigvalsh(np.cov(prior.T))
    anisotropic = eigen[-1]/max(eigen[0], 1e-12) >= 16
    removed = np.zeros(len(prior), bool)
    reason = 'insufficient_complete_surface_evidence'
    if anisotropic and len(measured) >= .5*len(prior):
        # Extent plus mean uncertainty defines an acceptance envelope, not a
        # claim that the Gaussian density is the surface of the physical body.
        covariance = np.cov(measured.T)+np.asarray(uncertainty)+np.eye(3)*.006**2
        delta = prior-measured.mean(0)
        distance = np.einsum('ni,ij,nj->n', delta, np.linalg.inv(covariance), delta)
        removed = (distance > chi2.ppf(.997, 3)) & (cKDTree(measured).query(prior)[0] > .008)
        reason = 'anisotropic_measured_extent_envelope'
    elif explained >= .9 and len(measured) >= .5*len(prior):
        combined = np.concatenate([prior, measured])
        cells = np.floor((combined-combined.min(0))/.012).astype(int)
        shape = cells.max(0)+1
        if np.prod(shape) > 2500000:
            return None, 0, reason
        occupied = np.zeros(shape, bool)
        occupied[tuple(cells.T)] = True
        labels, _ = label(occupied, structure=np.ones((3, 3, 3)))
        components = labels[tuple(cells.T)]
        anchors = set(components[len(prior):])
        counts = np.bincount(components[:len(prior)])
        removed = np.array([int(k) not in anchors and counts[k] < 48 for k in components[:len(prior)]])
        reason = 'detached_low_support_patch_rejection'
    else:
        return None, 0, reason
    return cmr.sample_cells(np.concatenate([prior[~removed], measured]), .002), int(removed.sum()), reason


def construct(context):
    data, jobs, kd, kc, scale = load_capture(context.manifest, context.image_dir, context.scene/'refined_instance')
    audit = dict(algorithm='VISTA_v16', ground_truth_used=False, vision_api_calls=0,
                 generated_points=0, component_sha256=component_sha256(), objects=[])
    if data.get('world_frame') != 'hypersim_world_z_up':
        audit['reason'] = '未声明可靠竖直轴，保留已有物体'
        (context.scene/'visible_instance_audit.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2)+'\n')
        return
    cache = getattr(context, 'native_contact_observations', None)
    if cache is None:
        views = Views(context, jobs, kd, kc, scale, {})
        native = NativeMasks(views)
        if not native.available:
            return
        groups = cmr.observations(context, jobs, kd, kc, scale, native)
    else:
        groups, views, native = cache
    audit['source_groups'] = len(groups)
    groups = reconcile_groups(groups)
    audit['reconciled_groups'] = len(groups)
    graph_path = context.scene/'topology_map.json'
    graph = json.loads(graph_path.read_text())
    nodes = graph['object_nodes']['nodes']
    cloud_path = context.scene/'instance_cloud_cleaned.ply'
    cloud = o3d.io.read_point_cloud(str(cloud_path))
    points = np.asarray(cloud.points)
    colors = np.rint(np.asarray(cloud.colors)*255).astype(int)
    ids = colors[:, 0]+255*colors[:, 1]+255**2*colors[:, 2]
    regions = {gid: points[ids == int(gid)] for gid in nodes}
    choices, proposals = {}, []
    for group in groups:
        proposal = measured_proposal(group, views, native)
        if proposal is None:
            continue
        q, evidence = proposal
        semantic = features(group)
        owner = owner_candidates(q, semantic, nodes, regions)
        proposals.append((group, q, evidence, owner))
        if owner is None:
            continue
        score, gid, a, b, agreement = owner
        own_distance=cKDTree(regions[gid]).query(q)[0]
        keep=np.ones(len(q),bool)
        for other,region in regions.items():
            if other==gid or len(region)<48 or np.ptp(region,axis=0).max()>.7:continue
            distance=cKDTree(region).query(q)[0]
            keep &= ~((distance<=.004)&(distance+.001<own_distance))
        q=q[keep]
        if len(q)<64:continue
        evidence['measured_points']=len(q)
        quality = (len(group['frames']), evidence['native_mean_confidence'])
        if gid not in choices or quality > choices[gid][0]:
            choices[gid] = (quality, group, q, evidence, owner)
    replacements = {}
    tracks_path = context.scene/'validated_object_tracks.json'
    tracks = json.loads(tracks_path.read_text())
    for gid, (_, group, q, evidence, owner) in choices.items():
        recovered, removed, reason = recover_surface(regions[gid], q, owner[3], evidence['mean_estimation_covariance'])
        if recovered is None:
            continue
        replacements[gid] = recovered
        nodes[gid]['visible_instance_evidence'] = dict(measured=True, **evidence)
        tracks[gid]['point_count'] = len(recovered)
        audit['objects'].append(dict(instance_id=gid, name=nodes[gid]['name'], native_name=group['name'],
            action=reason, semantic_agreement=owner[4], proposal_owned_fraction=owner[2],
            original_explained_fraction=owner[3], removed_points=removed,
            prior_points=len(regions[gid]), final_points=len(recovered), **evidence))
    # A known body owns its measured surface even when a neighboring root
    # accidentally contains a duplicate front-row patch. Preserve the unknown
    # remainder, which can be a separate object in the rear row.
    for gid, (_, group, q, evidence, _) in choices.items():
        if gid not in replacements:
            continue
        semantic = features(group)
        for other, prior in regions.items():
            if other == gid or nodes[other].get('atomic_instance_evidence') or nodes[other].get('geometry_hypothesis'):
                continue
            if len(prior) < 64 or np.ptp(prior, axis=0).max() > .7:
                continue
            if cmr.semantic_agreement(semantic, nodes[other]['text_embedding']) < .8:
                continue
            current = replacements.get(other, prior)
            shared = cKDTree(q).query(current)[0] <= .004
            if shared.mean() < .15 or shared.mean() > .8 or (~shared).sum() < 64:
                continue
            replacements[other] = current[~shared]
            tracks[other]['point_count'] = int((~shared).sum())
            audit['objects'].append(dict(instance_id=other, action='唯一原生表面归属，保留后排未知表面',
                unique_owner=gid, removed_duplicate_points=int(shared.sum()), generated_points=0))
    audit['accepted_proposals'] = len(proposals)
    audit['ambiguous_or_unowned_proposals'] = sum(owner is None for _, _, _, owner in proposals)
    if replacements:
        prior = Path(context.graph_geometry)
        if not prior.is_file():
            prior = cloud_path
        replace_regions(cloud_path, replacements, nodes)
        if prior != cloud_path:
            replace_regions(prior, replacements, nodes)
        graph_path.write_text(json.dumps(graph, indent=2)+'\n')
        tracks_path.write_text(json.dumps(tracks, indent=2)+'\n')
        context.graph_geometry, context.canonical_geometry_input = prior, graph_path
        from .canonical_geometry import publish
        publish(context)
    (context.scene/'visible_instance_audit.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2)+'\n')
    context.event('可见实例实测表面与前后排唯一归属完成', updated_objects=len(replacements), generated_points=0)


def component_sha256():
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()

"""Recover compact measured instances from signed, nonexclusive RGB-D masks.

This is a project adaptation of 2D-guided 3D proposals and view consensus,
not the Open3DIS learned superpoint network. All geometry is observed depth.
No annotations, instance identifiers, target counts or new VLM calls are used.
Detach this FINAL_GEOMETRY entry to retain the v14 construction behavior.
"""
import hashlib
import json
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
from scipy.ndimage import label
from scipy.spatial import cKDTree

from partaware.geometry import load_capture
from .fine_boundary import supported_samples
from .hierarchical_masks import normalize
from .instance_consensus import Views
from .native_assembly import NativeMasks
from .part_geometry import replace_regions
from .sam3_frontend import box_iou
from .visual_part_anchoring import acceptance as part_acceptance


def sample_cells(points, voxel=.003):
    _, first = np.unique(np.floor(points / voxel).astype(np.int64), axis=0, return_index=True)
    return points[np.sort(first)]


def compact_component(points):
    """Reject detached raster leakage without moving an observed coordinate."""
    if len(points) < 48 or np.ptp(points, axis=0).max() > 1.5:
        return None
    cells = np.floor((points - points.min(0)) / .012).astype(int)
    shape = cells.max(0) + 1
    if np.prod(shape) > 2500000:
        return None
    occupied = np.zeros(shape, bool)
    occupied[tuple(cells.T)] = True
    components, _ = label(occupied, structure=np.ones((3, 3, 3)))
    values = components[tuple(cells.T)]
    winner = np.argmax(np.bincount(values)[1:]) + 1
    selected = points[values == winner]
    if len(selected) < max(48, .5 * len(points)) or np.ptp(selected, axis=0).max() > .65:
        return None
    return sample_cells(selected)


def coverage(a, b, radius=.006):
    return float(np.mean(cKDTree(b).query(a)[0] <= radius)) if len(a) and len(b) else 0.


def semantic_agreement(left, right):
    a, b = np.asarray(left), np.asarray(right)
    return float(a @ b / max(np.linalg.norm(a)*np.linalg.norm(b), 1e-12))


def independent_entities(left, right):
    """Only distinct complete-image detections can establish multiplicity."""
    frames=set()
    for a in left['evidence']:
        if a['is_zoom_proposal']:continue
        for b in right['evidence']:
            if b['is_zoom_proposal'] or a['frame_id']!=b['frame_id']:continue
            if a['native_index']!=b['native_index'] and box_iou(a['segmentation_box'],b['segmentation_box'])<.25:
                frames.add(a['frame_id'])
    return len(frames)>=3


def projected_support(points, group, views, native):
    """Count independent visible-mask votes, not repeated sampling of a voxel.

    Thin curved surfaces need not land in the same 3 mm cell in three views.
    Every retained coordinate remains an actual depth sample. Occluded points
    and missing detections are unknown, rather than negative observations.
    """
    votes = np.zeros(len(points), int)
    for fid in sorted(group['frames']):
        indices, weights, _ = views.project(points, fid)
        indices = indices[weights >= .7]
        if not len(indices):
            continue
        pose = views.get(fid)[0]
        camera = (points[indices]-pose[:3, 3]) @ pose[:3, :3]
        uv = np.rint(camera[:, :2]/camera[:, 2, None]*[views.kc[0, 0], views.kc[1, 1]]
                     +[views.kc[0, 2], views.kc[1, 2]]).astype(int)
        native.get(fid)
        with np.load(native.source/'sam3_cache'/f'{fid}.npz', allow_pickle=False) as saved:
            if str(saved['signature']) != native.expected[fid]['signature']:
                raise ValueError('Native reprojection signature changed')
            masks = saved['packed_masks']
            local = sorted({e['native_index'] for e in group['evidence'] if e['frame_id']==fid})
            hit = np.any((masks[np.asarray(local)[:,None], uv[None,:,1], uv[None,:,0]//8]
                          >> (7-uv[None,:,0]%8)) & 1, axis=0)
        votes[indices[hit]] += 1
    return points[votes >= 3], votes


def association(a, b, same_frame=False):
    """Contact alone cannot identify an object; require shared measured surface."""
    gap = np.maximum(np.maximum(a.min(0) - b.max(0), b.min(0) - a.max(0)), 0)
    if np.linalg.norm(gap) > .006:
        return 0.
    left, right = coverage(a, b), coverage(b, a)
    if same_frame:
        return min(left, right) if min(left, right) >= .88 else 0.
    return (left + right) / 2 if min(left, right) >= .35 and max(left, right) >= .6 else 0.


def frame_samples(job, kd, kc, scale, stride=2):
    depth = cv2.imread(str(job['depth']), cv2.IMREAD_UNCHANGED)
    rgb = cv2.imread(str(job['rgb']))
    pose = np.loadtxt(job['pose'])
    v, u = np.mgrid[0:depth.shape[0]:stride, 0:depth.shape[1]:stride]
    z = depth[v, u].astype(float) / scale
    ur = (u - kd[0, 2]) * kc[0, 0] / kd[0, 0] + kc[0, 2]
    vr = (v - kd[1, 2]) * kc[1, 1] / kd[1, 1] + kc[1, 2]
    valid = (z > 0) & (ur >= 0) & (vr >= 0) & (ur < rgb.shape[1]) & (vr < rgb.shape[0])
    camera = np.column_stack(((u[valid] - kd[0, 2]) * z[valid] / kd[0, 0],
                              (v[valid] - kd[1, 2]) * z[valid] / kd[1, 1], z[valid]))
    return camera @ pose[:3, :3].T + pose[:3, 3], ur[valid].astype(int), vr[valid].astype(int), pose


def observations(context, jobs, kd, kc, scale, native):
    """Use complete masks and zoom masks before public exclusive-label loss."""
    groups = []
    public_cache = {j['frame_id']: json.loads((context.scene/'refined_instance'/f"{j['frame_id']}_instance.json").read_text()) for j in jobs}
    semantic = {}
    for records in public_cache.values():
        for record in records:
            semantic.setdefault(normalize(record['object_name']), record['bert_embedding'])
    for job in jobs:
        fid = job['frame_id']
        native.get(fid)
        q, u, v, pose = frame_samples(job, kd, kc, scale)
        public = public_cache[fid]
        with np.load(native.source / 'sam3_cache' / f'{fid}.npz', allow_pickle=False) as saved:
            if str(saved['signature']) != native.expected[fid]['signature']:
                raise ValueError('Native compact-mask signature changed')
            records = json.loads(str(saved['records']))
            packed = saved['packed_masks']
            for index, record in enumerate(records):
                if record['confidence'] < .7:
                    continue
                mask = ((packed[index, v, u // 8] >> (7 - u % 8)) & 1).astype(bool)
                points = compact_component(q[mask])
                if points is None:
                    continue
                name = normalize(record['object_name'])
                best = max(((box_iou(record['segmentation_box'], r['segmentation_box']), r)
                            for r in public if normalize(r['object_name']) == name or
                            (name in semantic and semantic_agreement(semantic[name], r['bert_embedding']) >= .8)),
                           key=lambda value: value[0], default=(0., None))
                evidence = dict(frame_id=fid, native_index=index, name=name,
                    confidence=float(record['confidence']), camera_center=pose[:3, 3].tolist(),
                    is_zoom_proposal=record['is_zoom_proposal'], segmentation_box=record['segmentation_box'],
                    local_record=best[1] if best[0] >= .35 else None)
                candidates = []
                sample = points[::max(1, len(points) // 1600)]
                for number, group in enumerate(groups):
                    if group['name'] != name and not (name in semantic and group['name'] in semantic
                            and semantic_agreement(semantic[name],semantic[group['name']]) >= .8):
                        continue
                    same = fid in group['frames']
                    score = association(sample, group['sample'], same)
                    if score:
                        candidates.append((score, number))
                if candidates:
                    _, winner = max(candidates)
                    group = groups[winner]
                    group['points'].append(points)
                    group['evidence'].append(evidence)
                    group['frames'].add(fid)
                    if len(points) > group['seed_size']:
                        group['sample'] = sample
                        group['seed_size'] = len(points)
                else:
                    groups.append(dict(name=name, sample=sample, seed_size=len(points), points=[points], evidence=[evidence], frames={fid}))
        if jobs.index(job) % 10 == 0:
            context.event('原生接触物体候选跨视角关联', frame=fid, candidate_groups=len(groups))
    return groups


def consensus(group, views=None, native=None):
    frames = sorted(group['frames'])
    if len(frames) < 4:
        return None, dict(accepted=False, reason='少于四个独立观测帧')
    mapping = {fid: index for index, fid in enumerate(frames)}
    points = np.concatenate(group['points'])
    view_ids = np.concatenate([np.full(len(p), mapping[e['frame_id']], int)
                               for p, e in zip(group['points'], group['evidence'])])
    selected, votes = supported_samples(points, view_ids, voxel=.003, minimum_views=3)
    vote_type = 'independent_observed_voxels'
    if len(selected) < 64 and views is not None:
        selected, votes = projected_support(sample_cells(points,.002), group, views, native)
        vote_type = 'independent_depth_visible_native_mask_reprojections'
    centers = np.array([e['camera_center'] for e in group['evidence']])
    baseline = float(np.linalg.norm(centers.max(0) - centers.min(0)))
    confidence = float(np.mean([e['confidence'] for e in group['evidence']]))
    local = [e['local_record'] for e in group['evidence'] if e['local_record'] is not None]
    evidence = dict(accepted=False, frames=frames, camera_baseline_m=baseline,
        native_mean_confidence=confidence, measured_points=len(selected), feature_observations=len(local),
        minimum_point_views=3, generated_points=0)
    evidence['point_vote_type'] = vote_type
    if len(selected) < 64 or baseline < .1 or confidence < .75 or len(local) < 3:
        evidence['reason'] = '实测表面、视角基线或真实缓存特征不足'
        return None, evidence
    if np.ptp(selected, axis=0).max() > .65:
        evidence['reason'] = '跨视角候选超出紧凑物体范围'
        return None, evidence
    evidence.update(accepted=True, reason='原生非互斥掩码与三视角实测共识')
    return sample_cells(selected, .002), evidence


def construct(context):
    data, jobs, kd, kc, scale = load_capture(context.manifest, context.image_dir, context.scene / 'refined_instance')
    report = dict(algorithm='CMR_native_contact_instance_recovery_v15', ground_truth_used=False,
        vision_api_calls=0, generated_points=0, component_sha256=component_sha256(), objects=[], candidates=[])
    if data.get('world_frame') != 'hypersim_world_z_up':
        report['reason'] = '未声明可靠竖直轴，保留原实例'
        (context.scene / 'contact_instance_audit.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
        return
    views = Views(context, jobs, kd, kc, scale, {})
    native = NativeMasks(views)
    if not native.available:
        return
    graph_path = context.scene / 'topology_map.json'
    graph = json.loads(graph_path.read_text())
    nodes = graph['object_nodes']['nodes']
    cloud_path = context.scene / 'instance_cloud_cleaned.ply'
    cloud = o3d.io.read_point_cloud(str(cloud_path))
    points = np.asarray(cloud.points)
    colors = np.rint(np.asarray(cloud.colors) * 255).astype(int)
    ids = colors[:, 0] + 255 * colors[:, 1] + 255**2 * colors[:, 2]
    geometry = {gid: points[ids == int(gid)] for gid in nodes}
    tracks_path = context.scene / 'validated_object_tracks.json'
    tracks = json.loads(tracks_path.read_text())
    groups = observations(context, jobs, kd, kc, scale, native)
    context.native_contact_observations = (groups, views, native)
    proposals = []
    for group in groups:
        recovered, evidence = consensus(group, views, native)
        report['candidates'].append(dict(name=group['name'], **evidence))
        if recovered is not None:
            proposals.append((len(group['frames']), group, recovered, evidence))
    replacements = {}
    updated = set()
    next_id = max(map(int, nodes)) + 1
    # A whole root is split only by multiple stable, simultaneously independent
    # native entities, never by one incomplete mask of an occluded curved body.
    consumed=set()
    for gid,region in list(geometry.items()):
        if nodes[gid].get('atomic_instance_evidence') or nodes[gid].get('geometry_hypothesis') or np.ptp(region,axis=0).max()>.7:
            continue
        candidates=[(index,g,q,e) for index,(_,g,q,e) in enumerate(proposals)
            if len(g['frames'])>=4 and semantic_agreement(nodes[gid]['text_embedding'],
                np.mean([r['local_record']['bert_embedding'] for r in g['evidence'] if r['local_record'] is not None],axis=0)) >= .8
            and coverage(q,region,.02)>=.65 and coverage(region,q,.015)<.8]
        chosen=[]
        for item in sorted(candidates,key=lambda item:-len(item[1]['frames'])):
            _,group,recovered,evidence=item
            compatible=True
            for _,old_group,old_points,_ in chosen:
                if max(coverage(recovered,old_points),coverage(old_points,recovered))>.1:
                    compatible=False;break
                if not independent_entities(group,old_group):compatible=False;break
            if compatible:chosen.append(item)
        if len(chosen)<2 or coverage(region,np.concatenate([item[2] for item in chosen]),.015)<.8:
            continue
        for number,(index,group,recovered,evidence) in enumerate(chosen):
            child=gid if number==0 else str(next_id)
            if number:next_id+=1
            local=[e['local_record'] for e in group['evidence'] if e['local_record'] is not None]
            nodes[child]={**nodes[gid],'id':child,
                'text_embedding':np.mean([r['bert_embedding'] for r in local],axis=0).tolist(),
                'visual_embedding':np.mean([r['feature'] for r in local],axis=0).tolist(),
                'contact_instance_recovery':dict(measured=True,source_root=gid,multiview_entity_split=True,supporting_frames=evidence['frames'])}
            observations_local=[dict(frame_id=e['frame_id'],local_id=e['local_record']['frame_instance_id'],
                confidence=e['confidence'],name=group['name'],mask_quality=e['local_record'].get('sam_quality_score'))
                for e in group['evidence'] if e['local_record'] is not None]
            tracks[child]={**tracks[gid],'observations':observations_local,'observed_frames':evidence['frames'],
                'name_votes':dict(Counter(o['name'] for o in observations_local)),'point_count':len(recovered)}
            geometry[child]=replacements[child]=recovered;updated.add(child);consumed.add(index)
            report['objects'].append(dict(instance_id=child,source_root=gid,name=group['name'],
                action='独立同帧掩码与实测联合覆盖拆分混合根实例',**evidence))
    for index,(_, group, recovered, evidence) in sorted(enumerate(proposals),key=lambda item:-item[1][0]):
        if index in consumed:continue
        # A repeated-rim model already owns its independent measured instances.
        protected = [gid for gid, node in nodes.items() if node.get('atomic_instance_evidence')]
        if any(coverage(recovered, geometry[gid], .01) >= .3 for gid in protected):
            continue
        choices = []
        for gid, region in geometry.items():
            if np.ptp(region, axis=0).max() > .7 or normalize(nodes[gid]['name']) != group['name']:
                continue
            explained = coverage(region, recovered, .01)
            proposal_owned = coverage(recovered, region, .01)
            if explained >= .8 or (explained >= .15 and proposal_owned >= .65):
                choices.append((explained, gid))
        if len(choices) > 1:
            continue
        if choices:
            _, gid = choices[0]
            region = geometry[gid]
            if gid in updated:
                continue
            # Add genuine boundaries while retaining every original trusted point.
            added = recovered[cKDTree(region).query(recovered)[0] > .001]
            mixed = coverage(region, recovered, .01) < .8
            if mixed:
                eigenvalues, axes = np.linalg.eigh(np.cov(region.T))
                # Occluded curved bodies are not incomplete flat support masks.
                # Do not shrink cups, appliances, stored bodies or completed parts.
                if (abs(axes[2,0]) < .97 or eigenvalues[0]/max(eigenvalues[1],1e-12) > .025
                        or np.ptp(region[:,2]) > .07 or nodes[gid].get('geometry_hypothesis')):
                    continue
            # A repeated native mask can identify one object inside a mixed root.
            # Keep only repeatedly measured native coordinates for that root.
            if len(added) < .25 * len(region) and not mixed:
                continue
            combined = sample_cells(recovered,.002) if mixed else np.concatenate([region,sample_cells(added,.002)])
            if np.max(np.ptp(combined, axis=0) - np.ptp(region, axis=0)) > .12:
                continue
            replacements[gid] = combined
            updated.add(gid)
            geometry[gid] = combined
            tracks[gid]['point_count'] = len(combined)
            nodes[gid]['contact_instance_recovery'] = dict(measured=True, supporting_frames=evidence['frames'])
            report['objects'].append(dict(instance_id=gid, action='原生多视角掩码纠正混合表面归属' if mixed else '密集恢复独立实测表面',
                name=group['name'], added_points=len(added), removed_points=len(region)-len(combined) if mixed else 0, **evidence))
            continue
        # Fragmentary native views cannot establish a new root identity.
        if len(group['frames']) < 6 or evidence['native_mean_confidence'] < .75:
            continue
        part_owners={}
        for part_id,part in graph.get('part_nodes',{}).items():
            body=str(part.get('parent_id'))
            point_file=context.scene/'parts'/f'{part_id}.points.npy'
            if body not in nodes or not point_file.is_file():continue
            proof=part_acceptance(part,recovered,geometry[body],group['name'],nodes[body]['name'],group['frames'])
            if proof is None:continue
            part_points=np.load(point_file,allow_pickle=False)
            if coverage(recovered,part_points,.015)>=.8:
                part_owners.setdefault(body,[]).append((part_id,part_points,proof))
        if len(part_owners)==1:
            body,parts=next(iter(part_owners.items()))
            added=recovered[cKDTree(geometry[body]).query(recovered)[0]>.001]
            geometry[body]=replacements[body]=np.concatenate([geometry[body],sample_cells(added,.002)])
            tracks[body]['point_count']=len(geometry[body])
            for part_id,part_points,proof in parts:
                part_added=recovered[cKDTree(part_points).query(recovered)[0]>.001]
                merged=np.concatenate([part_points,sample_cells(part_added,.002)])
                np.save(context.scene/'parts'/f'{part_id}.points.npy',merged)
                graph['part_nodes'][part_id].update(point_count=len(merged),native_contact_evidence=proof)
                report['objects'].append(dict(instance_id=body,part_id=part_id,name=group['name'],
                    action='原生实例确认为已验证部件，补入父体并避免重复根节点',**proof,**evidence))
            continue
        # A small same-identity patch of an existing body is not another root.
        parent_owned = False
        local = [e['local_record'] for e in group['evidence'] if e['local_record'] is not None]
        semantic = np.mean([r['bert_embedding'] for r in local], axis=0)
        for gid, region in geometry.items():
            related = semantic_agreement(nodes[gid]['text_embedding'],semantic) >= .4
            # A locally named subassembly cannot duplicate an already owned body.
            # The wider distance accounts for the historical coarser cloud,
            # while semantic evidence keeps separate objects on supports intact.
            if related and coverage(recovered,region,.02) >= .6:
                parent_owned = True
                break
            if normalize(nodes[gid]['name']) != group['name']:
                continue
            if coverage(recovered, region, .01) > .6:
                parent_owned = True
                break
            low, high = region.min(0)-.015, region.max(0)+.015
            if (np.prod(np.maximum(np.ptp(region,axis=0),.001)) > 4*np.prod(np.maximum(np.ptp(recovered,axis=0),.001))
                    and np.mean(np.all((recovered>=low)&(recovered<=high),axis=1))>.98):
                parent_owned = True
                break
        if parent_owned:
            continue
        # Never publish another copy of an already measured compact identity.
        compact = [q for q in geometry.values() if np.ptp(q, axis=0).max() < .7]
        if compact and coverage(recovered, np.concatenate(compact), .006) > .25:
            continue
        local = [e['local_record'] for e in group['evidence'] if e['local_record'] is not None]
        semantic = np.mean([r['bert_embedding'] for r in local], axis=0)
        visual = np.mean([r['feature'] for r in local], axis=0)
        if semantic.shape != (384,) or visual.shape != (256,):
            raise ValueError('Recovered proposal has incompatible genuine cached features')
        gid = str(next_id); next_id += 1
        node = dict(id=gid, name=group['name'], text_embedding=semantic.tolist(), visual_embedding=visual.tolist(),
                    contact_instance_recovery=dict(measured=True, supporting_frames=evidence['frames']))
        nodes[gid] = node
        geometry[gid] = replacements[gid] = recovered
        observation_records = [dict(frame_id=e['frame_id'], local_id=e['local_record']['frame_instance_id'],
            confidence=e['confidence'], name=group['name'], mask_quality=e['local_record'].get('sam_quality_score'))
            for e in group['evidence'] if e['local_record'] is not None]
        tracks[gid] = dict(observations=observation_records, observed_frames=evidence['frames'],
            name_votes=dict(Counter(o['name'] for o in observation_records)), point_count=len(recovered),
            confidence=evidence['native_mean_confidence'] * .7, confidence_type='uncalibrated_native_multiview_quality',
            contact_instance_recovery=True)
        report['objects'].append(dict(instance_id=gid, name=group['name'], action='恢复被互斥归属遗漏的独立物体', **evidence))
    if replacements:
        prior = Path(getattr(context, 'graph_geometry', cloud_path))
        if not prior.is_file():prior=cloud_path
        replace_regions(cloud_path, replacements, nodes)
        if prior != cloud_path:
            replace_regions(prior, replacements, nodes)
        tracks_path.write_text(json.dumps(tracks, indent=2)+'\n')
        graph_path.write_text(json.dumps(graph, indent=2)+'\n')
        context.graph_geometry = prior
        context.canonical_geometry_input = graph_path
        from .canonical_geometry import publish
        publish(context)
    (context.scene / 'contact_instance_audit.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
    context.event('非互斥原生掩码与多视角深度恢复接触小物体', updated_objects=len(replacements), generated_points=0)


def component_sha256():
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()

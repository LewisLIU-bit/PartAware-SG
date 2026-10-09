"""Retain observed subinstances under a unique measured whole-body anchor.

This geometric adapter does not predict hidden cabinet interiors or semantic
door labels. No category list, ground truth, target count or new VLM call is
used. Remove its MEASURED_REFINEMENT entry to detach it from v12.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation

from .part_geometry import replace_regions


def cosine(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return float(a @ b / max(np.linalg.norm(a)*np.linalg.norm(b), 1e-12))


def boundary_relation(source, body, rotation, origin, semantic, shared_frames, measured_face=False):
    """A planar subinstance must lie on one measured volumetric body boundary."""
    if len(source) < 32 or len(body) < 32 or semantic < .8 or len(shared_frames) < 3:
        return None
    local_body = (body-origin) @ rotation
    local_source = (source-origin) @ rotation
    lower, upper = local_body.min(0), local_body.max(0)
    extent = upper-lower
    source_extent = np.ptp(local_source, axis=0)
    volume_ratio = np.prod(extent)/max(np.prod(source_extent), 1e-12)
    if extent.min() < .15 or volume_ratio < 8:
        return None
    inclusion = np.mean(np.all((local_source >= lower-.025) & (local_source <= upper+.025), axis=1))
    if inclusion < .98:
        return None
    values, vectors = np.linalg.eigh(np.cov(source.T))
    flatness = (values[1]-values[0])/max(values[1], 1e-12)
    normal = vectors[:, 0]
    axis = int(np.argmax(np.abs(normal @ rotation)))
    alignment = float(abs(normal @ rotation[:, axis]))
    coordinate = float(np.median(local_source[:, axis]))
    boundary_gap = min(abs(coordinate-lower[axis]), abs(coordinate-upper[axis]))
    face_evidence = None
    if measured_face and boundary_gap > .035:
        from scipy.spatial import cKDTree
        distances = cKDTree(body).query(source)[0]
        bins, counts = np.unique(np.floor(local_body[:, axis]/.006).astype(int), return_counts=True)
        for index in np.argsort(counts)[::-1]:
            values = local_body[np.abs(local_body[:, axis]-(bins[index]+.5)*.006) <= .006, axis]
            if len(values) < .15*len(body):
                continue
            face_gap = abs(coordinate-float(np.median(values)))
            if face_gap <= .01 and np.mean(distances <= .015) >= .8:
                face_evidence = dict(plane_support_fraction=len(values)/len(body),
                                     source_shared_surface_fraction=float(np.mean(distances <= .015)),
                                     measured_plane_gap_m=face_gap, extreme_box_gap_m=boundary_gap)
                boundary_gap = face_gap
                break
    # A broad face is retained as a part; a volumetric neighboring object is not.
    if (flatness < .9 or alignment < .95 or boundary_gap > .035
            or source_extent[axis] > .2*extent[axis]
            or np.delete(source_extent, axis).min() < .1):
        return None
    return dict(semantic_cosine=semantic, inclusion=float(inclusion), volume_ratio=float(volume_ratio),
                plane_flatness=float(flatness), boundary_normal_alignment=alignment,
                boundary_gap_m=float(boundary_gap), shared_observed_frames=sorted(shared_frames),
                measured_body_face=face_evidence)


def unique_parents(candidates):
    """Ambiguous parentage is retained at object level, never resolved by score."""
    selected = {source: values[0] for source, values in candidates.items() if len(values) == 1}
    # Avoid dependent merges and cycles; anchors must remain canonical objects.
    return {source: value for source, value in selected.items() if value[0] not in selected}


def construct(context):
    graph_path = context.scene/'topology_map.json'
    graph = json.loads(graph_path.read_text())
    report = dict(algorithm='unique_measured_body_boundary_hierarchy_v12', ground_truth_used=False,
                  generated_points=0, vision_api_calls=0, point_coordinates_preserved=True, objects=[],
                  component_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    metadata = json.loads(context.manifest.read_text()) if context.manifest else {}
    if metadata.get('world_frame') != 'hypersim_world_z_up':
        report['reason'] = '未声明米制竖直轴，保留原物体层级'
    else:
        nodes = graph['object_nodes']['nodes']
        cloud_path = context.scene/'instance_cloud_cleaned.ply'
        cloud = o3d.io.read_point_cloud(str(cloud_path))
        points = np.asarray(cloud.points)
        colors = np.rint(np.asarray(cloud.colors)*255).astype(int)
        ids = colors[:, 0]+255*colors[:, 1]+255**2*colors[:, 2]
        geometry = {gid: points[ids == int(gid)] for gid in nodes}
        tracks = json.loads((context.scene/'validated_object_tracks.json').read_text())
        import pipeline_components as components
        hierarchy_module = getattr(components, 'HIERARCHY_VALIDATION', None)
        hierarchy = None
        if hierarchy_module is not None:
            from .instance_consensus import Views
            from partaware.geometry import load_capture
            _, jobs, kd, kc, scale = load_capture(context.manifest)
            mapping, records = {}, {}
            for gid, track in tracks.items():
                for observation in track['observations']:
                    mapping.setdefault(observation['frame_id'], {})[observation['local_id']] = int(gid)
            for job in jobs:
                fid = job['frame_id']
                values = json.loads((context.scene/'refined_instance'/f'{fid}_instance.json').read_text())
                for record in values:
                    record['instance_id'] = mapping.get(fid, {}).get(record['frame_instance_id'], -1)
                records[fid] = values
            hierarchy = hierarchy_module.Evidence(Views(context, jobs, kd, kc, scale, records), tracks, nodes)
        candidates = {}
        for source, p in geometry.items():
            for body, q in geometry.items():
                if source == body:
                    continue
                orientation = nodes[body]['shape']['orientation']
                rotation = Rotation.from_quat([orientation[k] for k in ['x', 'y', 'z', 'w']]).as_matrix()
                similarity = cosine(nodes[source]['text_embedding'], nodes[body]['text_embedding'])
                native_parent = (hierarchy is not None and similarity < .8
                    and tracks[body].get('proposal_validation', {}).get('native_hierarchy_confirmed', False))
                shared = set(tracks[source]['observed_frames']) & set(tracks[body]['observed_frames'])
                evidence = boundary_relation(p, q, rotation, np.asarray(nodes[body]['position']),
                                             1. if native_parent else similarity, shared,
                                             measured_face=native_parent)
                if evidence is not None and native_parent:
                    native_evidence = hierarchy.measure(p, source, identity_gid=body)
                    if not native_evidence.get('confirmed', False):
                        evidence = None
                    else:
                        evidence.update(semantic_cosine=similarity,
                                        semantic_override='native_parent_identity_and_measured_boundary',
                                        native_parent_evidence=native_evidence)
                if evidence is not None:
                    candidates.setdefault(source, []).append((body, evidence))
        selected = unique_parents(candidates)
        replacements, remap = {}, {}
        (context.scene/'parts').mkdir(exist_ok=True)
        for source, (body, evidence) in selected.items():
            part_id = f'assembly_{source}_panel'
            p = geometry[source]
            frames = evidence['shared_observed_frames']
            part = dict(id=part_id, name=f'{nodes[body]["name"]}: observed panel', node_type='part', parent_id=body,
                position=p.mean(0).tolist(), extent=np.ptp(p, axis=0).tolist(), confidence=tracks[source]['confidence'],
                status='confirmed', observed_frames=frames, point_count=len(p),
                observations=tracks[source]['observations'], semantic_embedding=None,
                semantic_feature_space='geometry_only_no_visual_embedding', source_object_id=source,
                source_name=nodes[source]['name'], evidence_type='unique_measured_boundary_multiview',
                semantic_part_label_inferred=False)
            graph.setdefault('part_nodes', {})[part_id] = part
            graph.setdefault('part_relations', []).append(dict(source_id=part_id, target_id=body,
                description='part_of', evidence_frames=len(frames)))
            np.save(context.scene/'parts'/f'{part_id}.points.npy', p)
            for existing in graph['part_nodes'].values():
                if str(existing.get('parent_id')) == source:
                    existing['parent_id'] = body
                if source in existing.get('parent_evidence', {}):
                    existing['parent_evidence'][body] = existing['parent_evidence'].get(body, 0)+existing['parent_evidence'].pop(source)
            for edge in graph['part_relations']:
                if str(edge['target_id']) == source:
                    edge['target_id'] = body
            replacements[body] = np.concatenate([replacements.get(body, geometry[body]), p])
            tracks[body]['observations'] += tracks[source]['observations']
            tracks[body]['observed_frames'] = sorted(set(tracks[body]['observed_frames']+tracks[source]['observed_frames']))
            tracks[body]['point_count'] = len(replacements[body])
            tracks[body].setdefault('assembly_source_ids', [body]).append(source)
            remap[source] = body
            graph.setdefault('object_identity_aliases', {})[source] = dict(canonical_id=body,
                source_name=nodes[source]['name'], role='observed_panel_component', part_id=part_id, evidence=evidence)
            report['objects'].append(dict(source_id=source, body_id=body, part_id=part_id,
                measured_points=len(p), reason='唯一完整实测边界锚点；保留原分实例为可查询面板部件', **evidence))
        if remap:
            for gid in remap:
                del nodes[gid]
                del tracks[gid]
            for alias in graph.get('object_identity_aliases', {}).values():
                canonical = str(alias.get('canonical_id'))
                if canonical in remap:
                    alias['canonical_id'] = remap[canonical]
            for file in sorted((context.scene/'refined_instance').glob('*_updated_instance.json')):
                records = json.loads(file.read_text())
                for record in records:
                    gid = str(record.get('instance_id', -1))
                    if gid in remap:
                        record['instance_id'] = int(remap[gid])
                file.write_text(json.dumps(records, indent=2)+'\n')
            geometry_path = getattr(context, 'graph_geometry', cloud_path)
            replace_regions(cloud_path, replacements, nodes)
            if geometry_path != cloud_path:
                replace_regions(geometry_path, replacements, nodes)
            (context.scene/'validated_object_tracks.json').write_text(json.dumps(tracks, indent=2)+'\n')
            graph['object_granularity_policy'] = dict(policy='whole_body_with_queryable_observed_subinstances',
                predicted_from='unique_measured_boundary_and_multiview_identity', ground_truth_used=False)
            graph_path.write_text(json.dumps(graph, indent=2)+'\n')
            context.graph_geometry, context.canonical_geometry_input = geometry_path, graph_path
            from . import canonical_geometry
            canonical_geometry.publish(context)
            graph = json.loads(graph_path.read_text())
            confirmed = {i: part for i, part in graph['part_nodes'].items() if part['status'] == 'confirmed'}
            graph['scene_graph'] = dict(schema_version=1,
                nodes={**{i: {**node, 'node_type': 'object'} for i, node in graph['object_nodes']['nodes'].items()}, **confirmed},
                edges=[edge for h in graph['edge_hypotheses'].values() for edge in h['edges'].values()]+graph['part_relations'])
            graph_path.write_text(json.dumps(graph, indent=2)+'\n')
            (context.scene/'parts/partaware_graph.json').write_text(json.dumps(graph, indent=2)+'\n')
    (context.scene/'part_body_assembly_audit.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
    context.event('实测整物体与面板层级完成', parts=len(report['objects']), generated_points=0)

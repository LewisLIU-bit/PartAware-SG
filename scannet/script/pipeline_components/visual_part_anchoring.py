"""Resolve root fragments using confirmed VLPart identity and unique parentage.

VPA consumes visual part predictions already produced by the ordinary pipeline.
Names alone, containment alone or GT granularity never establish a part. Source
coordinates remain available as a queryable geometric part and in its parent.
Detach its MEASURED_REFINEMENT entry to retain independent root hypotheses.
"""
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree

from .part_geometry import replace_regions
from .hierarchical_masks import normalize


def acceptance(part, source, body, root_name, body_name, source_frames):
    parent = str(part.get('parent_id'))
    prefix, separator, suffix = part['name'].partition(':')
    supported = int(part.get('parent_evidence', {}).get(parent, 0))
    inspected = int(part.get('parent_evidence_frames', 0))
    ratio = supported/max(inspected, 1)
    if (part.get('status') != 'confirmed' or not separator
            or normalize(suffix) != normalize(root_name)
            or normalize(prefix) != normalize(body_name)
            or supported < 5 or inspected < supported or ratio < .5
            or len(set(part.get('observed_frames', [])) & set(source_frames)) < 3
            or min(len(source), len(body)) < 32):
        return None
    lower, upper = body.min(0), body.max(0)
    inclusion = float(np.mean(np.all((source >= lower-.02) & (source <= upper+.02), axis=1)))
    source_volume = np.prod(np.maximum(np.ptp(source, axis=0), .001))
    body_volume = np.prod(np.maximum(np.ptp(body, axis=0), .001))
    contact = float(cKDTree(body).query(source)[0].min())
    if inclusion < .98 or source_volume > .5*body_volume or contact > .015:
        return None
    return dict(parent_mask_support_views=supported, parent_mask_support_fraction=ratio,
                inclusion=inclusion, measured_body_contact_m=contact,
                source_to_body_volume_ratio=float(source_volume/body_volume))


def construct(context):
    path = context.scene/'topology_map.json'
    graph = json.loads(path.read_text()); nodes = graph['object_nodes']['nodes']
    tracks_path = context.scene/'validated_object_tracks.json'
    tracks = json.loads(tracks_path.read_text())
    cloud_path = context.scene/'instance_cloud_cleaned.ply'
    cloud = o3d.io.read_point_cloud(str(cloud_path)); all_points = np.asarray(cloud.points)
    colors = np.rint(np.asarray(cloud.colors)*255).astype(int)
    ids = colors[:,0]+255*colors[:,1]+255**2*colors[:,2]
    geometry = {gid:all_points[ids == int(gid)] for gid in nodes}
    candidates = {}
    for part_id, part in graph.get('part_nodes', {}).items():
        body = str(part.get('parent_id'))
        points_path = context.scene/'parts'/f'{part_id}.points.npy'
        if body not in nodes or not points_path.is_file():
            continue
        part_points = np.load(points_path)
        if len(part_points) < 32:
            continue
        tree = cKDTree(part_points)
        for source in nodes:
            if source == body or source not in tracks:
                continue
            evidence = acceptance(part, geometry[source], geometry[body], nodes[source]['name'],
                nodes[body]['name'], tracks[source]['observed_frames'])
            if evidence is None:
                continue
            coverage = float(np.mean(tree.query(geometry[source])[0] <= .015))
            if coverage < .8:
                continue
            evidence.update(part_mask_surface_coverage=coverage, visual_part_id=part_id)
            candidates.setdefault(source, {}).setdefault(body, []).append(evidence)
    selected = {source:next(iter(values.items())) for source,values in candidates.items() if len(values) == 1}
    selected = {source:(body,evidence) for source,(body,evidence) in selected.items() if body not in selected}
    report = dict(algorithm='VPA_confirmed_visual_part_unique_parent_v13', ground_truth_used=False,
                  vision_api_calls=0, generated_points=0, source_coordinates_preserved=True,
                  objects=[], component_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    remap, replacements = {}, {}
    for source, (body, evidence) in selected.items():
        original = copy.deepcopy(nodes[source]); region = geometry[source]
        part_id = f'owned_{source}_part'
        graph['part_nodes'][part_id] = dict(id=part_id, name=f'{nodes[body]["name"]}: {original["name"]}',
            node_type='part', parent_id=body, status='confirmed', position=region.mean(0).tolist(),
            extent=np.ptp(region,axis=0).tolist(), point_count=len(region),
            source_object_id=source, source_name=original['name'],
            confidence=tracks[source]['confidence'], observed_frames=tracks[source]['observed_frames'],
            observations=tracks[source]['observations'], visual_part_evidence=evidence,
            semantic_embedding=None, semantic_feature_space='geometry_only_no_visual_embedding',
            source_object_features={'visual_embedding':original['visual_embedding'],
                                    'text_embedding':original['text_embedding']})
        np.save(context.scene/'parts'/f'{part_id}.points.npy', region)
        graph.setdefault('part_relations', []).append(dict(source_id=part_id, target_id=body,
            description='part_of', evidence_frames=max(e['parent_mask_support_views'] for e in evidence)))
        replacements[body] = np.concatenate([replacements.get(body,geometry[body]), region])
        remap[source] = body
        graph.setdefault('object_identity_aliases', {})[source] = dict(canonical_id=body,
            source_name=original['name'], role='confirmed_visual_part', evidence=evidence)
        tracks[body].setdefault('visual_part_source_ids', []).append(source)
        # Part observations are retained on the part, not mislabeled as whole
        # body observations in the detector's original local-label interface.
        del nodes[source]; del tracks[source]
        report['objects'].append(dict(source_id=source, body_id=body, part_id=part_id,
                                      measured_points=len(region), evidence=evidence))
    if remap:
        for part in graph.get('part_nodes', {}).values():
            parent = str(part.get('parent_id'))
            if parent in remap:
                part['parent_id'] = remap[parent]
        for relation in graph.get('part_relations', []):
            relation['target_id'] = remap.get(str(relation['target_id']), relation['target_id'])
        for alias in graph.get('object_identity_aliases', {}).values():
            alias['canonical_id'] = remap.get(str(alias['canonical_id']), alias['canonical_id'])
        for relation in graph.get('object_relations', []):
            for key in ['source_id','target_id']:
                relation[key] = remap.get(str(relation[key]), relation[key])
        for file in (context.scene/'refined_instance').glob('*_updated_instance.json'):
            values = json.loads(file.read_text())
            for value in values:
                source = str(value.get('instance_id',-1))
                if source in remap:
                    value['instance_id'] = int(remap[source])
            file.write_text(json.dumps(values,indent=2)+'\n')
        geometry_path = Path(context.graph_geometry)
        # Preserve any previously generated parent geometry as well as every
        # original source coordinate when changing its canonical owner.
        if geometry_path != cloud_path:
            completed_cloud = o3d.io.read_point_cloud(str(geometry_path))
            p = np.asarray(completed_cloud.points);c = np.rint(np.asarray(completed_cloud.colors)*255).astype(int)
            keys = c[:,0]+255*c[:,1]+255**2*c[:,2]
            complete_replacements = {}
            for source, body in remap.items():
                complete_replacements[body] = np.concatenate([
                    complete_replacements.get(body,p[keys == int(body)]),p[keys == int(source)]])
            replace_regions(geometry_path, complete_replacements, nodes)
        replace_regions(cloud_path, replacements, nodes)
        for body, region in replacements.items():
            tracks[body]['point_count'] = len(region)
        tracks_path.write_text(json.dumps(tracks,indent=2)+'\n')
        # Remove obsolete root entries from the combined graph while keeping
        # the queryable geometric parts and the new canonical aliases.
        for source in remap:
            graph.get('scene_graph',{}).get('nodes',{}).pop(source,None)
        path.write_text(json.dumps(graph,indent=2)+'\n')
        context.canonical_geometry_input = path
        from .canonical_geometry import publish
        publish(context)
    (context.scene/'visual_part_anchoring_audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    context.event('视觉部件与唯一父体消解重复根节点', parts=len(remap), generated_points=0)

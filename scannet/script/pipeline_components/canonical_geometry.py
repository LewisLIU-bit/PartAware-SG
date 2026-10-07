"""Publish measured or validated completed geometry without repeated filtering."""
import json
import numpy as np
import open3d as o3d
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation


def fit_box(points, upright=False):
    if not upright:
        cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
        try:
            box = cloud.get_oriented_bounding_box(robust=True)
            center, matrix, extent = np.asarray(box.center), np.asarray(box.R), np.asarray(box.extent)
        except RuntimeError:
            matrix = np.eye(3)
            lower, upper = points.min(0), points.max(0)
            center, extent = (lower+upper)/2, upper-lower
    else:
        # Minimal XY rectangle preserves the declared vertical gravity axis.
        xy = np.unique(points[:, :2], axis=0)
        angles = [0.]
        if len(xy) >= 3:
            try:
                hull = xy[ConvexHull(xy).vertices]
                edges = np.roll(hull, -1, axis=0)-hull
                angles = np.unique(np.arctan2(edges[:, 1], edges[:, 0]) % (np.pi/2))
            except Exception:
                pass
        best = None
        for theta in angles:
            c, s = np.cos(theta), np.sin(theta)
            matrix = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
            local = points @ matrix
            lower, upper = local.min(0), local.max(0)
            extent = upper-lower
            area = extent[0]*extent[1]
            if best is None or area < best[0]:
                best = (area, (lower+upper)/2 @ matrix.T, matrix, extent)
        _, center, matrix, extent = best
    extent = np.maximum(extent, 1e-4)
    quat = Rotation.from_matrix(matrix).as_quat()
    shape = {'length': float(extent[0]), 'width': float(extent[1]), 'height': float(extent[2]),
             'orientation': dict(zip(['x', 'y', 'z', 'w'], quat.tolist()))}
    return center.tolist(), shape


def synchronize_recovered_tracks(scene, nodes):
    path = scene/'validated_object_tracks.json'
    if not path.exists(): return
    tracks = json.loads(path.read_text())
    recovered = {gid for gid in nodes if nodes[gid].get('structural_surface_recovery')}
    if not recovered: return
    ownership = {}
    for file in sorted((scene/'refined_instance').glob('*_updated_instance.json')):
        fid = file.name.removesuffix('_updated_instance.json')
        for record in json.loads(file.read_text()):
            gid = str(record.get('instance_id', -1))
            if gid in recovered:
                ownership[(fid, int(record['frame_instance_id']))] = gid
    seen = {gid: {(o['frame_id'], int(o['local_id'])) for o in tracks[gid]['observations']} for gid in recovered}
    raw = json.loads((scene/'object_tracks.json').read_text())
    for original in raw.values():
        for observation in original['observations']:
            key = (observation['frame_id'], int(observation['local_id']))
            gid = ownership.get(key)
            if gid is not None and key not in seen[gid]:
                tracks[gid]['observations'].append(observation)
                seen[gid].add(key)
    for gid in recovered:
        tracks[gid]['observed_frames'] = sorted({o['frame_id'] for o in tracks[gid]['observations']})
    path.write_text(json.dumps(tracks, indent=2)+'\n')


def publish(context):
    source = getattr(context, 'canonical_geometry_input', context.scene/'topology_map_cleaned.json')
    graph = json.loads(source.read_text())
    cloud = o3d.io.read_point_cloud(str(context.graph_geometry))
    points, colors = np.asarray(cloud.points), np.rint(np.asarray(cloud.colors)*255).astype(int)
    ids = colors[:, 0]+255*colors[:, 1]+255*255*colors[:, 2]
    metadata = json.loads(context.manifest.read_text()) if context.manifest else {}
    upright = metadata.get('world_frame') == 'hypersim_world_z_up'
    nodes = graph['object_nodes']['nodes'] or {}
    for gid in list(nodes):
        local = points[ids == int(gid)]
        if len(local) < 16:
            del nodes[gid]
            continue
        if str(nodes[gid].get('id', gid)) != str(gid):
            raise ValueError(f'Object node identifier disagrees with graph key: {gid}')
        nodes[gid]['id'] = str(gid)
        nodes[gid]['position'], nodes[gid]['shape'] = fit_box(local, upright)
    synchronize_recovered_tracks(context.scene, nodes)
    # Store an observation-only geometric hypothesis for the completion ablation.
    measured_cloud = o3d.io.read_point_cloud(str(context.scene/'instance_cloud_cleaned.ply'))
    measured_points = np.asarray(measured_cloud.points)
    measured_colors = np.rint(np.asarray(measured_cloud.colors)*255).astype(int)
    measured_ids = measured_colors[:, 0]+255*measured_colors[:, 1]+255*255*measured_colors[:, 2]
    import copy
    observed_graph = copy.deepcopy(graph)
    for gid, node in observed_graph['object_nodes']['nodes'].items():
        local = measured_points[measured_ids == int(gid)]
        node['position'], node['shape'] = fit_box(local, upright)
    edges = {}
    keys = list(nodes)
    for i, a in enumerate(keys):
        for b in keys[i+1:]:
            delta = np.asarray(nodes[b]['position'])-nodes[a]['position']
            distance = float(np.linalg.norm(delta))
            if distance < context.edge_threshold and distance > 1e-8:
                edges[f'{a}_{b}'] = {'source_id': a, 'target_id': b, 'distance': distance,
                    'direction': (delta/distance).tolist(), 'description': 'next to'}
    graph['edge_hypotheses'] = {'default_hypothesis': {'id': 'default_hypothesis', 'confidence': 1., 'edges': edges}}
    graph['geometry_provenance'] = {'source': context.graph_geometry.name,
        'bbox_algorithm': 'minimum_area_upright_box' if upright else 'oriented_surface_box',
        'observed_ply': 'instance_cloud_cleaned.ply', 'generated_points_used_for_association': False}
    observed_edges = {}
    observed_nodes = observed_graph['object_nodes']['nodes']
    observed_keys = list(observed_nodes)
    for i, a in enumerate(observed_keys):
        for b in observed_keys[i+1:]:
            delta = np.asarray(observed_nodes[b]['position'])-observed_nodes[a]['position']
            distance = float(np.linalg.norm(delta))
            if 1e-8 < distance < context.edge_threshold:
                observed_edges[f'{a}_{b}'] = {'source_id': a, 'target_id': b, 'distance': distance,
                    'direction': (delta/distance).tolist(), 'description': 'next to'}
    observed_graph['edge_hypotheses'] = {'default_hypothesis': {'id': 'default_hypothesis', 'confidence': 1., 'edges': observed_edges}}
    observed_graph['geometry_provenance'] = {**graph['geometry_provenance'], 'source': 'instance_cloud_cleaned.ply'}
    for current in [graph, observed_graph]:
        if 'scene_graph' in current:
            scene_nodes = current['scene_graph'].setdefault('nodes', {})
            for gid, node in current['object_nodes']['nodes'].items():
                scene_nodes[gid] = {**scene_nodes.get(gid, {}), **node, 'node_type': 'object'}
            current['scene_graph']['edges'] = [e for h in current['edge_hypotheses'].values()
                for e in h['edges'].values()] + current.get('part_relations', [])
    (context.scene/'topology_map_observed.json').write_text(json.dumps(observed_graph, indent=2)+'\n')
    (context.scene/'topology_map.json').write_text(json.dumps(graph, indent=2)+'\n')
    context.event('正式图几何与空间关系同步完成', objects=len(nodes), edges=len(edges), source=context.graph_geometry.name)

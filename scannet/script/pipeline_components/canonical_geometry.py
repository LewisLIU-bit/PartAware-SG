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
        nodes[gid]['position'], nodes[gid]['shape'] = fit_box(local, upright)
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
    (context.scene/'topology_map_observed.json').write_text(json.dumps(observed_graph, indent=2)+'\n')
    (context.scene/'topology_map.json').write_text(json.dumps(graph, indent=2)+'\n')
    context.event('正式图几何与空间关系同步完成', objects=len(nodes), edges=len(edges), source=context.graph_geometry.name)

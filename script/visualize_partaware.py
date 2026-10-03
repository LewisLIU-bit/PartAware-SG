"""Display observed part clouds, object nodes, and part-of links without changing data."""
import argparse
import colorsys
import json
from pathlib import Path
import numpy as np
import open3d as o3d


def position(node):
    value = np.asarray(node['position'], dtype=float)
    if value.shape != (3,) or not np.isfinite(value).all():
        raise ValueError('Node positions must be finite world-coordinate vectors')
    return value


def sphere(center, radius, color):
    mesh = o3d.geometry.TriangleMesh.create_sphere(radius=radius, resolution=12)
    mesh.translate(center)
    mesh.paint_uniform_color(color)
    mesh.compute_vertex_normals()
    return mesh


def build_geometry(args):
    graph_path = Path(args.graph).expanduser().resolve()
    graph = json.loads(graph_path.read_text())
    objects = graph['object_nodes']['nodes']
    parts = graph.get('part_nodes', {})
    selected = {key: value for key, value in parts.items()
                if args.include_provisional or value['status'] == 'confirmed'}
    geometry = []
    if args.base_cloud:
        base = o3d.io.read_point_cloud(str(Path(args.base_cloud).expanduser().resolve()))
        if base.is_empty():
            raise ValueError('The base point cloud is empty or unreadable')
        base.paint_uniform_color([0.55, 0.55, 0.55])
        geometry.append(base)
    for key, node in objects.items():
        geometry.append(sphere(position(node), args.object_radius, [0.1, 0.4, 0.95]))
        print(f'Object {key}: {node["name"]}')
    point_count = 0
    for index, (key, node) in enumerate(selected.items()):
        color = colorsys.hsv_to_rgb((index * 0.61803398875) % 1, 0.85, 0.95)
        geometry.append(sphere(position(node), args.part_radius, color))
        if not args.nodes_only:
            cloud_path = graph_path.parent / f'{key}.points.npy'
            points = np.load(cloud_path, allow_pickle=False)
            if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
                raise ValueError(f'Invalid observed part point cloud: {cloud_path}')
            point_count += len(points)
            cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
            cloud.paint_uniform_color(color)
            geometry.append(cloud)
        print(f'Part {key}: {node["name"]}; parent={node["parent_id"]}; status={node["status"]}')
    endpoints, lines = [], []
    for edge in graph.get('part_relations', []):
        part_id, parent_id = str(edge['source_id']), str(edge['target_id'])
        if part_id not in selected:
            continue
        if parent_id not in objects:
            raise ValueError(f'Unknown parent object: {parent_id}')
        offset = len(endpoints)
        endpoints.extend([position(selected[part_id]), position(objects[parent_id])])
        lines.append([offset, offset + 1])
    if lines:
        links = o3d.geometry.LineSet(o3d.utility.Vector3dVector(endpoints),
                                    o3d.utility.Vector2iVector(lines))
        links.paint_uniform_color([0.1, 0.85, 0.2])
        geometry.append(links)
    object_links = 0
    if args.show_object_edges:
        hypotheses = graph.get('edge_hypotheses') or {}
        if hypotheses:
            hypothesis = max(hypotheses.values(), key=lambda item: item.get('confidence', 0))
            endpoints, lines = [], []
            for edge in (hypothesis.get('edges') or {}).values():
                source, target = str(edge['source_id']), str(edge['target_id'])
                if source not in objects or target not in objects:
                    raise ValueError('An object relation references an unknown node')
                offset = len(endpoints)
                endpoints.extend([position(objects[source]), position(objects[target])])
                lines.append([offset, offset + 1])
            object_links = len(lines)
            if lines:
                links = o3d.geometry.LineSet(o3d.utility.Vector3dVector(endpoints),
                                            o3d.utility.Vector2iVector(lines))
                links.paint_uniform_color([0.2, 0.4, 0.85])
                geometry.append(links)
    stats = {'objects': len(objects), 'displayed_parts': len(selected),
             'part_of_links': sum(e['source_id'] in selected for e in graph.get('part_relations', [])),
             'object_links': object_links, 'observed_part_points': point_count}
    print(json.dumps(stats))
    return geometry, stats


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--graph', required=True, help='Generated partaware_graph.json')
    parser.add_argument('--base-cloud', help='Optional baseline PLY in the same world frame')
    parser.add_argument('--include-provisional', action='store_true', help='Also display single-frame or unconfirmed tracks')
    parser.add_argument('--nodes-only', action='store_true', help='Hide observed part point clouds')
    parser.add_argument('--show-object-edges', action='store_true', help='Show the highest-confidence saved object relation hypothesis')
    parser.add_argument('--object-radius', type=float, default=0.07)
    parser.add_argument('--part-radius', type=float, default=0.03)
    parser.add_argument('--check-only', action='store_true', help='Validate geometry and print counts without opening a window')
    args = parser.parse_args()
    if not np.isfinite([args.object_radius, args.part_radius]).all() or min(args.object_radius, args.part_radius) <= 0:
        parser.error('Node radii must be finite and positive')
    geometry, _ = build_geometry(args)
    if not args.check_only:
        o3d.visualization.draw_geometries(geometry, window_name='PartAware-SG: object and part graph',
                                        width=1280, height=800)


if __name__ == '__main__':
    main()

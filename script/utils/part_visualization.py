"""Small part overlays for the existing ScanNet-SG map renderer."""
import json
from pathlib import Path

import numpy as np
import open3d as o3d


def _position(node, bias):
    value = np.asarray(node['position'], dtype=float)
    if value.shape != (3,) or not np.isfinite(value).all():
        raise ValueError('Expected a finite world-coordinate node position')
    return value + np.array([bias, 0, 0])


def _dashed_link(start, end, color):
    # Gaps distinguish hierarchy edges from the original object relations.
    count = max(2, int(np.linalg.norm(end - start) / 0.045))
    fractions = np.linspace(0, 1, count * 2 + 1)[:-1]
    points = start + fractions[:, None] * (end - start)
    lines = np.arange(count * 2).reshape(-1, 2)
    link = o3d.geometry.LineSet(o3d.utility.Vector3dVector(points),
                                o3d.utility.Vector2iVector(lines))
    link.paint_uniform_color(color)
    return link


def build_part_overlay(graph_path, tracking_colors, radius, bias=0.0,
                       show_points=False, include_provisional=False):
    """Return typed geometries and pickable nodes without altering the base graph."""
    if not np.isfinite(radius) or radius <= 0:
        raise ValueError('Part radius must be finite and positive')
    path = Path(graph_path)
    graph = json.loads(path.read_text())
    objects = graph['object_nodes']['nodes']
    selected = {key: node for key, node in graph.get('part_nodes', {}).items()
                if include_provisional or (node['status'] == 'confirmed' and node['parent_id'] is not None)}
    geometries, picks = [], []
    point_count, link_count = 0, 0
    for key, node in selected.items():
        parent = str(node['parent_id'])
        parent_color = np.asarray(tracking_colors.get(int(parent) if parent.isdecimal() else -1,
                                                      [0.55, 0.55, 0.55]))
        color = np.clip(parent_color * 0.72 + 0.12, 0, 1)
        center = _position(node, bias)
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
        sphere.translate(center)
        sphere.paint_uniform_color(color)
        sphere.compute_vertex_normals()
        geometries.append(('mesh', sphere))
        picks.append({'id': key, 'name': f"{node['name']} (part of {node['parent_id']})",
                      'position': center, 'kind': 'part'})
        if show_points:
            point_path = path.parent / f'{key}.points.npy'
            if not point_path.is_file():
                # The canonical graph lives above the standalone part artifacts.
                point_path = path.parent / 'parts' / f'{key}.points.npy'
            points = np.load(point_path, allow_pickle=False)
            if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
                raise ValueError(f'Invalid part point cloud: {key}')
            cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points + [bias, 0, 0]))
            cloud.paint_uniform_color(color)
            geometries.append(('points', cloud))
            point_count += len(points)
    for edge in graph.get('part_relations', []):
        source, target = str(edge['source_id']), str(edge['target_id'])
        if source not in selected:
            continue
        if target not in objects or str(selected[source]['parent_id']) != target:
            raise ValueError('Part relation does not agree with its parent object')
        color = np.asarray(tracking_colors.get(int(target), [0.55, 0.55, 0.55])) * 0.6 + 0.2
        geometries.append(('line', _dashed_link(_position(selected[source], bias),
                                              _position(objects[target], bias), color)))
        link_count += 1
    return geometries, picks, {'displayed_parts': len(selected), 'part_of_links': link_count,
                               'observed_part_points': point_count}

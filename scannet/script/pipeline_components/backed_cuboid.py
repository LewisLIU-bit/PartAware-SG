"""Infer an occluded cuboid bounded by an independently measured rear plane.

Adapts structural cuboid constraints from CubeSLAM/Total3DUnderstanding.
This is a geometric hypothesis, not measured surface or a learned CAD model.
No class dimensions, scene coordinates, annotations or text prompts are used.
"""
from pathlib import Path
import hashlib, json
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree
from partaware.geometry import load_capture, voxel_downsample
from .instance_consensus import Views


def fit_hypothesis(observed, background, cameras):
    if len(observed) < 512 or np.ptp(observed[:, 2]) < .5:
        return None, {'reason': '实测点数或竖直尺度不足'}
    cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(observed))
    plane, indices = cloud.segment_plane(.012, 3, 500)
    normal = np.array(plane[:3]); normal /= np.linalg.norm(normal)
    if len(indices)/len(observed) < .6 or abs(normal[2]) > .08:
        return None, {'reason': '没有占比足够的近竖直正面'}
    normal[2] = 0; normal /= np.linalg.norm(normal)
    front = observed[indices]
    if np.median((cameras-front.mean(0)) @ normal) < 0: normal *= -1
    tangent = np.cross([0, 0, 1], normal)
    rotation = np.column_stack([normal, tangent, [0, 0, 1]])
    local = observed @ rotation; face = front @ rotation
    lower, upper = local.min(0), local.max(0)
    front_depth = float(np.median(face[:, 0])); width, height = np.ptp(face[:, 1:3], axis=0)
    if width < .3 or height < .5 or np.ptp(face[:, 0]) > .05:
        return None, {'reason': '正面几何不足以支持长方体假设'}
    grid = np.clip(((face[:, 1:3]-face[:, 1:3].min(0))/[width, height]*20).astype(int), 0, 19)
    coverage = len(np.unique(grid, axis=0))/400
    if coverage < .65:
        return None, {'reason': '正面缺乏连续矩形覆盖', 'front_grid_coverage': coverage}
    if upper[0]-lower[0] > .6*min(width, height):
        return None, {'reason': '当前实测体积已有充分纵深'}
    bg = background @ rotation
    valid = (bg[:, 0] < front_depth-.1) & (bg[:, 0] > front_depth-min(1.2, width))
    valid &= (bg[:, 1] >= lower[1]-.25*width) & (bg[:, 1] <= upper[1]+.25*width)
    valid &= (bg[:, 2] >= lower[2]-.1) & (bg[:, 2] <= upper[2]+.25)
    bg = bg[valid]
    if len(bg) < 256:
        return None, {'reason': '没有足够的实测后方结构边界'}
    plane, indices = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(bg)).segment_plane(.012, 3, 500)
    parallel = abs(float(plane[0]))/max(np.linalg.norm(plane[:3]), 1e-9)
    rear = bg[indices]
    rear_depth = float(np.median(rear[:, 0])); depth = front_depth-rear_depth
    if parallel < .99 or len(indices)/len(bg) < .45 or np.ptp(rear[:, 1]) < .8*width:
        return None, {'reason': '后方结构边界不平行、不充分或横向覆盖不足'}
    # A visible side must already extend toward the inferred rear; a pure sheet
    # alone cannot establish attachment to a distant wall.
    side_support = int(np.sum(local[:, 0] < front_depth-.15*depth))
    if side_support < 128 or lower[0] > front_depth-.4*depth:
        return None, {'reason': '缺少朝向后方边界的实测侧面，纵深仍不可辨识'}
    if rear_depth >= lower[0]-.03:
        return None, {'reason': '结构边界不会提供有意义的新增纵深'}
    lower[0] = rear_depth+.01
    # Door handles and isolated protrusions are measured evidence, not evidence
    # that the entire body has that enlarged rectangular section.
    upper[0] = front_depth
    lower[1:] = np.quantile(face[:, 1:], .005, axis=0)
    upper[1:] = np.quantile(face[:, 1:], .995, axis=0)
    axes = [np.linspace(a, b, max(2, int(np.ceil((b-a)/.02))+1)) for a, b in zip(lower, upper)]
    surfaces = []
    for fixed in range(3):
        others = [i for i in range(3) if i != fixed]
        a, b = np.meshgrid(axes[others[0]], axes[others[1]])
        for coordinate in [lower[fixed], upper[fixed]]:
            # Keep the measured front and do not manufacture a new front texture.
            if fixed == 0 and coordinate == upper[0]: continue
            p = np.empty((a.size, 3)); p[:, fixed] = coordinate
            p[:, others[0]], p[:, others[1]] = a.ravel(), b.ravel()
            surfaces.append(p @ rotation.T)
    return np.concatenate(surfaces), {'front_inlier_fraction': len(front)/len(observed),
        'front_grid_coverage': coverage, 'rear_plane_parallelism': parallel,
        'rear_plane_points': len(rear), 'side_support_points': side_support,
        'inferred_depth_m': depth, 'inferred_local_bounds': [lower.tolist(), upper.tolist()],
        'rear_attachment_assumption': True}


def verify(candidate, gid, views, other):
    seen = np.zeros(len(candidate), int); free = seen.copy(); different = seen.copy()
    for fid in sorted(views.jobs):
        pose, raw_depth, mask = views.get(fid); depth = raw_depth/views.scale
        camera = (candidate-pose[:3, 3]) @ pose[:3, :3]; z = camera[:, 2]
        uv = np.rint(camera[:, :2]/np.maximum(z[:, None], 1e-8)*[views.kd[0, 0], views.kd[1, 1]]
                     + [views.kd[0, 2], views.kd[1, 2]]).astype(int)
        valid = (z > 0) & (uv[:, 0] >= 0) & (uv[:, 1] >= 0) & (uv[:, 0] < depth.shape[1]) & (uv[:, 1] < depth.shape[0])
        ids = np.flatnonzero(valid); measured = depth[uv[ids, 1], uv[ids, 0]]
        known = measured > 0; ids = ids[known]; measured = measured[known]
        tolerance = .012+.003*measured
        seen[ids] += 1; free[ids[z[ids] < measured-tolerance]] += 1
        cuv = np.rint(camera[ids, :2]/z[ids, None]*[views.kc[0, 0], views.kc[1, 1]]
                      + [views.kc[0, 2], views.kc[1, 2]]).astype(int)
        visible = (np.abs(z[ids]-measured) <= tolerance) & (cuv[:, 0] >= 0) & (cuv[:, 1] >= 0)
        visible &= (cuv[:, 0] < mask.shape[1]) & (cuv[:, 1] < mask.shape[0])
        indices = ids[visible]; labels = mask[cuv[visible, 1], cuv[visible, 0]]
        mapping = {int(r['frame_instance_id']): str(r.get('instance_id', -1)) for r in views.records[fid]}
        conflict = np.array([mapping.get(int(k), '-1') not in [gid, '-1'] for k in labels], dtype=bool)
        different[indices[conflict]] += 1
    foreign = cKDTree(other).query(candidate)[0] < .02 if len(other) else np.zeros(len(candidate), bool)
    contradicted = (free > 0) | (different > 0) | foreign
    repeated = free >= 2
    evidence = {'free_space_conflict_fraction': float(np.mean(free > 0)),
                'repeated_free_space_conflict_fraction': float(repeated.mean()),
                'foreign_object_conflict_fraction': float(np.mean((different > 0) | foreign)),
                'checked_frames': len(views.jobs), 'generated_points_are_measurements': False}
    if evidence['repeated_free_space_conflict_fraction'] > .03 or evidence['foreign_object_conflict_fraction'] > .05:
        return None, {**evidence, 'reason': '推断体积与实测自由空间或其他物体冲突'}
    # The fit bounds must also survive, rather than letting point trimming hide
    # a large invalid primitive. Single-frame conflicts cannot become evidence.
    if contradicted.mean() > .1:
        return None, {**evidence, 'reason': '被否定的候选表面过多'}
    return candidate[~contradicted], evidence


def construct(context):
    report = {'algorithm': 'measured_rear_plane_bounded_cuboid_v9', 'ground_truth_used': False,
              'language_prompts': 0, 'objects': [], 'accepted_objects': 0,
              'component_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    data, jobs, kd, kc, scale = load_capture(context.manifest, context.image_dir, context.scene/'refined_instance')
    source = context.scene/'instance_cloud_with_background.ply'
    if data.get('world_frame') != 'hypersim_world_z_up' or not source.exists():
        report['reason'] = '缺少已验证竖直轴或背景实测云，保留基础几何'
        (context.scene/'cuboid_completion_audit.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n'); return
    o3d.utility.random.seed(9)
    cloud = o3d.io.read_point_cloud(str(source)); p = np.asarray(cloud.points)
    colors = np.rint(np.asarray(cloud.colors)*255).astype(int)
    background = p[np.all(colors == 0, axis=1)]
    cloud = o3d.io.read_point_cloud(str(context.scene/'instance_cloud_cleaned.ply'))
    observed = np.asarray(cloud.points); encoded = np.rint(np.asarray(cloud.colors)*255).astype(int)
    ids = encoded[:, 0]+255*encoded[:, 1]+255**2*encoded[:, 2]
    records = {j['frame_id']: json.loads((context.scene/'refined_instance'/f"{j['frame_id']}_updated_instance.json").read_text()) for j in jobs}
    views = Views(context, jobs, kd, kc, scale, records)
    graph = json.loads((context.scene/'topology_map_cleaned.json').read_text()); nodes = graph['object_nodes']['nodes']
    cameras = np.array([np.loadtxt(j['pose'])[:3, 3] for j in jobs])
    base = o3d.io.read_point_cloud(str(context.graph_geometry)); additions = []; addition_colors = []
    for gid in nodes:
        local = observed[ids == int(gid)]
        candidate, fit = fit_hypothesis(local, background, cameras)
        entry = {'instance_id': gid, 'name': nodes[gid]['name'], 'status': 'not_eligible' if candidate is None else 'rejected', **fit}
        report['objects'].append(entry)
        if candidate is None: continue
        accepted, evidence = verify(candidate, gid, views, observed[ids != int(gid)])
        entry.update(evidence)
        if accepted is None: continue
        accepted = accepted[cKDTree(local).query(accepted)[0] > .015]
        if len(accepted) < 128: entry['reason'] = '没有足够新增隐蔽表面'; continue
        additions.append(accepted)
        color = [int(gid)%255, (int(gid)//255)%255, (int(gid)//255**2)%255]
        addition_colors.append(np.tile(color, (len(accepted), 1))/255)
        entry.update(status='accepted', generated_points=len(accepted))
        nodes[gid]['geometry_hypothesis'] = {'type': 'rear_plane_bounded_cuboid', 'measured': False,
                                           'rear_attachment_assumption': True, 'audit': 'cuboid_completion_audit.json'}
    if additions:
        output = o3d.geometry.PointCloud()
        output.points = o3d.utility.Vector3dVector(np.concatenate([np.asarray(base.points), *additions]))
        output.colors = o3d.utility.Vector3dVector(np.concatenate([np.asarray(base.colors), *addition_colors]))
        path = context.scene/'instance_cloud_completed.ply'
        if not o3d.io.write_point_cloud(str(path), output): raise IOError('Could not publish inferred cuboid geometry')
        context.graph_geometry = path
        (context.scene/'topology_map_cleaned.json').write_text(json.dumps(graph, indent=2)+'\n')
    report['accepted_objects'] = sum(e['status'] == 'accepted' for e in report['objects'])
    (context.scene/'cuboid_completion_audit.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
    context.event('结构约束完整体积验收完成', accepted_objects=report['accepted_objects'], qwen_api_calls=0)

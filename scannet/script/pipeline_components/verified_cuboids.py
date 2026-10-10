"""Verify rear-bounded shape hypotheses after measured body/part assembly.

The cached class selects a cuboid prior, not dimensions. Observed faces and
scene boundaries determine dimensions. All capture views provide support,
free-space contradictions, or unknown evidence. No trained text-conditioned
completion network is claimed by this geometric implementation.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree
from threadpoolctl import threadpool_limits

from partaware.geometry import load_capture
from .backed_cuboid import verify
from .instance_consensus import Views


def cuboid_identity(name):
    tokens = set(name.casefold().replace('_', ' ').split())
    return bool(tokens & {'cabinet', 'cupboard', 'refrigerator', 'fridge', 'wardrobe'})


def fit(observed, background, cameras):
    if len(observed) < 512 or np.ptp(observed[:, 2]) < .5:
        return None, {'reason': '实测表面不足'}
    sample = observed[::max(1, len(observed)//12000)]
    initial = len(sample)
    faces = []
    for _ in range(4):
        if len(sample) < 128:
            break
        plane, indices = o3d.geometry.PointCloud(
            o3d.utility.Vector3dVector(sample)).segment_plane(.008, 3, 500)
        face = sample[indices]
        sample = np.delete(sample, indices, axis=0)
        normal = np.asarray(plane[:3])
        if abs(normal[2]) > .08 or len(face) < .2*initial:
            continue
        normal[2] = 0.
        normal /= np.linalg.norm(normal)
        if np.median((cameras-face.mean(0)) @ normal) < 0:
            normal *= -1
        tangent = np.cross([0., 0., 1.], normal)
        rotation = np.column_stack([normal, tangent, [0., 0., 1.]])
        local, front = observed @ rotation, face @ rotation
        width, height = np.ptp(front[:, 1:], axis=0)
        if width < .3 or height < .5:
            continue
        cells = np.clip(((front[:, 1:]-front[:, 1:].min(0))/[width, height]*20).astype(int), 0, 19)
        coverage = len(np.unique(cells, axis=0))/400
        if coverage < .65:
            continue
        level = float(np.median(front[:, 0]))
        lower, upper = local.min(0), local.max(0)
        bg = background @ rotation
        inside = (bg[:, 0] < level-.1) & (bg[:, 0] > level-min(1.2, width))
        inside &= np.all((bg[:, 1:] >= lower[1:]-.1) & (bg[:, 1:] <= upper[1:]+.1), axis=1)
        rear = bg[inside]
        if len(rear) < 256:
            continue
        rear_options = []
        remaining = rear
        # A broad floor or side wall may dominate this crop. Search parallel
        # physical boundaries instead of accepting the largest arbitrary plane.
        for _ in range(8):
            if len(remaining) < 256:
                break
            plane, indices = o3d.geometry.PointCloud(
                o3d.utility.Vector3dVector(remaining)).segment_plane(.009, 3, 500)
            boundary = remaining[indices]
            remaining = np.delete(remaining, indices, axis=0)
            parallel = abs(plane[0])/np.linalg.norm(plane[:3])
            if parallel >= .99 and len(boundary) >= 256 and np.ptp(boundary[:, 1]) >= .8*width:
                rear_options.append((len(boundary), boundary, parallel))
        if not rear_options:
            continue
        _, rear, parallel = max(rear_options, key=lambda value: value[0])
        depth = level-float(np.median(rear[:, 0]))
        side = (local[:, 0] < level-.15*depth) & (local[:, 0] > level-depth-.02)
        if side.sum() < 128 or local[:, 0].min() > level-.4*depth:
            continue
        # Already measured sides are retained. Missing rear/side surfaces can
        # still be inferred even when the measured box has sufficient depth.
        lower[0] = min(lower[0], level-depth+.01)
        upper[0] = level
        lower[1:] = np.quantile(front[:, 1:], .005, axis=0)
        upper[1:] = np.quantile(front[:, 1:], .995, axis=0)
        axes = [np.linspace(a, b, max(2, int(np.ceil((b-a)/.015))+1)) for a, b in zip(lower, upper)]
        surfaces = []
        for fixed in range(3):
            others = [i for i in range(3) if i != fixed]
            a, b = np.meshgrid(axes[others[0]], axes[others[1]])
            for coordinate in [lower[fixed], upper[fixed]]:
                if fixed == 0 and coordinate == upper[0]:
                    continue
                points = np.empty((a.size, 3))
                points[:, fixed] = coordinate
                points[:, others[0]], points[:, others[1]] = a.ravel(), b.ravel()
                surfaces.append(points @ rotation.T)
        faces.append((coverage*len(face), np.concatenate(surfaces), dict(
            front_support_fraction=len(face)/initial, front_grid_coverage=coverage,
            rear_parallelism=float(parallel), rear_measured_points=len(rear),
            measured_side_points=int(side.sum()), inferred_depth_m=depth,
            prior='cached_identity_cuboid_with_measured_rear_attachment',
            learned_text_conditioning=False)))
    if not faces:
        return None, {'reason': '缺少连续正面、实测侧面或后方边界'}
    _, candidate, evidence = max(faces, key=lambda value: value[0])
    return candidate, evidence


def construct(context):
    report = dict(algorithm='identity_prior_all_view_cuboid_verification_v14', objects=[],
                  ground_truth_used=False, vision_api_calls=0,
                  learned_text_conditioning=False, generated_points_are_measurements=False,
                  component_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    data, jobs, kd, kc, scale = load_capture(context.manifest, context.image_dir, context.scene/'refined_instance')
    if data.get('world_frame') != 'hypersim_world_z_up':
        report['reason'] = '未声明已验证的竖直轴'
    else:
        o3d.utility.random.seed(14)
        graph_path = context.scene/'topology_map.json'
        graph = json.loads(graph_path.read_text())
        nodes = graph['object_nodes']['nodes']
        tracks = json.loads((context.scene/'validated_object_tracks.json').read_text())
        cloud = o3d.io.read_point_cloud(str(context.scene/'instance_cloud_cleaned.ply'))
        points = np.asarray(cloud.points)
        colors = np.rint(np.asarray(cloud.colors)*255).astype(int)
        ids = colors[:, 0]+255*colors[:, 1]+255**2*colors[:, 2]
        raw = o3d.io.read_point_cloud(str(context.scene/'instance_cloud_with_background.ply'))
        background = np.asarray(raw.points)[np.all(np.asarray(raw.colors) == 0., axis=1)]
        records = {j['frame_id']: json.loads((context.scene/'refined_instance'/f'{j["frame_id"]}_updated_instance.json').read_text()) for j in jobs}
        views = Views(context, jobs, kd, kc, scale, records)
        cameras = np.array([np.loadtxt(j['pose'])[:3, 3] for j in jobs])
        from .image_shape import remove_owned_surface
        base = o3d.io.read_point_cloud(str(context.graph_geometry))
        owned_path = context.scene/'verified_shape_generated.ply'
        if owned_path.is_file():
            base, removed = remove_owned_surface(base, o3d.io.read_point_cloud(str(owned_path)), cloud)
            report['replaced_owned_hypothesis_points'] = removed
        additions, addition_colors = [], []
        for gid, node in nodes.items():
            if not cuboid_identity(node['name']):
                continue
            track = tracks.get(gid, {})
            if track.get('confidence', 0.) < .4 or len(track.get('observed_frames', [])) < 3:
                report['objects'].append(dict(instance_id=gid, name=node['name'], status='rejected',
                    reason='缓存身份置信度或实测观测帧数不足'))
                continue
            local = points[ids == int(gid)]
            # OpenMP RANSAC scheduling otherwise changes seeded samples.
            with threadpool_limits(limits=1):
                candidate, evidence = fit(local, background, cameras)
            entry = dict(instance_id=gid, name=node['name'], status='rejected',
                identity_confidence=track['confidence'], identity_observed_frames=len(track['observed_frames']), **evidence)
            report['objects'].append(entry)
            if candidate is None:
                continue
            accepted, check = verify(candidate, gid, views, points[ids != int(gid)])
            entry.update(check)
            if accepted is None:
                continue
            accepted = accepted[cKDTree(local).query(accepted)[0] > .012]
            # Independent views must corroborate part of the proposed surface;
            # occlusion and absence of a detector label do not increase support.
            support = np.zeros(len(accepted), int)
            supporting_frames = []
            for fid in sorted(views.jobs):
                indices, _, _ = views.project(accepted, fid)
                support[indices] += 1
                if len(indices) >= 32:
                    supporting_frames.append(fid)
            corroborated = int(np.sum(support >= 3))
            entry.update(depth_corroborated_surface_points=corroborated,
                         depth_corroborating_frames=supporting_frames,
                         checked_frames_without_detection=True)
            if len(accepted) < 128 or corroborated < 128 or len(supporting_frames) < 3:
                entry['reason'] = '独立深度表面支持不足，推断保持未接受'
                continue
            additions.append(accepted)
            color = [int(gid)%255, (int(gid)//255)%255, (int(gid)//255**2)%255]
            addition_colors.append(np.tile(color, (len(accepted), 1))/255)
            entry.update(status='accepted', generated_points=len(accepted))
            node['geometry_hypothesis'] = dict(type='identity_prior_verified_rear_bounded_cuboid',
                measured=False, audit='verified_shape_audit.json', learned_text_conditioning=False)
        if additions:
            output = o3d.geometry.PointCloud()
            output.points = o3d.utility.Vector3dVector(np.concatenate([np.asarray(base.points), *additions]))
            output.colors = o3d.utility.Vector3dVector(np.concatenate([np.asarray(base.colors), *addition_colors]))
            path = context.scene/'instance_cloud_completed.ply'
            if not o3d.io.write_point_cloud(str(path), output):
                raise IOError('Could not publish verified shape hypotheses')
            owned = o3d.geometry.PointCloud()
            owned.points = o3d.utility.Vector3dVector(np.concatenate(additions))
            owned.colors = o3d.utility.Vector3dVector(np.concatenate(addition_colors))
            if not o3d.io.write_point_cloud(str(owned_path), owned):
                raise IOError('Could not publish owned shape evidence')
            context.graph_geometry = path
            graph_path.write_text(json.dumps(graph, indent=2)+'\n')
        elif owned_path.is_file():
            o3d.io.write_point_cloud(str(context.graph_geometry), base)
            owned_path.unlink()
            for node in nodes.values():
                if node.get('geometry_hypothesis', {}).get('type') == 'identity_prior_verified_rear_bounded_cuboid':
                    node.pop('geometry_hypothesis')
            graph_path.write_text(json.dumps(graph, indent=2)+'\n')
    report['accepted_objects'] = sum(entry['status'] == 'accepted' for entry in report['objects'])
    (context.scene/'verified_shape_audit.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
    context.event('最终实测物体的形状假设与全视角检验完成', accepted_objects=report['accepted_objects'])

"""Selective AdaPoinTr completion, with measured-surface and free-space guards."""
import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree


def construct(context):
    context.execute([str(context.repo/'.venv-completion/bin/python'), str(Path(__file__)),
        '--processed-scene', str(context.scene),
        *(['--manifest', str(context.manifest)] if context.manifest else ['--image-dir', str(context.image_dir)])],
        'AdaPoinTr 选择性形状补全与深度自由空间验收')
    report = json.loads((context.scene/'completion_audit.json').read_text())
    if report['accepted_objects']:
        context.graph_geometry = context.scene/'instance_cloud_completed.ply'


def accept_candidate(observed, candidate, views, track, fid_subset):
    """Never interpret an unobserved or occluded region as empty space."""
    from partaware.geometry import voxel_downsample
    candidate = voxel_downsample(candidate, .02)
    scale = max(float(np.linalg.norm(np.ptp(observed, axis=0))), .1)
    epsilon = max(.025, .015*scale)
    obs_dist = cKDTree(candidate).query(observed)[0]
    completeness = float(np.mean(cKDTree(observed).query(candidate)[0] <= epsilon))
    expansion = np.ptp(candidate, axis=0)/np.maximum(np.ptp(observed, axis=0), .03)
    evidence = {'estimated_completeness': completeness, 'anchor_coverage': float(np.mean(obs_dist <= epsilon)),
                'anchor_distance_p90_m': float(np.quantile(obs_dist, .9)), 'axis_expansion': expansion.tolist()}
    if evidence['anchor_coverage'] < .8 or not .1 <= completeness <= .85 or np.max(expansion) > 3:
        return None, {**evidence, 'reason': '形状先验与实测表面不相容，或不需要补全'}
    conflicts = np.zeros(len(candidate), int)
    examined = np.zeros(len(candidate), int)
    mask_conflicts = np.zeros(len(candidate), int)
    mask_examined = np.zeros(len(candidate), int)
    for fid in fid_subset:
        pose, depth, mask = views.get(fid)
        camera = (candidate-pose[:3, 3]) @ pose[:3, :3]
        z = camera[:, 2]
        uv = np.rint(camera[:, :2]/np.maximum(z[:, None], 1e-8)*[views.kd[0, 0], views.kd[1, 1]]
                     + [views.kd[0, 2], views.kd[1, 2]]).astype(int)
        valid = (z > 0) & (uv[:, 0] >= 0) & (uv[:, 1] >= 0) & (uv[:, 0] < depth.shape[1]) & (uv[:, 1] < depth.shape[0])
        ids = np.flatnonzero(valid)
        measured = depth[uv[ids, 1], uv[ids, 0]]/views.scale
        tolerance = .025+.01*measured
        known = measured > 0
        examined[ids[known]] += 1
        conflicts[ids[known & (z[ids] < measured-tolerance)]] += 1
        color_uv = np.rint(camera[:, :2]/np.maximum(z[:, None], 1e-8)*[views.kc[0, 0], views.kc[1, 1]]
                           + [views.kc[0, 2], views.kc[1, 2]]).astype(int)
        visible = known & (np.abs(z[ids]-measured) <= tolerance)
        projected = color_uv[ids]
        visible &= (projected[:, 0] >= 0) & (projected[:, 1] >= 0) & (projected[:, 0] < mask.shape[1]) & (projected[:, 1] < mask.shape[0])
        visible_ids = ids[visible]
        labels = mask[color_uv[visible_ids, 1], color_uv[visible_ids, 0]]
        own_ids = {int(o['local_id']) for o in track['observations'] if o['frame_id'] == fid}
        mask_examined[visible_ids] += 1
        different_object = (labels != 0) & ~np.isin(labels, list(own_ids))
        mask_conflicts[visible_ids[different_object]] += 1

    violation = float(np.mean(conflicts[examined > 0] > 0)) if np.any(examined > 0) else 1.
    evidence['free_space_violation_fraction'] = violation
    mask_violation = float(np.mean(mask_conflicts[mask_examined > 0] > 0)) if np.any(mask_examined > 0) else 0.
    evidence['visible_mask_conflict_fraction'] = mask_violation
    if mask_violation > .05:
        return None, {**evidence, 'reason': '生成可见表面与独立物体的掩码归属冲突'}

    if violation > .03:
        return None, {**evidence, 'reason': '生成表面侵入传感器观测到的自由空间'}
    additions = candidate[(conflicts == 0) & (mask_conflicts == 0) & (cKDTree(observed).query(candidate)[0] > .015)]
    if len(additions) < 32:
        return None, {**evidence, 'reason': '没有足够可信的新表面'}
    return additions, evidence


def align_candidate(observed, candidate):
    """Fit a bounded Sim(3) transform with observed-to-prior correspondences."""
    source = candidate.copy()
    sample = observed[::max(1, len(observed)//2048)]
    cumulative_scale = 1.
    for _ in range(8):
        distances, indices = cKDTree(candidate).query(sample)
        keep = distances <= np.quantile(distances, .8)
        a, b = candidate[indices[keep]], sample[keep]
        ca, cb = a.mean(0), b.mean(0)
        u, singular, vt = np.linalg.svd((a-ca).T @ (b-cb))
        correction = np.eye(3)
        correction[-1, -1] = np.linalg.det(u @ vt)
        rotation = u @ correction @ vt
        fitted_scale = float(np.sum(singular*np.diag(correction))/max(np.sum((a-ca)**2), 1e-8))
        fitted_scale = np.clip(fitted_scale, .8/cumulative_scale, 1.25/cumulative_scale)
        cumulative_scale *= fitted_scale
        candidate = (candidate-ca) @ rotation*fitted_scale+cb
    return candidate, {'sim3_scale': cumulative_scale,
                       'prior_center_shift_m': float(np.linalg.norm(candidate.mean(0)-source.mean(0)))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest')
    parser.add_argument('--image-dir')
    parser.add_argument('--processed-scene', required=True)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(repo/'scannet/script'))
    from partaware.geometry import load_capture
    from pipeline_components.instance_consensus import Views
    scene = Path(args.processed_scene).resolve()
    data, jobs, kd, kc, scale = load_capture(args.manifest, args.image_dir, scene/'refined_instance')
    generated_path = scene/'instance_cloud_completed.ply'
    if generated_path.exists():
        generated_path.unlink()
    if (data.get('world_frame') != 'hypersim_world_z_up' or data.get('length_unit') != 'meter'
            or not (scene/'object_tracks.json').is_file()):
        (scene/'completion_audit.json').write_text(json.dumps({'algorithm': 'official_adapointr_pcn_observation_constrained_v4',
            'accepted_objects': 0, 'objects': [], 'reason': '输入坐标未验证或原始融合没有观测轨迹，保留基础几何'}, ensure_ascii=False, indent=2)+'\n')
        return
    track_path = scene/'validated_object_tracks.json'
    if not track_path.exists():
        track_path = scene/'object_tracks.json'
    tracks = json.loads(track_path.read_text())
    records = {j['frame_id']: json.loads((scene/'refined_instance'/f"{j['frame_id']}_updated_instance.json").read_text()) for j in jobs}
    context = type('Context', (), {'scene': scene})()
    views = Views(context, jobs, kd, kc, scale, records)
    cloud = o3d.io.read_point_cloud(str(scene/'instance_cloud_cleaned.ply'))
    points, colors = np.asarray(cloud.points), np.asarray(cloud.colors)
    encoded = np.rint(colors*255).astype(int)
    ids = encoded[:, 0]+255*encoded[:, 1]+255*255*encoded[:, 2]
    lock_dir = Path.home()/'.cache/partaware-sg'
    lock_dir.mkdir(parents=True, exist_ok=True)
    audit, additions, addition_colors = [], [], []
    with (lock_dir/'gpu.lock').open('a') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        from pipeline_components.completion_model import CompletionModel
        model = CompletionModel(repo)
        for gid, track in tracks.items():
            observed = points[ids == int(gid)]
            name = max(track['name_votes'], key=track['name_votes'].get).lower()
            supported_class = any(word in name for word in ['chair', 'table', 'desk', 'cabinet', 'sofa', 'lamp'])
            entry = {'instance_id': gid, 'name': name, 'observed_points': len(observed)}
            eligibility = {'supported_training_family': supported_class, 'minimum_points': len(observed) >= 128,
                'minimum_views': len(track['observed_frames']) >= 3, 'minimum_quality': track['confidence'] >= .4,
                'minimum_reprojection': track.get('reprojection_support', 0) >= .55,
                'bounded_extent': bool(len(observed) and np.max(np.ptp(observed, axis=0)) <= 3)}
            entry['eligibility'] = eligibility
            if not all(eligibility.values()):
                audit.append({**entry, 'status': 'not_eligible', 'reason': '未满足模型准入条件',
                              'failed_conditions': [k for k, valid in eligibility.items() if not valid]})
                continue
            candidate = model.predict(observed)
            candidate, alignment = align_candidate(observed, candidate)
            selected = track['observed_frames'][::max(1, len(track['observed_frames'])//12)][:12]
            extra, evidence = accept_candidate(observed, candidate, views, track, selected)
            evidence.update(alignment)
            if extra is None:
                audit.append({**entry, 'status': 'rejected', **evidence})
                print('补全候选被观测约束拒绝', gid, evidence, flush=True)
                continue
            additions.append(extra)
            color = np.array([int(gid)%255, (int(gid)//255)%255, (int(gid)//255//255)%255])/255
            addition_colors.append(np.tile(color, (len(extra), 1)))
            audit.append({**entry, 'status': 'accepted', 'generated_points': len(extra), **evidence})
            print('补全候选通过验收', gid, '新增点', len(extra), flush=True)
    if additions:
        completed = o3d.geometry.PointCloud()
        completed.points = o3d.utility.Vector3dVector(np.concatenate([points, *additions]))
        completed.colors = o3d.utility.Vector3dVector(np.concatenate([colors, *addition_colors]))
        if not o3d.io.write_point_cloud(str(scene/'instance_cloud_completed.ply'), completed):
            raise IOError('Unable to publish completed geometry')
    report = {'algorithm': 'official_adapointr_pcn_observation_constrained_v4',
        'accepted_objects': sum(x['status'] == 'accepted' for x in audit), 'objects': audit,
        'observed_geometry': 'instance_cloud_cleaned.ply', 'generated_geometry': 'instance_cloud_completed.ply' if additions else None,
        'generated_points_are_observation_evidence': False,
        'model_sha256': hashlib.sha256((repo/'checkpoints/completion/AdaPoinTr_PCN.pth').read_bytes()).hexdigest(),
        'limitation': 'PCN geometry prior has no text conditioning; labels gate applicability only. No complete-mesh GT was used.'}
    (scene/'completion_audit.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
    print('选择性补全完成', '接受物体数', report['accepted_objects'], flush=True)


if __name__ == '__main__':
    main()

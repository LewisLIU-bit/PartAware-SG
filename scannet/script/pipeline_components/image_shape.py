"""Image-conditioned whole-object hypotheses anchored to measured RGB-D.

Generated surfaces never serve as tracking evidence. Geometry is published only
after all-view free-space and foreign-instance checks; rejected models are kept
in the audit, not turned into measurements. No class-specific dimensions exist.
"""
import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import sys
import itertools
import cv2
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree


def construct(context):
    command = [str(context.repo/'.venv-completion/bin/python'), str(Path(__file__)),
               '--processed-scene', str(context.scene)]
    command += ['--manifest', str(context.manifest)] if context.manifest else ['--image-dir', str(context.image_dir)]
    context.execute(command, '图像条件完整形状生成、实测尺度对齐与全部视角验收')
    audit = json.loads((context.scene/'image_shape_audit.json').read_text())
    if audit['accepted_objects']:
        context.graph_geometry = context.scene/'instance_cloud_completed.ply'


def initial_alignment(observed, prior, pose):
    """Use image axes for size, leaving unobserved thickness to the prior."""
    basis = np.column_stack([-pose[:3, 2], pose[:3, 0], -pose[:3, 1]])
    local = observed @ basis
    lower, upper = np.quantile(local, [.01, .99], axis=0)
    plower, pupper = np.quantile(prior, [.01, .99], axis=0)
    visible_scale = (upper-lower)[1:]/np.maximum((pupper-plower)[1:], 1e-5)
    # Fit measured image-plane dimensions independently. The hidden dimension
    # uses their geometric mean, rather than the observed near-zero thickness.
    depth_scale = float(np.sqrt(np.prod(visible_scale)))
    scales = np.array([depth_scale, *visible_scale])
    translation = (lower+upper)/2-scales*(plower+pupper)/2
    translation[0] = upper[0]-scales[0]*pupper[0]
    return (prior*scales+translation) @ basis.T, depth_scale


def aligned_hypotheses(observed, prior, pose):
    """Resolve canonical orientation without shrinking the hidden axis to zero."""
    from pipeline_components.shape_completion import align_candidate
    sample = observed[::max(1, len(observed)//2048)]
    choices = []
    for permutation in itertools.permutations(range(3)):
        for signs in itertools.product([-1, 1], repeat=3):
            rotation = np.eye(3)[list(permutation)]*np.asarray(signs)[:, None]
            if np.linalg.det(rotation) < .5:
                continue
            candidate, scale = initial_alignment(sample, prior @ rotation, pose)
            distance = cKDTree(candidate).query(sample)[0]
            loss = float(np.mean(np.sort(distance)[:max(16, int(.8*len(distance)))]))
            choices.append((loss, candidate, scale, rotation.tolist()))
    choices.sort(key=lambda value: value[0])
    fitted = []
    for _, candidate, scale, rotation in choices[:6]:
        candidate, transform = align_candidate(sample, candidate)
        candidate, deformation = anchor_deformation(sample, candidate)
        transform.update(deformation)
        distance = cKDTree(candidate).query(sample)[0]
        fitted.append((float(np.quantile(distance, .75)), candidate,
                       {'initial_metric_scale': scale, 'canonical_rotation': rotation, **transform}))
    return sorted(fitted, key=lambda value: value[0])


def anchor_deformation(observed, candidate):
    """Bound a smooth local displacement field; leave distant hidden shape alone.

    This is a project-specific kernel interpolation, not the CPD EM algorithm.
    Only generated coordinates move; observed coordinates are never replaced.
    """
    extent = max(float(np.linalg.norm(np.ptp(observed, axis=0))), .02)
    original = candidate.copy()
    sigma = .12*extent
    for _ in range(3):
        distance, indices = cKDTree(candidate).query(observed)
        good = distance < .12*extent
        if np.count_nonzero(good) < 16: break
        unique, inverse = np.unique(indices[good], return_inverse=True)
        displacement = np.zeros((len(unique), 3))
        counts = np.bincount(inverse)
        np.add.at(displacement, inverse, observed[good]-candidate[indices[good]])
        displacement /= counts[:, None]
        distances, neighbours = cKDTree(candidate[unique]).query(candidate, k=min(8, len(unique)))
        if distances.ndim == 1:
            distances, neighbours = distances[:, None], neighbours[:, None]
        weights = np.exp(-distances**2/(2*sigma**2))
        correction = np.sum(weights[:, :, None]*displacement[neighbours], axis=1)/np.maximum(weights.sum(1)[:, None], 1e-8)
        correction[distances[:, 0] > 2*sigma] = 0
        candidate += .7*correction
    displacement = np.linalg.norm(candidate-original, axis=1)
    if displacement.max() > .15*extent:
        return original, {'deformation_rejected': True, 'reason': 'Generated displacement exceeds the bounded shape correction'}
    return candidate, {'deformation_rejected': False,
        'deformation_max_m': float(displacement.max()), 'deformation_rms_m': float(np.sqrt(np.mean(displacement**2)))}


def evaluate_prior(observed, candidate, gid, views, foreign):
    from partaware.geometry import voxel_downsample
    extent = float(np.linalg.norm(np.ptp(observed, axis=0)))
    spacing = float(np.clip(extent/160., .002, .012))
    candidate = voxel_downsample(candidate, spacing)
    anchor_tol = max(.008, .035*extent)
    distance = cKDTree(candidate).query(observed[::max(1, len(observed)//4096)])[0]
    evidence = {'measured_anchor_fraction': float(np.mean(distance <= anchor_tol)),
                'anchor_distance_p90_m': float(np.quantile(distance, .9)),
                'anchor_tolerance_m': anchor_tol, 'sampling_resolution_m': spacing}
    if evidence['measured_anchor_fraction'] < .75:
        return None, {**evidence, 'reason': '图像生成形状无法覆盖足够实测锚点'}
    free = np.zeros(len(candidate), int)
    different = free.copy()
    visible = free.copy()
    for view_index, fid in enumerate(views.jobs):
        pose, raw, mask = views.get(fid)
        camera = (candidate-pose[:3, 3]) @ pose[:3, :3]
        z = camera[:, 2]
        duv = np.rint(camera[:, :2]/np.maximum(z[:, None], 1e-8)*[views.kd[0, 0], views.kd[1, 1]]
                       +[views.kd[0, 2], views.kd[1, 2]]).astype(int)
        ok = (z > 0) & (duv[:, 0] >= 0) & (duv[:, 1] >= 0) & (duv[:, 0] < raw.shape[1]) & (duv[:, 1] < raw.shape[0])
        indices = np.flatnonzero(ok)
        depth = raw[duv[indices, 1], duv[indices, 0]]/views.scale
        known = depth > 0
        indices, depth = indices[known], depth[known]
        tolerance = .012+.003*depth
        free[indices[z[indices] < depth-tolerance]] += 1
        uv = np.rint(camera[indices, :2]/z[indices, None]*[views.kc[0, 0], views.kc[1, 1]]
                     +[views.kc[0, 2], views.kc[1, 2]]).astype(int)
        measured = (np.abs(z[indices]-depth) <= tolerance) & (uv[:, 0] >= 0) & (uv[:, 1] >= 0)
        measured &= (uv[:, 0] < mask.shape[1]) & (uv[:, 1] < mask.shape[0])
        ix = indices[measured]
        mapping = {int(r['frame_instance_id']): str(r.get('instance_id', -1)) for r in views.records[fid]}
        labels = mask[uv[measured, 1], uv[measured, 0]]
        visible[ix] += 1
        different[ix[np.array([mapping.get(int(k), '-1') not in [gid, '-1'] for k in labels], bool)]] += 1
        # Contradiction counts only increase; a failed whole-model gate cannot
        # become valid after examining additional views. Accepted shapes still
        # require every input frame, so this changes cost rather than evidence.
        if np.mean(free > 0) > .03 or np.mean(different >= 2) > .05:
            return None, {**evidence, 'free_space_fraction': float(np.mean(free > 0)),
                'foreign_fraction': float(np.mean(different >= 2)),
                'checked_frames': view_index+1, 'early_rejected': True,
                'reason': '已观测的模型冲突超过验收上限，追加视角不能撤销已有反证'}

    collision = cKDTree(foreign).query(candidate)[0] < spacing if len(foreign) else np.zeros(len(candidate), bool)
    from pipeline_components.room_envelope import outside_envelope
    structural_veto = outside_envelope(candidate, getattr(views, 'room_envelope', []))
    veto = (free > 0) | (different >= 2) | collision | structural_veto
    evidence.update(free_space_fraction=float(np.mean(free > 0)),
        repeated_free_space_fraction=float(np.mean(free >= 2)),
        foreign_fraction=float(np.mean((different >= 2) | collision)),
        total_rejected_surface_fraction=float(veto.mean()), checked_frames=len(views.jobs))
    evidence['room_envelope_violation_fraction'] = float(structural_veto.mean())
    # A contradicted whole model cannot be rescued by retaining unseen remnants.
    if (evidence['free_space_fraction'] > .03 or evidence['foreign_fraction'] > .05
            or evidence['total_rejected_surface_fraction'] > .10):
        return None, {**evidence, 'reason': '完整生成模型与实测场景矛盾过多，拒绝裁剪后残余隐藏点'}
    # Measured visible surfaces replace approximate generated visible surfaces.
    # Keep only occluded/unseen prior surfaces; do not publish a model point in
    # known free space even once. This is visibility carving, not relaxed depth.
    hidden = visible == 0
    additions = candidate[~veto & hidden & (cKDTree(observed).query(candidate)[0] > spacing*1.5)]
    evidence.update(visibility_carved_points=int(veto.sum()),
        published_free_space_points=0, generated_visible_surfaces_published=False,
        hidden_hypothesis_points=len(additions), measured_surfaces_preserved=True)
    if len(additions) and np.max(np.ptp(np.concatenate([observed, additions]), axis=0)) > 2*max(np.max(np.ptp(observed, axis=0)), .05):
        return None, {**evidence, 'reason': '未知形状超出实测尺度所支持的有界范围'}
    if len(additions) < 128:
        return None, {**evidence, 'reason': '通过验收的新增隐藏表面不足'}
    return additions, evidence



def remove_owned_surface(base, owned, measured):
    """Replace only this worker's earlier hypotheses on a repeated build.

    Exact XYZ and encoded instance color form a row key. Measured rows are
    protected even if a newly observed surface coincides with an old hypothesis.
    """
    def keys(cloud):
        values = np.ascontiguousarray(np.column_stack([
            np.asarray(cloud.points), np.rint(np.asarray(cloud.colors)*255)]))
        return values.view(np.dtype([('row', np.float64, (6,))])).reshape(-1)
    base_keys = keys(base)
    remove = np.isin(base_keys, keys(owned)) & ~np.isin(base_keys, keys(measured))
    result = o3d.geometry.PointCloud()
    result.points = o3d.utility.Vector3dVector(np.asarray(base.points)[~remove])
    result.colors = o3d.utility.Vector3dVector(np.asarray(base.colors)[~remove])
    return result, int(remove.sum())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--processed-scene', required=True)
    parser.add_argument('--manifest'); parser.add_argument('--image-dir')
    parser.add_argument('--maximum-objects', type=int, default=0, help='Bounded isolated research preview only')
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(repo/'scannet/script'))
    from partaware.geometry import load_capture
    from pipeline_components.instance_consensus import Views
    from pipeline_components.proposal_validation import masked_crop
    scene = Path(args.processed_scene).resolve()
    data, jobs, kd, kc, scale = load_capture(args.manifest, args.image_dir, scene/'refined_instance')
    audit = {'algorithm': 'Hunyuan3D_mini_image_anchored_visibility_fused_v12', 'objects': [],
             'accepted_objects': 0, 'ground_truth_used': False, 'qwen_api_calls': 0,
             'generated_points_are_measurements': False, 'component_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    if data.get('world_frame') != 'hypersim_world_z_up' or not (scene/'validated_object_tracks.json').is_file():
        audit['reason'] = '缺少已验证坐标或轨迹；保留基础接口'
        (scene/'image_shape_audit.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2)+'\n'); return
    tracks = json.loads((scene/'validated_object_tracks.json').read_text())
    records = {j['frame_id']: json.loads((scene/'refined_instance'/f"{j['frame_id']}_updated_instance.json").read_text()) for j in jobs}
    views = Views(type('Context', (), {'scene': scene})(), jobs, kd, kc, scale, records)
    graph = json.loads((scene/'topology_map.json').read_text()); nodes = graph['object_nodes']['nodes']
    cloud = o3d.io.read_point_cloud(str(scene/'instance_cloud_cleaned.ply'))
    observed = np.asarray(cloud.points); colors = np.rint(np.asarray(cloud.colors)*255).astype(int)
    ids = colors[:, 0]+255*colors[:, 1]+255**2*colors[:, 2]
    from pipeline_components.room_envelope import fit_envelope
    background_path = scene/'instance_cloud_with_background.ply'
    views.room_envelope = []
    if background_path.is_file():
        native = o3d.io.read_point_cloud(str(background_path))
        native_colors = np.rint(np.asarray(native.colors)*255).astype(int)
        background = np.asarray(native.points)[np.all(native_colors == 0, axis=1)]
        cameras = np.array([np.loadtxt(job['pose'])[:3, 3] for job in jobs])
        views.room_envelope = fit_envelope(background, observed, cameras)
    audit['measured_room_envelope'] = views.room_envelope
    base_path = scene/'instance_cloud_completed.ply'
    base = o3d.io.read_point_cloud(str(base_path if base_path.exists() else scene/'instance_cloud_cleaned.ply'))
    owned_path = scene/'image_shape_generated.ply'
    replaced_points = 0
    if owned_path.is_file():
        base, replaced_points = remove_owned_surface(base, o3d.io.read_point_cloud(str(owned_path)), cloud)
        for node in nodes.values():
            if node.get('geometry_hypothesis', {}).get('type') == 'image_conditioned_hunyuan_mini_two_view':
                node.pop('geometry_hypothesis')
    audit['replaced_owned_hypothesis_points'] = replaced_points
    cache = scene/'image_shape_cache'; cache.mkdir(exist_ok=True)
    digest = hashlib.sha256()
    with (repo/'checkpoints/hunyuan_mini/model.fp16.safetensors').open('rb') as weights_file:
        for chunk in iter(lambda: weights_file.read(8*1024*1024), b''):
            digest.update(chunk)
    weights_sha = digest.hexdigest()
    audit['model_sha256'] = weights_sha
    model = None; additions = []; color_additions = []
    lock_path = Path.home()/'.cache/partaware-sg/gpu.lock'
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open('a') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        for gid, node in nodes.items():
            if args.maximum_objects and len(audit['objects']) >= args.maximum_objects:
                break
            track = tracks.get(gid, {}); local = observed[ids == int(gid)]
            entry = {'instance_id': gid, 'name': node['name'], 'measured_points': len(local), 'status': 'not_eligible'}
            audit['objects'].append(entry)
            if len(local) < 64 or len(track.get('observed_frames', [])) < 3 or track.get('confidence', 0) < .3:
                entry['reason'] = '独立实测点、视角或置信度不足'; continue
            centered = local-local.mean(0)
            _, _, principal = np.linalg.svd(centered[::max(1, len(centered)//4096)], full_matrices=False)
            spans = np.diff(np.quantile(centered @ principal.T, [.02, .98], axis=0), axis=0)[0]
            ratio = float(spans.min()/max(spans.max(), 1e-8))
            entry['observed_thin_axis_ratio'] = ratio
            if ratio >= .15:
                entry['reason'] = '实测点云已具有三维跨度，不因形状先验强行扩张'; continue
            crops = []
            for observation in track['observations']:
                fid = observation['frame_id']; pose, _, mask = views.get(fid)
                region = mask == int(observation['local_id']); area = int(region.sum())
                if area >= 64:
                    crops.append((area, fid, region, pose))
            if not crops:
                entry['reason'] = '没有有效的原图实例区域'; continue
            crops.sort(key=lambda value: -value[0])
            distinct = []
            for crop in crops:
                if not distinct:
                    distinct.append(crop); continue
                first_pose = distinct[0][3]
                angle = np.arccos(np.clip(first_pose[:3, 2] @ crop[3][:3, 2], -1, 1))
                if angle >= np.deg2rad(10) or np.linalg.norm(first_pose[:3, 3]-crop[3][:3, 3]) >= .15:
                    distinct.append(crop); break
            crops = distinct
            if len(crops) < 2:
                entry['reason'] = '缺少两个不同视角的图像先验核验'; continue
            accepted = []
            entry['candidates'] = []
            for area, fid, region, pose in crops[:2]:
                image = masked_crop(cv2.imread(str(views.jobs[fid]['rgb'])), region)
                if image is None: continue
                # Pad identically for every class; do not stretch missing depth.
                border = max(4, int(.09*image.shape[0]))
                image = cv2.copyMakeBorder(image, border, border, border, border, cv2.BORDER_CONSTANT, value=(127, 127, 127))
                digest = hashlib.sha256(image.tobytes()+weights_sha.encode()).hexdigest()
                cached = cache/f'{digest}.npz'
                if cached.exists():
                    saved = np.load(cached); prior = saved['points']; metadata = json.loads(str(saved['metadata']))
                else:
                    if model is None:
                        from pipeline_components.image_shape_model import DiffusionShapeModel
                        model = DiffusionShapeModel(repo)
                    from PIL import Image
                    try:
                        prior, metadata = model.predict(Image.fromarray(image))
                    except ValueError as error:
                        entry['candidates'].append({'frame_id': fid, 'reason': str(error)}); continue
                    np.savez_compressed(cached, points=prior, metadata=json.dumps(metadata))
                extra = None
                for _, candidate, alignment in aligned_hypotheses(local, prior, pose):
                    extra, evidence = evaluate_prior(local, candidate, gid, views, observed[ids != int(gid)])
                    entry['candidates'].append({'frame_id': fid, 'crop_pixels': area, **alignment, **metadata, **evidence})
                    if extra is not None:
                        break
                if extra is None:
                    # Two accepted independent images are mandatory. Once the
                    # first selected image fails, this pair cannot be accepted.
                    entry['status'] = 'rejected'; break
                accepted.append((extra, fid))
            if len(accepted) == 2:
                first, second = accepted[0][0], accepted[1][0]
                tolerance = max(.006, .025*float(np.linalg.norm(np.ptp(local, axis=0))))
                da = cKDTree(second).query(first)[0]; db = cKDTree(first).query(second)[0]
                agreement = min(float(np.mean(da <= tolerance)), float(np.mean(db <= tolerance)))
                entry['two_view_shape_agreement'] = agreement
                extra = first[da <= tolerance]
                if agreement >= .75 and len(extra) >= 128:
                    additions.append(extra)
                    color = np.array([int(gid)%255, int(gid)//255%255, int(gid)//255**2%255])/255.
                    color_additions.append(np.tile(color, (len(extra), 1)))
                    entry.update(status='accepted', generated_points=len(extra), accepted_frames=[x[1] for x in accepted])
                    node['geometry_hypothesis'] = {'type': 'image_conditioned_hunyuan_mini_two_view', 'measured': False, 'audit': 'image_shape_audit.json'}
                else:
                    entry.update(status='rejected', reason='两个图像先验的隐藏形状不一致')
            else:
                entry.update(status='rejected', reason='没有两个通过完整模型实测核验的独立图像先验')
            audit['accepted_objects'] = sum(x['status'] == 'accepted' for x in audit['objects'])
            audit['status'] = 'in_progress'
            (scene/'image_shape_audit.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2)+'\n')
            print('图像形状验收', gid, node['name'], entry['status'], flush=True)
    if additions:
        merged = o3d.geometry.PointCloud()
        merged.points = o3d.utility.Vector3dVector(np.concatenate([np.asarray(base.points), *additions]))
        merged.colors = o3d.utility.Vector3dVector(np.concatenate([np.asarray(base.colors), *color_additions]))
        if not o3d.io.write_point_cloud(str(base_path), merged): raise IOError(base_path)
        owned = o3d.geometry.PointCloud()
        owned.points = o3d.utility.Vector3dVector(np.concatenate(additions))
        owned.colors = o3d.utility.Vector3dVector(np.concatenate(color_additions))
        if not o3d.io.write_point_cloud(str(owned_path), owned): raise IOError(owned_path)
        (scene/'topology_map.json').write_text(json.dumps(graph, indent=2)+'\n')
    elif replaced_points:
        if not o3d.io.write_point_cloud(str(base_path), base): raise IOError(base_path)
        owned_path.unlink()
        (scene/'topology_map.json').write_text(json.dumps(graph, indent=2)+'\n')
    audit['accepted_objects'] = sum(x['status'] == 'accepted' for x in audit['objects'])
    audit['status'] = 'completed'
    audit['limitation'] = '单图生成模型不直接接收残缺点云或类别文本；隐藏表面仅为待核验假设，不能保证恢复真实形状。'
    (scene/'image_shape_audit.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2)+'\n')


if __name__ == '__main__':
    main()

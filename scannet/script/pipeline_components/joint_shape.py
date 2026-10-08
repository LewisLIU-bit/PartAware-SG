"""Joint measured-point, RGB and cached-text whole-object completion."""
import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import sys
import cv2
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline_components.image_shape import evaluate_prior, remove_owned_surface


def construct(context):
    command = [str(context.repo/'.venv-completion/bin/python'), str(Path(__file__)),
               '--processed-scene', str(context.scene)]
    command += ['--manifest', str(context.manifest)] if context.manifest else ['--image-dir', str(context.image_dir)]
    context.execute(command, '残缺点云、图像和缓存身份联合条件补全与多视角验收')
    audit = json.loads((context.scene/'joint_shape_audit.json').read_text())
    if audit['accepted_objects']:
        context.graph_geometry = context.scene/'instance_cloud_completed.ply'


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
    audit = {'algorithm': 'MGPC_partial_RGB_cached_text_visibility_v12', 'objects': [],
             'accepted_objects': 0, 'ground_truth_used': False, 'qwen_api_calls': 0,
             'generated_points_are_measurements': False, 'component_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    if data.get('world_frame') != 'hypersim_world_z_up' or not (scene/'validated_object_tracks.json').is_file():
        audit['reason'] = '缺少已验证坐标或轨迹；保留基础接口'
        (scene/'joint_shape_audit.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2)+'\n'); return
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
    owned_path = scene/'joint_shape_generated.ply'
    replaced_points = 0
    if owned_path.is_file():
        base, replaced_points = remove_owned_surface(base, o3d.io.read_point_cloud(str(owned_path)), cloud)
        for node in nodes.values():
            if node.get('geometry_hypothesis', {}).get('type') == 'joint_MGPC_measured_RGB_text_two_view':
                node.pop('geometry_hypothesis')
    audit['replaced_owned_hypothesis_points'] = replaced_points
    cache = scene/'joint_shape_cache'; cache.mkdir(exist_ok=True)
    digest = hashlib.sha256()
    with (repo/'checkpoints/mgpc/ckpt_8192.pt').open('rb') as weights_file:
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
                from partaware.geometry import project_mask, voxel_downsample
                job = views.jobs[fid]
                native_points, _ = project_mask(region, cv2.imread(str(job['rgb'])),
                    cv2.imread(str(job['depth']), -1), pose, kd, kc, scale, stride=1, max_depth=0)
                native_points = voxel_downsample(native_points, .002)
                if len(native_points) < 64:
                    entry['candidates'].append({'frame_id': fid, 'reason': 'Insufficient paired RGB-D input'})
                    continue
                digest = hashlib.sha256(image.tobytes()+native_points.astype(np.float32).tobytes()+pose.tobytes()+node['name'].encode()+weights_sha.encode()+Path(__file__).with_name('joint_shape_model.py').read_bytes()).hexdigest()
                cached = cache/f'{digest}.npz'
                if cached.exists():
                    saved = np.load(cached); prior = saved['points']; metadata = json.loads(str(saved['metadata']))
                else:
                    if model is None:
                        from pipeline_components.joint_shape_model import JointShapeModel
                        model = JointShapeModel(repo)
                    from PIL import Image
                    try:
                        prior, metadata = model.predict(native_points, Image.fromarray(image), node['name'], pose)
                    except ValueError as error:
                        entry['candidates'].append({'frame_id': fid, 'reason': str(error)}); continue
                    np.savez_compressed(cached, points=prior, metadata=json.dumps(metadata))
                extra = None
                from pipeline_components.joint_constraints import measured_surface_projection
                corrected, correction = measured_surface_projection(local, prior, gid, views)
                if corrected is None:
                    entry['candidates'].append({'frame_id': fid, **metadata, **correction})
                    entry['status'] = 'rejected'; break
                for candidate in [corrected]:
                    alignment = {'alignment': 'network_camera_frame_input_centric', **correction}
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
                    node['geometry_hypothesis'] = {'type': 'joint_MGPC_measured_RGB_text_two_view', 'measured': False, 'audit': 'joint_shape_audit.json'}
                else:
                    entry.update(status='rejected', reason='两个图像先验的隐藏形状不一致')
            else:
                entry.update(status='rejected', reason='没有两个通过完整模型实测核验的独立图像先验')
            audit['accepted_objects'] = sum(x['status'] == 'accepted' for x in audit['objects'])
            audit['status'] = 'in_progress'
            (scene/'joint_shape_audit.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2)+'\n')
            print('联合条件形状验收', gid, node['name'], entry['status'], flush=True)
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
    audit['limitation'] = '网络直接接收残缺点云、图像与缓存语义；隐藏表面仍为预测假设，不能保证其等于真实形状。'
    (scene/'joint_shape_audit.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2)+'\n')


if __name__ == '__main__':
    main()

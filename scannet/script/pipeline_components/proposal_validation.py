"""Observed proposal consolidation and masked-CLIP SMS validation.

Adapts inclusion removal and SMS from Details Matter (ICCV 2025), and
under-segmentation evidence from MaskClustering. No annotation is read.
"""
import argparse
import copy
import fcntl
import hashlib
import json
from pathlib import Path
import sys
import cv2
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree


def construct(context):
    command = [sys.executable, str(Path(__file__)), '--processed-scene', str(context.scene)]
    command += ['--manifest', str(context.manifest)] if context.manifest else ['--image-dir', str(context.image_dir)]
    context.execute(command, '多视角候选验收、包含残片合并与标准化语义过滤')


def standardized_maximum(similarities):
    """Standardize each proposal's best class with that class's scene statistics."""
    best = similarities.argmax(axis=1)
    mean, deviation = similarities.mean(0), similarities.std(0)
    return (similarities[np.arange(len(best)), best]-mean[best])/np.maximum(deviation[best], 1e-6)


def inclusion(points, container, tolerance=.015):
    lower, upper = container.min(0)-tolerance, container.max(0)+tolerance
    return float(np.mean(np.all((points >= lower) & (points <= upper), axis=1)))


def sample_frames(frames, maximum=24):
    frames = sorted(set(frames))
    return [frames[i] for i in np.linspace(0, len(frames)-1, min(len(frames), maximum)).astype(int)] if frames else []


def resolve(gid, remap):
    while gid in remap:
        gid = remap[gid]
    return gid


def raw_pair_evidence(a, b, views, frames):
    support, conflict, visible = 0, 0, 0
    for fid in sample_frames(frames):
        _, wa, la = views.project(a, fid)
        _, wb, lb = views.project(b, fid)
        if len(la) < 16 or len(lb) < 16:
            continue
        visible += 1
        labels = (set(la) | set(lb))-{0}
        pa = {int(k): float(wa[la == k].sum()/wa.sum()) for k in labels}
        pb = {int(k): float(wb[lb == k].sum()/wb.sum()) for k in labels}
        if max((min(pa[k], pb[k]) for k in labels), default=0) >= .8:
            support += 1
        elif labels:
            ka, kb = max(pa, key=pa.get), max(pb, key=pb.get)
            if ka != kb and pa[ka] >= .8 and pb[kb] >= .8:
                conflict += 1
    return {'whole_mask_support_views': support, 'independent_separation_views': conflict,
            'joint_visible_views': visible, 'whole_mask_consensus': support/max(visible, 1)}


def masked_crop(image, mask):
    ys, xs = np.where(mask)
    if len(xs) < 32:
        return None
    x0, y0, x1, y1 = xs.min(), ys.min(), xs.max()+1, ys.max()+1
    content = cv2.cvtColor(image[y0:y1, x0:x1], cv2.COLOR_BGR2RGB)
    content[~mask[y0:y1, x0:x1]] = 127
    side = max(content.shape[:2])
    square = np.full((side, side, 3), 127, np.uint8)
    dy, dx = (side-content.shape[0])//2, (side-content.shape[1])//2
    square[dy:dy+content.shape[0], dx:dx+content.shape[1]] = content
    return square


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest')
    parser.add_argument('--image-dir')
    parser.add_argument('--processed-scene', required=True)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(repo/'scannet/script'))
    from partaware.geometry import load_capture, voxel_downsample
    from pipeline_components.instance_consensus import Views
    scene = Path(args.processed_scene).resolve()
    data, jobs, kd, kc, scale = load_capture(args.manifest, args.image_dir, scene/'refined_instance')
    tracks_path = scene/'object_tracks.json'
    if not tracks_path.is_file():
        (scene/'object_validation.json').write_text(json.dumps({'algorithm': 'masked_clip_sms_inclusion_v5',
            'reason': '原始融合没有轨迹证据，保留基础输出', 'qwen_api_calls': 0}, ensure_ascii=False, indent=2)+'\n')
        return
    tracks = json.loads(tracks_path.read_text())
    graph_path = scene/'topology_map_cleaned.json'
    graph = json.loads(graph_path.read_text())
    nodes = graph['object_nodes']['nodes'] or {}
    cloud_path = scene/'instance_cloud_cleaned.ply'
    cloud = o3d.io.read_point_cloud(str(cloud_path))
    points = np.asarray(cloud.points)
    colors = np.rint(np.asarray(cloud.colors)*255).astype(int)
    ids = colors[:, 0]+255*colors[:, 1]+255**2*colors[:, 2]
    geometry = {gid: points[ids == int(gid)] for gid in nodes}
    records, local_mapping = {}, {}
    # Restore raw fusion ownership so graph-stage recovery is reproducible.
    for gid, track in tracks.items():
        for observation in track['observations']:
            local_mapping.setdefault(observation['frame_id'], {})[int(observation['local_id'])] = int(gid)
    for job in jobs:
        fid = job['frame_id']
        values = json.loads((scene/'refined_instance'/f'{fid}_instance.json').read_text())
        for record in values:
            record['instance_id'] = local_mapping.get(fid, {}).get(int(record['frame_instance_id']), -1)
        records[fid] = values
    context = type('Context', (), {'scene': scene})()
    views = Views(context, jobs, kd, kc, scale, records)
    class OriginalViews(Views):
        def get(self, fid):
            pose, depth, current = super().get(fid)
            cache = scene/'frontend_cache'/f'{fid}.npz'
            if fid not in self.source_masks:
                if cache.is_file():
                    with np.load(cache) as value:
                        self.source_masks[fid] = value['source_mask'].copy()
                else:
                    self.source_masks[fid] = current
            return pose, depth, self.source_masks[fid]
    original = OriginalViews(context, jobs, kd, kc, scale, records)
    original.source_masks = {}
    remap, audit = {}, []
    # Prefer a fuller observed object, rather than a high-scoring small fragment.
    keys = sorted(nodes, key=lambda gid: -len(geometry[gid]))
    for big in keys:
        if big in remap:
            continue
        for small in keys:
            if small == big or small in remap or len(geometry[small]) > len(geometry[big]):
                continue
            bbox_inclusion = inclusion(geometry[small], geometry[big])
            if bbox_inclusion < .95:
                continue
            a = np.asarray(nodes[big]['text_embedding'])
            b = np.asarray(nodes[small]['text_embedding'])
            similarity = float(a @ b/max(np.linalg.norm(a)*np.linalg.norm(b), 1e-8))
            if similarity < .6:
                continue
            coverage = float(np.mean(cKDTree(geometry[big]).query(geometry[small])[0] <= .03))
            reverse_coverage = float(np.mean(cKDTree(geometry[small]).query(geometry[big])[0] <= .03))
            strong_surface = similarity >= .8 and min(coverage, reverse_coverage) >= .5 and max(coverage, reverse_coverage) >= .8
            evidence = raw_pair_evidence(geometry[big][::max(1, len(geometry[big])//2048)],
                geometry[small][::max(1, len(geometry[small])//2048)], original,
                tracks[big]['observed_frames']+tracks[small]['observed_frames'])
            if evidence['independent_separation_views'] and not strong_surface:
                continue
            if not (strong_surface or coverage >= .99 or (evidence['whole_mask_support_views'] >= 3 and evidence['whole_mask_consensus'] >= .8)):
                continue
            remap[small] = big
            # Inclusion removal keeps the full measured geometry; duplicate fringe
            # points must not expand an already supported box. Complementary
            # observations may add missing surfaces only within the containing region.
            if not strong_surface and coverage < .99:
                geometry[big] = voxel_downsample(np.concatenate([geometry[big], geometry[small]]), .01)
            nb, ns = len(tracks[big]['observations']), len(tracks[small]['observations'])
            for field in ['visual_embedding', 'text_embedding']:
                nodes[big][field] = ((nb*np.asarray(nodes[big][field])+ns*np.asarray(nodes[small][field]))/(nb+ns)).tolist()
            tracks[big]['observations'] += tracks[small]['observations']
            tracks[big]['observed_frames'] = sorted(set(tracks[big]['observed_frames']+tracks[small]['observed_frames']))
            tracks[big]['confidence'] = (nb*tracks[big]['confidence']+ns*tracks[small]['confidence'])/(nb+ns)
            audit.append({'message': '完整实例吸收同义包含残片', 'source_id': small, 'target_id': big,
                          'surface_inclusion_3cm': coverage, 'reverse_surface_coverage_3cm': reverse_coverage,
                          'strong_surface_duplicate': strong_surface, 'bbox_inclusion': bbox_inclusion, 'semantic': similarity, **evidence})
    active = [gid for gid in keys if gid not in remap]
    metrics, crops = {}, []
    for gid in active:
        track, local = tracks[gid], geometry[gid]
        sample = local[::max(1, len(local)//2048)]
        visible, mixed, support = 0, 0, 0
        crop_views = []
        selected = sample_frames(track['observed_frames'], 20)
        selected += [f for f in sample_frames(views.jobs, 20) if f not in selected]
        for fid in selected:
            indices, weights, labels = views.project(sample, fid)
            fraction = len(indices)/max(len(sample), 1)
            if len(indices) < 16 or fraction < .3:
                continue
            visible += 1
            mapping = {r['frame_instance_id']: resolve(str(r['instance_id']), remap) for r in records[fid]}
            globals = np.array([mapping.get(int(k), '-1') for k in labels])
            own = globals == gid
            mass = {g: float(weights[globals == g].sum()/weights.sum()) for g in set(globals) if g != '-1'}
            distribution = sorted(mass.values(), reverse=True)
            mixed += int(len(distribution) >= 2 and distribution[1] >= .2 and distribution[0] < .8)
            support += int(float(weights[own].sum()/weights.sum()) >= .4)
            if fid in track['observed_frames'] and np.any(own):
                crop_views.append((fraction, fid))
        metrics[gid] = {'visible_views': visible, 'support_views': support,
            'mixed_views': mixed, 'undersegmented_view_fraction': mixed/max(visible, 1), 'view_detection_rate': support/max(visible, 1)}
        for fraction, fid in sorted(crop_views, reverse=True)[:5]:
            image = cv2.imread(str(views.jobs[fid]['rgb']))
            mask = views.get(fid)[2]
            member_ids = [r['frame_instance_id'] for r in records[fid] if resolve(str(r['instance_id']), remap) == gid]
            crop = masked_crop(image, np.isin(mask, member_ids))
            if crop is not None:
                crops.append((gid, fraction, crop))
    vocabulary = {key for value in json.loads((repo/'scannet/script/ram/hypersim_indoor_57.json').read_text()) for key in value}
    for values in records.values():
        vocabulary.update(r['object_name'].replace('_', ' ').strip().lower() for r in values)
    names = sorted(vocabulary-{'wall', 'floor', 'ceiling'})
    import torch
    import clip
    from PIL import Image
    lock_dir = Path.home()/'.cache/partaware-sg'
    lock_dir.mkdir(parents=True, exist_ok=True)
    embeddings, counts = {}, {}
    with (lock_dir/'gpu.lock').open('a') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model, preprocess = clip.load('RN50', device=device, download_root=str(repo/'checkpoints/clip'))
        with torch.inference_mode():
            text = model.encode_text(clip.tokenize([f'a photo of a {name} in a room.' for name in names+
                                                  ['wall', 'floor', 'ceiling', 'background texture']], truncate=True).to(device)).float()
            text /= text.norm(dim=1, keepdim=True)
            for start in range(0, len(crops), 32):
                batch = crops[start:start+32]
                image = torch.stack([preprocess(Image.fromarray(crop)) for _, _, crop in batch]).to(device)
                features = model.encode_image(image).float()
                features /= features.norm(dim=1, keepdim=True)
                for (gid, weight, _), vector in zip(batch, features.cpu().numpy()):
                    embeddings[gid] = embeddings.get(gid, np.zeros(1024))+vector*weight
                    counts[gid] = counts.get(gid, 0)+1
        text = text.cpu().numpy()
    valid = [gid for gid in active if gid in embeddings]
    if valid:
        features = np.array([embeddings[gid]/max(np.linalg.norm(embeddings[gid]), 1e-8) for gid in valid])
        scores = features @ text.T
        sms = standardized_maximum(scores[:, :len(names)]) if len(valid) >= 5 else np.zeros(len(valid))
        for i, gid in enumerate(valid):
            metrics[gid].update(sms=float(sms[i]), semantic_best_class=names[int(scores[i, :len(names)].argmax())],
                semantic_object_score=float(scores[i, :len(names)].max()),
                semantic_background_score=float(scores[i, len(names):].max()), semantic_views=counts[gid])
    survivors = []
    for gid in active:
        m = metrics[gid]
        reasons = []
        if m['mixed_views'] >= 3 and m['undersegmented_view_fraction'] > .2:
            reasons.append('多个独立实例混合于一个候选')
        if m['visible_views'] >= 5 and m['view_detection_rate'] < .2:
            reasons.append('可见视角缺少足够检测支持')
        reliable_geometry = tracks[gid].get('reprojection_support', 0) >= .6
        m['strong_observed_geometry'] = reliable_geometry
        if m.get('sms', 0) < 0 and not reliable_geometry:
            reasons.append('物体语义分数低于该类别场景平均水平')
        if ('semantic_object_score' in m and m['semantic_object_score']-m['semantic_background_score'] < .02
                and not reliable_geometry):
            reasons.append('物体与结构背景的语义对比不足')
        audit.append({'message': '物体候选验收', 'instance_id': gid, 'accepted': not reasons,
                      'reasons': reasons, **m})
        if not reasons:
            survivors.append(gid)
            tracks[gid]['proposal_validation'] = m
            tracks[gid]['point_count'] = len(geometry[gid])
    if not survivors:
        raise RuntimeError('No object passed observed proposal validation; inspect evidence')
    for fid, values in records.items():
        for record in values:
            gid = resolve(str(record['instance_id']), remap)
            record['instance_id'] = int(gid) if gid in survivors else -1
        (scene/'refined_instance'/f'{fid}_updated_instance.json').write_text(json.dumps(values, indent=2)+'\n')
    output_points = np.concatenate([geometry[gid] for gid in survivors])
    output_colors = np.concatenate([np.tile([int(gid)%255, (int(gid)//255)%255, (int(gid)//255**2)%255],
                                         (len(geometry[gid]), 1))/255 for gid in survivors])
    output = o3d.geometry.PointCloud()
    output.points = o3d.utility.Vector3dVector(output_points)
    output.colors = o3d.utility.Vector3dVector(output_colors)
    if not o3d.io.write_point_cloud(str(cloud_path), output):
        raise IOError('Could not publish validated observed geometry')
    graph['object_nodes']['nodes'] = {gid: nodes[gid] for gid in survivors}
    graph_path.write_text(json.dumps(graph, indent=2)+'\n')
    (scene/'validated_object_tracks.json').write_text(json.dumps({gid: tracks[gid] for gid in survivors}, indent=2)+'\n')
    report = {'algorithm': 'masked_clip_sms_inclusion_v5', 'candidate_objects': len(keys),
        'merged_fragments': len(remap), 'accepted_objects': len(survivors),
        'rejected_objects': len(active)-len(survivors), 'remap': remap, 'qwen_api_calls': 0,
        'thresholds': {'bbox_inclusion': .95, 'surface_inclusion': .99, 'background_margin': .02, 'whole_mask_consensus': .8, 'sms': 0, 'mixed_view_fraction': .2, 'view_detection_rate': .2, 'strong_reprojection_support': .6},
        'semantic_model': 'OpenAI RN50 masked square crops; not Alpha-CLIP', 'vocabulary': names,
        'model_sha256': hashlib.sha256((repo/'checkpoints/clip/RN50.pt').read_bytes()).hexdigest(), 'evidence': audit}
    (scene/'object_validation.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
    print('候选验收完成', {k: report[k] for k in ['candidate_objects', 'merged_fragments', 'accepted_objects', 'rejected_objects']}, flush=True)


if __name__ == '__main__':
    main()

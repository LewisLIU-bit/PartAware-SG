"""OP3DSG-inspired object fusion exporting the original ScanNet-SG interface."""
from dataclasses import dataclass, field
from collections import Counter
import csv
import json
from pathlib import Path
import cv2
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree
from partaware.geometry import load_capture, project_mask, voxel_downsample
from partaware.fusion import normalize, color_histogram


@dataclass
class Observation:
    record: dict
    mask: np.ndarray
    points: np.ndarray
    color: np.ndarray
    semantic: np.ndarray
    visual: np.ndarray


@dataclass
class Track:
    id: int
    points: np.ndarray
    semantic_sum: np.ndarray
    visual_sum: np.ndarray
    color: np.ndarray
    observations: list = field(default_factory=list)
    frames: set = field(default_factory=set)
    names: Counter = field(default_factory=Counter)
    fine_scale: bool = False

    @property
    def semantic(self):
        return normalize(self.semantic_sum)


def coverages(a, b, radius=0.1):
    return (float(np.mean(cKDTree(b).query(a)[0] <= radius)),
            float(np.mean(cKDTree(a).query(b)[0] <= radius)))


def supported_floor(samples, background, frames):
    """Port the existing C++ floor guards; a low table alone is insufficient."""
    if len(samples) < 1000:
        return None
    heights = samples[:, 2]
    low_height = np.sort(heights)[len(heights)//50]
    bins, counts = np.unique(np.floor(heights/.02).astype(int), return_counts=True)
    for cell, count in zip(bins, counts):
        center = (cell+.5)*.02
        if center > low_height+.03:
            break
        if count < 100:
            continue
        local = heights[np.abs(heights-center) <= .02]
        if len(local) < 300:
            continue
        candidate = float(np.partition(local, len(local)//2)[len(local)//2])
        inliers = np.abs(heights-candidate) <= .01
        bg = inliers & background
        cells = np.unique(np.floor(samples[bg, :2]/.1).astype(int), axis=0)
        _, support = np.unique(frames[bg], return_counts=True)
        below = float(np.mean(heights < candidate-.03))
        if inliers.sum() >= 300 and len(cells) >= 100 and np.count_nonzero(support >= 30) >= 3 and below <= .02:
            return {'height_m': candidate, 'inlier_points': int(inliers.sum()),
                    'background_cells': len(cells), 'supported_frames': int(np.count_nonzero(support >= 30)),
                    'below_fraction': below, 'removal_band_m': .01}
    return None


def estimate_floor(context, data, jobs, kd, kc, scale):
    if not (data.get('dataset') == 'hypersim' and data.get('world_frame') == 'hypersim_world_z_up'
            and data.get('length_unit') == 'meter'):
        context.event('跳过几何地板估计', reason='输入未声明已验证的米制 Hypersim Z-up 坐标')
        return None
    points, flags, indices = [], [], []
    for index, job in enumerate(jobs):
        mask = cv2.imread(str(context.scene/'refined_instance'/f"{job['frame_id']}.png"), cv2.IMREAD_UNCHANGED)
        rgb, depth = cv2.imread(str(job['rgb'])), cv2.imread(str(job['depth']), cv2.IMREAD_UNCHANGED)
        pose = np.loadtxt(job['pose'])
        coarse_cache = context.scene/'fine_frontend_cache'/f"{job['frame_id']}.npz"
        if coarse_cache.is_file():
            with np.load(coarse_cache) as cached:
                mask = cached['source_mask'].copy()
        if mask is None or rgb is None or depth is None:
            raise ValueError(f"Unreadable floor sample: {job['frame_id']}")
        for region, is_background in [(mask == 0, True), (mask != 0, False)]:
            sampled, _ = project_mask(region, rgb, depth, pose, kd, kc, scale,
                                      stride=max(16, context.stride), max_depth=context.max_depth)
            points.append(sampled)
            flags.append(np.full(len(sampled), is_background, bool))
            indices.append(np.full(len(sampled), index, int))
    floor = supported_floor(np.concatenate(points), np.concatenate(flags), np.concatenate(indices))
    if floor:
        samples, flags, indices = np.concatenate(points), np.concatenate(flags), np.concatenate(indices)
        levels = [floor['height_m']]
        heights = samples[:, 2]
        bins, counts = np.unique(np.floor(heights[flags]/.02).astype(int), return_counts=True)
        for cell, count in zip(bins, counts):
            center = (cell+.5)*.02
            if count < 300 or not floor['height_m']+.03 < center < floor['height_m']+.35:
                continue
            local = flags & (np.abs(heights-center) < .02)
            candidate = float(np.median(heights[local]))
            if min(abs(candidate-h) for h in levels) < .025:
                continue
            bg = flags & (np.abs(heights-candidate) <= .01)
            cells = np.unique(np.floor(samples[bg, :2]/.1).astype(int), axis=0)
            _, support = np.unique(indices[bg], return_counts=True)
            local_count = np.count_nonzero(flags & (np.abs(heights-candidate) <= .02))
            sharp_mode = int(bg.sum())/max(local_count, 1)
            rectangle_cells = np.prod(np.maximum(np.ptp(cells, axis=0)+1, 1)) if len(cells) else 1
            filled_fraction = len(cells)/rectangle_cells
            if (len(cells) >= 100 and np.count_nonzero(support >= 30) >= 3
                    and sharp_mode >= .8 and filled_fraction >= .25):
                levels.append(candidate)
        floor['levels_m'] = levels
    context.event('几何地板估计通过' if floor else '跳过几何地板清除',
                  **(floor or {'reason': '低处水平面缺少足够背景覆盖或跨帧支持'}))
    (context.scene/'floor_filter.json').write_text(json.dumps({'algorithm': 'multiframe_floor_and_supported_low_platforms_v4',
                                                             'estimate': floor}, indent=2)+'\n')
    return floor


from pipeline_components.fine_instances import sampling_resolution, association_radius

def update(track, observation, fid):
    record = observation.record
    track.points = voxel_downsample(np.concatenate([track.points, observation.points]),
        sampling_resolution(observation.points) if record.get('fine_scale_instance') else .01)
    track.fine_scale = track.fine_scale or bool(record.get('fine_scale_instance'))
    # Keep sums of unit observations, never repeatedly normalize an accumulated mean.
    track.semantic_sum += observation.semantic
    track.visual_sum += observation.visual
    track.color = (track.color * len(track.observations) + observation.color) / (len(track.observations) + 1)
    track.names[record['object_name']] += 1
    track.frames.add(fid)
    record['instance_id'] = track.id
    track.observations.append({'frame_id': fid, 'local_id': record['frame_instance_id'],
                               'confidence': record['confidence'], 'name': record['object_name'],
                               'mask_quality': record.get('sam_quality_score')})


def fuse(context, association=None):
    scene = context.scene
    data, jobs, kd, kc, scale = load_capture(context.manifest, context.image_dir, scene / 'refined_instance')
    floor = estimate_floor(context, data, jobs, kd, kc, scale)
    tracks, records_by_frame, audit, background = [], {}, [], []
    for job in jobs:
        fid = job['frame_id']
        folder = scene / 'refined_instance'
        records = json.loads((folder / f'{fid}_instance.json').read_text())
        mask = cv2.imread(str(folder / f'{fid}.png'), cv2.IMREAD_UNCHANGED)
        rgb = cv2.imread(str(job['rgb']))
        depth = cv2.imread(str(job['depth']), cv2.IMREAD_UNCHANGED)
        pose = np.loadtxt(job['pose'])
        if any(x is None for x in [mask, rgb, depth]):
            raise ValueError(f'Unreadable capture: {fid}')
        frame = {'pose': pose, 'depth': depth, 'kd': kd, 'kc': kc, 'scale': scale}
        bg, _ = project_mask(mask == 0, rgb, depth, pose, kd, kc, scale, stride=4)
        background.append(voxel_downsample(bg, .03))
        coarse_mask = mask
        coarse_cache = scene/'fine_frontend_cache'/f'{fid}.npz'
        if coarse_cache.is_file():
            with np.load(coarse_cache) as cached:
                coarse_mask = cached['source_mask'].copy()
        observations = []
        for record in records:
            record['instance_id'] = -1
            evidence_mask = mask if record.get('fine_scale_instance') else coarse_mask
            region = evidence_mask == int(record['frame_instance_id'])
            points, colors = project_mask(region, rgb, depth, pose, kd, kc, scale,
                                           stride=context.stride, max_depth=context.max_depth)
            if floor:
                keep = np.min(np.abs(points[:, 2:3]-np.asarray(floor.get('levels_m', [floor['height_m']]))), axis=1) > floor['removal_band_m']
                removed = int(np.count_nonzero(~keep))
                if removed:
                    background.append(voxel_downsample(points[~keep], .03))
                    audit.append({'message': '清除几何地板上的实例标签', 'frame_id': fid,
                                  'local_id': record['frame_instance_id'], 'removed_points': removed})
                points, colors = points[keep], colors[keep]
            if len(points) < 16:
                continue
            visual = np.asarray(record['feature'], float)
            semantic = np.asarray(record['bert_embedding'], float)
            if visual.shape != (256,) or semantic.shape != (384,):
                raise ValueError('Expected separate DINO-256 and SBERT-384 feature spaces')
            observations.append(Observation(record, region, voxel_downsample(points, sampling_resolution(points) if record.get('fine_scale_instance') else .01),
                                            color_histogram(colors), normalize(semantic), visual))
        scores = np.full((len(observations), len(tracks)), -1e6)
        evidence = {}
        track_trees = [cKDTree(t.points) for t in tracks]
        track_bounds = [(t.points.min(0), t.points.max(0)) for t in tracks]
        for i, observation in enumerate(observations):
            observation_tree = cKDTree(observation.points)
            lower, upper = observation.points.min(0), observation.points.max(0)
            for j, track in enumerate(tracks):
                # Independently verified fine instances cannot rename or enlarge
                # a coarse object through an asymmetric surface-overlap match.
                if bool(observation.record.get('fine_scale_instance')) != track.fine_scale:
                    continue
                q = float(observation.semantic @ track.semantic)
                if fid in track.frames:
                    continue
                if np.linalg.norm(observation.points.mean(0) - track.points.mean(0)) > 3:
                    continue
                tlower, tupper = track_bounds[j]
                separation = np.maximum(np.maximum(lower-tupper, tlower-upper), 0)
                radius = (association_radius(observation.points, track.points)
                    if observation.record.get('fine_scale_instance') and track.fine_scale else .1)
                if np.linalg.norm(separation) > radius+1e-6:
                    continue
                a = float(np.mean(track_trees[j].query(observation.points)[0] <= radius))
                b = float(np.mean(observation_tree.query(track.points)[0] <= radius))
                g = max(a, b)
                base = g + (q + 1) / 2
                if g < .2 or (q < .8 and min(a, b) < .7) or (q >= .8 and base < 1.2):
                    continue
                details = {'geometry': g, 'semantic': q, 'directional_coverage': [a, b], 'association_radius_m': radius}
                if association:
                    projection = association.visibility_score(observation, track, frame)
                    if projection is None:
                        continue
                    details.update(projection)
                    scores[i, j] = association.score(base, projection)
                else:
                    scores[i, j] = base
                evidence[i, j] = details
        if association:
            assignments = association.assign(scores)
        else:
            assignments, used = {}, set()
            for i in range(len(observations)):
                valid = [j for j in range(len(tracks)) if j not in used and scores[i, j] > 0]
                if valid:
                    j = max(valid, key=lambda j: scores[i, j])
                    assignments[i] = j
                    used.add(j)
        for i, observation in enumerate(observations):
            if i in assignments:
                j = assignments[i]
                track = tracks[j]
                details = evidence[i, j]
                action = '关联历史物体'
            else:
                track = Track(len(tracks) + 1, np.empty((0, 3)), np.zeros(384), np.zeros(256), observation.color)
                tracks.append(track)
                details, action = {}, '新增物体轨迹'
            update(track, observation, fid)
            audit.append({'message': action, 'frame_id': fid, 'local_id': observation.record['frame_instance_id'],
                          'global_id': track.id, **details})
        records_by_frame[fid] = records
        context.event('主流程物体关联完成', frame_id=fid, observations=len(observations), tracks=len(tracks), matched=len(assignments))
    import pipeline_components as components
    consensus = getattr(components, 'INSTANCE_REFINEMENT', None)
    remap = consensus.process(context, tracks, records_by_frame, data, jobs, kd, kc, scale, audit) if consensus else {}
    alive = {t.id: t for t in tracks if t.id not in remap and len(t.frames) >= 2 and len(t.points) >= 16}
    for fid, records in records_by_frame.items():
        for record in records:
            gid = remap.get(record['instance_id'], record['instance_id'])
            record['instance_id'] = gid if gid in alive else -1
        (scene / 'refined_instance' / f'{fid}_updated_instance.json').write_text(json.dumps(records, indent=2) + '\n')
    if not alive:
        raise RuntimeError('No multi-frame object survived fusion')
    points, colors, rgb_colors = [], [], []
    rng = np.random.default_rng(0)
    for track in alive.values():
        points.append(track.points)
        gid = track.id
        encoded = np.array([gid % 255, (gid // 255) % 255, (gid // 255 // 255) % 255]) / 255
        colors.append(np.tile(encoded, (len(track.points), 1)))
        rgb_colors.append(np.tile(rng.uniform(.2, 1, 3), (len(track.points), 1)))
    for filename, palette in [('instance_cloud.ply', colors), ('instance_cloud_with_background.ply', colors),
                              ('instance_cloud_colored.ply', rgb_colors)]:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(np.concatenate(points))
        pcd.colors = o3d.utility.Vector3dVector(np.concatenate(palette))
        if filename == 'instance_cloud_with_background.ply' and background:
            bg = voxel_downsample(np.concatenate(background), .03)
            pcd.points = o3d.utility.Vector3dVector(np.concatenate([np.concatenate(points), bg]))
            pcd.colors = o3d.utility.Vector3dVector(np.concatenate([np.concatenate(palette), np.zeros_like(bg)]))
        if not o3d.io.write_point_cloud(str(scene / filename), pcd):
            raise IOError(filename)
    with (scene / 'instance_name_map.csv').open('w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['instance_id', 'name'])
        writer.writerows((t.id, t.names.most_common(1)[0][0]) for t in alive.values())
    for filename, feature in [('averaged_instance_features.json', 'visual_sum'), ('instance_bert_embeddings.json', 'semantic_sum')]:
        values = [{'instance_id': t.id, 'feature': (getattr(t, feature) / len(t.observations)).tolist()} for t in alive.values()]
        (scene / filename).write_text(json.dumps(values, indent=2) + '\n')
    (scene / 'object_tracks.json').write_text(json.dumps({str(t.id): {'observations': t.observations,
        'observed_frames': sorted(t.frames), 'name_votes': dict(t.names), 'point_count': len(t.points),
        'fine_scale_instance': t.fine_scale,
        **getattr(t, 'quality', {}),
        'confidence': getattr(t, 'quality', {}).get('confidence', float(np.mean([x['confidence'] for x in t.observations])))} for t in alive.values()}, indent=2) + '\n')
    (scene / 'object_association_zh.jsonl').write_text(''.join(json.dumps(x, ensure_ascii=False) + '\n' for x in audit))
    context.event('主流程物体融合完成', confirmed_objects=len(alive), recovered_histories=len(remap),
                  association='visibility_hungarian' if association else 'op3dsg_greedy')

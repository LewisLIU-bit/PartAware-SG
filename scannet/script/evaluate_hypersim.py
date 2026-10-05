"""Independent observed-Hypersim evaluation; never use GT during construction."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import cv2
import csv
import h5py
import numpy as np
import open3d as o3d
from scipy.optimize import linear_sum_assignment
from scipy.spatial.transform import Rotation
from partaware.geometry import load_capture, project_mask

NYU40 = ['wall', 'floor', 'cabinet', 'bed', 'chair', 'sofa', 'table', 'door', 'window',
         'bookshelf', 'picture', 'counter', 'blinds', 'desk', 'shelves', 'curtain', 'dresser',
         'pillow', 'mirror', 'floor mat', 'clothes', 'ceiling', 'books', 'refrigerator',
         'television', 'paper', 'towel', 'shower curtain', 'box', 'whiteboard', 'person',
         'night stand', 'toilet', 'sink', 'lamp', 'bathtub', 'bag', 'otherstructure',
         'otherfurniture', 'otherprop']


def read_hdf(path):
    with h5py.File(path, 'r') as file:
        return file['dataset'][:]


def voxel_keys(points, size=.01):
    grid = np.floor(points / size).astype(np.int64) + (1 << 20)
    if np.any(grid < 0) or np.any(grid >= 1 << 21):
        raise ValueError('Scene coordinates exceed the voxel key domain')
    return np.unique((grid[:, 0] << 42) | (grid[:, 1] << 21) | grid[:, 2])


def bounds(points):
    if len(points) < 4:
        return points.min(0)-1e-4, points.max(0)+1e-4
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    try:
        box = pcd.get_oriented_bounding_box()
        return box.get_min_bound(), box.get_max_bound()
    except RuntimeError:
        return points.min(0)-1e-4, points.max(0)+1e-4


def box_overlap(gt, prediction):
    size = np.maximum(np.minimum(gt[1], prediction[1]) - np.maximum(gt[0], prediction[0]), 0)
    intersection = float(np.prod(size))
    vg, vp = float(np.prod(gt[1]-gt[0])), float(np.prod(prediction[1]-prediction[0]))
    return intersection / max(vg+vp-intersection, 1e-12), intersection / max(vp, 1e-12)


def node_bounds(node):
    """Evaluate the saved graph shape exactly as the common viewer presents it."""
    shape = node['shape']
    orientation = shape['orientation']
    rotation = Rotation.from_quat([orientation[k] for k in ['x', 'y', 'z', 'w']]).as_matrix()
    center = np.asarray(node['position'], float)
    if 'radius' in shape:
        axis = rotation[:, 2]
        half = shape['radius'] * np.sqrt(np.maximum(1-axis**2, 0)) + shape['height']/2 * np.abs(axis)
        return center-half, center+half
    corners = np.array([[x, y, z] for x in [-.5, .5] for y in [-.5, .5] for z in [-.5, .5]])
    box = (corners * np.array([shape['length'], shape['width'], shape['height']])) @ rotation.T + center
    return box.min(0), box.max(0)


def average_precision(matrix, confidences, threshold):
    used, hits = set(), []
    for i in sorted(range(len(confidences)), key=lambda i: (-confidences[i], i)):
        available = [j for j in range(matrix.shape[1]) if j not in used and matrix[i, j] >= threshold]
        if available:
            best = max(available, key=lambda j: matrix[i, j])
            used.add(best)
            hits.append(1)
        else:
            hits.append(0)
    if not matrix.shape[1]:
        return None
    if not hits:
        return 0.
    true = np.cumsum(hits)
    precision, recall = true / np.arange(1, len(hits)+1), true / matrix.shape[1]
    precision = np.maximum.accumulate(precision[::-1])[::-1]
    increments = np.diff(np.r_[0, recall])
    return float(np.sum(increments * precision))


def object_count_metrics(predicted, annotated):
    """Report symmetric count error without encoding infinity as invalid JSON."""
    if predicted < 0 or annotated < 0:
        raise ValueError('Object counts cannot be negative')
    ratio = predicted / annotated if annotated else None
    error = abs(math.log(ratio)) if ratio is not None and ratio > 0 else None
    return {'predicted': predicted, 'annotated': annotated, 'ratio': ratio,
            'absolute_log_ratio': error, 'log_base': 'e',
            'status': 'undefined_no_ground_truth' if not annotated else
                      'infinite_no_predictions' if not predicted else 'finite',
            'absolute_count_error': abs(predicted-annotated),
            'scope': 'observable official object instances; exclude wall, floor, ceiling and parts',
            'limitation': 'Missed and duplicate objects can cancel; pair with one-to-one localization metrics.'}


def evaluate(args):
    manifest = Path(args.manifest).expanduser().resolve()
    payload = json.loads(manifest.read_text())
    if payload['dataset'] != 'hypersim' or not payload.get('ground_truth_source'):
        raise ValueError('Official Hypersim labels must be prepared independently')
    native = Path(payload['ground_truth_source'])
    scene = Path(args.processed_scene).expanduser().resolve()
    graph_path = Path(args.graph_file).expanduser().resolve() if args.graph_file else scene / 'topology_map.json'
    graph = json.loads(graph_path.read_text())
    nodes = graph['object_nodes']['nodes'] or {}
    _, jobs, kd, kc, scale = load_capture(manifest)
    geometry_source = Path(args.geometry_source).expanduser().resolve() if args.geometry_source else scene
    prediction_ply = geometry_source / 'instance_cloud_cleaned.ply'
    if not prediction_ply.exists():
        prediction_ply = geometry_source / 'instance_cloud.ply'
    if not prediction_ply.is_file():
        raise FileNotFoundError(prediction_ply)
    cloud = o3d.io.read_point_cloud(str(prediction_ply))
    points = np.asarray(cloud.points)
    rgb_ids = np.round(np.asarray(cloud.colors) * 255).astype(np.int64)
    ids = rgb_ids[:, 0] + 255 * rgb_ids[:, 1] + 255**2 * rgb_ids[:, 2]
    predictions = []
    tracks_path = scene / 'validated_object_tracks.json'
    if not tracks_path.exists():
        tracks_path = scene / 'object_tracks.json'
    tracks = json.loads(tracks_path.read_text()) if tracks_path.exists() else {}
    for key, node in nodes.items():
        region = points[ids == int(key)]
        if not len(region):
            raise ValueError(f'Canonical graph node {key} has no cleaned points')
        observations = []
        if key not in tracks:
            for job in jobs:
                path = geometry_source / 'refined_instance' / f"{job['frame_id']}_updated_instance.json"
                if path.exists():
                    observations += [float(x['confidence']) for x in json.loads(path.read_text()) if str(x.get('instance_id')) == key]
        confidence = float(tracks[key]['confidence']) if key in tracks else float(np.mean(observations)) if observations else 0.
        predictions.append({'id': key, 'label': node['name'], 'keys': voxel_keys(region),
                            'bounds': node_bounds(node), 'confidence': confidence})
    gt_points, class_votes = defaultdict(list), defaultdict(Counter)
    label_digest = hashlib.sha256()
    manifest_frames = {f['frame_id']: f for f in payload['frames']}
    for index, job in enumerate(jobs):
        entry = manifest_frames[job['frame_id']]
        source = native / f"images/scene_{entry['source_camera']}_geometry_hdf5/frame.{entry['source_camera_frame']:04d}"
        semantic_path = Path(str(source)+'.semantic.hdf5')
        instance_path = Path(str(source)+'.semantic_instance.hdf5')
        semantic, instance = read_hdf(semantic_path), read_hdf(instance_path)
        label_digest.update(semantic_path.read_bytes())
        label_digest.update(instance_path.read_bytes())
        rgb = cv2.imread(str(job['rgb']))
        depth = cv2.imread(str(job['depth']), cv2.IMREAD_UNCHANGED)
        pose = np.loadtxt(job['pose'])
        if semantic.shape != rgb.shape[:2] or instance.shape != rgb.shape[:2]:
            raise ValueError('Official label resolution disagrees with input RGB')
        for gid in np.unique(instance):
            if gid < 0:
                continue
            region = instance == gid
            classes, counts = np.unique(semantic[region], return_counts=True)
            valid = [(int(c), int(n)) for c, n in zip(classes, counts) if 1 <= c <= 40 and c not in (1, 2, 22)]
            if not valid:
                continue
            mask = region & (semantic > 0) & ~np.isin(semantic, [1, 2, 22])
            projected, _ = project_mask(mask, rgb, depth, pose, kd, kc, scale, stride=2)
            if len(projected):
                gt_points[int(gid)].append(projected.astype(np.float32))
                class_votes[int(gid)].update(dict(valid))
        if (index+1) % 10 == 0:
            print('官方真值投影完成', index+1, '/', len(jobs), flush=True)
    truth = []
    metadata = list(csv.DictReader((native / '_detail/metadata_scene.csv').read_text().splitlines()))
    asset_scale = float(metadata[0]['meters_per_asset_unit']) if 'meters_per_asset_unit' in metadata[0] else float(
        next(r['parameter_value'] for r in metadata if r['parameter_name'] == 'meters_per_asset_unit'))
    mesh = native / '_detail/mesh'
    box_center = read_hdf(mesh / 'metadata_semantic_instance_bounding_box_object_aligned_2d_positions.hdf5') * asset_scale
    box_extent = read_hdf(mesh / 'metadata_semantic_instance_bounding_box_object_aligned_2d_extents.hdf5') * asset_scale
    box_rotation = read_hdf(mesh / 'metadata_semantic_instance_bounding_box_object_aligned_2d_orientations.hdf5')
    corners = np.array([[x, y, z] for x in [-.5, .5] for y in [-.5, .5] for z in [-.5, .5]])
    for gid in sorted(gt_points):
        region = np.concatenate(gt_points[gid])
        keys = voxel_keys(region)
        if len(keys) < 100:
            continue
        if gid >= len(box_center) or not np.isfinite(box_center[gid]).all() or not np.isfinite(box_extent[gid]).all():
            raise ValueError(f'Missing independent official box for semantic instance {gid}')
        box = (corners * box_extent[gid]) @ box_rotation[gid].T + box_center[gid]
        truth.append({'id': gid, 'label': NYU40[class_votes[gid].most_common(1)[0][0]-1],
                      'keys': keys, 'bounds': (box.min(0), box.max(0))})
    del gt_points
    if not truth:
        raise RuntimeError('No observed GT object passed the fixed visibility criterion')
    matrix = np.zeros((len(predictions), len(truth)))
    bbox_iou, bbox_iop = matrix.copy(), matrix.copy()
    for i, prediction in enumerate(predictions):
        for j, gt in enumerate(truth):
            intersection = len(np.intersect1d(prediction['keys'], gt['keys'], assume_unique=True))
            matrix[i, j] = intersection / max(len(prediction['keys'])+len(gt['keys'])-intersection, 1)
            bbox_iou[i, j], bbox_iop[i, j] = box_overlap(gt['bounds'], prediction['bounds'])
    confidences = [x['confidence'] for x in predictions]
    spatial = {}
    bbox_spatial = {}
    for threshold in [.25, .5, .75]:
        # Match the maximum number of valid pairs, then prefer the greater overlap.
        value = np.where(matrix >= threshold, 1 + matrix / (min(matrix.shape)+1), -1e6)
        padded = np.concatenate([value, np.zeros((len(predictions), len(predictions)))], axis=1)
        rows, columns = linear_sum_assignment(-padded)
        pairs = [(int(i), int(j)) for i, j in zip(rows, columns) if j < len(truth) and matrix[i, j] >= threshold]
        tp, fp, fn = len(pairs), len(predictions)-len(pairs), len(truth)-len(pairs)
        p, r = tp/max(tp+fp, 1), tp/max(tp+fn, 1)
        spatial[str(threshold)] = {'TP': tp, 'FP': fp, 'FN': fn, 'precision': p, 'recall': r,
                                  'F1': 2*p*r/max(p+r, 1e-12),
                                  'matches': [{'prediction_id': predictions[i]['id'], 'gt_id': truth[j]['id'],
                                               'iou': float(matrix[i, j])} for i, j in pairs]}
        value = np.where(bbox_iou >= threshold, 1 + bbox_iou / (min(bbox_iou.shape)+1), -1e6)
        padded = np.concatenate([value, np.zeros((len(predictions), len(predictions)))], axis=1)
        rows, columns = linear_sum_assignment(-padded)
        pairs = [(int(i), int(j)) for i, j in zip(rows, columns) if j < len(truth) and bbox_iou[i, j] >= threshold]
        tp, fp, fn = len(pairs), len(predictions)-len(pairs), len(truth)-len(pairs)
        p, r = tp/max(tp+fp, 1), tp/max(tp+fn, 1)
        bbox_spatial[str(threshold)] = {'TP': tp, 'FP': fp, 'FN': fn, 'precision': p, 'recall': r,
                                       'F1': 2*p*r/max(p+r, 1e-12), 'matches': [
                                           {'prediction_id': predictions[i]['id'], 'gt_id': truth[j]['id'],
                                            'iou': float(bbox_iou[i, j])} for i, j in pairs]}
    # Reuse OP3DSG's label-space rank and permissive per-GT matching protocol.
    import clip
    import torch
    model, _ = clip.load('ViT-B/16', device='cpu', download_root=str(Path(__file__).resolve().parents[2] / 'checkpoints/clip'))
    labels = NYU40
    queries = [x['label'].replace('_', ' ').replace('-', ' ').replace(':', ' ') for x in predictions]
    with torch.no_grad():
        embeddings = model.encode_text(clip.tokenize(labels+queries, truncate=True)).float()
        embeddings /= embeddings.norm(dim=1, keepdim=True)
    retrieval = np.argsort((embeddings[40:] @ embeddings[:40].T).numpy(), axis=1)[:, ::-1]
    ranks = np.array([[int(np.where(retrieval[i] == labels.index(gt['label']))[0][0])+1 for gt in truth]
                      for i in range(len(predictions))]).reshape((len(predictions), len(truth)))
    recalls = {}
    for name, overlaps, threshold in [('IoU>0', bbox_iou, 0), ('IoU>0.10', bbox_iou, .1), ('IoP>0.25', bbox_iop, .25)]:
        recalls[name] = {f'R@{k}': float(np.mean(np.any((overlaps > threshold) & (ranks <= k), axis=0))) for k in [1, 3, 5]}
    report = {'protocol': 'observed_hypersim_adaptation_not_official_scannet_or_unigraph3d',
              'input_manifest': str(manifest), 'processed_scene': str(scene), 'frames': len(jobs),
              'voxel_m': .01, 'projection_stride': 2, 'minimum_observed_gt_voxels': 100,
              'excluded_nyu40_ids': [1, 2, 22], 'gt_objects': len(truth), 'predicted_objects': len(predictions),
              'object_count_consistency': object_count_metrics(len(predictions), len(truth)),
              'label_space': labels, 'clip_model': 'openai_ViT-B/16_text',
              'graph_path': str(graph_path), 'graph_sha256': hashlib.sha256(graph_path.read_bytes()).hexdigest(), 'gt_labels_sha256': label_digest.hexdigest(),
              'prediction_ply': str(prediction_ply), 'prediction_ply_sha256': hashlib.sha256(prediction_ply.read_bytes()).hexdigest(),
              'gt_bbox_source': 'official_mesh_object_aligned_2d_boxes_converted_to_aabb',
              'prediction_bbox_source': 'canonical_topology_node_shape_and_position_converted_to_aabb',
              'geometry_only_box_AP25': average_precision(bbox_iou, confidences, .25),
              'geometry_only_box_AP50': average_precision(bbox_iou, confidences, .5),
              'geometry_only_box_AP75': average_precision(bbox_iou, confidences, .75),
              'geometry_only_box_AP': float(np.mean([average_precision(bbox_iou, confidences, t) for t in np.arange(.5, 1, .05)])),
              'one_to_one_bbox_geometry': bbox_spatial, 'op3dsg_adapted_object_label_recall': recalls,
              'voxel_occupancy_diagnostic': {'AP25': average_precision(matrix, confidences, .25),
                  'AP50': average_precision(matrix, confidences, .5), 'one_to_one': spatial,
                  'limitation': 'Exact occupied-cell overlap is sensitive to output point density; do not interpret it as official mask AP or evidence of algorithmic improvement.'},
              'part_precision': None, 'relation_triplet_recall': None,
              'unavailable_reason': 'Hypersim official labels do not annotate parts, part_of or relation triplets.',
              'ground_truth_objects': [{'id': g['id'], 'label': g['label'], 'voxels': len(g['keys']),
                                        'bounds': np.asarray(g['bounds']).tolist()} for g in truth],
              'prediction_objects': [{'id': p['id'], 'label': p['label'], 'voxels': len(p['keys']),
                                      'confidence': p['confidence'], 'bounds': np.asarray(p['bounds']).tolist()} for p in predictions]}
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + '\n')
    print('独立物体评价完成', {key: report[key] for key in ['gt_objects', 'predicted_objects', 'geometry_only_box_AP25', 'geometry_only_box_AP50', 'geometry_only_box_AP75']}, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--processed-scene', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--graph-file', help='Explicit saved graph hypothesis for a geometry ablation')
    parser.add_argument('--geometry-source', help='Explicit geometry source for older experiments that reused another object map')
    evaluate(parser.parse_args())

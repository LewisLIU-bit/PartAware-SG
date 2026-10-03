"""RGB-D projection matching the existing InstanceCloudGenerator convention."""
from pathlib import Path
import json
import cv2
import numpy as np


def load_capture(manifest=None, image_dir=None, instance_dir=None):
    if manifest:
        path = Path(manifest).expanduser().resolve()
        data = json.loads(path.read_text(encoding='utf-8'))
        if data.get('format') != 'scannet_sg_input':
            raise ValueError('Expected a scannet_sg_input manifest')
        if data.get('pose_convention') != 'T_world_from_camera':
            raise ValueError('Only T_world_from_camera poses are supported')
        if data.get('length_unit') != 'meter' or data.get('depth_type') != 'optical_axis_z':
            raise ValueError('Metric optical-axis depth is required')
        base = path.parent
        frames = data['frames']
        info = base / data['camera_info']
    else:
        base = Path(image_dir).expanduser().resolve()
        info = base / '_info.txt'
        ids = sorted((p.stem for p in Path(instance_dir).glob('*.png') if p.stem.isdecimal()), key=int)
        frames = [{'frame_id': fid, 'rgb': f'frame-{int(fid):06d}.color.jpg',
                   'depth': f'frame-{int(fid):06d}.depth.pgm',
                   'pose': f'frame-{int(fid):06d}.pose.txt'} for fid in ids]
        data = {'dataset': 'scannet', 'scene_id': base.name,
                'pose_convention': 'T_world_from_camera', 'length_unit': 'meter',
                'depth_type': 'optical_axis_z'}
    if not frames:
        raise ValueError('No input frames')
    seen = set()
    jobs = []
    for frame in frames:
        fid = frame['frame_id']
        if not isinstance(fid, str) or not fid or fid in ('.', '..') or any(c in fid for c in '/\\\0'):
            raise ValueError(f'Unsafe frame ID: {fid!r}')
        if fid in seen:
            raise ValueError(f'Duplicate frame ID: {fid}')
        seen.add(fid)
        job = {'frame_id': fid}
        for key in ('rgb', 'depth', 'pose'):
            p = (base / frame[key]).resolve()
            if not p.is_file():
                raise FileNotFoundError(p)
            job[key] = p
        jobs.append(job)
    metadata = {}
    for line in info.read_text().splitlines():
        if '=' in line:
            key, value = line.split('=', 1)
            metadata[key.strip()] = value.strip()
    def intrinsics(key):
        values = np.fromstring(metadata[key], sep=' ')
        if values.size not in (9, 16) or not np.isfinite(values).all():
            raise ValueError(f'Invalid camera intrinsics: {key}')
        matrix = values.reshape(3, 3) if values.size == 9 else values.reshape(4, 4)[:3, :3]
        if matrix[0, 0] <= 0 or matrix[1, 1] <= 0:
            raise ValueError('Focal lengths must be positive')
        return matrix
    kd = intrinsics('m_calibrationDepthIntrinsic')
    kc = intrinsics('m_calibrationColorIntrinsic')
    scale = float(metadata['m_depthShift'])
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError('Depth scale must be positive')
    if manifest and not np.isclose(float(data['depth_scale']), scale):
        raise ValueError('Manifest and camera metadata depth scales disagree')
    return data, jobs, kd, kc, scale


def project_mask(mask, rgb, depth, pose, kd, kc, scale, stride=2, max_depth=0):
    if stride < 1 or max_depth < 0:
        raise ValueError('Invalid sampling parameters')
    if depth.ndim != 2 or depth.dtype != np.uint16:
        raise ValueError('Depth must be a uint16 single-channel image')
    if pose.shape != (4, 4) or not np.isfinite(pose).all() or not np.allclose(pose[3], [0, 0, 0, 1]):
        raise ValueError('Invalid camera-to-world pose')
    if not np.allclose(pose[:3, :3].T @ pose[:3, :3], np.eye(3), atol=2e-3):
        raise ValueError('Pose rotation must be orthonormal')
    if mask.shape != rgb.shape[:2]:
        raise ValueError('RGB and mask shapes disagree')
    v, u = np.mgrid[0:depth.shape[0]:stride, 0:depth.shape[1]:stride]
    z = depth[v, u].astype(np.float64) / scale
    x = (u - kd[0, 2]) * z / kd[0, 0]
    y = (v - kd[1, 2]) * z / kd[1, 1]
    # Depth and color are registered, as required by the baseline reader.
    ur = ((u - kd[0, 2]) * kc[0, 0] / kd[0, 0] + kc[0, 2])
    vr = ((v - kd[1, 2]) * kc[1, 1] / kd[1, 1] + kc[1, 2])
    valid = (z > 0) & (ur >= 0) & (vr >= 0) & (ur < mask.shape[1]) & (vr < mask.shape[0])
    if max_depth:
        valid &= z <= max_depth
    ui, vi = ur[valid].astype(int), vr[valid].astype(int)
    keep = mask[vi, ui].astype(bool)
    camera = np.column_stack((x[valid][keep], y[valid][keep], z[valid][keep]))
    points = camera @ pose[:3, :3].T + pose[:3, 3]
    return points, rgb[vi[keep], ui[keep]]


def voxel_downsample(points, voxel):
    points = np.asarray(points, dtype=float).reshape(-1, 3)
    points = points[np.isfinite(points).all(axis=1)]
    if voxel <= 0:
        raise ValueError('Voxel size must be positive')
    if not len(points):
        return points
    _, inverse = np.unique(np.floor(points / voxel).astype(np.int64), axis=0, return_inverse=True)
    counts = np.bincount(inverse)
    return np.column_stack([np.bincount(inverse, weights=points[:, i]) / counts for i in range(3)])


def denoise_largest_cluster(points, colors, eps, min_samples=10):
    """Optional ConceptGraphs-style cleanup with matching point/color indices."""
    if eps <= 0 or min_samples < 2:
        raise ValueError('DBSCAN parameters must be positive')
    from sklearn.cluster import DBSCAN
    labels = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=1).fit_predict(points)
    valid = labels >= 0
    if not valid.any():
        return points, colors
    counts = np.bincount(labels[valid])
    keep = labels == int(counts.argmax())
    if keep.sum() < 5:
        return points, colors
    return points[keep], colors[keep]


def subtract_contained_masks(masks, labels, parent_ids, containment=0.9):
    """Clean contained sibling masks only within a resolved parent instance."""
    if not len(masks) == len(labels) == len(parent_ids):
        raise ValueError('Mask, label, and parent counts disagree')
    result = [mask.copy() for mask in masks]
    areas = [int(mask.sum()) for mask in masks]
    for i, large in enumerate(masks):
        for j, small in enumerate(masks):
            if i == j or areas[j] == 0 or areas[i] <= 1.5 * areas[j]:
                continue
            if parent_ids[i] is None or parent_ids[i] != parent_ids[j]:
                continue
            if labels[i] == labels[j] or labels[i].split(':', 1)[0] != labels[j].split(':', 1)[0]:
                continue
            if np.count_nonzero(large & small) / areas[j] >= containment:
                result[i] &= ~small
    return result

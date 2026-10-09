"""Orient partial bodies from measured orthogonal faces, never from GT boxes.

An L-shaped visible surface can have a smaller diagonal minimum-area rectangle
than its physical body. This adapter uses two independently measured vertical
planes to recover the body axes. Extents still enclose only measured points.
Detach BOX_FITTING to retain the original minimum-area solver.
"""
import hashlib
from pathlib import Path

import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation


def fit(points, upright, fallback):
    """Require two broad orthogonal faces; retain all original measurements."""
    center, shape = fallback(points, upright)
    evidence = dict(accepted=False, generated_points=0, point_coordinates_preserved=True,
                    planes=[], reason='insufficient_orthogonal_measured_faces')
    if not upright or len(points) < 256:
        return center, shape, evidence
    sample = points[::max(1, len(points)//4096)]
    initial = len(sample)
    vertical = []
    for _ in range(4):
        if len(sample) < 64:
            break
        o3d.utility.random.seed(29)
        cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(sample))
        plane, indices = cloud.segment_plane(.003, 3, 400)
        normal = np.asarray(plane[:3])
        support = len(indices)/initial
        face = sample[indices]
        lateral = np.array([-normal[1], normal[0], 0.])
        width = np.ptp(face @ lateral)
        height = np.ptp(face[:, 2])
        detail = dict(normal=normal.tolist(), support_fraction=support,
                      face_width_m=float(width), face_height_m=float(height))
        evidence['planes'].append(detail)
        if support >= .15 and abs(normal[2]) <= .1 and min(width, height) >= .1:
            vertical.append((support, normal))
        sample = np.delete(sample, indices, axis=0)
    pairs = [(a, b) for i, a in enumerate(vertical) for b in vertical[i+1:]
             if abs(a[1] @ b[1]) <= .1]
    if not pairs:
        return center, shape, evidence
    a, b = max(pairs, key=lambda pair: pair[0][0]+pair[1][0])
    normal = max([a, b], key=lambda item: item[0])[1]
    theta = float(np.arctan2(normal[1], normal[0]) % (np.pi/2))
    c, s = np.cos(theta), np.sin(theta)
    matrix = np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])
    local = points @ matrix
    lower, upper = local.min(0), local.max(0)
    extent = np.maximum(upper-lower, 1e-4)
    center = ((lower+upper)/2 @ matrix.T).tolist()
    quaternion = Rotation.from_matrix(matrix).as_quat()
    shape = dict(length=float(extent[0]), width=float(extent[1]), height=float(extent[2]),
                 orientation=dict(zip(['x', 'y', 'z', 'w'], quaternion.tolist())))
    evidence.update(accepted=True, reason='two_broad_orthogonal_measured_faces',
                    yaw_radians=theta, total_plane_support=a[0]+b[0])
    return center, shape, evidence


def component_sha256():
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()

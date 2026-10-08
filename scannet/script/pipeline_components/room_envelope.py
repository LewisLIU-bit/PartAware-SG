"""Conservative measured room halfspaces for hidden-surface hypotheses."""
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree


def fit_envelope(background, observed, cameras):
    if len(background) < 1000 or len(cameras) < 3: return []
    if len(observed):
        background = background[cKDTree(observed).query(background)[0] > .03]
    cloud = o3d.geometry.PointCloud(); cloud.points = o3d.utility.Vector3dVector(background)
    cloud = cloud.voxel_down_sample(.025)
    planes = []
    o3d.utility.random.seed(12)
    for _ in range(16):
        if len(cloud.points) < 1000: break
        plane, inliers = cloud.segment_plane(.008, 3, 600)
        points = np.asarray(cloud.points)[inliers]
        cloud = cloud.select_by_index(inliers, invert=True)
        if len(points) < 1000: continue
        normal = np.asarray(plane[:3]); normal /= np.linalg.norm(normal)
        basis = np.linalg.svd(points-points.mean(0), full_matrices=False)[2][:2].T
        projected = points @ basis
        low, high = np.quantile(projected, [.01, .99], axis=0)
        if np.min(high-low) < 1.: continue
        signed = cameras @ normal+plane[3]
        sign = 1. if np.median(signed) >= 0 else -1.
        if np.mean(sign*signed > .05) < .9: continue
        planes.append({'normal': (normal*sign).tolist(), 'offset': float(plane[3]*sign),
            'basis': basis.tolist(), 'lower': low.tolist(), 'upper': high.tolist(),
            'measured_support_points': len(points), 'camera_side_fraction': float(np.mean(sign*signed > .05))})
    return planes


def outside_envelope(points, planes):
    forbidden = np.zeros(len(points), bool)
    for plane in planes:
        uv = points @ np.asarray(plane['basis'])
        inside_patch = np.all((uv >= np.asarray(plane['lower'])-.03) & (uv <= np.asarray(plane['upper'])+.03), axis=1)
        signed = points @ np.asarray(plane['normal'])+plane['offset']
        forbidden |= inside_patch & (signed < -.012)
    return forbidden

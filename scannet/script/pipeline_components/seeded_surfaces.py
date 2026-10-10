"""Propose physical surfaces from cached identity seeds, then verify raw depth.

Adapted from detect-then-group object representations and scene-constrained
completion. This is a geometric adapter, not the published trained networks.
No annotations, object identifiers, fixed class sizes, or new VLM calls are used.
"""
import json
from pathlib import Path

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree

from . import structural_surfaces



def closed_faces(points, normal):
    """Separate dense closed fronts from connected sparse rack posts."""
    from scipy.ndimage import binary_closing, label
    horizontal = np.cross(normal, [0., 0., 1.])
    horizontal /= np.linalg.norm(horizontal)
    planar = np.column_stack([points @ horizontal, points[:, 2]])
    bins = np.floor((planar-planar.min(0))/.06).astype(int)
    occupied = np.zeros(bins.max(0)+1, bool)
    occupied[bins[:, 0], bins[:, 1]] = True
    dense = occupied.mean(1) >= .6
    # Close narrow door seams, without converting open racks into closed faces.
    dense = binary_closing(dense, structure=np.ones(2), border_value=1)
    components, count = label(dense)
    return [points[components[bins[:, 0]] == index] for index in range(1, count+1)
            if np.sum(components == index)*.06 >= .5]


def proposals(raw, regions, tracks):
    """Local semantic seeds prevent large floor planes starving object proposals."""
    sample = np.asarray(o3d.geometry.PointCloud(
        o3d.utility.Vector3dVector(raw)).voxel_down_sample(.012).points)
    hypotheses = []
    for gid, track in tracks.items():
        name = max(track['name_votes'], key=track['name_votes'].get)
        kind = structural_surfaces.family(name)
        if kind is None or track['confidence'] < .4 or len(track['observed_frames']) < 3:
            continue
        cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(regions[gid])).voxel_down_sample(.012)
        original = len(cloud.points)
        for _ in range(4):
            if len(cloud.points) < 128:
                break
            plane, indices = cloud.segment_plane(.008, 3, 400)
            face = np.asarray(cloud.points)[indices].copy()
            cloud = cloud.select_by_index(indices, invert=True)
            normal = np.asarray(plane[:3])
            valid = abs(normal[2]) > .995 if kind == 'support' else abs(normal[2]) < .01
            if not valid or len(face) < max(128, .2*original):
                continue
            if normal[np.argmax(abs(normal))] < 0:
                normal = -normal
            offset = -float(np.median(face @ normal))
            if any(k == kind and abs(n @ normal) > .9995 and abs(d-offset) < .012
                   for n, d, k in hypotheses):
                continue
            hypotheses.append((normal, offset, kind))
    results = []
    for normal, offset, kind in hypotheses:
        face = sample[abs(sample @ normal+offset) < .009]
        if len(face) < 700:
            continue
        labels = np.asarray(o3d.geometry.PointCloud(
            o3d.utility.Vector3dVector(face)).cluster_dbscan(.045, 3))
        for label in np.unique(labels):
            if label < 0:
                continue
            points = face[labels == label]
            if len(points) < 700:
                continue
            _, basis = np.linalg.eigh(np.cov(points.T))
            extent = np.ptp(points @ basis[:, 1:], axis=0)
            if min(extent) < .5 or max(extent) < 1.:
                continue
            # A cached identity must touch this connected face; sharing a plane
            # with a distant wall or neighboring object is insufficient.
            tree = cKDTree(points)
            if not any(structural_surfaces.family(max(t['name_votes'], key=t['name_votes'].get)) == kind
                       and len(t['observed_frames']) >= 3 and len(regions[g]) >= 256
                       and np.sum(tree.query(regions[g])[0] < .025) >= 256
                       for g, t in tracks.items()):
                continue
            for closed in closed_faces(points, normal) if kind == 'storage' else [points]:
                if len(closed) >= 700:
                    results.append((closed, normal, kind))
    return results


def recovery_gate(verified, candidate, evidence, kind):
    if len(verified) < 512 or evidence['camera_baseline_m'] < .08:
        return False
    if kind != 'support':
        return len(verified) >= .65*len(candidate)
    # A discarded residual pixel is not evidence against a different verified
    # pixel. Require a substantial connected, repeatedly measured slab surface.
    labels = np.asarray(o3d.geometry.PointCloud(
        o3d.utility.Vector3dVector(verified)).cluster_dbscan(.045, 3))
    for label in np.unique(labels):
        if label < 0:
            continue
        component = verified[labels == label]
        if len(component) >= 512 and np.ptp(component[:, :2], axis=0).max() >= .5:
            evidence['acceptance_basis'] = 'connected_individually_depth_verified_support_surface'
            evidence['support_component_points'] = len(component)
            return True
    return False


def construct(context):
    metadata = json.loads(context.manifest.read_text()) if context.manifest else {}
    if metadata.get('world_frame') != 'hypersim_world_z_up':
        return structural_surfaces.construct(context)
    cloud = o3d.io.read_point_cloud(str(context.scene/'instance_cloud_with_background.ply'))
    points = np.asarray(cloud.points)
    encoded = np.rint(np.asarray(cloud.colors)*255).astype(int)
    ids = encoded[:, 0]+255*encoded[:, 1]+255**2*encoded[:, 2]
    tracks = json.loads((context.scene/'object_tracks.json').read_text())
    regions = {gid: points[ids == int(gid)] for gid in tracks}
    o3d.utility.random.seed(14)
    def original_samples(points, voxel):
        # A voxel centroid can lie off a measured thin surface and fail strict
        # reprojection. Keep an actual sensor sample instead of averaging XYZ.
        _, first = np.unique(np.floor(points/voxel).astype(np.int64), axis=0, return_index=True)
        return points[np.sort(first)]
    structural_surfaces.construct(context, proposal_builder=lambda raw: proposals(raw, regions, tracks),
                                  sample_builder=original_samples, recovery_gate=recovery_gate)

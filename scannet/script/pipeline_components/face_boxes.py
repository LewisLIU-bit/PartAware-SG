"""Use a broad measured vertical face for upright flat-object box axes.

Extents enclose the same actual points; this never inflates a box from GT.
Two orthogonal faces retain the historical solver. Thin, noisy curtains and
open shelves need one broad stable face rather than a diagonal hull edge.
"""
import hashlib
from pathlib import Path
import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation
from . import plane_boxes


def fit(points,upright,fallback):
    center,shape,evidence=plane_boxes.fit(points,upright,fallback)
    if evidence['accepted'] or not upright or len(points)<512:
        return center,shape,evidence
    sample=points[::max(1,len(points)//4096)]
    o3d.utility.random.seed(15)
    cloud=o3d.geometry.PointCloud(o3d.utility.Vector3dVector(sample))
    plane,indices=cloud.segment_plane(.006,3,600)
    normal=np.asarray(plane[:3])
    face=sample[indices]
    support=len(indices)/len(sample)
    width=np.ptp(face@np.array([-normal[1],normal[0],0.]))
    height=np.ptp(face[:,2])
    if abs(normal[2])>.05 or support<.45 or min(width,height)<.2:
        return center,shape,evidence
    theta=float(np.arctan2(normal[1],normal[0])%(np.pi/2))
    c,s=np.cos(theta),np.sin(theta)
    matrix=np.array([[c,-s,0.],[s,c,0.],[0.,0.,1.]])
    local=points@matrix
    low,high=local.min(0),local.max(0)
    extent=np.maximum(high-low,1e-4)
    center=((low+high)/2@matrix.T).tolist()
    shape=dict(length=float(extent[0]),width=float(extent[1]),height=float(extent[2]),
               orientation=dict(zip(('x','y','z','w'),Rotation.from_matrix(matrix).as_quat().tolist())))
    evidence.update(accepted=True,reason='one_broad_stable_measured_vertical_face',
                    yaw_radians=theta,total_plane_support=support)
    return center,shape,evidence


def component_sha256():
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def construct(context):
    # Fit broad-face axes only after the unchanged historical construction.
    import sys
    import pipeline_components as registry
    registry.BOX_FITTING=sys.modules[__name__]
    context.canonical_geometry_input=context.scene/'topology_map.json'
    from .canonical_geometry import publish
    publish(context)

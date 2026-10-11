"""Regularize broad observed planar patches without declaring new measurements.

Actual observations stay immutable in cleaned.ply. A separate reconstructed
cloud combines bounded plane projections with untouched edges/shape hypotheses.
This is local denoising, not hidden-surface completion or neural generation.
"""
from pathlib import Path
import hashlib,json
import numpy as np,open3d as o3d
from scipy.spatial import cKDTree
from threadpoolctl import threadpool_limits


def regularize(q):
    if len(q)<4096:return None
    sample=q[::max(1,len(q)//12000)]
    with threadpool_limits(limits=1):
        o3d.utility.random.seed(18)
        plane,indices=o3d.geometry.PointCloud(o3d.utility.Vector3dVector(sample)).segment_plane(.004,3,600)
    if len(indices)/len(sample)<.8:return None
    face=sample[indices];basis=np.linalg.svd(face-face.mean(0),full_matrices=False)[2][:2].T
    if np.min(np.ptp(face@basis,axis=0))<.5:return None
    n=np.asarray(plane[:3]);norm=np.linalg.norm(n);n/=norm;offset=plane[3]/norm
    signed=q@n+offset;core=abs(signed)<=.004
    result=q.copy();result[core]-=signed[core,None]*n
    return result,dict(original_points=len(q),regularized_points=int(core.sum()),
        maximum_displacement_m=float(np.max(abs(signed[core]))),rms_residual_before_m=float(np.sqrt(np.mean(signed[core]**2))),
        plane_normal=n.tolist(),plane_offset=float(offset),measured_plane_support=len(indices)/len(sample),
        observed_geometry_unchanged=True,reconstructed_coordinates_are_new_measurements=False,hidden_points_generated=0)


def construct(context):
    graph=json.loads((context.scene/'topology_map.json').read_text());nodes=graph['object_nodes']['nodes']
    observed=o3d.io.read_point_cloud(str(context.scene/'instance_cloud_cleaned.ply'));p=np.asarray(observed.points);colors=np.rint(np.asarray(observed.colors)*255).astype(int);ids=colors[:,0]+255*colors[:,1]+255**2*colors[:,2]
    base=o3d.io.read_point_cloud(str(context.graph_geometry));b=np.asarray(base.points).copy();bc=np.rint(np.asarray(base.colors)*255).astype(int);bid=bc[:,0]+255*bc[:,1]+255**2*bc[:,2]
    report=dict(algorithm='MEND_measured_envelope_normal_denoising_v18',construction_uses_ground_truth=False,
        node_or_category_whitelist=False,component_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),objects=[])
    for gid,node in nodes.items():
        q=p[ids==int(gid)];r=regularize(q)
        if r is None:continue
        smooth,evidence=r;index=np.flatnonzero(bid==int(gid));distance,nearest=cKDTree(q).query(b[index])
        # Only replace exact measurement anchors. Preserve every generated prior,
        # boundary, small item and other identity in the reconstructed source.
        exact=distance<1e-7;b[index[exact]]=smooth[nearest[exact]]
        node['surface_regularization']=dict(**evidence,audit='surface_regularization_audit.json')
        report['objects'].append(dict(instance_id=gid,name=node['name'],**evidence))
    if report['objects']:
        output=o3d.geometry.PointCloud(o3d.utility.Vector3dVector(b));output.colors=base.colors
        path=context.scene/'instance_cloud_regularized.ply';assert o3d.io.write_point_cloud(str(path),output)
        context.graph_geometry=path;context.canonical_geometry_input=context.scene/'topology_map.json'
        context.canonical_geometry_input.write_text(json.dumps(graph,indent=2)+'\n')
        from .canonical_geometry import publish
        publish(context)
    (context.scene/'surface_regularization_audit.json').write_text(json.dumps(report,indent=2)+'\n')
    context.event('大面积实测平面局部重建去噪',objects=len(report['objects']),maximum_displacement_m=.004,hidden_completion=False)

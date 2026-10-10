"""RSE: Robust Surface Envelopes after visible identity assignment.

FAST-MCD is used only after an independently evidenced ownership operation or
a dominant horizontal measured surface. It is not a multimodal row detector.
All retained coordinates are measured; no category names or annotations enter.
"""
import hashlib,json
from pathlib import Path
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree
from scipy.stats import chi2
from sklearn.covariance import MinCovDet
from .contact_instances import semantic_agreement
from .part_geometry import replace_regions


def robust_residual(points, connectivity=True):
    """Defer a correction when a dominant supported remainder is absent."""
    if len(points)<64:return points,dict(accepted=False,reason='insufficient_residual')
    cloud=o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    components=np.asarray(cloud.cluster_dbscan(eps=.02,min_points=1,print_progress=False))
    if np.count_nonzero(np.bincount(components)>=48)>1:
        return points,dict(accepted=False,reason='multiple_supported_unknown_components')
    try:
        model=MinCovDet(support_fraction=.75,random_state=16).fit(points[::max(1,len(points)//4096)])
    except (ValueError,np.linalg.LinAlgError):
        return points,dict(accepted=False,reason='degenerate_robust_support')
    cov=model.raw_covariance_+np.eye(3)*.006**2
    delta=points-model.raw_location_
    distance=np.einsum('ni,ij,nj->n',delta,np.linalg.inv(cov),delta)
    keep=distance<=chi2.ppf(.997,3)
    if keep.sum()<64 or keep.mean()<.7:return points,dict(accepted=False,reason='no_dominant_residual')
    selected=points[keep]
    if connectivity:
        cloud=o3d.geometry.PointCloud(o3d.utility.Vector3dVector(selected))
        components=np.asarray(cloud.cluster_dbscan(eps=.02,min_points=1,print_progress=False))
        sizes=np.bincount(components)
        supported=sizes[components]>=48
        if supported.sum()>=64 and supported.mean()>=.7:selected=selected[supported]
    return selected,dict(accepted=True,raw_mcd_location=model.raw_location_.tolist(),
        raw_mcd_covariance=model.raw_covariance_.tolist(),removed_points=len(points)-len(selected),
        generated_points=0,posterior_probability_calibrated=False)


def horizontal_envelope(points):
    if len(points)<256:return points,None
    sample=points[::max(1,len(points)//4096)]
    o3d.utility.random.seed(16)
    cloud=o3d.geometry.PointCloud(o3d.utility.Vector3dVector(sample))
    plane,indices=cloud.segment_plane(.003,3,600)
    fraction=len(indices)/len(sample)
    if abs(plane[2])<.97 or fraction<.5:return points,None
    try:
        model=MinCovDet(support_fraction=.75,random_state=16).fit(sample)
    except (ValueError,np.linalg.LinAlgError):
        return points,None
    eigen,axes=np.linalg.eigh(model.raw_covariance_)
    if eigen[0]/max(eigen[1],1e-12)>.025 or abs(axes[2,0])<.97:return points,None
    recovered,evidence=robust_residual(points,False)
    if len(recovered)<.85*len(points):return points,None
    evidence.update(measured_plane=plane.tolist(),plane_support_fraction=fraction)
    return recovered,evidence


def construct(context):
    graph_path=context.scene/'topology_map.json'
    graph=json.loads(graph_path.read_text());nodes=graph['object_nodes']['nodes']
    cloud_path=context.scene/'instance_cloud_cleaned.ply'
    cloud=o3d.io.read_point_cloud(str(cloud_path));points=np.asarray(cloud.points)
    colors=np.rint(np.asarray(cloud.colors)*255).astype(int)
    ids=colors[:,0]+255*colors[:,1]+255**2*colors[:,2]
    regions={gid:points[ids==int(gid)] for gid in nodes}
    audit=dict(algorithm='RSE_v16_FAST_MCD',ground_truth_used=False,vision_api_calls=0,generated_points=0,
               component_sha256=component_sha256(),objects=[])
    metadata=json.loads(context.manifest.read_text()) if context.manifest else {}
    if metadata.get('world_frame')!='hypersim_world_z_up':
        audit['reason']='未声明可靠竖直轴，保留已有物体'
        (context.scene/'residual_ownership_audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2)+'\n')
        return
    replacements={}
    evidence_nodes={gid for gid,node in nodes.items() if node.get('visible_instance_evidence')}
    for gid in sorted(evidence_nodes,key=int):
        q=regions[gid]
        if len(q)<64 or np.ptp(q,axis=0).max()>.7:continue
        refined,proof=horizontal_envelope(q)
        if proof and len(refined)<len(q):
            replacements[gid]=refined
            audit['objects'].append(dict(instance_id=gid,action='水平主表面稳健包络排除少量混入点',**proof))
        for other,prior in regions.items():
            if other==gid or other in evidence_nodes or nodes[other].get('geometry_hypothesis') or nodes[other].get('atomic_instance_evidence'):continue
            if len(prior)<64 or np.ptp(prior,axis=0).max()>.7 or semantic_agreement(nodes[gid]['text_embedding'],nodes[other]['text_embedding'])<.8:continue
            current=replacements.get(other,prior)
            shared=cKDTree(q).query(current)[0]<=.01
            if shared.mean()<.05 or shared.mean()>.8 or (~shared).sum()<64:continue
            residual,proof=robust_residual(current[~shared])
            replacements[other]=residual
            audit['objects'].append(dict(instance_id=other,unique_owner=gid,
                action='量化尺度表面归属与未知余量稳健审核',removed_duplicate_points=int(shared.sum()),**proof))
    if replacements:
        tracks_path=context.scene/'validated_object_tracks.json';tracks=json.loads(tracks_path.read_text())
        for gid,q in replacements.items():
            tracks[gid]['point_count']=len(q)
            nodes[gid]['residual_ownership']=dict(measured=True,generated_points=0)
        tracks_path.write_text(json.dumps(tracks,indent=2)+'\n')
        graph_path.write_text(json.dumps(graph,indent=2)+'\n')
        replace_regions(cloud_path,replacements,nodes)
        prior=Path(context.graph_geometry)
        if prior.is_file() and prior!=cloud_path:replace_regions(prior,replacements,nodes)
        context.canonical_geometry_input=graph_path
    (context.scene/'residual_ownership_audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2)+'\n')
    context.event('发布前主表面与前后排未知余量稳健审核',objects=len(replacements),generated_points=0)


def component_sha256():return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()

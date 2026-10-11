"""BASE: Bottom Anchors and Support Evidence.

Observed residual slabs under independently reconstructed compact bodies are
geometry proposals, not proof of a hidden object count. A planar parent, raised
bottom anchors, connected depth and three independent measured cell votes are
required. Names and scene/GT identifiers are not used for proposal construction.
"""
import json,hashlib
from pathlib import Path
import numpy as np,open3d as o3d
from scipy.spatial import cKDTree
from .contact_instances import frame_samples,sample_cells
from .fine_boundary import supported_samples
from partaware.geometry import load_capture
from .part_geometry import replace_regions
from .instance_consensus import Views
from .native_assembly import NativeMasks
from . import contact_instances as cmr
from .support_publication import publish_verified


def raised_groups(regions):
    proposals=[]
    for pid,parent in regions.items():
        if len(parent)<4096 or np.ptp(parent[:,:2],axis=0).min()<.5:continue
        sample=parent[::max(1,len(parent)//4096)];o3d.utility.random.seed(19)
        plane,inside=o3d.geometry.PointCloud(o3d.utility.Vector3dVector(sample)).segment_plane(.003,3,500)
        if abs(plane[2])<.99 or len(inside)<.6*len(sample):continue
        height=-plane[3]/plane[2];anchors=[]
        for gid,q in regions.items():
            if gid==pid or len(q)<64 or np.ptp(q,axis=0).max()>.4:continue
            base=float(np.quantile(q[:,2],.02));offset=base-height
            if not .008<=offset<=.05:continue
            if np.linalg.norm(np.maximum(np.maximum(q.min(0)[:2]-parent.max(0)[:2],parent.min(0)[:2]-q.max(0)[:2]),0))>.03:continue
            anchors.append((gid,q,base))
        groups=[]
        for item in anchors:
            compatible=[g for g in groups if any(abs(item[2]-old[2])<=.008 and np.linalg.norm(item[1][:,:2].mean(0)-old[1][:,:2].mean(0))<=.18 for old in g)]
            if not compatible:groups.append([item])
            else:
                target=compatible[0];target.append(item)
                for other in compatible[1:]:target.extend(other);groups.remove(other)
        for group in groups:
            if len(group)<2:continue
            points=np.concatenate([q for _,q,_ in group]);low=points.min(0);high=points.max(0);base=float(np.median([b for _,_,b in group]));low[:2]-=.035;high[:2]+=.035;low[2]=height+.004;high[2]=base+.003
            if np.ptp(points[:,:2],axis=0).max()>.65:continue
            proposals.append(dict(parent=pid,anchors=[gid for gid,_,_ in group],lower=low,upper=high,parent_plane=plane,parent_height=height,points=[],view_ids=[],frames=[]))
    return proposals


def discover(context,regions):
    proposals=raised_groups(regions)
    context.event('BASE底部锚点组',groups=[dict(parent=p['parent'],anchors=p['anchors'],lower=p['lower'].tolist(),upper=p['upper'].tolist()) for p in proposals])
    if not proposals:return []
    _,jobs,kd,kc,scale=load_capture(context.manifest,context.image_dir,context.scene/'refined_instance')
    compact=np.concatenate([q for q in regions.values() if len(q) and np.ptp(q,axis=0).max()<=.4]);tree=cKDTree(compact)
    for vi,job in enumerate(jobs):
        q,_,_,_=frame_samples(job,kd,kc,scale,stride=1)
        for proposal in proposals:
            selected=q[np.all((q>=proposal['lower'])&(q<=proposal['upper']),axis=1)]
            if len(selected)<64:continue
            selected=selected[tree.query(selected)[0]>.004]
            if len(selected)<64:continue
            selected=sample_cells(selected,.002);proposal['points'].append(selected);proposal['view_ids'].append(np.full(len(selected),vi));proposal['frames'].append(job['frame_id'])
    accepted=[]
    for proposal in proposals:
        if len(proposal['frames'])<3:continue
        q,_=supported_samples(np.concatenate(proposal['points']),np.concatenate(proposal['view_ids']),voxel=.003,minimum_views=3)
        context.event('BASE残余实测共识',anchors=proposal['anchors'],points=len(q),span=np.ptp(q,axis=0).tolist() if len(q) else [])
        if len(q)<128:continue
        labels=np.asarray(o3d.geometry.PointCloud(o3d.utility.Vector3dVector(q)).cluster_dbscan(.012,3));good=[k for k in set(labels) if k>=0]
        if not good:continue
        winner=max(good,key=lambda k:np.count_nonzero(labels==k));q=q[labels==winner];span=np.ptp(q,axis=0)
        if len(q)<128 or span[:2].min()<.04 or span[:2].max()<.08 or span[2]<.008 or span.max()>.65:continue
        remaining=q.copy();faces=[]
        for _ in range(4):
            if len(remaining)<64:break
            plane,ix=o3d.geometry.PointCloud(o3d.utility.Vector3dVector(remaining)).segment_plane(.002,3,500)
            face=remaining[ix];remaining=np.delete(remaining,ix,axis=0)
            if abs(plane[2])>=.995 and len(face)>=64 and np.ptp(face[:,:2],axis=0).min()>=.04:faces.append((len(face),plane))
        context.event('BASE平行支撑面检验',anchors=proposal['anchors'],component_points=len(q),parallel_faces=len(faces))
        if not faces:continue
        support,plane=max(faces)
        centers=np.array([np.loadtxt(next(j['pose'] for j in jobs if j['frame_id']==fid))[:3,3] for fid in proposal['frames']])
        baseline=float(np.linalg.norm(centers.max(0)-centers.min(0)))
        if baseline<.15:continue
        proof=dict(parent_id=proposal['parent'],bottom_anchor_ids=proposal['anchors'],independent_frames=proposal['frames'],camera_baseline_m=baseline,measured_points=len(q),generated_points=0,minimum_point_views=3,parent_plane=proposal['parent_plane'].tolist(),slab_plane=plane.tolist(),plane_support_fraction=support/len(q),semantic_status='unidentified_measured_support_not_a_class_prediction')
        accepted.append((q,proof))
    return accepted


def construct(context):
    metadata=json.loads(context.manifest.read_text())
    if metadata.get('world_frame')!='hypersim_world_z_up':return
    graph_path=context.scene/'topology_map.json';graph=json.loads(graph_path.read_text());nodes=graph['object_nodes']['nodes'];path=context.scene/'instance_cloud_cleaned.ply';cloud=o3d.io.read_point_cloud(str(path));p=np.asarray(cloud.points);rgb=np.rint(np.asarray(cloud.colors)*255).astype(int);ids=rgb[:,0]+255*rgb[:,1]+255**2*rgb[:,2];regions={gid:p[ids==int(gid)] for gid in nodes}
    proposals=discover(context,regions)
    audit=dict(algorithm='BASE',ground_truth_used=False,vision_api_calls=0,generated_points=0,component_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),candidates=[proof for _,proof in proposals],objects=[])
    npz=context.scene/'support_residual_candidates.npz'
    np.savez_compressed(npz,**{str(i):q for i,(q,_) in enumerate(proposals)})
    (context.scene/'support_residual_audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    context.event('BASE支撑残余候选验证',candidates=len(proposals),generated_points=0)
    if proposals:publish_verified(context,proposals,regions,graph,p,ids)

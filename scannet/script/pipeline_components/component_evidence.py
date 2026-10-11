"""CORE: Component Ownership and Residual Evidence.

Quarantine remote minority fragments only when signed, jointly depth-visible
masks repeatedly isolate the dominant body. Unknown observations do not vote.
The raw cloud and an exact residual sidecar retain every quarantined sample.
"""
import hashlib,json
from pathlib import Path
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree
from partaware.geometry import load_capture
from .instance_consensus import Views
from .native_assembly import NativeMasks
from . import contact_instances as cmr
from .part_geometry import replace_regions


def disconnected_candidate(q):
    if len(q)<64 or len(q)>20000:return None
    radius=float(np.clip(np.median(cKDTree(q).query(q,k=2)[0][:,1])*3,.008,.03))
    cloud=o3d.geometry.PointCloud(o3d.utility.Vector3dVector(q))
    labels=np.asarray(cloud.cluster_dbscan(radius,3))
    eligible=[k for k in set(labels) if k>=0]
    if not eligible:return None
    winner=max(eligible,key=lambda k:np.sum(labels==k));core=q[labels==winner];residual=q[labels!=winner]
    if len(residual)<16 or len(core)/len(q)<.9:return None
    gap=float(cKDTree(core).query(residual)[0].min());diameter=float(np.linalg.norm(np.ptp(core,axis=0)))
    if gap<=max(.1,diameter):return None
    return core,residual,dict(connectivity_radius_m=radius,core_fraction=len(core)/len(q),fragment_gap_m=gap,core_diameter_m=diameter)


def acceptance(isolated,joint,baseline):
    return len(isolated)>=2 and len(joint)<3 and len(isolated)/max(len(isolated)+len(joint),1)>=.6 and baseline>=.15


def construct(context):
    _,jobs,kd,kc,scale=load_capture(context.manifest,context.image_dir,context.scene/'refined_instance')
    views=Views(context,jobs,kd,kc,scale,{});native=NativeMasks(views)
    if not native.available:return
    graph_path=context.scene/'topology_map.json';graph=json.loads(graph_path.read_text());nodes=graph['object_nodes']['nodes']
    tracks_path=context.scene/'validated_object_tracks.json';tracks=json.loads(tracks_path.read_text())
    path=context.scene/'instance_cloud_cleaned.ply';cloud=o3d.io.read_point_cloud(str(path));p=np.asarray(cloud.points);rgb=np.rint(np.asarray(cloud.colors)*255).astype(int);ids=rgb[:,0]+255*rgb[:,1]+255**2*rgb[:,2]
    audit=dict(algorithm='CORE',ground_truth_used=False,vision_api_calls=0,generated_points=0,component_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),objects=[])
    replacements={}
    for gid,node in nodes.items():
        if node.get('atomic_instance_evidence') or node.get('geometry_hypothesis'):continue
        result=disconnected_candidate(p[ids==int(gid)])
        if result is None:continue
        core,residual,proof=result;isolated=[];joint=[]
        for fid in tracks[gid]['observed_frames']:
            a,b=native.mass(core,fid),native.mass(residual,fid)
            if a is None or b is None:continue
            records=native.records(fid);public=json.loads((context.scene/'refined_instance'/f'{fid}_instance.json').read_text())
            semantics={r['object_name']:r['bert_embedding'] for r in public}
            compatible=[i for i,r in enumerate(records) if r['object_name'] in semantics and cmr.semantic_agreement(node['text_embedding'],semantics[r['object_name']])>=.8]
            if any(a[i]>=.8 and b[i]<.2 for i in compatible):isolated.append(fid)
            if any(min(a[i],b[i])>=.6 for i in compatible):joint.append(fid)
        centers=np.array([views.get(fid)[0][:3,3] for fid in isolated])
        baseline=float(np.linalg.norm(centers.max(0)-centers.min(0))) if len(centers)>1 else 0.
        accepted=acceptance(isolated,joint,baseline)
        row=dict(instance_id=gid,name=node['name'],accepted=accepted,isolated_jointly_visible_frames=isolated,whole_component_frames=joint,camera_baseline_m=baseline,core_points=len(core),residual_points=len(residual),**proof)
        audit['objects'].append(row)
        if not accepted:continue
        sidecar=context.scene/'unresolved_fragments'/f'{gid}.points.npy';sidecar.parent.mkdir(exist_ok=True);np.save(sidecar,residual)
        node['component_ownership_evidence']={**row,'residual_sidecar':str(sidecar.relative_to(context.scene)),'residual_status':'unresolved_measured_fragment_not_an_object','raw_coordinates_preserved':True}
        tracks[gid]['point_count']=len(core);tracks[gid]['component_ownership_evidence']=node['component_ownership_evidence'];replacements[gid]=core
    if replacements:
        replace_regions(path,replacements,nodes)
        if Path(context.graph_geometry)!=path:replace_regions(Path(context.graph_geometry),replacements,nodes)
        graph_path.write_text(json.dumps(graph,indent=2)+'\n');tracks_path.write_text(json.dumps(tracks,indent=2)+'\n');context.canonical_geometry_input=graph_path
        from .canonical_geometry import publish
        publish(context)
    (context.scene/'component_evidence_audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    context.event('CORE离散碎片归属与节点几何校正',corrected_roots=len(replacements))

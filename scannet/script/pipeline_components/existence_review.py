"""Withdraw identities contradicted by repeated visible ownership evidence.

No identifier, class, GT, target count or thinness deletion rule. Native
non-exclusive masks protect real small/occluded instances. Missing depth and
occlusion provide no negative vote. Coordinates remain in a residual sidecar.
"""
from pathlib import Path
import hashlib,json
import numpy as np,open3d as o3d
from partaware.geometry import load_capture
from .instance_consensus import Views
from .native_assembly import NativeMasks
from .visibility_ownership import measure
from .proposal_validation import resolve
from .whole_object_consensus import cosine


def eligible(e):
    return (e.get('linear_fragment_ratio',1.)<.02
            and e.get('semantic_identity_margin',0.)<-.02
            and e.get('semantic_object_background_margin',1.)<=.02
            and len(e.get('frame_evidence',[]))>=5 and e['visible_point_fraction']>=.8
            and e['stable_own_point_fraction']<.05
            and e['majority_background_point_fraction']>=.9
            and e['compatible_class_point_fraction']<.05
            and e['compatible_class_weighted_fraction']<.35)


def reject(e):
    return (eligible(e) and e['native_stable_point_fraction']<.05
            and e['holdout_background_views']>=5 and e['negative_camera_baseline_m']>=.15)


def native_evidence(sample,gid,node,views,native,source_frames):
    seen=np.zeros(len(sample),int);positive=seen.copy();mass=np.zeros(len(sample));total=mass.copy()
    negative=[];centers=[];evidence=[]
    # Use genuine public embeddings of native concept names, without a model request.
    prototypes={}
    for values in views.records.values():
        for r in values:
            if 'bert_embedding' in r:prototypes.setdefault(r['object_name'].casefold(),r['bert_embedding'])
    for fid in sorted(views.jobs):
        ix,w,labels=views.project(sample,fid)
        if len(ix)<16 or w.sum()<=0:continue
        pose=views.get(fid)[0];camera=(sample[ix]-pose[:3,3])@pose[:3,:3]
        uv=np.rint(camera[:,:2]/camera[:,2,None]*[views.kc[0,0],views.kc[1,1]]+[views.kc[0,2],views.kc[1,2]]).astype(int)
        masks=native.get(fid);records=native.records(fid);compatible=[]
        for i,r in enumerate(records):
            vector=prototypes.get(r['object_name'].casefold())
            if r['confidence']>=.5 and (r['object_name'].casefold()==node['name'].casefold()
                    or vector is not None and cosine(vector,node['text_embedding'])>=.8):compatible.append(i)
        hits=np.zeros(len(ix),bool)
        for i in compatible:hits|=((masks[i,uv[:,1],uv[:,0]//8]>>(7-uv[:,0]%8))&1).astype(bool)
        seen[ix]+=1;positive[ix[hits]]+=1;mass[ix]+=w*hits;total[ix]+=w
        native_mass=float((w*hits).sum()/w.sum());bg=float(w[labels==0].sum()/w.sum())
        is_negative=(fid not in source_frames and len(ix)/len(sample)>=.3 and native_mass<.2 and bg>=.8)
        if is_negative:negative.append(fid);centers.append(pose[:3,3])
        evidence.append(dict(frame_id=fid,visible_points=len(ix),native_mass=native_mass,background_mass=bg,holdout_negative=is_negative))
    stable=(positive>=3)&(mass/np.maximum(total,1e-9)>=.6)
    return dict(native_stable_point_fraction=float(stable.mean()),holdout_background_views=len(negative),
                negative_camera_baseline_m=float(np.max(np.linalg.norm(np.asarray(centers)-centers[0],axis=1))) if centers else 0.,
                native_frame_evidence=evidence)


def construct(context):
    data,jobs,kd,kc,scale=load_capture(context.manifest,context.image_dir,context.scene/'refined_instance')
    gp=context.scene/'topology_map.json';g=json.loads(gp.read_text());nodes=g['object_nodes']['nodes']
    tp=context.scene/'validated_object_tracks.json';tracks=json.loads(tp.read_text())
    records={j['frame_id']:json.loads((context.scene/'refined_instance'/f"{j['frame_id']}_updated_instance.json").read_text()) for j in jobs}
    views=Views(context,jobs,kd,kc,scale,records);native=NativeMasks(views)
    path=context.scene/'instance_cloud_cleaned.ply';cloud=o3d.io.read_point_cloud(str(path));p=np.asarray(cloud.points);rgb=np.rint(np.asarray(cloud.colors)*255).astype(int);ids=rgb[:,0]+255*rgb[:,1]+255**2*rgb[:,2]
    audit=dict(algorithm='ECHO_evidence_consensus_for_hypothesis_ownership_v18',construction_uses_ground_truth=False,
               node_or_category_whitelist=False,thinness_is_rejection=False,vision_api_calls=0,objects=[],
               component_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    removed=[]
    for gid,node in list(nodes.items()):
        q=p[ids==int(gid)]
        if len(q)<16:continue
        eigenvalues=np.linalg.eigvalsh(np.cov(q.T));linearity=float(eigenvalues[-2]/max(eigenvalues[-1],1e-12))
        semantic=tracks.get(gid,{}).get('proposal_validation',{})
        shape_evidence=dict(linear_fragment_ratio=linearity,
            semantic_identity_margin=semantic.get('semantic_identity_margin',0.),
            semantic_object_background_margin=semantic.get('semantic_object_score',1.)-semantic.get('semantic_background_score',0.))
        # Shape alone never withdraws a slender real tool. Unknown semantic
        # evidence also preserves identity; all three modalities must disagree.
        if not (linearity<.02 and shape_evidence['semantic_identity_margin']<-.02 and shape_evidence['semantic_object_background_margin']<=.02):continue
        evidence=measure(q,gid,views,records,resolve,{})
        evidence.update(shape_evidence)
        if not eligible(evidence):continue
        if not native.available:continue
        sample=q[::max(1,len(q)//2048)]
        evidence.update(native_evidence(sample,gid,node,views,native,set(tracks.get(gid,{}).get('observed_frames',[]))))
        accepted=reject(evidence);audit['objects'].append(dict(instance_id=gid,name=node['name'],withdrawn=accepted,**evidence))
        if not accepted:continue
        residual=context.scene/'unresolved_fragments'/f'{gid}.existence.points.npy';residual.parent.mkdir(exist_ok=True);np.save(residual,q)
        removed.append(gid);del nodes[gid]
        tracks[gid]['existence_status']='withdrawn_by_independent_observation';tracks[gid]['residual_points']=str(residual.relative_to(context.scene))
        for pid,part in g.get('part_nodes',{}).items():
            if str(part.get('parent_id',part.get('object_id','')))==gid:part['status']='unresolved'
    if removed:
        for path in dict.fromkeys([path,Path(context.graph_geometry)]):
            cloud=o3d.io.read_point_cloud(str(path));colors=np.rint(np.asarray(cloud.colors)*255).astype(int);current=colors[:,0]+255*colors[:,1]+255**2*colors[:,2]
            cloud=cloud.select_by_index(np.flatnonzero(~np.isin(current,[int(gid) for gid in removed])).tolist());assert o3d.io.write_point_cloud(str(path),cloud)
        g['object_relations']=[r for r in g.get('object_relations',[]) if str(r['source_id']) not in removed and str(r['target_id']) not in removed]
        g['part_relations']=[r for r in g.get('part_relations',[]) if str(r.get('target_id')) not in removed and str(r.get('source_id')) not in removed]
        gp.write_text(json.dumps(g,indent=2)+'\n');tp.write_text(json.dumps(tracks,indent=2)+'\n');context.canonical_geometry_input=gp
        from .canonical_geometry import publish
        publish(context)
    audit['withdrawn_objects']=len(removed);(context.scene/'existence_review_audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    context.event('独立可见反证撤销不稳定实例身份',withdrawn=len(removed),manual_deletions=0)

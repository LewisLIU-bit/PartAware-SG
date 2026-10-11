"""Visual verification and ownership of generic BASE measured proposals."""
import hashlib,json
from pathlib import Path
import cv2,numpy as np,open3d as o3d
from scipy.spatial import cKDTree
from partaware.geometry import load_capture
from .instance_consensus import Views
from .native_assembly import NativeMasks
from . import contact_instances as cmr
from .part_geometry import replace_regions


def prepare_jobs(context,proposals,views):
    jobs=[]
    for i,(q,_) in enumerate(proposals):
        candidates=[]
        for fid,job in views.jobs.items():
            ix,w,_=views.project(q,fid);ix=ix[w>=.7]
            if len(ix)<128:continue
            pose=views.get(fid)[0];camera=(q[ix]-pose[:3,3])@pose[:3,:3]
            uv=np.rint(camera[:,:2]/camera[:,2,None]*[views.kc[0,0],views.kc[1,1]]+[views.kc[0,2],views.kc[1,2]]).astype(int)
            lo,hi=uv.min(0),uv.max(0)+1;candidates.append(dict(candidate=i,frame_id=fid,rgb=str(job['rgb']),box=[*lo.tolist(),*hi.tolist()],points=len(ix),camera_center=pose[:3,3].tolist()))
        chosen=[]
        for job in sorted(candidates,key=lambda r:-r['points']):
            if chosen and min(np.linalg.norm(np.array(job['camera_center'])-old['camera_center']) for old in chosen)<.15:continue
            chosen.append(job)
            if len(chosen)==5:break
        jobs+=chosen
    path=context.scene/'support_residual_jobs.json';path.write_text(json.dumps(dict(repo=str(context.repo),candidates=[proof for _,proof in proposals],jobs=jobs),indent=2));return path


def verify(q,index,receipts,folder,views):
    verified=[];checks=[]
    for receipt in receipts['frames']:
        if receipt['candidate']!=index:continue
        fid=receipt['frame_id'];job=views.jobs[fid]
        if hashlib.sha256(Path(job['rgb']).read_bytes()).hexdigest()!=receipt['rgb_sha256']:raise ValueError('BASE RGB changed')
        ix,w,_=views.project(q,fid);ix=ix[w>=.7]
        pose=views.get(fid)[0];camera=(q[ix]-pose[:3,3])@pose[:3,:3];uv=np.rint(camera[:,:2]/camera[:,2,None]*[views.kc[0,0],views.kc[1,1]]+[views.kc[0,2],views.kc[1,2]]).astype(int)
        with np.load(folder/receipt['file'],allow_pickle=False) as saved:
            if str(saved['signature'])!=receipt['signature']:raise ValueError('BASE visual mask signature changed')
            masks=np.unpackbits(saved['packed_masks'],axis=-1,count=int(saved['shape'][-1])).astype(bool);scores=saved['scores']
        raw,u,v,_=cmr.frame_samples(job,views.kd,views.kc,views.scale,stride=1);choices=[]
        for k,(mask,score) in enumerate(zip(masks,scores)):
            recall=float(mask[uv[:,1],uv[:,0]].mean());samples=raw[mask[v,u]];purity=cmr.coverage(samples[::max(1,len(samples)//2048)],q,.015)
            choices.append(dict(mask_index=k,confidence=float(score),candidate_coverage=recall,measured_purity=purity))
        best=max(choices,key=lambda r:r['candidate_coverage']*r['measured_purity'],default=None)
        passed=best is not None and best['candidate_coverage']>=.7 and best['measured_purity']>=.6
        row=dict(frame_id=fid,accepted=passed,best=best,mask_file=str((folder/receipt['file']).relative_to(views.context.scene)),box=receipt['box']);checks.append(row)
        if passed:verified.append(row)
    return verified,checks


def existing_owner(q,verified,nodes,regions,views,native):
    choices=[]
    for gid,r in regions.items():
        if not len(r) or np.ptp(r,axis=0).max()>.7 or cKDTree(r).query(q)[0].min()>.015:continue
        frames=[]
        for row in verified:
            fid=row['frame_id'];left=native.mass(q,fid);right=native.mass(r[::max(1,len(r)//1024)],fid)
            if left is None or right is None:continue
            public=json.loads((views.context.scene/'refined_instance'/f'{fid}_instance.json').read_text());semantics={r['object_name']:r['bert_embedding'] for r in public}
            if any(min(left[k],right[k])>=.7 and record['object_name'] in semantics and cmr.semantic_agreement(semantics[record['object_name']],nodes[gid]['text_embedding'])>=.8 for k,record in enumerate(native.records(fid))):frames.append(fid)
        if len(frames)>=3:choices.append((len(frames),gid))
    return max(choices)[1] if choices else None


def existing_envelopes(q,formal_regions):
    return [gid for gid,r in formal_regions.items() if len(r) and np.ptp(r,axis=0).max()<=.7 and np.mean(np.all((q>=r.min(0)-.005)&(q<=r.max(0)+.005),axis=1))>=.95]


def publish_verified(context,proposals,regions,graph,p,ids):
    _,jobs,kd,kc,scale=load_capture(context.manifest,context.image_dir,context.scene/'refined_instance');views=Views(context,jobs,kd,kc,scale,{});native=NativeMasks(views);nodes=graph['object_nodes']['nodes']
    jobs_path=prepare_jobs(context,proposals,views);folder=context.scene/'support_residual_visual'
    context.execute([str(Path.home()/'miniconda3/envs/sg-sam3/bin/python'),str(Path(__file__).with_name('support_visual_worker.py')),'--jobs',str(jobs_path),'--cache-dir',str(folder)],'BASE自动几何框的本地视觉验证')
    receipts=json.loads((folder/'audit.json').read_text());tracks_path=context.scene/'validated_object_tracks.json';tracks=json.loads(tracks_path.read_text());next_id=max(map(int,nodes))+1;replacements={};unknown=[]
    formal=o3d.io.read_point_cloud(str(context.graph_geometry));fp=np.asarray(formal.points);fr=np.rint(np.asarray(formal.colors)*255).astype(int);fi=fr[:,0]+255*fr[:,1]+255**2*fr[:,2];formal_regions={gid:fp[fi==int(gid)] for gid in nodes}
    audit=json.loads((context.scene/'support_residual_audit.json').read_text());audit['publication_component_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    for i,(q,proof) in enumerate(proposals):
        verified,checks=verify(q,i,receipts,folder,views);proof['visual_verification']=checks
        if len(verified)<3:audit['objects'].append(dict(candidate=i,action='defer_unverified_surface',**proof));continue
        gid=existing_owner(q,verified,nodes,regions,views,native)
        if gid is not None:
            replacements[gid]=cmr.sample_cells(np.concatenate([replacements.get(gid,regions[gid]),q]),.002);tracks[gid]['point_count']=len(replacements[gid]);audit['objects'].append(dict(candidate=i,instance_id=gid,action='existing_whole_mask_measured_recovery',**proof));continue
        enclosed=existing_envelopes(q,formal_regions)
        if enclosed:
            side=context.scene/'unresolved_fragments'/f'support_{i}.points.npy';side.parent.mkdir(exist_ok=True);np.save(side,q);audit['objects'].append(dict(candidate=i,action='defer_existing_body_envelope',possible_owners=enclosed,residual_sidecar=str(side.relative_to(context.scene)),**proof));continue
        unknown.append((q,proof,verified))
    if unknown:
        from sentence_transformers import SentenceTransformer
        from groundingdino.util.inference import load_model
        from .yoloe_frontend import roi_features
        bert=SentenceTransformer(str(Path.home()/'.cache/torch/sentence_transformers/sentence-transformers_all-MiniLM-L6-v2'),device='cpu');semantic=bert.encode('support platform')
        root=context.repo/'scannet/script/thirdparty/Grounded-Segment-Anything';dino=load_model(str(root/'GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py'),str(root/'groundingdino_swint_ogc.pth'),device='cpu').eval()
        for q,proof,verified in unknown:
            local=[]
            for row in verified[:3]:
                record=dict(segmentation_box=row['box']);roi_features(dino,cv2.imread(str(views.jobs[row['frame_id']]['rgb'])),[record]);local.append(record['feature'])
            gid=str(next_id);next_id+=1;visual_feature=np.mean(local,axis=0)
            nodes[gid]=dict(id=gid,name='support platform',text_embedding=semantic.tolist(),visual_embedding=visual_feature.tolist(),support_residual_evidence=proof,semantic_status='geometric_role_only_category_not_identified')
            tracks[gid]=dict(observations=[],secondary_observations=verified,observed_frames=[r['frame_id'] for r in verified],name_votes={'support platform':len(verified)},point_count=len(q),confidence=.7*min(r['best']['confidence'] for r in verified),confidence_type='uncalibrated_minimum_independent_visual_verification')
            replacements[gid]=regions[gid]=q;parent=proof['parent_id'];current=replacements.get(parent,regions[parent]);shared=cKDTree(q).query(current)[0]<=.004
            if shared.any():replacements[parent]=current[~shared];tracks[parent]['point_count']=int((~shared).sum())
            for anchor in proof['bottom_anchor_ids']:graph.setdefault('object_relations',[]).append(dict(source_id=gid,target_id=anchor,description='supports',confidence=tracks[gid]['confidence']))
            audit['objects'].append(dict(instance_id=gid,name='support platform',action='new_verified_measured_support',**proof))
        del dino,bert
    if replacements:
        for product in ('instance_cloud_completed.ply','instance_cloud_regularized.ply'):
            destination=context.scene/product
            if not destination.exists():continue
            c=o3d.io.read_point_cloud(str(destination));xp=np.asarray(c.points);xc=np.rint(np.asarray(c.colors)*255).astype(int);xi=xc[:,0]+255*xc[:,1]+255**2*xc[:,2];edits={}
            for gid,q in replacements.items():
                old=p[ids==int(gid)];prior=xp[xi==int(gid)];extra=prior[cKDTree(old).query(prior)[0]>.0041] if len(old) and len(prior) else np.empty((0,3));edits[gid]=np.concatenate([q,extra])
            replace_regions(destination,edits,nodes)
        replace_regions(context.scene/'instance_cloud_cleaned.ply',replacements,nodes);graph_path=context.scene/'topology_map.json';graph_path.write_text(json.dumps(graph,indent=2)+'\n');tracks_path.write_text(json.dumps(tracks,indent=2)+'\n');context.canonical_geometry_input=graph_path
        from .canonical_geometry import publish
        publish(context)
    (context.scene/'support_residual_audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    context.event('BASE验证后支撑残余归属',candidates=len(proposals),new_instances=len(unknown),updated_objects=len(replacements),generated_points=0)

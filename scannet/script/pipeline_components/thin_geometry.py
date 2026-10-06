"""Recover under-resolved observed surfaces with automatic 3D SAM seeds.

Inspired by SAMPro3D's projected geometry prompts and SAI3D's multi-view
geometry/mask consensus. This adapter is not a reproduction of those methods.
No language prompt, category-specific branch, annotation or synthetic point
is used. Existing unrelated object regions are immutable during publication.
"""
from pathlib import Path
import hashlib,json,sys,fcntl
import cv2
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree
from partaware.geometry import load_capture,project_mask,voxel_downsample
from .part_geometry import replace_regions


def under_resolved(points):
    extent=np.ptp(points,axis=0)
    area=2*(extent[0]*extent[1]+extent[1]*extent[2]+extent[0]*extent[2])
    density=len(points)*.01**2/max(area,1e-9)
    return len(points)<512 and extent.max()>=.5 and density<.08, density


def depth_mask_votes(points,pose,depth,mask,kd,kc):
    """Pair each candidate depth sample with its own registered color label.

    A one-pixel depth neighborhood tolerates rasterization at thin boundaries;
    an unrelated neighbor's mask cannot validate a different depth sample.
    Occluded, off-image and unmeasured points abstain instead of voting no.
    """
    camera=(points-pose[:3,3])@pose[:3,:3]
    z=camera[:,2];safe=np.maximum(z,1e-8)
    uv=np.rint(camera[:,:2]/safe[:,None]*[kd[0,0],kd[1,1]]+[kd[0,2],kd[1,2]]).astype(int)
    seen=np.zeros(len(points),bool);positive=seen.copy()
    for dx,dy in [(0,0),(-1,0),(1,0),(0,-1),(0,1)]:
        u,v=uv[:,0]+dx,uv[:,1]+dy
        cu=np.rint((u-kd[0,2])*kc[0,0]/kd[0,0]+kc[0,2]).astype(int)
        cv=np.rint((v-kd[1,2])*kc[1,1]/kd[1,1]+kc[1,2]).astype(int)
        valid=(z>0)&(u>=0)&(v>=0)&(u<depth.shape[1])&(v<depth.shape[0])
        valid&=(cu>=0)&(cv>=0)&(cu<mask.shape[1])&(cv<mask.shape[0])
        indices=np.flatnonzero(valid)
        measured=depth[v[indices],u[indices]]
        good=(measured>0)&(np.abs(measured-z[indices])<=.004+.001*z[indices])
        indices=indices[good]
        seen[indices]=True
        positive[indices]|=mask[cv[indices],cu[indices]]
    return seen,positive


def supported(points,observations,kd,kc,minimum=3,ratio=.65):
    counts=np.zeros(len(points),np.int32);positive=counts.copy()
    for pose,depth,mask in observations:
        visible,own=depth_mask_votes(points,pose,depth,mask,kd,kc)
        counts+=visible;positive+=own
    keep=(positive>=minimum)&(positive/np.maximum(counts,1)>=ratio)
    return points[keep],{'candidate_points':len(points),'supported_points':int(keep.sum()),
        'minimum_distinct_views':minimum,'minimum_visible_mask_ratio':ratio}


def substantial_components(points,minimum=64):
    cloud=o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    labels=np.asarray(cloud.cluster_dbscan(eps=.025,min_points=3))
    good=[i for i in np.unique(labels) if i>=0 and (labels==i).sum()>=minimum]
    return points[np.isin(labels,good)]


def protected_residual(points,other,tolerance=.015):
    if not len(other):return points
    return points[cKDTree(other).query(points,workers=2)[0]>tolerance]


def connected_to_seeds(seeds,points,tolerance=.05):
    combined=np.concatenate([seeds,points])
    cloud=o3d.geometry.PointCloud(o3d.utility.Vector3dVector(combined))
    labels=np.asarray(cloud.cluster_dbscan(eps=tolerance,min_points=1))
    connected=combined[np.isin(labels,np.unique(labels[:len(seeds)]))]
    return voxel_downsample(connected,.005)


def geometry_pixels(seeds,pose,depth,kd,kc):
    # Exact central-pixel depth checks avoid choosing boundary/background seeds.
    camera=(seeds-pose[:3,3])@pose[:3,:3];z=camera[:,2]
    uv=np.rint(camera[:,:2]/np.maximum(z[:,None],1e-8)*[kc[0,0],kc[1,1]]+[kc[0,2],kc[1,2]]).astype(int)
    duv=np.rint(camera[:,:2]/np.maximum(z[:,None],1e-8)*[kd[0,0],kd[1,1]]+[kd[0,2],kd[1,2]]).astype(int)
    valid=(z>0)&(duv[:,0]>=0)&(duv[:,1]>=0)&(duv[:,0]<depth.shape[1])&(duv[:,1]<depth.shape[0])
    indices=np.flatnonzero(valid)
    valid[indices]&=np.abs(depth[duv[indices,1],duv[indices,0]]-z[indices])<=.004+.001*z[indices]
    return np.unique(uv[valid],axis=0)


def farthest_pixels(pixels,maximum=16):
    if not len(pixels):return pixels
    chosen=[0];distance=np.full(len(pixels),np.inf)
    for _ in range(min(maximum,len(pixels))-1):
        distance=np.minimum(distance,np.sum((pixels-pixels[chosen[-1]])**2,axis=1))
        chosen.append(int(distance.argmax()))
    return pixels[chosen].astype(float)


def select_geometry_masks(masks,scores,pixels):
    """Geometric seed coverage gates every candidate before model-score ranking."""
    audit=[];accepted=[]
    for mask,score in zip(masks,scores):
        coverage=float(mask[pixels[:,1],pixels[:,0]].mean())
        valid=float(score)>=.65 and coverage>=.85 and float(mask.mean())<=.3
        audit.append({'sam_score':float(score),'seed_coverage':coverage,'mask_pixels':int(mask.sum()),'accepted':valid})
        if valid:accepted.append(mask)
    return np.any(accepted,axis=0) if accepted else None,audit


def automatic_masks(context,seeds,source,kd,kc,scale,directory):
    """Use frozen observed 3D seeds, never text, to propose missing structure."""
    import torch
    sam_root=context.repo/'scannet/script/thirdparty/Grounded-Segment-Anything/segment_anything'
    sys.path.insert(0,str(sam_root))
    from segment_anything import sam_model_registry,SamPredictor
    checkpoint=context.repo/'scannet/script/thirdparty/Grounded-Segment-Anything/sam_vit_h_4b8939.pth'
    proposals=[];candidates=[];audit=[]
    eligible=[]
    for job,pose,depth,_ in source:
        pixels=geometry_pixels(seeds,pose,depth,kd,kc)
        if len(pixels)>=64:eligible.append((job,pose,depth,pixels))
    # Evenly sample valid observations across all cameras, avoiding early-view bias.
    indices=np.unique(np.linspace(0,len(eligible)-1,min(16,len(eligible))).astype(int)) if eligible else []
    lock_dir=Path.home()/'.cache/partaware-sg';lock_dir.mkdir(parents=True,exist_ok=True)
    with (lock_dir/'gpu.lock').open('a') as lock:
        fcntl.flock(lock.fileno(),fcntl.LOCK_EX)
        model=sam_model_registry['vit_h'](checkpoint=str(checkpoint)).to('cuda' if torch.cuda.is_available() else 'cpu')
        predictor=SamPredictor(model)
        for index in indices:
            job,pose,depth,pixels=eligible[index]
            rgb=cv2.imread(str(job['rgb']))
            pixels=pixels[(pixels[:,0]>=0)&(pixels[:,1]>=0)&(pixels[:,0]<rgb.shape[1])&(pixels[:,1]<rgb.shape[0])]
            if len(pixels)<64:continue
            selected=farthest_pixels(pixels)
            predictor.set_image(cv2.cvtColor(rgb,cv2.COLOR_BGR2RGB))
            lo,hi=pixels.min(0),pixels.max(0)
            margin=.1*float(max(hi-lo))
            box=np.r_[np.maximum(lo-margin,0),np.minimum(hi+margin,[rgb.shape[1]-1,rgb.shape[0]-1])]
            candidate_masks=[];candidate_scores=[]
            for region in [None,box]:
                masks,scores,_=predictor.predict(point_coords=selected,point_labels=np.ones(len(selected),int),
                    box=region,multimask_output=True)
                candidate_masks.extend(masks);candidate_scores.extend(scores)
            mask,mask_audit=select_geometry_masks(candidate_masks,candidate_scores,pixels)
            audit.append({'frame_id':job['frame_id'],'candidates':mask_audit,'accepted':mask is not None,
                'prompt_type':'projected_measured_3d_points_and_derived_roi','pixel_roi':box.tolist()})
            if mask is None:continue
            np.savez_compressed(directory/f'{job["frame_id"]}.npz',mask=mask)
            p,_=project_mask(mask,rgb,(depth*scale).round().astype(np.uint16),pose,kd,kc,scale,stride=1,max_depth=context.max_depth)
            candidates.append(p);proposals.append((pose,depth,mask))
        del predictor,model
        if torch.cuda.is_available():torch.cuda.empty_cache()
    return candidates,proposals,audit


def construct(context):
    graph_path=context.scene/'topology_map.json';graph=json.loads(graph_path.read_text())
    nodes=graph['object_nodes']['nodes']
    cloud_path=context.scene/'instance_cloud_cleaned.ply'
    cloud=o3d.io.read_point_cloud(str(cloud_path));points=np.asarray(cloud.points)
    colors=np.rint(np.asarray(cloud.colors)*255).astype(int);ids=colors[:,0]+255*colors[:,1]+255**2*colors[:,2]
    geometry={i:points[ids==int(i)] for i in nodes}
    report={'algorithm':'class_agnostic_measured_thin_surface_consensus_v8','ground_truth_used':False,
        'language_prompts':0,'qwen_api_calls':0,'generated_points':0,'objects':[],
        'component_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    triggers=[i for i,p in geometry.items() if under_resolved(p)[0]]
    raw_path=context.scene/'object_tracks.json'
    if not triggers or not raw_path.exists():
        report['reason']='没有满足稀疏几何触发条件的已验收物体，保留全部点云'
        (context.scene/'thin_geometry_audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
        context.event('细结构实测恢复检查完成',triggered_objects=0,changed_objects=0)
        return
    raw=json.loads(raw_path.read_text())
    semantic={str(r['instance_id']):np.asarray(r['feature']) for r in json.loads((context.scene/'instance_bert_embeddings.json').read_text())}
    _,jobs,kd,kc,scale=load_capture(context.manifest,context.image_dir,context.scene/'refined_instance')
    replacements={};directory=context.scene/'geometry_recovery';directory.mkdir(exist_ok=True)
    for gid in triggers:
        vector=np.asarray(nodes[gid]['text_embedding']);norm=np.linalg.norm(vector)
        cohort={i for i,v in semantic.items() if float(vector@v/max(norm*np.linalg.norm(v),1e-8))>=.65}
        entry={'instance_id':gid,'original_points':len(geometry[gid]),'surface_density':under_resolved(geometry[gid])[1],
            'cohort_tracks':sorted(cohort,key=int),'status':'rejected'}
        report['objects'].append(entry)
        if len(cohort&set(nodes))!=1:
            entry['reason']='存在多个独立已验收语义相近实例，不能唯一归属';continue
        frame_labels={}
        for i in cohort:
            for observation in raw[i]['observations']:
                frame_labels.setdefault(observation['frame_id'],set()).add(int(observation['local_id']))
        if len(frame_labels)<12:
            entry['reason']='独立缓存观测少于12帧';continue
        source=[];pool=[]
        for job in jobs:
            fid=job['frame_id']
            if fid not in frame_labels:continue
            pose=np.loadtxt(job['pose']);depth_raw=cv2.imread(str(job['depth']),-1)
            rgb=cv2.imread(str(job['rgb']));mask=cv2.imread(str(context.scene/'refined_instance'/f'{fid}.png'),-1)
            region=np.isin(mask,list(frame_labels[fid]))
            p,_=project_mask(region,rgb,depth_raw,pose,kd,kc,scale,stride=1,max_depth=context.max_depth)
            pool.append(p);source.append((job,pose,depth_raw/scale,region))
        candidates=voxel_downsample(np.concatenate(pool),.005)
        seeds,evidence=supported(candidates,[(p,d,m) for _,p,d,m in source],kd,kc)
        seeds=substantial_components(seeds)
        other=np.concatenate([p for i,p in geometry.items() if i!=gid]) if len(geometry)>1 else np.empty((0,3))
        seeds=protected_residual(seeds,other)
        entry.update(source_frames=len(source),dense_mask_evidence=evidence,verified_seed_points=len(seeds))
        if len(seeds)<max(128,3*len(geometry[gid])):
            entry['reason']='三视角确认的新表面不足，保持原物体';continue
        local=directory/gid;local.mkdir(exist_ok=True)
        context.event('实测稀疏物体启动几何种子分割',instance_id=gid,seed_points=len(seeds),source_frames=len(source))
        extra,observations,sam_audit=automatic_masks(context,seeds,source,kd,kc,scale,local)
        final=seeds
        if len(observations)>=3:
            expanded=voxel_downsample(np.concatenate(extra),.005)
            expanded=protected_residual(expanded,other)
            accepted,expansion_evidence=supported(expanded,observations,kd,kc,ratio=.65)
            final=connected_to_seeds(seeds,accepted)
            entry['expansion_evidence']=expansion_evidence
        final=protected_residual(final,other)
        entry.update(status='accepted',published_measured_points=len(final),sam_frames=sam_audit,
            final_bounds=[final.min(0).tolist(),final.max(0).tolist()],publication_voxel_m=.005)
        replacements[gid]=final
        nodes[gid]['measured_geometry_recovery']={'algorithm':report['algorithm'],'observed_frames':len(source),
            'measured_point_count':len(final),'synthetic_point_count':0}
    if replacements:
        geometry_path=getattr(context,'graph_geometry',cloud_path)
        replace_regions(cloud_path,replacements,nodes)
        if geometry_path!=cloud_path:replace_regions(geometry_path,replacements,nodes)
        graph_path.write_text(json.dumps(graph,indent=2)+'\n')
        context.graph_geometry=geometry_path;context.canonical_geometry_input=graph_path
        from . import canonical_geometry
        canonical_geometry.publish(context)
        graph=json.loads(graph_path.read_text())
        graph['scene_graph']['edges']=[e for h in graph['edge_hypotheses'].values() for e in h['edges'].values()]+graph.get('part_relations',[])
        graph['geometry_provenance']['thin_geometry_component']=__name__
        graph_path.write_text(json.dumps(graph,indent=2)+'\n')
        (context.scene/'parts/partaware_graph.json').write_text(json.dumps(graph,indent=2)+'\n')
    (context.scene/'thin_geometry_audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    context.event('几何种子与多视角实测深度恢复完成',changed_objects=len(replacements),other_objects_preserved=True,generated_points=0)

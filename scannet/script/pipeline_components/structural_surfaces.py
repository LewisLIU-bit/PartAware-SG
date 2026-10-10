"""Recover bounded measured structural objects suppressed by exclusive masks.

Large connected planes propose storage facades and support slabs. Cached
semantic observations, measured rear boundaries, and repeated depth jointly
validate them. This is a class-conditioned geometric adapter, not a learned
storage model. Names and dimensions are never supplied for a particular scene.
"""
from pathlib import Path
import copy,hashlib,json
import cv2,numpy as np,open3d as o3d
from scipy.spatial import cKDTree
from partaware.geometry import load_capture,voxel_downsample
from .thin_geometry import depth_mask_votes
from .part_geometry import replace_regions


def family(name):
    tokens=name.casefold().replace('_',' ').split()
    if any(t in tokens for t in ['shelf','shelves','cabinet','bookshelf']):return 'storage'
    if any(t in tokens for t in ['counter','countertop']):return 'support'
    return None


def plane_regions(raw):
    remaining=o3d.geometry.PointCloud(o3d.utility.Vector3dVector(raw)).voxel_down_sample(.012)
    proposals=[]
    for _ in range(28):
        if len(remaining.points)<700:break
        plane,ix=remaining.segment_plane(.009,3,500)
        if len(ix)<700:break
        cloud=remaining.select_by_index(ix);remaining=remaining.select_by_index(ix,invert=True)
        labels=np.asarray(cloud.cluster_dbscan(.045,3));p=np.asarray(cloud.points)
        for label in np.unique(labels):
            if label<0:continue
            q=p[labels==label]
            if len(q)<700:continue
            normal=np.array(plane[:3]);values,vectors=np.linalg.eigh(np.cov(q.T));basis=vectors[:,1:]
            extent=np.ptp(q@basis,axis=0)
            if min(extent)<.5 or max(extent)<1.:continue
            kind='support' if abs(normal[2])>.995 else 'storage' if abs(normal[2])<.01 else None
            if kind:proposals.append((q,normal,kind))
    return proposals


def rear_boundary(front,normal,background,cameras,kind):
    """A detached face needs an observed parallel rear, not assumed thickness."""
    if kind=='support':normal=np.array([0.,0.,1.])
    elif np.median((cameras-front.mean(0))@normal)<0:normal=-normal
    tangent=np.cross(normal,[0.,0.,1.]) if kind=='storage' else np.array([1.,0.,0.])
    tangent/=np.linalg.norm(tangent);second=np.cross(normal,tangent)
    rotation=np.column_stack([normal,tangent,second]);face=front@rotation
    lo,hi=face.min(0),face.max(0);level=float(np.median(face[:,0]))
    bg=background@rotation;maximum=.22 if kind=='support' else .8;minimum=.04 if kind=='support' else .12
    search=(bg[:,0]<level-minimum)&(bg[:,0]>level-maximum)
    search&=np.all((bg[:,1:]>=lo[1:]-.15)&(bg[:,1:]<=hi[1:]+.15),axis=1)
    q=bg[search]
    if len(q)<256:return None,{'reason':'没有足够实测后方/下方边界'}
    remaining=o3d.geometry.PointCloud(o3d.utility.Vector3dVector(q));options=[]
    for _ in range(8):
        if len(remaining.points)<256:break
        plane,ix=remaining.segment_plane(.009,3,300)
        rear=np.asarray(remaining.points)[ix];remaining=remaining.select_by_index(ix,invert=True)
        parallel=abs(float(plane[0]))/np.linalg.norm(plane[:3])
        if parallel>=.995 and len(ix)>=256:options.append((len(ix),rear,parallel))
    if not options:return None,{'reason':'后方边界不稳定或不平行'}
    _,rear,parallel=max(options,key=lambda item:item[0])
    depth=level-float(np.median(rear[:,0]))
    if not minimum<depth<maximum:return None,{'reason':'纵深没有满足可辨识范围'}
    return (rotation,level,depth,lo,hi),{'measured_rear_depth_m':depth,'rear_plane_points':len(rear),'rear_parallelism':parallel}


def measure(points,jobs,kd,kc,scale):
    count=np.zeros(len(points),int);free=count.copy();frame_ids=[];centers=[]
    for j in jobs:
        if 'registered_depth' not in j:
            j['registered_depth']=cv2.imread(str(j['depth']),-1).astype(np.float32)/scale
            j['pose_matrix']=np.loadtxt(j['pose'])
            j['mask_shape']=cv2.imread(str(j['rgb'])).shape[:2]
        pose,depth=j['pose_matrix'],j['registered_depth'];mask=np.ones(j['mask_shape'],bool)
        # Only depth visibility is used here; the positive identity comes from
        # cached object observations, not from treating a zero label as a wall.
        seen,_=depth_mask_votes(points,pose,depth,mask,kd,kc);count+=seen
        if seen.sum()>=64:frame_ids.append(j['frame_id']);centers.append(pose[:3,3])
        camera=(points-pose[:3,3])@pose[:3,:3];z=camera[:,2]
        uv=np.rint(camera[:,:2]/np.maximum(z[:,None],1e-8)*[kd[0,0],kd[1,1]]+[kd[0,2],kd[1,2]]).astype(int)
        valid=(z>0)&(uv[:,0]>=0)&(uv[:,1]>=0)&(uv[:,0]<depth.shape[1])&(uv[:,1]<depth.shape[0])
        ix=np.flatnonzero(valid);d=depth[uv[ix,1],uv[ix,0]]
        free[ix[(d>0)&(z[ix]<d-.012-.003*d)]]+=1
    keep=(count>=3)&(free<2)
    baseline=float(np.max(np.linalg.norm(np.array(centers)-centers[0],axis=1))) if centers else 0.
    return points[keep],{'depth_support_frames':frame_ids,'camera_baseline_m':baseline,'depth_confirmed_points':int(keep.sum()),
                        'candidate_points':len(points),'minimum_depth_views':3}


def storage_identity(front,fit,support,rawtracks,verified,fallback):
    """Infer closed storage from cached category evidence and measured sides.

    A zero detector label cannot establish a wall identity. Conversely, a flat
    sheet, an open rack, or repeated votes without a measured enclosure cannot
    establish a cabinet. No new text or model request is needed.
    """
    rotation,level,depth,lo,hi=fit
    local=front@rotation
    shape=np.maximum(1,np.ceil((hi[1:]-lo[1:])/.025).astype(int))
    cells=np.clip(np.floor((local[:,1:]-lo[1:])/.025).astype(int),0,shape-1)
    closure=len(np.unique(cells,axis=0))/int(np.prod(shape))
    body=verified@rotation
    inset=level-body[:,0]
    near_edge=np.any((abs(body[:,1:]-lo[1:])<.03)|(abs(body[:,1:]-hi[1:])<.03),axis=1)
    sides=near_edge&(inset>=.15*depth)&(inset<=depth-.015)
    reaches=bool(np.any(sides&(inset>=.5*depth)))
    frames=sorted({o['frame_id'] for gid,_ in support for o in rawtracks[gid]['observations']
        if any(token in o['name'].casefold().replace('_',' ').split() for token in ['cabinet','cupboard'])})
    accepted=closure>=.6 and int(sides.sum())>=128 and reaches and len(frames)>=3
    evidence={'front_occupancy':closure,'measured_side_points':int(sides.sum()),
        'side_reaches_half_depth':reaches,'cached_cabinet_frames':frames,
        'minimum_cabinet_frames':3,'minimum_front_occupancy':.6,
        'source_name':fallback,'inferred_name':'cabinet' if accepted else fallback,
        'accepted':accepted,'qwen_api_calls':0,'language_prompts':0}
    return ('cabinet' if accepted else fallback),evidence


def construct(context, proposal_builder=plane_regions, sample_builder=voxel_downsample, recovery_gate=None):
    data,jobs,kd,kc,scale=load_capture(context.manifest,context.image_dir,context.scene/'refined_instance')
    report={'algorithm':'bounded_cached_identity_planar_structural_recovery_v10','ground_truth_used':False,
        'qwen_api_calls':0,'language_prompts':0,'generated_points':0,'objects':[],
        'component_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    source=context.scene/'instance_cloud_with_background.ply'
    if data.get('world_frame')!='hypersim_world_z_up' or not source.exists():
        report['reason']='输入缺少竖直轴或实测背景，保持基本接口'
    else:
        o3d.utility.random.seed(10)
        cloud=o3d.io.read_point_cloud(str(source));raw=np.asarray(cloud.points);encoded=np.rint(np.asarray(cloud.colors)*255).astype(int)
        rawids=encoded[:,0]+255*encoded[:,1]+255**2*encoded[:,2];background=raw[rawids==0]
        rawtracks=json.loads((context.scene/'object_tracks.json').read_text())
        graph_path=context.scene/'topology_map_cleaned.json';graph=json.loads(graph_path.read_text());nodes=graph['object_nodes']['nodes']
        path=context.scene/'instance_cloud_cleaned.ply';cloud=o3d.io.read_point_cloud(str(path));p=np.asarray(cloud.points)
        enc=np.rint(np.asarray(cloud.colors)*255).astype(int);ids=enc[:,0]+255*enc[:,1]+255**2*enc[:,2]
        geometry={gid:p[ids==int(gid)] for gid in nodes};tracks_path=context.scene/'validated_object_tracks.json';tracks=json.loads(tracks_path.read_text())
        semantics={str(r['instance_id']):r['feature'] for r in json.loads((context.scene/'instance_bert_embeddings.json').read_text())}
        visuals={str(r['instance_id']):r['feature'] for r in json.loads((context.scene/'averaged_instance_features.json').read_text())}
        raw_regions={gid:raw[rawids==int(gid)] for gid in rawtracks};cameras=np.array([np.loadtxt(j['pose'])[:3,3] for j in jobs])
        protected=cKDTree(p);used=np.empty((0,3));replacements={};next_id=max(map(int,rawtracks))+1;reassigned={}
        for front,normal,kind in proposal_builder(raw):
            entry={'kind':kind,'front_points':len(front),'status':'rejected','front_bounds':[front.min(0).tolist(),front.max(0).tolist()]}
            report['objects'].append(entry)
            tree=cKDTree(front);support=[]
            for gid,track in rawtracks.items():
                name=max(track['name_votes'],key=track['name_votes'].get)
                if family(name)!=kind or len(track['observed_frames'])<3:continue
                q=raw_regions[gid]
                if not len(q):continue
                coverage=float(np.mean(tree.query(q)[0]<.025))
                if coverage>=.55 and len(q)*coverage>=256:
                    support.append((gid,len(q)*coverage*track['confidence']))
            if not support:entry['reason']='缺少三帧缓存的兼容物体身份';continue
            entry['cached_identity_tracks']=[g for g,_ in support]
            fit,evidence=rear_boundary(front,normal,background,cameras,kind);entry.update(evidence)
            if fit is None:continue
            rotation,level,depth,lo,hi=fit;local=raw@rotation
            bounded=(local[:,0]<=level+.015)&(local[:,0]>=level-depth+.012)
            bounded&=np.all((local[:,1:]>=lo[1:]-.015)&(local[:,1:]<=hi[1:]+.015),axis=1)
            # Slabs may be L-shaped: tangent occupancy is never replaced by the
            # enclosing rectangle, which could sweep through the room interior.
            if kind=='support':bounded&=cKDTree((front@rotation)[:,1:]).query(local[:,1:])[0]<.035
            candidate=raw[bounded]
            if len(used):candidate=candidate[cKDTree(used).query(candidate)[0]>.015]
            candidate=candidate[protected.query(candidate)[0]>.015]
            candidate=sample_builder(candidate,.01)
            if len(candidate)<512:entry['reason']='没有足够未分配的实测表面';continue
            verified,depth_evidence=measure(candidate,jobs,kd,kc,scale);entry.update(depth_evidence)
            adequate = (len(verified)>=512 and len(verified)>=.65*len(candidate)
                and depth_evidence['camera_baseline_m']>=.08) if recovery_gate is None else recovery_gate(verified, candidate, depth_evidence, kind)
            if not adequate:
                entry['reason']='三视角实测表面、相机基线不足或自由空间冲突';continue
            source_id=max(support,key=lambda a:a[1])[0];source_track=copy.deepcopy(rawtracks[source_id])
            parents=[]
            for key,old in geometry.items():
                if family(nodes[key]['name'])!=kind:continue
                coords=old@rotation
                contained=(coords[:,0]<=level+.03)&(coords[:,0]>=level-depth-.015)
                contained&=np.all((coords[:,1:]>=lo[1:]-.03)&(coords[:,1:]<=hi[1:]+.03),axis=1)
                if contained.mean()>=.9:parents.append(key)
            # A confirmed compatible whole object may gain residual surfaces;
            # independent existing objects are never removed or rewritten.
            gid=max(parents,key=lambda key:len(geometry[key])) if parents else str(next_id)
            if not parents:next_id+=1
            name=nodes[gid]['name'] if parents else max(source_track['name_votes'],key=source_track['name_votes'].get)
            if kind=='storage':
                name,identity=storage_identity(front,fit,support,rawtracks,verified,name)
                entry['storage_identity']=identity
            node=nodes[gid] if parents else {'id':gid,'name':name,'text_embedding':semantics[source_id],'visual_embedding':visuals[source_id]}
            node['name']=name
            if kind=='storage':node['storage_identity']=identity
            node['structural_surface_recovery']={'measured':True,'cached_identity_source':source_id,'audit':'structural_surface_audit.json'}
            if parents:
                verified=np.concatenate([geometry[gid],verified]);source_track=copy.deepcopy(tracks[gid])
            nodes[gid]=node;geometry[gid]=verified;replacements[gid]=verified
            source_track.update(instance_id=int(gid),point_count=len(verified),
                structural_surface_recovery={'source_tracks':[g for g,_ in support],'depth_frames':depth_evidence['depth_support_frames']})
            tracks[gid]=source_track;used=np.concatenate([used,verified])
            for source_key,_ in support:
                for observation in rawtracks[source_key]['observations']:
                    reassigned[(observation['frame_id'],int(observation['local_id']))]=gid
            entry.update(status='accepted',instance_id=gid,name=name,measured_points=len(verified),source_id=source_id,existing_parent=bool(parents),
                published_bounds=[verified.min(0).tolist(),verified.max(0).tolist()])
            context.event('结构表面通过缓存身份和多视角深度验收',instance_id=gid,name=name,measured_points=len(verified),existing_parent=bool(parents))
        if replacements:
            replace_regions(path,replacements,nodes)
            for j in jobs:
                records_path=context.scene/'refined_instance'/f'{j["frame_id"]}_updated_instance.json'
                records=json.loads(records_path.read_text())
                for record in records:
                    key=(j['frame_id'],int(record['frame_instance_id']))
                    if int(record.get('instance_id',-1))<0 and key in reassigned:
                        record['instance_id']=int(reassigned[key])
                records_path.write_text(json.dumps(records,indent=2)+'\n')
            graph_path.write_text(json.dumps(graph,indent=2)+'\n');tracks_path.write_text(json.dumps(tracks,indent=2)+'\n')
    (context.scene/'structural_surface_audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    context.event('缓存身份与几何边界联合恢复完成',accepted_objects=sum(o['status']=='accepted' for o in report['objects']),generated_points=0)

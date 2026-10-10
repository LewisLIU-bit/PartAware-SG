"""Recover measured suspension members with unique contact and depth consensus.

Geometry proposes narrow gravity-aligned members between an observed upper
facet and an independently measured ceiling. Detection absence abstains;
other accepted objects remain immutable. No category prompt or GT is used.
"""
from pathlib import Path
import hashlib,json
import cv2,numpy as np,open3d as o3d
from scipy.spatial import cKDTree
from partaware.geometry import load_capture,project_mask,voxel_downsample
from .thin_geometry import depth_mask_votes
from .part_geometry import replace_regions


def line_member(points):
    if len(points)<12:return None
    values,vectors=np.linalg.eigh(np.cov(points.T));axis=vectors[:,-1]
    span=np.ptp(points,axis=0)
    if abs(axis[2])<.98 or values[-1]<50*max(values[:-1].sum(),1e-9):return None
    if span[2]<.15 or max(span[:2])>.055:return None
    # Aligned endpoints do not establish a measured shaft across an empty gap.
    bins=np.clip(((points[:,2]-points[:,2].min())/span[2]*12).astype(int),0,11)
    if len(np.unique(bins))<10:return None
    return {'axis':axis.tolist(),'xy_center':np.median(points[:,:2],axis=0).tolist(),
            'z_bounds':[float(points[:,2].min()),float(points[:,2].max())],
            'vertical_extent_m':float(span[2]),'transverse_extent_m':span[:2].tolist()}


def connected_body_points(body,candidate):
    """Grow measured upper-body surfaces without jumping to isolated mounts."""
    if not len(candidate):return candidate
    cloud=o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.concatenate([body,candidate])))
    labels=np.asarray(cloud.cluster_dbscan(.018,1,print_progress=False))
    anchors=np.unique(labels[:len(body)])
    return candidate[np.isin(labels[len(body):],anchors)]


def upper_facet(points):
    remaining=o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points)).voxel_down_sample(.008)
    candidates=[]
    for _ in range(8):
        if len(remaining.points)<128:break
        plane,ix=remaining.segment_plane(.005,3,200)
        p=np.asarray(remaining.points)[ix]
        remaining=remaining.select_by_index(ix,invert=True)
        if abs(plane[2])<.98 or len(p)<128 or min(np.ptp(p[:,:2],axis=0))<.035:continue
        if max(np.ptp(p[:,:2],axis=0))<.3:continue
        z=float(np.median(p[:,2]))
        bottom,top=np.quantile(points[:,2],[.05,.95])
        if z<bottom+.55*(top-bottom):continue
        candidates.append((z,p))
    return max(candidates,key=lambda x:x[0]) if candidates else None


def ceiling_member_candidates(points,ceiling):
    """Seed near the ceiling so an attached crossbar cannot join all rods."""
    top=points[points[:,2]>=ceiling-.2]
    if len(top)<6:return []
    xy=np.column_stack([top[:,:2],np.zeros(len(top))])
    labels=np.asarray(o3d.geometry.PointCloud(o3d.utility.Vector3dVector(xy)).cluster_dbscan(.025,3))
    result=[]
    for label in np.unique(labels):
        if label<0:continue
        anchor=top[labels==label]
        if len(anchor)<6 or max(np.ptp(anchor[:,:2],axis=0))>.04:continue
        center=np.median(anchor[:,:2],axis=0)
        local=points[np.linalg.norm(points[:,:2]-center,axis=1)<=.022]
        if line_member(local) is not None:result.append(local)
    return result


def dense_member(member,jobs,kd,kc,scale,max_depth):
    center=np.array(member['xy_center']);low,high=member['z_bounds']
    endpoints=np.array([[*center,low],[*center,high]])
    pool=[];observations=[]
    for j in jobs:
        pose=np.loadtxt(j['pose']);rgb=cv2.imread(str(j['rgb']));depth=cv2.imread(str(j['depth']),-1)
        camera=(endpoints-pose[:3,3])@pose[:3,:3]
        if np.any(camera[:,2]<=0):continue
        uv=camera[:,:2]/camera[:,2,None]*[kc[0,0],kc[1,1]]+[kc[0,2],kc[1,2]]
        mask=np.zeros(rgb.shape[:2],np.uint8)
        width=int(np.ceil(kc[0,0]*.015/camera[:,2].min()))+2
        cv2.line(mask,tuple(np.rint(uv[0]).astype(int)),tuple(np.rint(uv[1]).astype(int)),1,width*2+1)
        if not mask.any():continue
        p,_=project_mask(mask>0,rgb,depth,pose,kd,kc,scale,stride=1,max_depth=max_depth)
        keep=(np.linalg.norm(p[:,:2]-center,axis=1)<=.018)&(p[:,2]>=low-.02)&(p[:,2]<=high+.02)
        if keep.any():pool.append(p[keep])
        observations.append((pose,depth/scale,np.ones(rgb.shape[:2],bool),j['frame_id']))
    if not pool:return np.empty((0,3)),{'depth_support_frames':[]}
    candidate=voxel_downsample(np.concatenate(pool),.003)
    count=np.zeros(len(candidate),int);frames=[];centers=[]
    for pose,depth,mask,fid in observations:
        visible,_=depth_mask_votes(candidate,pose,depth,mask,kd,kc);count+=visible
        if visible.sum()>=8:frames.append(fid);centers.append(pose[:3,3])
    baseline=float(np.max(np.linalg.norm(np.array(centers)-centers[0],axis=1))) if centers else 0.
    return candidate[count>=3],{'depth_support_frames':frames,'camera_baseline_m':baseline,'minimum_distinct_depth_views':3,
                                'dense_candidate_points':len(candidate)}


def measured_roi(lower,upper,jobs,kd,kc,scale,max_depth):
    """Revisit original pixels before fusion downsampling can erase thin rods."""
    corners=np.array([[x,y,z] for x in [lower[0],upper[0]] for y in [lower[1],upper[1]] for z in [lower[2],upper[2]]])
    pool=[]
    for j in jobs:
        pose=np.loadtxt(j['pose']);rgb=cv2.imread(str(j['rgb']));depth=cv2.imread(str(j['depth']),-1)
        camera=(corners-pose[:3,3])@pose[:3,:3]
        mask=np.zeros(rgb.shape[:2],np.uint8)
        if np.any(camera[:,2]<=.05):mask[:]=1
        else:
            uv=np.rint(camera[:,:2]/camera[:,2,None]*[kc[0,0],kc[1,1]]+[kc[0,2],kc[1,2]]).astype(np.int32)
            cv2.fillConvexPoly(mask,cv2.convexHull(uv),1)
        q,_=project_mask(mask>0,rgb,depth,pose,kd,kc,scale,stride=1,max_depth=max_depth)
        q=q[np.all((q>=lower)&(q<=upper),axis=1)]
        if len(q):pool.append(q)
    return voxel_downsample(np.concatenate(pool),.003) if pool else np.empty((0,3))


def construct(context, require_suspension=False):
    data,jobs,kd,kc,scale=load_capture(context.manifest,context.image_dir,context.scene/'refined_instance')
    report={'algorithm':'unique_ceiling_attachment_measured_members_v10','ground_truth_used':False,
        'qwen_api_calls':0,'language_prompts':0,'generated_points':0,'objects':[],
        'component_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    source=context.scene/'instance_cloud_with_background.ply'
    if data.get('world_frame')!='hypersim_world_z_up' or not source.exists():
        report['reason']='缺少实测背景或已声明重力轴，保留基本流程'
    else:
        o3d.utility.random.seed(10)
        raw=np.asarray(o3d.io.read_point_cloud(str(source)).points)
        top=raw[raw[:,2]>=np.quantile(raw[:,2],.93)]
        plane,ix=o3d.geometry.PointCloud(o3d.utility.Vector3dVector(top)).segment_plane(.01,3,500)
        if abs(plane[2])<.98 or len(ix)<512:
            report['reason']='无法确认独立天花板面'
        else:
            ceiling=float(np.median(top[ix,2]));report['measured_ceiling_height_m']=ceiling
            graph_path=context.scene/'topology_map.json';graph=json.loads(graph_path.read_text());nodes=graph['object_nodes']['nodes']
            cloud_path=context.scene/'instance_cloud_cleaned.ply';cloud=o3d.io.read_point_cloud(str(cloud_path))
            p=np.asarray(cloud.points);c=np.rint(np.asarray(cloud.colors)*255).astype(int);ids=c[:,0]+255*c[:,1]+255**2*c[:,2]
            regions={gid:p[ids==int(gid)] for gid in nodes};replacements={}
            for gid,body in regions.items():
                if len(body)<512:continue
                facet=upper_facet(body)
                if facet is None:continue
                z,face=facet
                span=np.ptp(face[:,:2],axis=0)
                if not .3<=ceiling-z<=1.0 or min(span)>.6 or max(span)<3*min(span):continue
                lower,upper=face[:,:2].min(0)-.02,face[:,:2].max(0)+.02
                q=measured_roi(np.r_[lower,z+.015],np.r_[upper,ceiling-.018],jobs,kd,kc,scale,context.max_depth)
                if len(q)<12:continue
                members = ceiling_member_candidates(q, ceiling)
                if require_suspension and not members:
                    report.setdefault('rejected_objects', []).append(dict(instance_id=gid,
                        reason='没有实测悬挂杆，水平面不自动获得灯体扩展资格'))
                    continue
                accepted=[];audit=[]
                other=np.concatenate([v for k,v in regions.items() if k!=gid]) if len(regions)>1 else np.empty((0,3))
                # Recover the actual cuboid body first. Its top and side faces may
                # have been absent from the cached object masks, yet remain in
                # original depth. A disconnected roof cannot enter this body.
                nearby=q[q[:,2]<=z+.25]
                if len(other):nearby=nearby[cKDTree(other).query(nearby)[0]>.015]
                from .structural_surfaces import measure
                verified,body_evidence=measure(nearby,jobs,kd,kc,scale)
                anchors=body[body[:,2]>=z-.05]
                verified=connected_body_points(anchors,verified)
                verified=verified[cKDTree(body).query(verified)[0]>.003]
                if len(verified)>=256 and body_evidence['camera_baseline_m']>=.08:
                    accepted.append(verified)
                    audit.append({'role':'upper_body','new_measured_points':len(verified),**body_evidence})
                    pid=f'upper_body_{gid}'
                    np.save(context.scene/'parts'/f'{pid}.points.npy',verified)
                    graph.setdefault('part_nodes',{})[pid]={'id':pid,'name':f'{nodes[gid]["name"]}: upper body',
                        'node_type':'part','parent_id':gid,'status':'confirmed','position':verified.mean(0).tolist(),
                        'extent':np.ptp(verified,axis=0).tolist(),'point_count':len(verified),
                        'observed_frames':body_evidence['depth_support_frames'],'confidence':.8,
                        'semantic_embedding':None,'semantic_feature_space':'geometry_only_no_visual_embedding',
                        'evidence_type':'seed_connected_multiview_measured_upper_body'}
                    graph.setdefault('part_relations',[]).append({'source_id':pid,'target_id':gid,'description':'part_of',
                        'evidence_frames':len(body_evidence['depth_support_frames'])})
                for member_points in members:
                    member=line_member(member_points)
                    if member is None:continue
                    bottom=member_points[np.argsort(member_points[:,2])[:max(3,len(member_points)//10)]]
                    distance=float(cKDTree(body).query(bottom)[0].min())
                    if distance>.06 or member['z_bounds'][1]<ceiling-.10:continue
                    partners=[k for k,b in regions.items() if len(b) and cKDTree(b).query(bottom)[0].min()<.06]
                    if partners!=[gid]:continue
                    dense,evidence=dense_member(member,jobs,kd,kc,scale,context.max_depth)
                    if len(dense)<32 or len(evidence['depth_support_frames'])<3 or evidence['camera_baseline_m']<.08:continue
                    if len(other):dense=dense[cKDTree(other).query(dense)[0]>.015]
                    dense=dense[cKDTree(body).query(dense)[0]>.003]
                    if len(dense)<32 or line_member(dense) is None:continue
                    accepted.append(dense);audit.append({**member,**evidence,'nearest_body_contact_m':distance,'new_measured_points':len(dense)})
                    pid=f'suspension_{gid}_{len(audit)}'
                    np.save(context.scene/'parts'/f'{pid}.points.npy',dense)
                    graph.setdefault('part_nodes',{})[pid]={'id':pid,'name':f'{nodes[gid]["name"]}: suspension member',
                        'node_type':'part','parent_id':gid,'status':'confirmed','position':dense.mean(0).tolist(),
                        'extent':np.ptp(dense,axis=0).tolist(),'point_count':len(dense),'observed_frames':evidence['depth_support_frames'],
                        'confidence':min(.95,len(evidence['depth_support_frames'])/10),'semantic_embedding':None,'semantic_feature_space':'geometry_only_no_visual_embedding',
                        'evidence_type':'unique_ceiling_contact_multiview_depth'}
                    graph.setdefault('part_relations',[]).append({'source_id':pid,'target_id':gid,'description':'part_of','evidence_frames':len(evidence['depth_support_frames'])})
                if accepted:
                    additions=voxel_downsample(np.concatenate(accepted),.003)
                    replacements[gid]=np.concatenate([body,additions])
                    report['objects'].append({'instance_id':gid,'name':nodes[gid]['name'],'members':audit,
                        'original_points':len(body),'added_measured_points':len(additions)})
                    nodes[gid]['suspension_recovery']={'measured':True,'point_count':len(additions),
                        'member_count':sum(m.get('role')!='upper_body' for m in audit),
                        'upper_body_recovered':any(m.get('role')=='upper_body' for m in audit)}
            if replacements:
                geometry_path=context.graph_geometry
                replace_regions(cloud_path,replacements,nodes)
                if geometry_path!=cloud_path:
                    completed=o3d.io.read_point_cloud(str(geometry_path));cp=np.asarray(completed.points)
                    cc=np.rint(np.asarray(completed.colors)*255).astype(int);ci=cc[:,0]+255*cc[:,1]+255**2*cc[:,2]
                    keep_completed={gid:np.concatenate([cp[ci==int(gid)],q[len(regions[gid]):]]) for gid,q in replacements.items()}
                    replace_regions(geometry_path,keep_completed,nodes)
                graph_path.write_text(json.dumps(graph,indent=2)+'\n')
                context.canonical_geometry_input=graph_path
                from .canonical_geometry import publish
                publish(context)
                graph=json.loads(graph_path.read_text());graph['scene_graph']['edges']=[e for h in graph['edge_hypotheses'].values() for e in h['edges'].values()]+graph.get('part_relations',[])
                for gid,q in graph.get('part_nodes',{}).items():
                    if q['status']=='confirmed':graph['scene_graph']['nodes'][gid]=q
                graph_path.write_text(json.dumps(graph,indent=2)+'\n');(context.scene/'parts/partaware_graph.json').write_text(json.dumps(graph,indent=2)+'\n')
                tracks_path=context.scene/'validated_object_tracks.json';tracks=json.loads(tracks_path.read_text())
                for gid,p in replacements.items():tracks[gid]['point_count']=len(p)
                tracks_path.write_text(json.dumps(tracks,indent=2)+'\n')
    (context.scene/'suspension_geometry_audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    context.event('唯一悬挂杆实测恢复完成',changed_objects=len(report['objects']),generated_points=0)

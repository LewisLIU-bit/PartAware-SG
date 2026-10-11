"""WISP: weak measured instances verified before admission, without GT."""
import json
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from . import contact_instances as cmr
from .visible_instances import features


def coframe_separation(a, b):
    frames=set()
    for x in a['evidence']:
        for y in b['evidence']:
            if x['frame_id']!=y['frame_id'] or x['native_index']==y['native_index']:
                continue
            u,v=x['segmentation_box'],y['segmentation_box']
            intersection=max(0,min(u[2],v[2])-max(u[0],v[0]))*max(0,min(u[3],v[3])-max(u[1],v[1]))
            smaller=min((u[2]-u[0])*(u[3]-u[1]),(v[2]-v[0])*(v[3]-v[1]))
            if intersection/max(smaller,1)<.2:
                frames.add(x['frame_id'])
    return frames


def agreement(a,b):
    if coframe_separation(a,b):return 0.
    x,y=a['sample'],b['sample']
    gap=np.maximum(np.maximum(x.min(0)-y.max(0),y.min(0)-x.max(0)),0)
    if np.linalg.norm(gap)>.006:return 0.
    left,right=cmr.coverage(x,y),cmr.coverage(y,x)
    if min(left,right)<.5 or max(left,right)<.7:return 0.
    semantic=cmr.semantic_agreement(features(a),features(b))
    if semantic<.3:return 0.
    va=np.mean([e['local_record']['feature'] for e in a['evidence'] if e['local_record'] is not None],axis=0)
    vb=np.mean([e['local_record']['feature'] for e in b['evidence'] if e['local_record'] is not None],axis=0)
    if cmr.semantic_agreement(va,vb)<.65:return 0.
    return min(left,right)


def associate(groups):
    """Undersegmented masks vote for children but cannot join their identities."""
    groups=[g for g in groups if features(g) is not None]
    trees=[cKDTree(g['sample']) for g in groups]
    centers=np.array([g['sample'].mean(0) for g in groups])
    neighbors=cKDTree(centers).query_ball_point(centers,.65)
    broad=set()
    for i,g in enumerate(groups):
        candidates=[]
        for j in neighbors[i]:
            h=groups[j]
            if i==j or len(g['sample'])<1.25*len(h['sample']):continue
            if cmr.semantic_agreement(features(g),features(h))<.3:continue
            if np.mean(trees[i].query(h['sample'])[0]<.006)>=.8:candidates.append(j)
        if any(coframe_separation(groups[a],groups[b]) and cmr.coverage(groups[a]['sample'],groups[b]['sample'])<.1
               for ia,a in enumerate(candidates) for b in candidates[ia+1:]):broad.add(i)
    edges=[]
    for i,g in enumerate(groups):
        if i in broad:continue
        for j in neighbors[i]:
            if j<=i or j in broad:continue
            score=agreement(g,groups[j])
            if score:edges.append((score,i,j))
    parent=list(range(len(groups)));members={i:{i} for i in range(len(groups)) if i not in broad}
    def find(i):
        while parent[i]!=i:i=parent[i]
        return i
    for score,i,j in sorted(edges,reverse=True):
        a,b=find(i),find(j)
        if a==b or any(coframe_separation(groups[x],groups[y]) for x in members[a] for y in members[b]):continue
        parent[b]=a;members[a]|=members.pop(b)
    result=[]
    for indices in members.values():
        if len(set.union(*(groups[i]['frames'] for i in indices)))<2:continue
        best=max(indices,key=lambda i:len(groups[i]['sample']));g=dict(groups[best]);g['evidence']=sum((groups[i]['evidence'] for i in indices),[]);g['points']=sum((groups[i]['points'] for i in indices),[]);g['frames']=set.union(*(groups[i]['frames'] for i in indices));g['sample']=cmr.sample_cells(np.concatenate(g['points']),.002)
        g['name']=Counter(e['name'] for e in g['evidence']).most_common(1)[0][0];result.append(g)
    return result, len(broad)


def witness_frames(group,others):
    result=set()
    for other in others:
        if other is group:continue
        if cmr.semantic_agreement(features(group),features(other))<.8:continue
        if np.linalg.norm(group['sample'].mean(0)-other['sample'].mean(0))>.65:continue
        if cmr.coverage(group['sample'],other['sample'])>.1:continue
        result.update(coframe_separation(group,other))
    return result


def source_resolution(group,native):
    areas={}
    for e in group['evidence']:
        fid=e['frame_id'];native.get(fid)
        with np.load(native.source/'sam3_cache'/f'{fid}.npz',allow_pickle=False) as f:
            if str(f['signature'])!=native.expected[fid]['signature']:raise ValueError('Mask resolution signature changed')
            area=int(np.unpackbits(f['packed_masks'][e['native_index']]).sum())
        areas[fid]=max(areas.get(fid,0),area)
    return areas


def compatible_mask_support(q,group,views,native):
    """Independent depth-visible containment; a broad mask is only a vote."""
    semantic=features(group);votes=np.zeros(len(q),int);frames=[]
    for fid in views.jobs:
        indices,weights,_=views.project(q,fid);indices=indices[weights>=.7]
        if len(indices)<16:continue
        pose=views.get(fid)[0];camera=(q[indices]-pose[:3,3])@pose[:3,:3]
        uv=np.rint(camera[:,:2]/camera[:,2,None]*[views.kc[0,0],views.kc[1,1]]+[views.kc[0,2],views.kc[1,2]]).astype(int)
        native.get(fid)
        public=json.loads((views.context.scene/'refined_instance'/f'{fid}_instance.json').read_text())
        labels={r['object_name']:r['bert_embedding'] for r in public}
        with np.load(native.source/'sam3_cache'/f'{fid}.npz',allow_pickle=False) as f:
            if str(f['signature'])!=native.expected[fid]['signature']:raise ValueError('Mask signature changed')
            records=json.loads(str(f['records']));packed=f['packed_masks']
            candidates=[i for i,r in enumerate(records) if r['confidence']>=.5 and r['object_name'] in labels and cmr.semantic_agreement(semantic,labels[r['object_name']])>=.3]
            hits=np.zeros(len(indices),bool)
            for i in candidates:
                local=((packed[i,uv[:,1],uv[:,0]//8]>>(7-uv[:,0]%8))&1).astype(bool)
                if np.mean(local)>=.7:hits|=local
            if hits.sum()>=16:frames.append(fid);votes[indices[hits]]+=1
    return q[votes>=3],frames


def propose(groups,views,native):
    merged,broad=associate(groups);accepted=[];audit=[]
    print('ASSOCIATED',len(merged),'BROAD',broad,flush=True)
    import open3d as o3d
    existing=np.asarray(o3d.io.read_point_cloud(str(views.context.scene/'instance_cloud_cleaned.ply')).points)
    tree=cKDTree(existing)
    for g in merged:
        # This branch rescues the confidence bucket discarded before CMR.
        # High-confidence identities retain the existing v16 admission path.
        if max(e['confidence'] for e in g['evidence'])>=.7:continue
        if np.mean(tree.query(g['sample'])[0]<.006)>=.25:continue
        witnesses=witness_frames(g,merged)
        if len(witnesses)<2:continue
        frames=sorted(g['frames']);local_frames={e['frame_id'] for e in g['evidence'] if e['local_record'] is not None}
        quality=float(np.mean([max(e['confidence'] for e in g['evidence'] if e['frame_id']==fid) for fid in frames]))
        centers=np.array([e['camera_center'] for e in g['evidence']]);baseline=float(np.linalg.norm(centers.max(0)-centers.min(0)))
        if len(local_frames)<2 or baseline<.15 or quality<.5:continue
        areas=source_resolution(g,native)
        if sum(area>=128 for area in areas.values())<2:continue
        print('VERIFY',g['name'],frames,len(g['sample']),flush=True)
        q,vote_frames=compatible_mask_support(cmr.sample_cells(np.concatenate(g['points']),.002),g,views,native)
        row=dict(name=g['name'],frames=frames,independent_separation_frames=sorted(witnesses),camera_baseline_m=baseline,native_mean_confidence=quality,maximum_identity_mask_confidence=max(e['confidence'] for e in g['evidence']),original_mask_pixels=areas,minimum_original_mask_pixels=128,corroboration_frames=vote_frames,measured_points=len(q),minimum_point_views=3,generated_points=0)
        audit.append(row)
        if len(q)>=64 and np.ptp(q,axis=0).max()<=.65:accepted.append((g,q,row))
    return accepted,dict(undersegmented_identity_groups=broad,associated_groups=len(merged),candidates=audit)

import hashlib
from partaware.geometry import load_capture
from .instance_consensus import Views
from .native_assembly import NativeMasks
from .part_geometry import replace_regions
from .sam3_frontend import box_iou
from .hierarchical_masks import normalize
from .visible_instances import simultaneous_separation
from .instance_reconciliation import associate as strong_associate


def collect(context, jobs, kd, kc, scale, native):
    groups = []
    semantic = {}
    public_cache = {}
    for job in jobs:
        public = json.loads((context.scene/'refined_instance'/f"{job['frame_id']}_instance.json").read_text())
        public_cache[job['frame_id']] = public
        for record in public:
            semantic.setdefault(normalize(record['object_name']), record['bert_embedding'])
    for job in jobs:
        fid = job['frame_id']
        native.get(fid)
        points, u, v, pose = cmr.frame_samples(job, kd, kc, scale, stride=1)
        with np.load(native.source/'sam3_cache'/f'{fid}.npz', allow_pickle=False) as saved:
            if str(saved['signature']) != native.expected[fid]['signature']:
                raise ValueError('Native proposal signature changed')
            records = json.loads(str(saved['records']))
            packed = saved['packed_masks']
            for index, record in enumerate(records):
                if record['confidence'] < .5:
                    continue
                mask = ((packed[index, v, u//8] >> (7-u%8)) & 1).astype(bool)
                q = cmr.compact_component(points[mask])
                if q is None:
                    continue
                name = normalize(record['object_name'])
                local_score, local = max(((box_iou(record['segmentation_box'], r['segmentation_box']), r)
                    for r in public_cache[fid] if normalize(r['object_name']) == name or
                    (name in semantic and cmr.semantic_agreement(semantic[name], r['bert_embedding']) >= .8)),
                    key=lambda pair: pair[0], default=(0., None))
                evidence = dict(frame_id=fid, native_index=index, name=name,
                    confidence=float(record['confidence']), camera_center=pose[:3, 3].tolist(),
                    is_zoom_proposal=record['is_zoom_proposal'], segmentation_box=record['segmentation_box'],
                    local_record=local if local_score >= .35 else None)
                sample = q[::max(1, len(q)//2000)]
                new = dict(name=name, sample=sample, seed_size=len(q), points=[q], evidence=[evidence], frames={fid})
                candidates = []
                for number, old in enumerate(groups):
                    if old['name'] != name and not (name in semantic and old['name'] in semantic
                            and cmr.semantic_agreement(semantic[name], semantic[old['name']]) >= .8):
                        continue
                    if simultaneous_separation(new, old):
                        continue
                    score = cmr.association(sample, old['sample'], fid in old['frames'])
                    if score:
                        candidates.append((score, number))
                if not candidates:
                    groups.append(new)
                    continue
                old = groups[max(candidates)[1]]
                old['points'].append(q)
                old['evidence'].append(evidence)
                old['frames'].add(fid)
                if len(q) > old['seed_size']:
                    old['sample'], old['seed_size'] = sample, len(q)
        if jobs.index(job) % 10 == 0:
            context.event('通用全分辨率掩码图观测', frame=fid, groups=len(groups))
    return strong_associate(groups)


def construct(context):
    metadata,jobs,kd,kc,scale=load_capture(context.manifest,context.image_dir,context.scene/'refined_instance')
    if metadata.get('world_frame')!='hypersim_world_z_up':return
    views=Views(context,jobs,kd,kc,scale,{});native=NativeMasks(views)
    if not native.available:return
    groups=collect(context,jobs,kd,kc,scale,native)
    proposals,verification=propose(groups,views,native)
    graph_path=context.scene/'topology_map.json';graph=json.loads(graph_path.read_text());nodes=graph['object_nodes']['nodes']
    tracks_path=context.scene/'validated_object_tracks.json';tracks=json.loads(tracks_path.read_text())
    import open3d as o3d
    cloud_path=context.scene/'instance_cloud_cleaned.ply';cloud=o3d.io.read_point_cloud(str(cloud_path));p=np.asarray(cloud.points);rgb=np.rint(np.asarray(cloud.colors)*255).astype(int);ids=rgb[:,0]+255*rgb[:,1]+255**2*rgb[:,2]
    regions={gid:p[ids==int(gid)] for gid in nodes};next_id=max(map(int,nodes))+1;replacements={}
    audit=dict(algorithm='WISP',ground_truth_used=False,vision_api_calls=0,generated_points=0,component_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),verification=verification,objects=[])
    for group,q,proof in proposals:
        semantic=features(group)
        owned=[(gid,cmr.coverage(q,region,.01)) for gid,region in regions.items() if len(region)]
        maximum=max(owned,key=lambda pair:pair[1]);union=cmr.coverage(q,np.concatenate(list(regions.values())),.006)
        row={**proof,'maximum_existing_coverage':maximum,'union_existing_coverage':union}
        if maximum[1]>=.3 or union>=.25:
            row['action']='defer_existing_owner';audit['objects'].append(row);continue
        local=[e['local_record'] for e in group['evidence'] if e['local_record'] is not None]
        visual=np.mean([r['feature'] for r in local],axis=0)
        if semantic.shape!=(384,) or visual.shape!=(256,):raise ValueError('Incompatible cached object features')
        gid=str(next_id);next_id+=1
        nodes[gid]=dict(id=gid,name=group['name'],text_embedding=semantic.tolist(),visual_embedding=visual.tolist(),weak_instance_evidence=proof)
        observations=[dict(frame_id=e['frame_id'],local_id=e['local_record']['frame_instance_id'],confidence=e['confidence'],name=group['name'],mask_quality=e['local_record'].get('sam_quality_score')) for e in group['evidence'] if e['local_record'] is not None]
        tracks[gid]=dict(observations=observations,observed_frames=proof['frames'],name_votes=dict(Counter(e['name'] for e in observations)),point_count=len(q),confidence=.7*proof['native_mean_confidence'],confidence_type='uncalibrated_verified_weak_multiview_quality')
        replacements[gid]=regions[gid]=q;row.update(action='new_measured_instance',instance_id=gid);audit['objects'].append(row)
    if replacements:
        replace_regions(cloud_path,replacements,nodes)
        if Path(context.graph_geometry)!=cloud_path:replace_regions(Path(context.graph_geometry),replacements,nodes)
        graph_path.write_text(json.dumps(graph,indent=2)+'\n');tracks_path.write_text(json.dumps(tracks,indent=2)+'\n');context.canonical_geometry_input=graph_path
        from .canonical_geometry import publish
        publish(context)
    (context.scene/'weak_instance_audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    context.event('WISP弱掩码多视角验证与独立实测实例恢复',new_instances=len(replacements))

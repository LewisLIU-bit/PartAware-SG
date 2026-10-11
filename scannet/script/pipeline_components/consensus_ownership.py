"""COVE: Consensus Ownership of Visible Evidence.

Generic signed-mask recovery with independent-view counting. Two-view evidence
has its own stronger gates; synthetic points and annotation identifiers are
never inputs. Unknown depth, occlusion and absent detections do not vote against
an identity. Features are real public-model outputs, not fabricated vectors.
"""
import hashlib,json
from collections import Counter
from pathlib import Path
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree
from partaware.geometry import load_capture
from . import contact_instances as cmr
from .weak_verification import collect,associate
from .visible_instances import features
from .instance_consensus import Views
from .native_assembly import NativeMasks
from .part_geometry import replace_regions


def admission(group):
    frames=sorted(group['frames'])
    qualities=[max(e['confidence'] for e in group['evidence'] if e['frame_id']==fid) for fid in frames]
    local_frames={e['frame_id'] for e in group['evidence'] if e['local_record'] is not None}
    centers=np.array([e['camera_center'] for e in group['evidence']])
    baseline=float(np.linalg.norm(centers.max(0)-centers.min(0)))
    if baseline<.15 or not local_frames:return None
    rare=(len(frames)==2 and np.mean(qualities)>=.75 and max(qualities)>=.9)
    normal=(len(frames)>=3 and len(local_frames)>=2 and np.mean(qualities)>=.65 and max(qualities)>=.8)
    if not (rare or normal):return None
    return dict(frames=frames,feature_frames=sorted(local_frames),camera_baseline_m=baseline,
        native_mean_confidence=float(np.mean(qualities)),native_max_confidence=max(qualities),
        native_min_confidence=min(qualities),
        minimum_point_views=2 if rare else 3,branch='strict_two_view' if rare else 'multiview')


def consistent_votes(points,group,views,packed_by_frame):
    positive=np.zeros(len(points),int);negative=np.zeros(len(points),int);rows=[]
    for fid in sorted(group['frames']):
        ix,w,_=views.project(points,fid);ix=ix[w>=.7]
        if not len(ix):continue
        pose=views.get(fid)[0];camera=(points[ix]-pose[:3,3])@pose[:3,:3]
        uv=np.rint(camera[:,:2]/camera[:,2,None]*[views.kc[0,0],views.kc[1,1]]+[views.kc[0,2],views.kc[1,2]]).astype(int)
        indices=sorted({e['native_index'] for e in group['evidence'] if e['frame_id']==fid})
        packed=packed_by_frame(fid)
        hit=np.any((packed[np.asarray(indices)[:,None],uv[None,:,1],uv[None,:,0]//8]>>(7-uv[None,:,0]%8))&1,axis=0)
        positive[ix[hit]]+=1;negative[ix[~hit]]+=1
        rows.append(dict(frame_id=fid,depth_visible_points=len(ix),inside_mask_points=int(hit.sum()),fraction=float(hit.mean())))
    return positive,negative,rows


def owner(q,semantic,nodes,regions):
    choices=[]
    for gid,region in regions.items():
        if len(region)<64 or len(region)>20000 or np.ptp(region,axis=0).max()>.7:continue
        if nodes[gid].get('atomic_instance_evidence') or nodes[gid].get('geometry_hypothesis'):continue
        sim=cmr.semantic_agreement(semantic,nodes[gid]['text_embedding'])
        if sim<.8:continue
        a,b=cmr.coverage(q,region,.01),cmr.coverage(region,q,.01)
        if a>=.5 and b>=.7:choices.append((a*b,gid,a,b,sim))
    choices.sort(reverse=True)
    if choices and (len(choices)==1 or choices[0][0]>=1.5*choices[1][0]):return choices[0]
    return None


def supported_trim(prior,measured,positive,negative):
    """A projected occluded point is absent from both vote arrays, so survives."""
    distance=cKDTree(measured).query(prior)[0]
    ratio=positive/np.maximum(positive+negative,1)
    remove=(ratio<.4)&(negative>=3)&(distance>.01)
    if remove.mean()>.3:return prior,np.zeros(len(prior),bool)
    return prior[~remove],remove


def axial_envelope(measured):
    """Robust circular cross-section from a broad measured arc, not its name."""
    zlo,zhi=np.quantile(measured[:,2],[.2,.8]);xy=measured[(measured[:,2]>=zlo)&(measured[:,2]<=zhi),:2]
    xy=xy[::max(1,len(xy)//2048)]
    if len(xy)<128:return None
    rng=np.random.default_rng(19);best=None
    for _ in range(256):
        a,b,c=xy[rng.choice(len(xy),3,replace=False)]
        matrix=2*np.array([b-a,c-a])
        if abs(np.linalg.det(matrix))<1e-8:continue
        center=np.linalg.solve(matrix,np.array([b@b-a@a,c@c-a@a]));radius=float(np.linalg.norm(a-center))
        if not .01<=radius<=.15:continue
        error=np.abs(np.linalg.norm(xy-center,axis=1)-radius);inliers=error<=.002
        if best is None or inliers.sum()>best[0]:best=(inliers.sum(),center,radius,inliers)
    if best is None or best[0]/len(xy)<.8:return None
    fit=xy[best[3]];matrix=np.column_stack([2*fit,np.ones(len(fit))]);solution=np.linalg.lstsq(matrix,np.sum(fit**2,axis=1),rcond=None)[0];center=solution[:2];radius=float(np.sqrt(max(solution[2]+center@center,0)))
    angles=np.sort(np.arctan2(fit[:,1]-center[1],fit[:,0]-center[0])%(2*np.pi));coverage=2*np.pi-np.diff(np.r_[angles,angles[0]+2*np.pi]).max();rms=float(np.sqrt(np.mean((np.linalg.norm(fit-center,axis=1)-radius)**2)))
    if coverage<2*np.pi/3 or rms>.002 or not .01<=radius<=.15:return None
    return dict(center_xy=center.tolist(),radius_m=radius,inlier_fraction=best[0]/len(xy),angular_coverage_radians=float(coverage),radial_rms_m=rms,margin_m=max(.005,3*rms),status='uncalibrated_measured_axial_shape_envelope')


def axial_trim(prior,measured,positive,negative):
    proof=axial_envelope(measured)
    if proof is None:return prior,np.zeros(len(prior),bool),None
    outside=np.linalg.norm(prior[:,:2]-proof['center_xy'],axis=1)>proof['radius_m']+proof['margin_m']
    ratio=positive/np.maximum(positive+negative,1)
    remove=outside&((positive<3)|((negative>=3)&(ratio<.4)))
    # A coherent attachment can be a handle or another genuine part. Preserve
    # it rather than forcing every object to fit a cylinder.
    if remove.any():
        labels=np.asarray(o3d.geometry.PointCloud(o3d.utility.Vector3dVector(prior[remove])).cluster_dbscan(.012,1))
        counts=np.bincount(labels);indexes=np.flatnonzero(remove);remove[indexes[counts[labels]>=32]]=False
    if remove.mean()>.3:return prior,np.zeros(len(prior),bool),None
    return prior[~remove],remove,proof


def construct(context):
    _,jobs,kd,kc,scale=load_capture(context.manifest,context.image_dir,context.scene/'refined_instance')
    views=Views(context,jobs,kd,kc,scale,{});native=NativeMasks(views)
    if not native.available:return
    groups=getattr(context,'full_native_groups',None)
    if groups is None:groups=collect(context,jobs,kd,kc,scale,native)
    groups,broad=associate(groups)
    graph_path=context.scene/'topology_map.json';graph=json.loads(graph_path.read_text());nodes=graph['object_nodes']['nodes']
    tracks_path=context.scene/'validated_object_tracks.json';tracks=json.loads(tracks_path.read_text())
    path=context.scene/'instance_cloud_cleaned.ply';cloud=o3d.io.read_point_cloud(str(path));p=np.asarray(cloud.points);rgb=np.rint(np.asarray(cloud.colors)*255).astype(int);ids=rgb[:,0]+255*rgb[:,1]+255**2*rgb[:,2]
    regions={gid:p[ids==int(gid)] for gid in nodes};replacements={};next_id=max(map(int,nodes))+1
    audit=dict(algorithm='COVE',ground_truth_used=False,vision_api_calls=0,generated_points=0,
        source_groups=len(groups),undersegmented_groups=broad,component_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),objects=[])
    cache={}
    def packed(fid):
        if fid not in cache:
            native.get(fid)
            with np.load(native.source/'sam3_cache'/f'{fid}.npz',allow_pickle=False) as f:
                if str(f['signature'])!=native.expected[fid]['signature']:raise ValueError('COVE signature changed')
                cache[fid]=f['packed_masks'].copy()
        return cache[fid]
    proposals=[]
    for group in groups:
        proof=admission(group)
        if proof is None:continue
        q=cmr.sample_cells(np.concatenate(group['points']),.002)
        if len(q)>12000 or np.ptp(q,axis=0).max()>.65:continue
        pos,neg,rows=consistent_votes(q,group,views,packed)
        stable=pos>=proof['minimum_point_views']
        strong=[r for r in rows if r['depth_visible_points']>=64 and r['fraction']>=.85]
        if np.count_nonzero(stable)<64 or len(strong)<2:continue
        if proof['branch']=='strict_two_view' and len(strong)!=2:continue
        # Keep measured, uncontradicted sides visible in only one of two views.
        # Identity still requires a shared stable core; occlusion is unknown.
        keep=stable|((pos>=1)&(neg==0)) if proof['branch']=='strict_two_view' else stable
        q=q[keep]
        proof.update(measured_points=len(q),projection_consensus=rows,generated_points=0)
        proposals.append((group,q,proof))
    context.event('COVE独立视角候选审核',proposals=len(proposals))
    chosen={};unowned=[]
    for group,q,proof in proposals:
        semantic=features(group);candidate=owner(q,semantic,nodes,regions)
        if candidate:
            gid=candidate[1];quality=(len(group['frames']),candidate[0])
            if gid not in chosen or quality>chosen[gid][0]:chosen[gid]=(quality,group,q,proof,candidate)
        else:unowned.append((group,q,proof))
    for gid,(_,group,q,proof,candidate) in chosen.items():
        prior=regions[gid];pos,neg,_=consistent_votes(prior,group,views,packed);kept,removed=supported_trim(prior,q,pos,neg)
        axial,axial_removed,shape_proof=axial_trim(prior,q,pos,neg)
        removed|=axial_removed;kept=prior[~removed]
        if not removed.any():continue
        new=cmr.sample_cells(np.concatenate([kept,q]),.002)
        replacements[gid]=new
        if removed.any():
            side=context.scene/'unresolved_fragments'/f'{gid}.cove.points.npy';side.parent.mkdir(exist_ok=True);np.save(side,prior[removed])
        row=dict(instance_id=gid,name=nodes[gid]['name'],action='verified_measured_surface_recovery',prior_points=len(prior),final_points=len(new),removed_unsupported_points=int(removed.sum()),axial_shape_envelope=shape_proof,semantic_agreement=candidate[4],**proof)
        nodes[gid]['consensus_ownership_evidence']=row;tracks[gid]['point_count']=len(new);audit['objects'].append(row)
    for group,q,proof in sorted(unowned,key=lambda x:-len(x[0]['frames'])):
        # Existing compact bodies keep their identities. Broad supports may own
        # a small independent object's pixels, and cannot veto its existence.
        compact=[r for r in regions.values() if len(r) and np.ptp(r,axis=0).max()<=.7]
        if compact and cmr.coverage(q,np.concatenate(compact),.006)>=.25:continue
        local=[e['local_record'] for e in group['evidence'] if e['local_record'] is not None]
        semantic=features(group);visual=np.mean([r['feature'] for r in local],axis=0)
        if semantic.shape!=(384,) or visual.shape!=(256,):raise ValueError('COVE requires genuine public features')
        # A separated visible patch of an existing semantic body is not a new
        # identity. Broad, semantically different supports still cannot veto.
        if any(cmr.semantic_agreement(semantic,nodes[pid]['text_embedding'])>=.8 and
               np.mean(np.all((q>=r.min(0)-.01)&(q<=r.max(0)+.01),axis=1))>=.8
               for pid,r in regions.items() if len(r)):continue
        gid=str(next_id);next_id+=1
        nodes[gid]=dict(id=gid,name=group['name'],text_embedding=semantic.tolist(),visual_embedding=visual.tolist(),consensus_ownership_evidence=proof)
        obs=[dict(frame_id=e['frame_id'],local_id=e['local_record']['frame_instance_id'],confidence=e['confidence'],name=group['name'],mask_quality=e['local_record'].get('sam_quality_score')) for e in group['evidence'] if e['local_record'] is not None]
        tracks[gid]=dict(observations=obs,observed_frames=proof['frames'],name_votes=dict(Counter(o['name'] for o in obs)),point_count=len(q),confidence=.7*proof['native_min_confidence'],confidence_type='uncalibrated_minimum_independent_view_quality')
        replacements[gid]=regions[gid]=q
        for parent,prior in list(regions.items()):
            if parent==gid or np.ptp(prior,axis=0).max()<=.7:continue
            current=replacements.get(parent,prior);shared=cKDTree(q).query(current)[0]<=.004
            if shared.any() and (~shared).sum()>=64:replacements[parent]=current[~shared];tracks[parent]['point_count']=int((~shared).sum())
        audit['objects'].append(dict(instance_id=gid,name=group['name'],action='new_measured_instance',**proof))
    if replacements:
        # Preserve completion-only samples in each separate product.
        for product in ('instance_cloud_completed.ply','instance_cloud_regularized.ply'):
            destination=context.scene/product
            if not destination.exists():continue
            c=o3d.io.read_point_cloud(str(destination));qp=np.asarray(c.points);cr=np.rint(np.asarray(c.colors)*255).astype(int);ci=cr[:,0]+255*cr[:,1]+255**2*cr[:,2]
            edits={}
            for gid,q in replacements.items():
                old=p[ids==int(gid)];prior=qp[ci==int(gid)]
                generated=prior[cKDTree(old).query(prior)[0]>.0041] if len(old) and len(prior) else np.empty((0,3))
                edits[gid]=np.concatenate([q,generated])
            replace_regions(destination,edits,nodes)
        replace_regions(path,replacements,nodes)
        graph_path.write_text(json.dumps(graph,indent=2)+'\n');tracks_path.write_text(json.dumps(tracks,indent=2)+'\n')
        context.canonical_geometry_input=graph_path
        from .canonical_geometry import publish
        publish(context)
    (context.scene/'consensus_ownership_audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    context.event('COVE独立实测小物体和表面归属恢复',updated_objects=len(replacements),generated_points=0)

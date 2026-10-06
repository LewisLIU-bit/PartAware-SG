"""Group uniquely attached axial supports as measured whole-object assemblies.

The policy exposes the observed body and supports as queryable parts while strict
object evaluation sees their predicted assembly. No GT grouping is consulted.
Names alone never authorize a merge. All measured coordinates are retained.
"""
from pathlib import Path
import json,hashlib
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree
from partaware.geometry import load_capture
from .instance_consensus import Views
from .part_geometry import replace_regions
from .thin_geometry import depth_mask_votes


def attachment(body,support):
    """Reject broad surfaces, loose stacks and off-axis touching objects."""
    be=np.ptp(body,axis=0);se=np.ptp(support,axis=0)
    if se[2]<.2 or se[2]<1.5*max(se[:2]) or min(be[:2])<.05:return None
    gap=float(body[:,2].min()-support[:,2].max())
    if not -.04<=gap<=.03:return None
    bottom=support[:,2].min();height=se[2]
    bands=[];centers=[]
    for lo,hi in zip(np.linspace(.2,.8,7)[:-1],np.linspace(.2,.8,7)[1:]):
        p=support[(support[:,2]>=bottom+lo*height)&(support[:,2]<bottom+hi*height)]
        if len(p)<16:return None
        bands.append(np.ptp(p[:,:2],axis=0));centers.append(p[:,:2].mean(0))
    width=np.median(bands,axis=0);center=np.mean(centers,axis=0)
    foot=support[support[:,2]<bottom+.15*height]
    if len(foot)<16:return None
    foot_width=np.ptp(foot[:,:2],axis=0)
    offset=float(np.linalg.norm(body[:,:2].mean(0)-center))
    drift=float(np.max(np.linalg.norm(np.array(centers)-center,axis=1)))
    contact=float(cKDTree(support).query(body)[0].min())
    if np.any(width>.7*be[:2]) or np.any(foot_width<1.4*width):return None
    if drift>.03 or offset>.15*np.linalg.norm(be[:2]) or contact>.02:return None
    return {'vertical_gap_m':gap,'nearest_contact_m':contact,'axis_offset_m':offset,
        'stem_drift_m':drift,'stem_xy_extent_m':width.tolist(),'foot_xy_extent_m':foot_width.tolist(),
        'body_xy_extent_m':be[:2].tolist()}


def coobserved(body,support,views,body_id,support_id,frames):
    evidence=[]
    selected=sorted(set(frames))
    selected=[selected[i] for i in np.linspace(0,len(selected)-1,min(24,len(selected))).astype(int)] if selected else []
    for fid in selected:
        mapping={int(r['frame_instance_id']):str(r.get('instance_id',-1)) for r in views.records[fid]}
        visible=[]
        for p,gid in [(body,body_id),(support,support_id)]:
            _,weights,labels=views.project(p[::max(1,len(p)//1024)],fid)
            own=np.array([mapping.get(int(k),'-1')==gid for k in labels])
            visible.append(len(labels)>=16 and float(weights[own].sum()/max(weights.sum(),1e-9))>=.6)
        if all(visible):evidence.append(fid)
    return evidence


def overhead_attachment(body,bar):
    """Require a narrow upper crossbar aligned with an observed head assembly."""
    be=np.ptp(body,axis=0);se=np.ptp(bar,axis=0)
    if se[2]>.18 or max(se[:2])<.3 or min(se[:2])>.35 or max(se[:2])<2*se[2]:return None
    if bar[:,2].min()<body[:,2].min()+.55*be[2] or bar[:,2].max()>body[:,2].max()+.05:return None
    axes=[]
    for p in [body,bar]:
        values,vectors=np.linalg.eigh(np.cov(p[:,:2].T))
        if values[1]<2.5*max(values[0],1e-9):return None
        axes.append(vectors[:,1])
    alignment=float(abs(axes[0]@axes[1]))
    distance=cKDTree(body).query(bar)[0]
    if alignment<.9 or distance.min()>.025 or (distance<.035).sum()<32:return None
    return {'nearest_contact_m':float(distance.min()),'horizontal_axis_alignment':alignment,
        'contact_points':int((distance<.035).sum()),'upper_bar_extent_m':se.tolist()}


def precise_coobserved(body,support,views,body_id,support_id,frames):
    """Check all available views using jointly registered depth and ownership."""
    evidence=[]
    for fid in sorted(set(frames)):
        pose,depth,mask=views.get(fid)
        mapping={int(r['frame_instance_id']):str(r.get('instance_id',-1)) for r in views.records[fid]}
        visible=[]
        for points,gid in [(body,body_id),(support,support_id)]:
            own=np.isin(mask,[k for k,v in mapping.items() if v==gid])
            seen,yes=depth_mask_votes(points[::max(1,len(points)//1024)],pose,depth/views.scale,own,views.kd,views.kc)
            visible.append(seen.sum()>=16 and yes.sum()/max(seen.sum(),1)>=.6)
        if all(visible):evidence.append(fid)
    return evidence


def construct(context):
    graph_path=context.scene/'topology_map.json';graph=json.loads(graph_path.read_text())
    report={'algorithm':'unique_measured_attachment_whole_object_policy_v8','ground_truth_used':False,
        'point_coordinates_preserved':True,'language_prompts':0,'objects':[],
        'component_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    data,jobs,kd,kc,scale=load_capture(context.manifest,context.image_dir,context.scene/'refined_instance')
    if data.get('world_frame')!='hypersim_world_z_up':
        report['reason']='未声明竖直轴，保留基础物体粒度'
        (context.scene/'assembly_audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n');return
    nodes=graph['object_nodes']['nodes'];cloud_path=context.scene/'instance_cloud_cleaned.ply'
    cloud=o3d.io.read_point_cloud(str(cloud_path));points=np.asarray(cloud.points)
    colors=np.rint(np.asarray(cloud.colors)*255).astype(int);ids=colors[:,0]+255*colors[:,1]+255**2*colors[:,2]
    geometry={i:points[ids==int(i)] for i in nodes}
    tracks=json.loads((context.scene/'validated_object_tracks.json').read_text())
    candidates={}
    for body,p in geometry.items():
        for support,q in geometry.items():
            if body==support:continue
            measured=attachment(p,q)
            if measured is not None:candidates.setdefault(support,[]).append((body,measured))
    unique={s:v[0] for s,v in candidates.items() if len(v)==1}
    # Both the body and support must have a unique geometric partner.
    unique={s:v for s,v in unique.items() if sum(b==v[0] for b,_ in unique.values())==1}
    unique={s:(b,m,'support') for s,(b,m) in unique.items()}
    overhead={}
    for support,q in geometry.items():
        for body,p in geometry.items():
            if body==support or not nodes[body].get('measured_geometry_recovery'):continue
            measured=overhead_attachment(p,q)
            if measured is not None:overhead.setdefault(support,[]).append((body,measured,'crossbar'))
    # Each crossbar must have one possible recovered parent; a parent may have several fragments.
    for support,values in overhead.items():
        if len(values)==1 and support not in unique:unique[support]=values[0]
    if unique:
        records={j['frame_id']:json.loads((context.scene/'refined_instance'/f"{j['frame_id']}_updated_instance.json").read_text()) for j in jobs}
        views=Views(context,jobs,kd,kc,scale,records)
        original_tracks=json.loads(json.dumps(tracks))
        recovery_records=json.loads(json.dumps(records))
        audit_path=context.scene/'thin_geometry_audit.json'
        raw_path=context.scene/'object_tracks.json'
        if audit_path.exists() and raw_path.exists():
            raw=json.loads(raw_path.read_text())
            for entry in json.loads(audit_path.read_text())['objects']:
                if entry['status']!='accepted':continue
                for source_id in entry['cohort_tracks']:
                    for observation in raw[source_id]['observations']:
                        for record in recovery_records[observation['frame_id']]:
                            if int(record['frame_instance_id'])==int(observation['local_id']):
                                record['instance_id']=int(entry['instance_id'])
        recovery_views=Views(context,jobs,kd,kc,scale,recovery_records)
        replacements={};remap={}
        for support,(body,measurement,role) in unique.items():
            if role=='crossbar':
                frames=precise_coobserved(geometry[body],geometry[support],recovery_views,body,support,list(recovery_views.jobs))
            else:
                frames=coobserved(geometry[body],geometry[support],views,body,support,
                    original_tracks[body]['observed_frames']+original_tracks[support]['observed_frames'])
            if len(frames)<3:continue
            original=geometry[body];base=geometry[support]
            # Concatenation preserves each original measurement, including contacts.
            replacements[body]=np.concatenate([replacements.get(body,original),base]);remap[support]=body
            for source_id,p,part_role in [(body,original,'body'),(support,base,role)]:
                pid=f'assembly_{source_id}_{part_role}'
                if pid in graph.get('part_nodes',{}):continue
                part={'id':pid,'name':f'{nodes[body]["name"]}: {part_role}','node_type':'part','parent_id':body,
                    'position':p.mean(0).tolist(),'extent':np.ptp(p,axis=0).tolist(),
                    'confidence':original_tracks[source_id]['confidence'],'status':'confirmed','observed_frames':frames,
                    'point_count':len(p),'observations':original_tracks[source_id]['observations'],
                    'semantic_embedding':None,'semantic_feature_space':'geometry_only_no_visual_embedding',
                    'source_object_id':source_id,'evidence_type':'unique_measured_contact_multiview'}
                graph.setdefault('part_nodes',{})[pid]=part
                graph.setdefault('part_relations',[]).append({'source_id':pid,'target_id':body,
                    'description':'part_of','evidence_frames':len(frames)})
                np.save(context.scene/'parts'/f'{pid}.points.npy',p)
            graph.setdefault('object_identity_aliases',{})[support]={'canonical_id':body,
                'source_name':nodes[support]['name'],'role':f'measured_{role}_component','evidence':measurement}
            for part in graph.get('part_nodes',{}).values():
                if str(part.get('parent_id'))==support:part['parent_id']=body
                if 'parent_evidence' in part and support in part['parent_evidence']:
                    part['parent_evidence'][body]=part['parent_evidence'].get(body,0)+part['parent_evidence'].pop(support)
            for edge in graph.get('part_relations',[]):
                if str(edge['target_id'])==support:edge['target_id']=body
            tracks[body]['observations']+=tracks[support]['observations']
            tracks[body]['observed_frames']=sorted(set(tracks[body]['observed_frames']+tracks[support]['observed_frames']))
            tracks[body]['point_count']=len(replacements[body])
            tracks[body].setdefault('assembly_source_ids',[body]).append(support)
            report['objects'].append({'body_id':body,'support_id':support,'canonical_id':body,
                'body_points':len(original),'support_points':len(base),'assembly_points':len(replacements[body]),
                'coobserved_frames':frames,'attachment_role':role,**measurement})
        if remap:
            for gid in remap:del nodes[gid];del tracks[gid]
            for fid,values in records.items():
                for record in values:
                    gid=str(record.get('instance_id',-1))
                    if gid in remap:record['instance_id']=int(remap[gid])
                (context.scene/'refined_instance'/f'{fid}_updated_instance.json').write_text(json.dumps(values,indent=2)+'\n')
            geometry_path=getattr(context,'graph_geometry',cloud_path)
            replace_regions(cloud_path,replacements,nodes)
            if geometry_path!=cloud_path:replace_regions(geometry_path,replacements,nodes)
            (context.scene/'validated_object_tracks.json').write_text(json.dumps(tracks,indent=2)+'\n')
            graph_path.write_text(json.dumps(graph,indent=2)+'\n')
            context.graph_geometry=geometry_path;context.canonical_geometry_input=graph_path
            from . import canonical_geometry
            canonical_geometry.publish(context)
            graph=json.loads(graph_path.read_text())
            parts={i:p for i,p in graph.get('part_nodes',{}).items() if p['status']=='confirmed'}
            graph['scene_graph']={'schema_version':1,'nodes':{**{i:{'id':i,'name':n['name'],'node_type':'object'} for i,n in nodes.items()},**parts},
                'edges':[e for h in graph['edge_hypotheses'].values() for e in h['edges'].values()]+graph.get('part_relations',[])}
            graph['object_granularity_policy']={'policy':'whole_object_with_unique_measured_attachments',
                'predicted_from':'measured_contact_axis_shape_and_three_view_ownership','ground_truth_used':False}
            graph_path.write_text(json.dumps(graph,indent=2)+'\n')
            (context.scene/'parts/partaware_graph.json').write_text(json.dumps(graph,indent=2)+'\n')
    (context.scene/'assembly_audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    context.event('唯一轴向支撑与上部横梁纳入整体并保留部件',assemblies=len(report['objects']),point_coordinates_preserved=True)

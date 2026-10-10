"""Restore dense actual depth to a bounded, repeatedly observed level surface.

This never creates plane samples. Mask identity, measured footprint and
independent-view votes are required; all unrelated coordinates are preserved.
"""
import hashlib,json
from pathlib import Path
import numpy as np,open3d as o3d
from scipy.spatial import cKDTree
from partaware.geometry import load_capture
from .instance_consensus import Views
from .native_assembly import NativeMasks
from .contact_instances import frame_samples, sample_cells
from .fine_boundary import supported_samples
from .hierarchical_masks import normalize
from .part_geometry import replace_regions


def construct(context):
    data,jobs,kd,kc,scale=load_capture(context.manifest,None,context.scene/'refined_instance')
    if data.get('world_frame')!='hypersim_world_z_up':return
    graph_path=context.scene/'topology_map.json';graph=json.loads(graph_path.read_text());nodes=graph['object_nodes']['nodes']
    path=context.scene/'instance_cloud_cleaned.ply'
    cloud=o3d.io.read_point_cloud(str(path));points=np.asarray(cloud.points)
    colors=np.rint(np.asarray(cloud.colors)*255).astype(int);ids=colors[:,0]+255*colors[:,1]+255**2*colors[:,2]
    tracks_path=context.scene/'validated_object_tracks.json';tracks=json.loads(tracks_path.read_text())
    candidates={}
    for gid,node in nodes.items():
        q=points[ids==int(gid)]
        if len(q)<1024 or np.ptp(q[:,:2],axis=0).min()<.2 or np.ptp(q[:,:2],axis=0).max()<.8:continue
        o3d.utility.random.seed(15)
        sample=q[::max(1,len(q)//4096)]
        plane,index=o3d.geometry.PointCloud(o3d.utility.Vector3dVector(sample)).segment_plane(.003,3,500)
        if abs(plane[2])<.995 or len(index)/len(sample)<.55:continue
        candidates[gid]=dict(original=q,plane=np.asarray(plane),tree=cKDTree(q),points=[],frames=[],identity=normalize(node['name']))
    native=NativeMasks(Views(context,jobs,kd,kc,scale,{}))
    if not native.available:return
    for view_id,job in enumerate(jobs):
        fid=job['frame_id'];q,u,v,_=frame_samples(job,kd,kc,scale,stride=1)
        for gid,candidate in candidates.items():
            signed=np.abs(q@candidate['plane'][:3]+candidate['plane'][3])
            eligible=np.flatnonzero(signed<=.003)
            if len(eligible)<64:continue
            eligible=eligible[candidate['tree'].query(q[eligible])[0]<=.035]
            if len(eligible)<64:continue
            mask=np.zeros(len(eligible),bool)
            records=native.records(fid);packed=native.get(fid)
            for index,record in enumerate(records):
                if normalize(record['object_name'])==candidate['identity'] and record['confidence']>=.5:
                    mask|=((packed[index,v[eligible],u[eligible]//8]>>(7-u[eligible]%8))&1).astype(bool)
            selected=sample_cells(q[eligible[mask]],.002)
            if len(selected):
                candidate['points'].append(selected);candidate['frames'].append(np.full(len(selected),view_id))
    replacements={};objects=[]
    for gid,candidate in candidates.items():
        if not candidate['points']:continue
        recovered,_=supported_samples(np.concatenate(candidate['points']),np.concatenate(candidate['frames']),voxel=.003,minimum_views=3)
        added=recovered[cKDTree(candidate['original']).query(recovered)[0]>.001]
        if len(added)<256:continue
        combined=np.concatenate([candidate['original'],sample_cells(added,.002)])
        replacements[gid]=combined;tracks[gid]['point_count']=len(combined)
        proof=dict(measured=True,original_points=len(candidate['original']),retained_points=len(combined),
                   additional_observed_samples=len(added),minimum_point_views=3,generated_points=0)
        nodes[gid]['surface_densification']=proof
        objects.append(dict(instance_id=gid,name=nodes[gid]['name'],**proof))
    if replacements:
        prior=Path(getattr(context,'graph_geometry',path))
        if not prior.is_file():prior=path
        replace_regions(path,replacements,nodes)
        if prior!=path:replace_regions(prior,replacements,nodes)
        tracks_path.write_text(json.dumps(tracks,indent=2)+'\n');graph_path.write_text(json.dumps(graph,indent=2)+'\n')
        context.graph_geometry=prior;context.canonical_geometry_input=graph_path
        from .canonical_geometry import publish
        publish(context)
    report=dict(algorithm='DSM_depth_surface_multiview_v15',objects=objects,ground_truth_used=False,vision_api_calls=0,
                component_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (context.scene/'surface_densification_audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    context.event('真实深度对有界实测水平面密集恢复',objects=len(objects),generated_points=0)

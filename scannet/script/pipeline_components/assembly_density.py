"""Densify measured crossbar surfaces after ownership has been established."""
from pathlib import Path
import hashlib,json
import numpy as np
import cv2,open3d as o3d
from scipy.spatial import cKDTree
from partaware.geometry import load_capture,project_mask,voxel_downsample
from .thin_geometry import supported,protected_residual
from .part_geometry import replace_regions


def construct(context):
    report={'algorithm':'verified_crossbar_dense_measurements_v9','ground_truth_used':False,
        'generated_points':0,'objects':[],'component_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    path=context.scene/'assembly_audit.json'
    entries=[e for e in json.loads(path.read_text()).get('objects',[]) if e.get('attachment_role')=='crossbar'] if path.exists() else []
    if not entries:
        (context.scene/'assembly_density_audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n');return
    _,jobs,kd,kc,scale=load_capture(context.manifest,context.image_dir,context.scene/'refined_instance')
    jobs={j['frame_id']:j for j in jobs};raw=json.loads((context.scene/'object_tracks.json').read_text())
    graph_path=context.scene/'topology_map.json';graph=json.loads(graph_path.read_text());nodes=graph['object_nodes']['nodes']
    cloud_path=context.scene/'instance_cloud_cleaned.ply';cloud=o3d.io.read_point_cloud(str(cloud_path))
    points=np.asarray(cloud.points);rgb=np.rint(np.asarray(cloud.colors)*255).astype(int);ids=rgb[:,0]+255*rgb[:,1]+255**2*rgb[:,2]
    replacements={}
    for entry in entries:
        source_id,target=entry['support_id'],entry['canonical_id']
        if source_id not in raw or target not in nodes:continue
        part_file=context.scene/'parts'/f'assembly_{source_id}_crossbar.points.npy'
        if not part_file.exists():continue
        original=np.load(part_file);frame_labels={}
        for o in raw[source_id]['observations']:frame_labels.setdefault(o['frame_id'],set()).add(int(o['local_id']))
        observations=[];pool=[]
        for fid,local_ids in sorted(frame_labels.items()):
            job=jobs[fid];pose=np.loadtxt(job['pose']);depth=cv2.imread(str(job['depth']),-1)
            mask=cv2.imread(str(context.scene/'refined_instance'/f'{fid}.png'),-1)
            region=np.isin(mask,list(local_ids));p,_=project_mask(region,cv2.imread(str(job['rgb'])),depth,pose,kd,kc,scale,stride=1,max_depth=context.max_depth)
            bounded=np.all((p>=original.min(0)-.02)&(p<=original.max(0)+.02),axis=1)
            pool.append(p[bounded]);observations.append((pose,depth/scale,region))
        if len(observations)<3:continue
        candidates=voxel_downsample(np.concatenate(pool),.005)
        accepted,evidence=supported(candidates,observations,kd,kc)
        accepted=accepted[cKDTree(original).query(accepted)[0]<=.03]
        accepted=protected_residual(accepted,points[ids!=int(target)])
        additions=accepted[cKDTree(points[ids==int(target)]).query(accepted)[0]>.003]
        report['objects'].append({'source_id':source_id,'canonical_id':target,**evidence,
            'added_measured_points':len(additions),'status':'accepted' if len(additions)>=64 else 'unchanged'})
        if len(additions)<64:continue
        replacements[target]=np.concatenate([replacements.get(target,points[ids==int(target)]),additions])
        part_id=next((k for k,v in graph.get('part_nodes',{}).items() if v.get('source_object_id')==source_id and k.endswith('_crossbar')),None)
        if part_id:
            np.save(context.scene/'parts'/f'{part_id}.points.npy',np.concatenate([original,additions]))
            graph['part_nodes'][part_id]['point_count']=len(original)+len(additions)
            graph['part_nodes'][part_id]['position']=np.concatenate([original,additions]).mean(0).tolist()
            graph['part_nodes'][part_id]['extent']=np.ptp(np.concatenate([original,additions]),axis=0).tolist()
    if replacements:
        geometry_path=context.graph_geometry
        replace_regions(cloud_path,replacements,nodes)
        if geometry_path!=cloud_path:replace_regions(geometry_path,replacements,nodes)
        graph_path.write_text(json.dumps(graph,indent=2)+'\n')
        tracks_path=context.scene/'validated_object_tracks.json'
        tracks=json.loads(tracks_path.read_text())
        for gid,region in replacements.items():tracks[gid]['point_count']=len(region)
        tracks_path.write_text(json.dumps(tracks,indent=2)+'\n')
        context.canonical_geometry_input=graph_path
        from . import canonical_geometry
        canonical_geometry.publish(context)
        graph=json.loads(graph_path.read_text())
        graph['scene_graph']['edges']=[e for h in graph['edge_hypotheses'].values() for e in h['edges'].values()]+graph.get('part_relations',[])
        (context.scene/'parts/partaware_graph.json').write_text(json.dumps(graph,indent=2)+'\n')
        graph_path.write_text(json.dumps(graph,indent=2)+'\n')
    (context.scene/'assembly_density_audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    context.event('连接结构的密集实测恢复完成',changed_objects=len(replacements),generated_points=0)

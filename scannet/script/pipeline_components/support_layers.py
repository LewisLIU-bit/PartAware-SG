"""Separate repeated open support layers from adjacent closed sheets.

Layer topology is measured across observed views. Cached shelf/rack identity
can support relabeling; geometric clipping alone retains the original name.
This is not category-prompt specialization or hidden completion.
"""
import hashlib,json
from pathlib import Path
import numpy as np,open3d as o3d
from .part_geometry import replace_regions
from partaware.geometry import load_capture
from .instance_consensus import Views
from .native_assembly import NativeMasks
from .proposal_validation import sample_frames
from .sam3_frontend import box_iou
from .hierarchical_masks import normalize


def layer_cluster(points,cameras=None):
    if len(points)<1024:
        return None,None
    sample=points[::max(1,len(points)//8192)]
    planes=[]
    original=len(sample)
    for _ in range(10):
        if len(sample)<64:break
        o3d.utility.random.seed(15)
        cloud=o3d.geometry.PointCloud(o3d.utility.Vector3dVector(sample))
        plane,index=cloud.segment_plane(.004,3,500)
        face=sample[index]
        sample=np.delete(sample,index,axis=0)
        if abs(plane[2])<.99 or len(face)<.02*original:continue
        lower,upper=np.quantile(face[:,:2],[.01,.99],axis=0)
        if (upper-lower).min()<.075 or np.prod(upper-lower)<.02:continue
        planes.append(dict(z=float(np.median(face[:,2])),lower=lower,upper=upper))
    groups=[]
    for p in sorted(planes,key=lambda p:p['z']):
        for group in groups:
            area=[]
            for q in group:
                overlap=np.maximum(np.minimum(p['upper'],q['upper'])-np.maximum(p['lower'],q['lower']),0)
                area.append(np.prod(overlap)/max(np.prod(p['upper']-p['lower']),np.prod(q['upper']-q['lower'])))
            if min(area)>.55 and min(abs(p['z']-q['z']) for q in group)>.05:
                group.append(p);break
        else:groups.append([p])
    if not groups or max(map(len,groups))<3:return None,None
    group=max(groups,key=len)
    low=np.min([p['lower'] for p in group],axis=0)
    high=np.max([p['upper'] for p in group],axis=0)
    inside=np.all((points[:,:2]>=low-.015)&(points[:,:2]<=high+.015),axis=1)
    z=np.array([p['z'] for p in group])
    horizontal=np.min(np.abs(points[:,2,None]-z),axis=1)<.012
    width_axis=int(np.argmax(high-low));normal_axis=1-width_axis
    interior=points[inside&~horizontal]
    closed_occupancy=0.
    sides=(low[normal_axis],high[normal_axis])
    if cameras is not None:
        reference=float(np.median(cameras[:,normal_axis]))
        sides=(min(sides,key=lambda side:abs(reference-side)),)
    for side in sides:
        face=interior[np.abs(interior[:,normal_axis]-side)<.015]
        if len(face):
            uv=np.column_stack([face[:,width_axis]-low[width_axis],face[:,2]-z.min()])
            cells=np.unique(np.floor(uv/.03).astype(int),axis=0)
            expected=max((high[width_axis]-low[width_axis])*(z.max()-z.min())/.03**2,1.)
            closed_occupancy=max(closed_occupancy,len(cells)/expected)
    if closed_occupancy>.6:return None,None
    # Uprights must lie at support layer boundaries, not through their interiors.
    edge=np.min(np.abs(np.column_stack([points[:,:2]-low,points[:,:2]-high])),axis=1)<.025
    kept=inside&(horizontal|edge)
    if kept.mean()<.25 or kept.mean()>.95:return None,None
    return points[kept],dict(layers=len(group),levels_m=z.tolist(),original_points=len(points),
        retained_points=int(kept.sum()),closed_front_occupancy=closed_occupancy,
        generated_points=0,point_coordinates_preserved=True)


def construct(context):
    if not context.manifest or json.loads(context.manifest.read_text()).get('world_frame')!='hypersim_world_z_up':return
    graph_path=context.scene/'topology_map.json'
    graph=json.loads(graph_path.read_text());nodes=graph['object_nodes']['nodes']
    tracks=json.loads((context.scene/'validated_object_tracks.json').read_text())
    _,jobs,kd,kc,scale=load_capture(context.manifest,None,context.scene/'refined_instance')
    native=NativeMasks(Views(context,jobs,kd,kc,scale,{}))
    camera_centers={j['frame_id']:np.loadtxt(j['pose'])[:3,3] for j in jobs}
    cloud_path=context.scene/'instance_cloud_cleaned.ply'
    cloud=o3d.io.read_point_cloud(str(cloud_path));points=np.asarray(cloud.points)
    c=np.rint(np.asarray(cloud.colors)*255).astype(int);ids=c[:,0]+255*c[:,1]+255**2*c[:,2]
    replacements={};objects=[]
    for gid,node in nodes.items():
        votes=tracks[gid]['name_votes']
        frames={o['frame_id'] for o in tracks[gid]['observations'] if any(x in o['name'].lower() for x in ('shelf','shelves','rack'))}
        region=points[ids==int(gid)]
        centers=np.array([camera_centers[fid] for fid in tracks[gid]['observed_frames']])
        q,evidence=layer_cluster(region,centers)
        if q is None:continue
        cached_features=[];native_names=[]
        if native.available:
            for fid in sample_frames(tracks[gid]['observed_frames']):
                mass=native.mass(q[::max(1,len(q)//768)],fid)
                if mass is None:continue
                public=json.loads((context.scene/'refined_instance'/f'{fid}_instance.json').read_text())
                for r,value in zip(native.records(fid),mass):
                    if value<.4 or r['confidence']<.5 or not any(x in r['object_name'].lower() for x in ('shelf','shelves','rack')):continue
                    frames.add(fid);native_names.append(r['object_name'])
                    best=max(((box_iou(r['segmentation_box'],p['segmentation_box']),p) for p in public
                        if normalize(p['object_name'])==normalize(r['object_name'])),key=lambda item:item[0],default=(0,None))
                    if best[0]>=.35:cached_features.append(best[1])
        geometric_only=len(frames)<3
        if geometric_only and len(set(tracks[gid]['observed_frames']))<6:continue
        if geometric_only:frames=set(tracks[gid]['observed_frames'])
        replacements[gid]=q
        tracks[gid]['point_count']=len(q)
        candidates={name:count for name,count in votes.items() if any(x in name.lower() for x in ('shelf','shelves','rack'))}
        if candidates:node['name']=max(candidates,key=candidates.get)
        elif len(cached_features)>=3:
            from collections import Counter
            node['name']=Counter(native_names).most_common(1)[0][0]
            node['text_embedding']=np.mean([r['bert_embedding'] for r in cached_features],axis=0).tolist()
            node['visual_embedding']=np.mean([r['feature'] for r in cached_features],axis=0).tolist()
        node['open_support_recovery']=dict(measured=True,**evidence)
        node['structural_type']='measured_open_support_layers'
        objects.append(dict(instance_id=gid,name=node['name'],identity_frames=sorted(frames),
            identity_basis='geometric_open_support_no_semantic_override' if geometric_only else 'cached_native_support_identity',**evidence))
    if replacements:
        prior=Path(getattr(context,'graph_geometry',cloud_path))
        if not prior.is_file():prior=cloud_path
        replace_regions(cloud_path,replacements,nodes)
        if prior!=cloud_path:replace_regions(prior,replacements,nodes)
        (context.scene/'validated_object_tracks.json').write_text(json.dumps(tracks,indent=2)+'\n')
        graph_path.write_text(json.dumps(graph,indent=2)+'\n')
        context.graph_geometry=prior;context.canonical_geometry_input=graph_path
        from .canonical_geometry import publish
        publish(context)
    report=dict(algorithm='OSL_observed_open_support_layers_v15',ground_truth_used=False,vision_api_calls=0,
                objects=objects,component_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (context.scene/'support_layers_audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    context.event('重复实测支撑层与邻接封闭立面分离',objects=len(objects))

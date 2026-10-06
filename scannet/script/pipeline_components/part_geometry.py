"""Publish depth- and mask-verified sink components into observed object geometry.

Only measured part points enter this stage. No mesh, shape prior, ground truth,
synthetic surface or enlarged box is used to generate geometry.
"""
from pathlib import Path
import json, hashlib
import numpy as np
import open3d as o3d
from partaware.geometry import load_capture, voxel_downsample
from pipeline_components.instance_consensus import Views


def eligible(part, parent_id, parent_name):
    prefix, separator, suffix = part['name'].partition(':')
    canonical = lambda s: ' '.join(s.casefold().replace('_', ' ').split())
    return (part['status'] == 'confirmed' and str(part.get('parent_id')) == parent_id
        and separator and canonical(prefix) == canonical(parent_name)
        and suffix.strip() in ['basin','faucet','drain']
        and len(part['observed_frames']) >= 3
        and part.get('parent_evidence', {}).get(parent_id, 0) >= 3
        and part.get('parent_evidence', {}).get(parent_id, 0)/max(part.get('parent_evidence_frames',0),1) >= .5)


def verify(points, views, frames, directory, track_ids):
    counts = np.zeros(len(points), int)
    audits = []
    for fid in sorted(set(frames)):
        indices, weights, _ = views.project(points, fid)
        if len(indices) < 16:
            continue
        records = json.loads((directory/f'{fid}.json').read_text())
        selected = [r['mask_index'] for r in records if r['track_id'] in track_ids]
        if not selected:
            continue
        with np.load(directory/f'{fid}.npz') as saved:
            mask = np.any(saved['masks'][selected], axis=0)
        pose, _, _ = views.get(fid)
        camera = (points[indices]-pose[:3,3]) @ pose[:3,:3]
        uv = np.rint(camera[:,:2]/camera[:,2,None]*[views.kc[0,0],views.kc[1,1]]+[views.kc[0,2],views.kc[1,2]]).astype(int)
        inside = mask[uv[:,1],uv[:,0]]
        counts[indices[inside]] += 1
        audits.append({'frame_id': fid, 'depth_consistent_points': len(indices), 'part_mask_supported_points': int(inside.sum())})
    return points[counts >= 3], audits


def replace_regions(cloud_path, replacements, object_ids):
    """Replace measured regions while preserving other objects and completions."""
    cloud = o3d.io.read_point_cloud(str(cloud_path))
    points = np.asarray(cloud.points)
    colors = np.rint(np.asarray(cloud.colors)*255).astype(int)
    ids = colors[:,0]+255*colors[:,1]+255**2*colors[:,2]
    output_points, output_colors = [], []
    for gid in object_ids:
        region = replacements.get(gid, points[ids == int(gid)])
        output_points.append(region)
        color = [int(gid)%255, (int(gid)//255)%255, (int(gid)//255**2)%255]
        output_colors.append(np.tile(color, (len(region), 1))/255)
    output = o3d.geometry.PointCloud()
    output.points = o3d.utility.Vector3dVector(np.concatenate(output_points))
    output.colors = o3d.utility.Vector3dVector(np.concatenate(output_colors))
    if not o3d.io.write_point_cloud(str(cloud_path), output):
        raise IOError('Could not publish measured part-supported cloud')


def construct(context):
    graph_path = context.scene/'topology_map.json'
    graph = json.loads(graph_path.read_text())
    data,jobs,kd,kc,scale = load_capture(context.manifest,context.image_dir,context.scene/'refined_instance')
    report = {'algorithm': 'measured_sink_part_mask_depth_consensus_v7', 'ground_truth_used': False,
        'generated_points': 0, 'qwen_api_calls': 0, 'objects': [],
        'component_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    if data.get('world_frame') != 'hypersim_world_z_up':
        report['reason'] = '未声明米制竖直轴，保留基础几何'
        (context.scene/'part_geometry_audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
        return
    cloud_path = context.scene/'instance_cloud_cleaned.ply'
    cloud = o3d.io.read_point_cloud(str(cloud_path))
    points = np.asarray(cloud.points);colors = np.rint(np.asarray(cloud.colors)*255).astype(int)
    ids = colors[:,0]+255*colors[:,1]+255**2*colors[:,2]
    records = {j['frame_id']:json.loads((context.scene/'refined_instance'/f"{j['frame_id']}_updated_instance.json").read_text()) for j in jobs}
    views = Views(context,jobs,kd,kc,scale,records)
    replacements = {}
    for gid,node in graph['object_nodes']['nodes'].items():
        if 'sink' not in node['name'].casefold().replace('_',' '):
            continue
        selected = [p for p in graph.get('part_nodes',{}).values() if eligible(p,gid,node['name'])]
        if not selected:
            continue
        original = points[ids == int(gid)]
        part_points = np.concatenate([np.load(context.scene/'parts'/f"{p['id']}.points.npy") for p in selected])
        # Bound observed component growth to the neighborhood of the seed basin.
        span = np.ptp(original,axis=0)
        lower,upper = original.min(0).copy(),original.max(0).copy()
        margin = np.maximum(span[:2],.15)
        lower[:2] -= margin; upper[:2] += margin
        lower[2] -= .5; upper[2] += .5
        bounded = part_points[np.all((part_points >= lower)&(part_points <= upper),axis=1)]
        candidates = voxel_downsample(bounded,.01)
        if not len(candidates):
            continue
        accepted,evidence = verify(candidates,views,[f for p in selected for f in p['observed_frames']],
            context.scene/'parts/frame_parts',{p['id'] for p in selected})
        if not len(accepted):
            continue
        combined = np.concatenate([original,accepted])
        temporary = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(combined))
        labels = np.array(temporary.cluster_dbscan(eps=.035,min_points=1,print_progress=False))
        anchors = np.unique(labels[:len(original)])
        connected = combined[np.isin(labels,anchors)]
        final = voxel_downsample(connected,.01)
        if np.max(np.ptp(final,axis=0)) > 2.:
            report['objects'].append({'instance_id':gid,'status':'rejected','reason':'连接后的实测部件超出2米水槽范围'})
            continue
        replacements[gid] = final
        report['objects'].append({'instance_id':gid,'status':'accepted','observed_seed_points':len(original),
            'candidate_part_points':len(candidates),'three_view_supported_points':len(accepted),
            'published_measured_points':len(final),'part_ids':[p['id'] for p in selected],
            'frame_evidence':evidence})
    if replacements:
        geometry_path = getattr(context, 'graph_geometry', cloud_path)
        replace_regions(cloud_path, replacements, graph['object_nodes']['nodes'])
        if geometry_path != cloud_path:
            replace_regions(geometry_path, replacements, graph['object_nodes']['nodes'])
        context.graph_geometry = geometry_path
        context.canonical_geometry_input=graph_path
        from pipeline_components import canonical_geometry
        canonical_geometry.publish(context)
        graph=json.loads(graph_path.read_text())
        graph['scene_graph']['edges']=[e for h in graph['edge_hypotheses'].values() for e in h['edges'].values()]+graph['part_relations']
        graph['geometry_provenance']['measured_part_geometry_component']='pipeline_components.part_geometry'
        graph_path.write_text(json.dumps(graph,indent=2)+'\n')
        (context.scene/'parts/partaware_graph.json').write_text(json.dumps(graph,indent=2)+'\n')
    (context.scene/'part_geometry_audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    context.event('三视角部件掩码与实测深度审核后发布整体水槽几何',objects=len(replacements),generated_points=0)

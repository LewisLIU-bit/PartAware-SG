"""Remove only measured outer-room sheets from object ownership.

Room planes come from the RGB-D background, never from annotation classes.
All other measured coordinates, hypotheses and part identities are retained.
"""
import hashlib
import json
from pathlib import Path
import numpy as np
import open3d as o3d
from partaware.geometry import load_capture
from .room_envelope import fit_envelope
from .part_geometry import replace_regions


def fit_room_sheets(background,cameras):
    cloud=o3d.geometry.PointCloud(o3d.utility.Vector3dVector(background)).voxel_down_sample(.02)
    planes=[]
    for _ in range(16):
        if len(cloud.points)<800:break
        o3d.utility.random.seed(15)
        plane,index=cloud.segment_plane(.006,3,600)
        points=np.asarray(cloud.points)[index]
        cloud=cloud.select_by_index(index,invert=True)
        normal=np.asarray(plane[:3]);normal/=np.linalg.norm(normal)
        lateral=np.array([-normal[1],normal[0],0.])
        width=np.ptp(points@lateral);height=np.ptp(points[:,2])
        if len(points)<800 or abs(normal[2])>.05 or width<1. or height<.3:continue
        signed=cameras@normal+plane[3];sign=1. if np.median(signed)>=0 else -1.
        if np.mean(sign*signed>.05)<.9:continue
        basis=np.column_stack([lateral,[0.,0.,1.]])
        low,high=np.quantile(points@basis,[.001,.999],axis=0)
        planes.append(dict(normal=(normal*sign).tolist(),offset=float(plane[3]*sign),basis=basis.tolist(),
            lower=low.tolist(),upper=high.tolist(),measured_support_points=len(points),
            camera_side_fraction=float(np.mean(sign*signed>.05))))
    return outer_planes(planes)


def outer_planes(planes):
    vertical = [p for p in planes if abs(p['normal'][2]) < .05]
    # Parallel patches behind a closer furniture face define the room shell.
    return [p for p in vertical if not any(
        np.dot(p['normal'], q['normal']) > .99 and q['offset'] > p['offset']+.08
        for q in vertical)]


def sheet_evidence(points, planes):
    if len(points) < 256:
        return None
    for p in planes:
        normal = np.asarray(p['normal'])
        signed = points@normal+p['offset']
        near = np.abs(signed) <= .01
        uv = points@np.asarray(p['basis'])
        patch = np.all((uv >= np.asarray(p['lower'])-.03)&(uv <= np.asarray(p['upper'])+.03), axis=1)
        thickness = float(np.ptp(points@normal))
        extent = np.ptp(uv, axis=0)
        if np.mean(near&patch) >= .95 and thickness < .02 and extent.min() > .3:
            return dict(room_sheet_fraction=float(np.mean(near&patch)), thickness_m=thickness,
                        observed_room_plane=p, evidence_type='observed_outer_room_sheet')
    return None


def construct(context):
    metadata, jobs, _, _, _ = load_capture(context.manifest)
    report = dict(algorithm='ROS_room_outer_surface_ownership_v15', ground_truth_used=False,
                  vision_api_calls=0, generated_points=0, objects=[],
                  component_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    if metadata.get('world_frame') != 'hypersim_world_z_up':
        return
    graph_path = context.scene/'topology_map.json'
    graph = json.loads(graph_path.read_text())
    nodes = graph['object_nodes']['nodes']
    cloud_path = context.scene/'instance_cloud_cleaned.ply'
    cloud = o3d.io.read_point_cloud(str(cloud_path))
    points = np.asarray(cloud.points)
    colors = np.rint(np.asarray(cloud.colors)*255).astype(int)
    ids = colors[:,0]+255*colors[:,1]+255**2*colors[:,2]
    background = np.asarray(o3d.io.read_point_cloud(str(context.scene/'instance_cloud_with_background.ply')).points)
    cameras = np.array([np.loadtxt(j['pose'])[:3,3] for j in jobs])
    planes = fit_room_sheets(background,cameras)
    report['outer_planes'] = planes
    removed = []
    for gid, node in nodes.items():
        proof = sheet_evidence(points[ids == int(gid)], planes)
        if proof:
            removed.append(gid)
            report['objects'].append(dict(instance_id=gid, name=node['name'], action='退回背景，不进入物体层', **proof))
    if removed:
        tracks_path = context.scene/'validated_object_tracks.json'
        tracks = json.loads(tracks_path.read_text())
        for gid in removed:
            del nodes[gid]
            tracks.pop(gid, None)
        for part in graph.get('part_nodes', {}).values():
            if str(part.get('parent_id')) in removed:
                part['parent_id'] = None
                part['status'] = 'provisional'
        graph['part_relations'] = [e for e in graph.get('part_relations', []) if str(e['target_id']) not in removed]
        replace_regions(cloud_path, {}, nodes)
        prior = Path(getattr(context, 'graph_geometry', cloud_path))
        if not prior.is_file():prior=cloud_path
        if prior != cloud_path:
            replace_regions(prior, {}, nodes)
        tracks_path.write_text(json.dumps(tracks, indent=2)+'\n')
        graph_path.write_text(json.dumps(graph, indent=2)+'\n')
        context.graph_geometry = prior
        context.canonical_geometry_input = graph_path
        from .canonical_geometry import publish
        publish(context)
    (context.scene/'boundary_ownership_audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    context.event('实测最外层房间面恢复背景归属', removed_objects=len(removed))

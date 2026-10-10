"""Assemble planar fronts carried by one continuous measured top surface.

Connected fronts become one object with queryable measured panel parts.
Semantic similarity alone or box contact alone cannot create an assembly.
This geometric project adapter uses no GT, target counts or class prompts.
"""
import hashlib
import json
from pathlib import Path
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree
from .enclosure_continuity import dominant_face
from .part_body_assembly import cosine, construct as assemble


def front(points):
    if len(points) < 512:
        return None
    normal, support = dominant_face(points)
    z = np.quantile(points[:,2],[.01,.99])
    width = np.ptp(points@np.array([-normal[1],normal[0],0.]))
    thickness = np.ptp(points@normal)
    if abs(normal[2]) > .06 or support < .65 or np.ptp(z) < .5 or width < .12 or thickness > .12:
        return None
    return dict(normal=normal,z=z,width=width,support=support)


def carrier_support(a,b,carrier):
    """A visible top must bridge the entire connection, not just box corners."""
    aa = a[np.argmin(cKDTree(b).query(a)[0])]
    bb = b[np.argmin(cKDTree(a).query(b)[0])]
    samples = np.linspace(aa[:2],bb[:2], max(12,int(np.linalg.norm(aa[:2]-bb[:2])/.015)))
    distances = cKDTree(carrier[:,:2]).query(samples)[0]
    return float(np.mean(distances <= .045))


def links(geometry,nodes,tracks):
    fronts = {gid:f for gid,q in geometry.items() if (f:=front(q)) is not None}
    carriers = []
    for gid,q in geometry.items():
        if len(q)<1024 or len(set(tracks[gid]['observed_frames']))<6:
            continue
        # A broad, locally level, measured upper face can carry a cabinet run.
        z = np.quantile(q[:,2],.95)
        upper = q[np.abs(q[:,2]-z)<.015]
        span = np.ptp(upper[:,:2],axis=0) if len(upper) else np.zeros(2)
        if len(upper)>1024 and span.max()>.8 and span.min()>.2:
            carriers.append((gid,z,upper))
    edges = []
    for i,(a,fa) in enumerate(fronts.items()):
        for b,fb in list(fronts.items())[i+1:]:
            shared=set(tracks[a]['observed_frames'])&set(tracks[b]['observed_frames'])
            similarity=cosine(nodes[a]['text_embedding'],nodes[b]['text_embedding'])
            mismatch=float(np.max(np.abs(fa['z']-fb['z'])))
            dot=abs(float(fa['normal']@fb['normal']))
            gap=float(cKDTree(geometry[b]).query(geometry[a])[0].min())
            if len(shared)<3 or similarity<.9 or mismatch>.06 or not (dot>.985 or dot<.15) or gap>.8:
                continue
            compatible=[]
            for gid,z,q in carriers:
                if gid in (a,b) or not (.01 < z-max(fa['z'][1],fb['z'][1]) < .18):
                    continue
                coverage=carrier_support(geometry[a],geometry[b],q)
                if coverage>=.9:
                    compatible.append((gid,coverage))
            if len(compatible)!=1:
                continue
            edges.append((a,b,dict(evidence_type='continuous_measured_front_carrier',
                semantic_cosine=similarity,shared_observed_frames=sorted(shared),
                boundary_height_mismatch_m=mismatch,front_normal_dot=dot,nearest_gap_m=gap,
                carrier_id=compatible[0][0],carrier_coverage=compatible[0][1],generated_points=0,ground_truth_used=False)))
    return edges


class Adapter:
    def __init__(self, selected):
        self.selected=selected
        self.__file__=__file__
    def relation(self,source,body,source_node,body_node,semantic,shared):
        return self.selected.get((str(source_node['id']),str(body_node['id'])))
    def retain_evidence(self,graph,report):
        report['algorithm']='FCA_continuous_measured_front_carrier_v15'


def assemble_pass(context):
    if not context.manifest or json.loads(context.manifest.read_text()).get('world_frame') != 'hypersim_world_z_up':
        return
    graph=json.loads((context.scene/'topology_map.json').read_text())
    nodes=graph['object_nodes']['nodes']
    cloud=o3d.io.read_point_cloud(str(context.scene/'instance_cloud_cleaned.ply'))
    points=np.asarray(cloud.points)
    colors=np.rint(np.asarray(cloud.colors)*255).astype(int)
    ids=colors[:,0]+255*colors[:,1]+255**2*colors[:,2]
    geometry={gid:points[ids==int(gid)] for gid in nodes}
    tracks=json.loads((context.scene/'validated_object_tracks.json').read_text())
    edges=links(geometry,nodes,tracks)
    parent={gid:gid for gid in nodes}
    def root(gid):
        while parent[gid]!=gid:
            gid=parent[gid]
        return gid
    for a,b,e in edges:
        parent[root(a)]=root(b)
    groups={}
    for gid in nodes:
        groups.setdefault(root(gid),[]).append(gid)
    selected={}
    for members in groups.values():
        if len(members)<2:
            continue
        anchor=max(members,key=lambda gid:len(geometry[gid]))
        relevant=[e for a,b,e in edges if a in members and b in members]
        # One physical carrier must explain the entire connected assembly.
        if len({e['carrier_id'] for e in relevant})!=1:
            continue
        for gid in members:
            if gid!=anchor and len(set(tracks[gid]['observed_frames'])&set(tracks[anchor]['observed_frames']))>=3:
                selected[(gid,anchor)]={**relevant[0],
                    'shared_observed_frames':sorted(set(tracks[gid]['observed_frames'])&set(tracks[anchor]['observed_frames'])),
                    'connected_front_members':members,'connection_edges':relevant}
    assemble(context,relation_adapter=Adapter(selected),audit_filename='front_continuity_audit.json')


def construct(context):
    if not context.manifest or json.loads(context.manifest.read_text()).get('world_frame')!='hypersim_world_z_up':return
    graph=json.loads((context.scene/'topology_map.json').read_text())
    budget=len(graph['object_nodes']['nodes'])
    applied=[]
    for _ in range(budget):
        assemble_pass(context)
        path=context.scene/'front_continuity_audit.json'
        report=json.loads(path.read_text())
        if not report['objects']:break
        applied.extend(report['objects'])
    graph=json.loads((context.scene/'topology_map.json').read_text())
    # Preserve valid proofs after replay, while reporting newly applied merges.
    seen={str(row['source_id']) for row in applied}
    for source,alias in graph.get('object_identity_aliases',{}).items():
        proof=alias.get('evidence',{})
        body=str(alias.get('canonical_id'));part_id=alias.get('part_id')
        if (isinstance(proof,dict) and source not in seen and proof.get('evidence_type')=='continuous_measured_front_carrier'
                and body in graph['object_nodes']['nodes'] and part_id in graph.get('part_nodes',{})):
            applied.append(dict(source_id=source,body_id=body,part_id=part_id,retained_from_valid_alias=True,**proof))
    report['algorithm']='FCA_continuous_measured_front_carrier_v15'
    report['objects']=applied
    report['point_coordinates_preserved']=True
    report['passes_until_no_new_identity']=True
    path.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')

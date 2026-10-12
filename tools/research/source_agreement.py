"""SAGE: actual reconstructed-surface ownership and evidence-linked graph.

Requires an ordinary, unlabeled reconstructed mesh and signed class-agnostic
peer predictions on that mesh. Never opens aggregation or semantic GT files.
No production registry/default change. Geometry hypotheses do not name objects;
new names are linked to actual cached image observations, otherwise unknown.
"""
import copy,hashlib,json
from pathlib import Path
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree
from proposal_ensemble import propose
from mask_proposals import overlap
from partaware.geometry import load_capture
from pipeline_components.instance_consensus import Views

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def smooth(labels,segments):
 _,group=np.unique(segments,return_inverse=True);count=np.bincount(group)
 out=np.full(labels.shape,-1,int)
 for label in np.unique(labels):
  if label<0:continue
  positive=np.bincount(group[labels==label],minlength=len(count))
  out[(positive>count*.5)[group]]=label
 return out

def encode(ids):
 ids=np.asarray(ids,int)
 if np.any(ids<0) or np.any(ids>=255**3):raise ValueError('Instance color encoding overflow')
 return np.column_stack([ids%255,ids//255%255,ids//255**2])/255.

def observation_name(points,views):
 points=points[::max(1,len(points)//1000)];votes={};evidence=[];examples={}
 for fid in views.jobs:
  ix,w,labels=views.project(points,fid);ix=ix[w>=.7];labels=labels[w>=.7]
  if len(ix)<32:continue
  public=json.loads((views.context.scene/'refined_instance'/f'{fid}_instance.json').read_text())
  candidates=[]
  for record in public:
   local_id=int(record['frame_instance_id'])
   fraction=float(np.mean(labels==local_id))
   if fraction>=.5:candidates.append((fraction,record))
  if not candidates:continue
  fraction,record=max(candidates,key=lambda r:r[0]);name=record['object_name']
  votes[name]=votes.get(name,0)+fraction;examples[name]=record
  evidence.append(dict(frame_id=fid,name=name,visible_points=len(ix),mask_fraction=fraction,
      camera_center=views.get(fid)[0][:3,3].tolist()))
 if not votes:return None,dict(status='no_image_name_evidence',views=[])
 name=max(votes,key=votes.get);selected=[e for e in evidence if e['name']==name]
 centers=np.array([e['camera_center'] for e in selected]);baseline=float(np.linalg.norm(np.ptp(centers,axis=0)))
 if len(selected)<2 or baseline<.15:return None,dict(status='insufficient_independent_name_evidence',views=selected)
 record=examples[name]
 return dict(name=name,text_embedding=record.get('bert_embedding',[]),visual_embedding=record.get('feature',[])),dict(status='cached_image_name_supported',views=selected,camera_baseline_m=baseline)

def construct(context):
 mesh_path=Path(context.research_mesh);segment_path=Path(context.research_segments)
 peer_path=Path(context.research_peer);peer_receipt=context.research_peer_receipt
 if sha(peer_path)!=peer_receipt['prediction_sha256']:raise ValueError('Peer prediction hash changed')
 if sha(context.manifest)!=peer_receipt['preparation']['manifest_sha256']:raise ValueError('Peer frames/poses differ from pipeline manifest')
 mesh=o3d.io.read_triangle_mesh(str(mesh_path));xyz=np.asarray(mesh.vertices)
 segments=np.array(json.loads(segment_path.read_text())['segIndices'])
 if len(segments)!=len(xyz) or not np.isfinite(xyz).all():raise ValueError('Mesh/segments geometry mismatch')
 with np.load(peer_path,allow_pickle=False) as f:b,sb=f['masks'].copy(),f['scores'].copy()
 if b.shape!=(len(sb),len(xyz)) or b.dtype!=bool:raise ValueError('Peer mesh domain mismatch')
 gp=context.scene/'topology_map.json';graph=json.loads(gp.read_text());old_nodes=graph['object_nodes']['nodes']
 tracks_path=context.scene/'validated_object_tracks.json';tracks=json.loads(tracks_path.read_text())
 old_path=context.graph_geometry;cloud=o3d.io.read_point_cloud(str(old_path));p=np.asarray(cloud.points)
 rgb=np.rint(np.asarray(cloud.colors)*255).astype(int);ids=rgb[:,0]+255*rgb[:,1]+255**2*rgb[:,2]
 distance,index=cKDTree(p).query(xyz,workers=2);transferred=ids[index].copy();transferred[distance>.05]=-1
 transferred=smooth(transferred,segments)
 keys=list(old_nodes);a=np.array([transferred==int(gid) for gid in keys]);sa=np.array([tracks.get(gid,{}).get('confidence',old_nodes[gid].get('confidence',0)) for gid in keys])
 masks,scores,audit=propose(a,sa,b,sb,context.research_ensemble_mode,points=xyz)
 if np.any(masks.sum(0)>1):raise ValueError('Published point ownership must be exclusive')
 # One ordinary geometric segment has one published owner, consistent with
 # the mesh representation used by the independently released ESAM baseline.
 ownership=np.full(len(xyz),-1,int)
 for i,mask in enumerate(masks):ownership[mask]=i
 ownership=smooth(ownership,segments)
 _,jobs,kd,kc,scale=load_capture(context.manifest,context.image_dir,context.scene/'refined_instance')
 views=Views(context,jobs,kd,kc,scale,{})
 nodes={};new_tracks={};mapping={};pieces=[];labels=[];next_id=max(map(int,keys),default=0)+1;objects=[]
 kept=audit['kept'];pair_iou=overlap(masks,a)
 for i,source_index in enumerate(kept):
  local=xyz[ownership==i]
  if len(local)<16:continue
  if source_index<len(a):
   gid=keys[source_index];node=copy.deepcopy(old_nodes[gid]);track=copy.deepcopy(tracks.get(gid,{}));mapping[gid]=gid
   name_proof=dict(status='existing_verified_image_identity',source_node=gid)
  else:
   gid=str(next_id);next_id+=1;j=int(pair_iou[i].argmax()) if len(a) else None
   native=keys[j] if j is not None and pair_iou[i,j]>=.5 else None
   if native is not None:
    node=copy.deepcopy(old_nodes[native]);track=copy.deepcopy(tracks.get(native,{}));mapping.setdefault(native,gid)
    name_proof=dict(status='geometric_match_to_cached_identity',source_node=native,mask_IoU=float(pair_iou[i,j]))
   else:
    name_data,name_proof=observation_name(local,views)
    node=name_data or dict(name='unlabeled object',text_embedding=[],visual_embedding=[])
    track=dict(observations=[],observed_frames=[e['frame_id'] for e in name_proof['views']])
   node.pop('structural_surface_recovery',None)
  node.update(id=gid,confidence=float(scores[i]),geometry_evidence=dict(source='unlabeled_reconstructed_surface',generated_points=0,source_agreement= True,name_evidence=name_proof))
  track.update(confidence=float(scores[i]),confidence_type='uncalibrated_cross_model_mask_agreement',point_count=len(local))
  nodes[gid]=node;new_tracks[gid]=track;pieces.append(local);labels.append(np.full(len(local),int(gid)))
  objects.append(dict(node_id=gid,source='native' if source_index<len(a) else 'ESAM_geometric_peer',source_index=source_index if source_index<len(a) else source_index-len(a),points=len(local),name_evidence=name_proof))
 # Retain real camera-sampled small/mesh-unrepresented native objects. They
 # are not ghosts merely because the reconstructed mesh has no >=100pt mask.
 retained_unrepresented=[]
 for j,gid in enumerate(keys):
  if a[j].sum()>=100 or gid in nodes:continue
  local=p[ids==int(gid)]
  if len(local)<16:continue
  nodes[gid]=copy.deepcopy(old_nodes[gid]);new_tracks[gid]=copy.deepcopy(tracks.get(gid,{}));mapping[gid]=gid
  pieces.append(local);labels.append(np.full(len(local),int(gid)));retained_unrepresented.append(gid)
 out=o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.concatenate(pieces)));out.colors=o3d.utility.Vector3dVector(encode(np.concatenate(labels)))
 path=context.scene/'instance_cloud_sage.ply';o3d.io.write_point_cloud(str(path),out)
 # The canonical publisher's observation graph uses the same real output.
 o3d.io.write_point_cloud(str(context.scene/'instance_cloud_cleaned.ply'),out)
 graph['object_nodes']['nodes']=nodes;graph['object_identity_aliases']={}
 for pid,part in graph.get('part_nodes',{}).items():
  parent=str(part.get('parent_id',''))
  if parent in mapping:
   part['parent_id']=mapping[parent]
  elif parent and parent not in nodes:part['status']='provisional'
 for relation in graph.get('part_relations',[]):
  if str(relation.get('target_id')) in mapping:relation['target_id']=mapping[str(relation['target_id'])]
 graph['part_relations']=[r for r in graph.get('part_relations',[]) if str(r.get('target_id')) in nodes]
 graph['object_relations']=[r for r in graph.get('object_relations',[]) if r['source_id'] in nodes and r['target_id'] in nodes]
 gp.write_text(json.dumps(graph,indent=2)+'\n');tracks_path.write_text(json.dumps(new_tracks,indent=2)+'\n')
 context.graph_geometry=path
 receipt=dict(algorithm='SAGE',policy=context.research_ensemble_mode,GT_in_construction=False,training=False,new_vision_API_calls=0,mesh_sha256=sha(mesh_path),geometric_segments_sha256=sha(segment_path),peer_prediction_sha256=sha(peer_path),peer_checkpoint_sha256=peer_receipt['checkpoint_sha256'],ownership_exclusive=True,generated_hidden_points=0,geometry_source='ordinary reconstructed mesh plus retained camera samples',objects=objects,retained_mesh_unrepresented=retained_unrepresented,proposal_audit=audit,component_sha256=sha(__file__))
 (context.scene/'source_agreement_audit.json').write_text(json.dumps(receipt,indent=2)+'\n')
 context.event('SAGE 跨来源候选与实测重建表面归属发布',objects=len(nodes),points=len(out.points),GT_used=False)

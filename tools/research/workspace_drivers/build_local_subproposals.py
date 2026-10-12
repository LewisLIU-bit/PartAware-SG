"""Independent-view local mask discovery; never reads GT or expected counts.

Unlike one-mask refinement, retain distinct SAM proposals that agree across
separated camera centers. Candidates are observed mesh vertices, not generated
shape. This is mask-space research until actual pipeline replay is validated.
"""
import hashlib,json,sys
from pathlib import Path
import cv2,numpy as np,open3d as o3d
from scipy.optimize import linear_sum_assignment
ROOT=Path(__file__).resolve().parents[1];REPO=Path('/home/lewisliu/PartAware-SG')
sys.path[:0]=[str(REPO/'scannet/script'),str(ROOT/'v20_work/research')]
from partaware.geometry import load_capture
from projection_verification import project
from mask_proposals import overlap
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
results=[]
for item in json.loads((ROOT/'v20_work/projection_preparation.json').read_text())['results']:
 target=Path(item['target']);jobs_path=target/'projection_text_jobs.json';jobs=json.loads(jobs_path.read_text());receipt_path=target/'sam3_text_box_receipt.json';receipt=json.loads(receipt_path.read_text())
 assert receipt['complete'] and sha(jobs_path)==receipt['jobs_sha256']
 assert sha(jobs['mesh'])==jobs['mesh_sha256'] and sha(jobs['peer'])==jobs['peer_sha256']
 xyz=np.asarray(o3d.io.read_triangle_mesh(jobs['mesh']).vertices)
 with np.load(jobs['peer'],allow_pickle=False) as f:original=f['masks'].copy();scores=f['scores'].copy()
 _,captures,kd,kc,scale=load_capture(Path(jobs['manifest']));captures={j['frame_id']:j for j in captures}
 observations={};signed={(r['frame_id'],r['peer_index']):r for r in receipt['rows']}
 for frame in jobs['jobs']:
  fid=frame['frame_id'];capture=captures[fid];pose=np.loadtxt(capture['pose']);depth=cv2.imread(str(capture['depth']),cv2.IMREAD_UNCHANGED);h,w=cv2.imread(frame['rgb']).shape[:2]
  indices,uv=project(xyz,pose,kd,kc,depth,scale,h,w)
  for prompt in frame['prompts']:
   i=prompt['peer_index'];cache=target/'sam3_text_box_cache'/f'{fid}_{i}.npz';assert sha(cache)==signed[fid,i]['sha256']
   with np.load(cache,allow_pickle=False) as f:masks=f['masks'].copy();sam_scores=f['scores'].copy()
   if not len(masks):continue
   seed=original[i,indices]
   if seed.sum()<32:continue
   hits=masks[:,uv[:,1],uv[:,0]];intersection=hits[:,seed].sum(1);coverage=intersection/seed.sum();purity=intersection/np.maximum(hits.sum(1),1)
   choices=np.flatnonzero((coverage>=.1)&(purity>=.5)&(hits.sum(1)>=100))
   candidates=[];conf=[]
   for j in choices:
    m=np.zeros(len(xyz),bool);m[indices[hits[j]]]=True;candidates.append(m);conf.append(float(sam_scores[j]))
   if candidates:observations.setdefault(i,[]).append((fid,np.array(candidates),np.array(conf)))
 additions=[];addition_scores=[];proofs=[]
 for parent,frames in observations.items():
  clusters=[]
  for fid,candidates,conf in frames:
   if not clusters:
    clusters=[dict(mask=m.copy(),views=[fid],scores=[float(s)]) for m,s in zip(candidates,conf)];continue
   agreement=overlap(np.array([c['mask'] for c in clusters]),candidates)
   rows,cols=linear_sum_assignment(-agreement);used=set()
   for r,c in zip(rows,cols):
    if agreement[r,c]<.25:continue
    clusters[r]['mask']|=candidates[c];clusters[r]['views'].append(fid);clusters[r]['scores'].append(float(conf[c]));used.add(c)
   clusters.extend(dict(mask=m.copy(),views=[fid],scores=[float(conf[c])]) for c,m in enumerate(candidates) if c not in used)
  for cluster in clusters:
   if len(cluster['views'])<2 or cluster['mask'].sum()<100:continue
   # Independent camera centers were selected before SAM inference.
   additions.append(cluster['mask']);addition_scores.append(np.mean(cluster['scores']));proofs.append(dict(parent_peer_index=parent,independent_frames=cluster['views'],observed_vertices=int(cluster['mask'].sum()),model_scores=cluster['scores']))
 merged=np.concatenate([original,np.array(additions)]) if additions else original
 merged_scores=np.r_[scores,addition_scores]
 cp=target/'verified_local_peer_masks.npz';np.savez_compressed(cp,masks=merged,scores=merged_scores)
 audit=dict(complete=True,scene=jobs['scene'],GT_in_construction=False,training=False,new_API_calls=0,full_geometry_replay=False,input_peer_sha256=jobs['peer_sha256'],sam3_receipt_sha256=sha(receipt_path),output_sha256=sha(cp),generated_points=0,local_candidates=proofs,code_sha256=sha(__file__))
 (target/'verified_local_peer_receipt.json').write_text(json.dumps(audit,indent=2)+'\n');results.append(audit)
 print('INDEPENDENT_LOCAL_SUBPROPOSALS',jobs['scene'],len(additions),flush=True)
(ROOT/'v20_work/verified_local_peer_geometry.json').write_text(json.dumps(dict(complete=True,GT_in_construction=False,full_geometry_replay=False,results=results),indent=2)+'\n')

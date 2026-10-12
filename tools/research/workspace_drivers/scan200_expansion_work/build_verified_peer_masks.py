"""GT-free visible-depth refinement of peer proposals from frozen box evidence."""
import argparse,hashlib,json,sys
from pathlib import Path
import cv2,numpy as np,open3d as o3d
ROOT=Path(__file__).resolve().parents[1];REPO=Path('/home/lewisliu/PartAware-SG')
sys.path[:0]=[str(REPO/'scannet/script'),str(ROOT/'v20_work/research')]
from partaware.geometry import load_capture
from projection_verification import project,refine
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
parser=argparse.ArgumentParser();parser.add_argument('--text',action='store_true');parser.add_argument('--positive-only',action='store_true');args=parser.parse_args()
prompt_prefix='text_' if args.text else ''
result_prefix=prompt_prefix+('positive_' if args.positive_only else '')
items=json.loads((ROOT/'scan200_expansion_work/projection_preparation.json').read_text())['results'];out=[]
for item in items:
 target=Path(item['target']);job_path=target/('projection_text_jobs.json' if args.text else 'projection_jobs.json');jobs=json.loads(job_path.read_text())
 receipt_path=target/f'sam3_{prompt_prefix}box_receipt.json';receipt=json.loads(receipt_path.read_text());assert receipt['complete'] and receipt['jobs_sha256']==sha(job_path)
 assert sha(jobs['mesh'])==jobs['mesh_sha256'] and sha(jobs['peer'])==jobs['peer_sha256']
 xyz=np.asarray(o3d.io.read_triangle_mesh(jobs['mesh']).vertices)
 with np.load(jobs['peer'],allow_pickle=False) as f:original=f['masks'].copy();scores=f['scores'].copy()
 _,captures,kd,kc,scale=load_capture(Path(jobs['manifest']));captures={j['frame_id']:j for j in captures}
 positive=np.zeros(original.shape,np.uint8);visible=np.zeros(original.shape,np.uint8);observations=[]
 for frame in jobs['jobs']:
  fid=frame['frame_id'];capture=captures[fid];pose=np.loadtxt(capture['pose']);depth=cv2.imread(str(capture['depth']),cv2.IMREAD_UNCHANGED);h,w=cv2.imread(frame['rgb']).shape[:2]
  indices,uv=project(xyz,pose,kd,kc,depth,scale,h,w)
  for prompt in frame['prompts']:
   i=prompt['peer_index'];cache=target/f'sam3_{prompt_prefix}box_cache'/f'{fid}_{i}.npz'
   signed=next(r for r in receipt['rows'] if r['frame_id']==fid and r['peer_index']==i)
   assert sha(cache)==signed['sha256']
   with np.load(cache,allow_pickle=False) as f:masks=f['masks'].copy();model_scores=f['scores'].copy()
   if not len(masks):continue
   seed=original[i,indices]
   if seed.sum()<32:continue
   hits=masks[:,uv[:,1],uv[:,0]]
   coverage=hits[:,seed].mean(1);purity=hits[:,seed].sum(1)/np.maximum(hits.sum(1),1)
   quality=np.sqrt(coverage*purity)*model_scores;best=int(quality.argmax())
   if coverage[best]<.6:continue
   visible[i,indices]+=1;positive[i,indices]+=hits[best].astype(np.uint8)
   observations.append(dict(peer_index=i,frame_id=fid,seed_coverage=float(coverage[best]),visible_mask_purity=float(purity[best]),model_score=float(model_scores[best])))
 refined=refine(original,positive,visible)
 if args.positive_only:refined|=original
 supported=positive.sum(1)/np.maximum(visible.sum(1),1)
 observed=visible.sum(1)>0
 # Keep quality ranking unchanged first; geometry and sorting must be isolated.
 cp=target/f'verified_{result_prefix}peer_masks.npz';np.savez_compressed(cp,masks=refined,scores=scores)
 audit=dict(complete=True,GT_in_construction=False,full_geometry_replay=False,training=False,new_API_calls=0,scene=jobs['scene'],
  input_peer_sha256=jobs['peer_sha256'],sam3_receipt_sha256=sha(receipt_path),output_sha256=sha(cp),text_from_cached_GPT_only=args.text,positive_only=args.positive_only,
  score_ranking_unchanged=True,added_observed_vertices=int((refined&~original).sum()),removed_independently_contradicted_vertices=int((original&~refined).sum()),
  unknown_occluded_vertices_preserved=bool(np.all(refined[visible==0]==original[visible==0])),observations=observations,
  candidate_visible_support_fraction=supported.tolist(),generated_points=0,code_sha256=sha(__file__))
 (target/f'verified_{result_prefix}peer_receipt.json').write_text(json.dumps(audit,indent=2)+'\n');out.append(audit)
 print('VERIFIED_PEER_GEOMETRY',jobs['scene'],audit['added_observed_vertices'],audit['removed_independently_contradicted_vertices'],flush=True)
(ROOT/f'scan200_expansion_work/verified_{result_prefix}peer_geometry.json').write_text(json.dumps(dict(complete=True,full_geometry_replay=False,GT_in_construction=False,results=out),indent=2)+'\n')

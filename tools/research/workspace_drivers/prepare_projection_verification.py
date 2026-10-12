"""Prepare GT-free geometric prompts from existing peer masks and source poses."""
import hashlib,json,sys
from pathlib import Path
import cv2,numpy as np,open3d as o3d
ROOT=Path(__file__).resolve().parents[1];REPO=Path('/home/lewisliu/PartAware-SG')
sys.path[:0]=[str(REPO/'scannet/script'),str(ROOT/'v20_work/research')]
from partaware.geometry import load_capture
from projection_verification import project
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
reg=json.loads((ROOT/'scan200_work/esam_results.json').read_text());out=[]
for item in reg['results']:
 scene=item['scene'];raw=Path('/home/lewisliu/datasets/scannet/scans')/scene;info=json.loads((ROOT/'scan200_work'/f'{scene}_input.json').read_text())
 assert sha(info['manifest'])==item['preparation']['manifest_sha256']
 mesh=raw/f'{scene}_vh_clean_2.ply';xyz=np.asarray(o3d.io.read_triangle_mesh(str(mesh)).vertices)
 _,jobs,kd,kc,scale=load_capture(Path(info['manifest']));peer=Path(item['output'])/'predictions.npz'
 assert sha(peer)==item['prediction_sha256']
 with np.load(peer,allow_pickle=False) as f:masks=f['masks'].copy()
 captures={}
 for job in jobs:
  rgb=cv2.imread(str(job['rgb']));captures[job['frame_id']]=(job,np.loadtxt(job['pose']),cv2.imread(str(job['depth']),cv2.IMREAD_UNCHANGED),rgb.shape[:2])
 scheduled={};selected=[]
 for i,mask in enumerate(masks):
  seed=xyz[mask];seed=seed[::max(1,len(seed)//1000)];ranked=[]
  if len(seed)<32:continue
  for fid,(job,pose,depth,(h,w)) in captures.items():
   ix,uv=project(seed,pose,kd,kc,depth,scale,h,w)
   if len(ix)<32 or len(ix)/len(seed)<.15:continue
   low=np.quantile(uv,.02,axis=0);high=np.quantile(uv,.98,axis=0);extent=np.maximum(high-low,2)
   low=np.maximum(low-.05*extent,0);high=np.minimum(high+.05*extent,[w-1,h-1])
   if np.any(high-low<2):continue
   box=np.r_[(low+high)/2/[w,h],(high-low)/[w,h]].tolist()
   ranked.append((len(ix),fid,pose[:3,3],box))
  centers=[];chosen=[]
  for _,fid,center,box in sorted(ranked,key=lambda r:-r[0]):
   if centers and min(np.linalg.norm(center-c) for c in centers)<.15:continue
   centers.append(center);chosen.append(fid)
   scheduled.setdefault(fid,[]).append(dict(peer_index=i,box=box,seed_points=int(mask.sum())))
   if len(centers)==3:break
  selected.append(dict(peer_index=i,independent_views=chosen))
 target=Path('/home/lewisliu/datasets/scannet-sg-processed/research_cycle/projection_verification/scannet')/scene;target.mkdir(parents=True,exist_ok=True)
 tasks=[dict(frame_id=fid,rgb=str(captures[fid][0]['rgb']),rgb_sha256=sha(captures[fid][0]['rgb']),prompts=scheduled[fid]) for fid in sorted(scheduled)]
 payload=dict(scene=scene,jobs=tasks,selected=selected,peer=str(peer),peer_sha256=sha(peer),mesh=str(mesh),mesh_sha256=sha(mesh),manifest=info['manifest'],manifest_sha256=sha(info['manifest']),GT_in_construction=False,new_GPT_calls=0,selection='up to three highest-coverage camera centers separated by >=15cm; no object count prior')
 (target/'projection_jobs.json').write_text(json.dumps(payload,indent=2)+'\n');out.append(dict(scene=scene,target=str(target),selected_frames=len(tasks),geometric_prompts=sum(len(t['prompts']) for t in tasks)))
 print('PREPARED_INDEPENDENT_PROJECTED_PROMPTS',out[-1],flush=True)
(ROOT/'v20_work/projection_preparation.json').write_text(json.dumps(dict(complete=True,GT_in_construction=False,results=out),indent=2)+'\n')

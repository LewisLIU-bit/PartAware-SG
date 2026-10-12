"""GT-free cross-domain source admission using existing original image evidence.

Missing image evidence defers a peer identity; it does not prove nonexistence
and never deletes measured structure points. This is a candidate screen.
"""
import hashlib,json,sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np,open3d as o3d
from scipy.spatial import cKDTree
ROOT=Path(__file__).resolve().parents[1];REPO=Path('/home/lewisliu/PartAware-SG')
sys.path[:0]=[str(REPO/'scannet/script'),str(ROOT/'v20_work/research')]
import proposal_ensemble
from core_ownership import propose as core
from bounded_overlap import overlap
from source_agreement import smooth,observation_name
from partaware.geometry import load_capture
from pipeline_components.instance_consensus import Views
proposal_ensemble.overlap=overlap
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
registration=json.loads((ROOT/'v20_work/four_domain_confirmation_registration.json').read_text());results=[]
for item in registration['items']:
 source=Path(item['source']);target=Path('/home/lewisliu/datasets/scannet-sg-processed/research_cycle/image_confirmed_peer')/item['dataset']/item['scene'];target.mkdir(parents=True,exist_ok=True)
 for k,digest in item['input_hashes'].items():assert sha(item[k])==digest
 if item['domain_kind']=='unlabeled_mesh':xyz=np.asarray(o3d.io.read_triangle_mesh(item['domain']).vertices)
 else:
  with np.load(item['domain'],allow_pickle=False) as f:xyz=f['xyz'].copy()
 with np.load(item['peer'],allow_pickle=False) as f:b=f['masks'].copy();sb=f['scores'].copy()
 assert b.shape==(len(sb),len(xyz))
 _,jobs,kd,kc,scale=load_capture(Path(item['manifest']))
 views=Views(SimpleNamespace(scene=source),jobs,kd,kc,scale,{})
 evidence=[];confirmed=[]
 for i,mask in enumerate(b):
  name,proof=observation_name(xyz[mask],views)
  confirmed.append(name is not None);evidence.append(dict(peer_index=i,confirmed=name is not None,name=name,proof=proof))
 print('IMAGE_PEER_ADMISSION',item['scene'],sum(confirmed),'of',len(confirmed),flush=True)
 graph=json.loads((source/'topology_map.json').read_text());keys=list(graph['object_nodes']['nodes']);cloud_name=graph['geometry_provenance']['source'];assert Path(cloud_name).name==cloud_name
 cloud=o3d.io.read_point_cloud(str(source/cloud_name));p=np.asarray(cloud.points);rgb=np.rint(np.asarray(cloud.colors)*255).astype(np.int64);ids=rgb[:,0]+255*rgb[:,1]+255**2*rgb[:,2]
 distance,index=cKDTree(p).query(xyz,workers=2);labels=ids[index];labels[distance>.05]=-1
 segments=None
 if item['segments']:
  segments=np.array(json.loads(Path(item['segments']).read_text())['segIndices']);labels=smooth(labels,segments)
 a=np.array([labels==int(k) for k in keys]);tracks=json.loads((source/'validated_object_tracks.json').read_text());sa=np.array([tracks.get(k,{}).get('confidence',graph['object_nodes']['nodes'][k].get('confidence',0)) for k in keys])
 # Candidate geometry which was deferred remains in the signed source peer.
 masks,scores,audit=core(a,sa,b[np.array(confirmed)],sb[np.array(confirmed)],points=xyz)
 if segments is not None:
  ownership=np.full(len(xyz),-1,int)
  for i,mask in enumerate(masks):ownership[mask]=i
  ownership=smooth(ownership,segments);masks=np.array([ownership==i for i in range(len(scores))])
 path=target/'masks.npz';np.savez_compressed(path,masks=masks,scores=scores)
 (target/'image_evidence.json').write_text(json.dumps(evidence,indent=2)+'\n')
 row=dict(scene=item['scene'],dataset=item['dataset'],masks_file=str(path),masks_sha256=sha(path),confirmed_peer=sum(confirmed),provisional_peer=len(confirmed)-sum(confirmed),inputs=item['input_hashes'],audit=audit,image_evidence_sha256=sha(target/'image_evidence.json'))
 results.append(row)
 (ROOT/'v20_work/image_confirmed_construction.json').write_text(json.dumps(dict(complete=False,GT_in_construction=False,full_pipeline_replay=False,generated_points=0,physical_point_deletions=0,new_API_calls=0,training=False,results=results),indent=2)+'\n')
 del views,a,b,masks,cloud,p
result=json.loads((ROOT/'v20_work/image_confirmed_construction.json').read_text());result['complete']=True;result['builder_sha256']=sha(__file__)
(ROOT/'v20_work/image_confirmed_construction.json').write_text(json.dumps(result,indent=2)+'\n');print('FOUR_DOMAIN_IMAGE_CONFIRMATION_SCREEN_BUILT',flush=True)

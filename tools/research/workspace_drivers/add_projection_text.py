"""Attach existing image-derived names through geometric identity; never GT."""
import hashlib,json,sys
from pathlib import Path
import numpy as np,open3d as o3d
from scipy.spatial import cKDTree
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'v20_work/research'),'/home/lewisliu/PartAware-SG/scannet/script','/home/lewisliu/PartAware-SG/tools/research']
from source_agreement import smooth
from mask_proposals import overlap
reg=json.loads((ROOT/'scan200_work/evaluation_results.json').read_text())
for item in json.loads((ROOT/'v20_work/projection_preparation.json').read_text())['results']:
 target=Path(item['target']);jobs=json.loads((target/'projection_jobs.json').read_text());scene=item['scene']
 source=Path(next(r for r in reg['results'] if r['scene_id']==scene and r['profile']=='v19')['processed_scene'])
 graph=json.loads((source/'topology_map.json').read_text());keys=list(graph['object_nodes']['nodes'])
 cloud=o3d.io.read_point_cloud(str(source/graph['geometry_provenance']['source']));xyz=np.asarray(o3d.io.read_triangle_mesh(jobs['mesh']).vertices)
 rgb=np.rint(np.asarray(cloud.colors)*255).astype(int);ids=rgb[:,0]+255*rgb[:,1]+255**2*rgb[:,2]
 distance,index=cKDTree(np.asarray(cloud.points)).query(xyz,workers=2);labels=ids[index].copy();labels[distance>.05]=-1
 raw=Path('/home/lewisliu/datasets/scannet/scans')/scene
 labels=smooth(labels,np.array(json.loads((raw/f'{scene}_vh_clean_2.0.010000.segs.json').read_text())['segIndices']))
 native=np.array([labels==int(k) for k in keys])
 with np.load(jobs['peer'],allow_pickle=False) as f:peer=f['masks'].copy()
 iou=overlap(peer,native)
 for frame in jobs['jobs']:
  for prompt in frame['prompts']:
   j=int(iou[prompt['peer_index']].argmax());agreement=float(iou[prompt['peer_index'],j])
   prompt['text']=graph['object_nodes']['nodes'][keys[j]]['name'] if agreement>=.2 else None
   prompt['name_evidence']=dict(source='cached_GPT_image_identity',node=keys[j],mask_IoU=agreement,accepted=agreement>=.2)
 jobs['name_source_graph_sha256']=hashlib.sha256((source/'topology_map.json').read_bytes()).hexdigest()
 (target/'projection_text_jobs.json').write_text(json.dumps(jobs,indent=2)+'\n')
 print('ATTACHED_CACHED_NAMES',scene,flush=True)

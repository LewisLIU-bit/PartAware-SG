"""Independent four-domain screen, without feedback to the candidate builder."""
import hashlib,json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scan200_work'))
import evaluate_scan200 as ref
from scannet_utils.evaluate_semantic_instance import scannet_eval
ref.verify_evaluator()
construction=json.loads((ROOT/'v20_work/image_confirmed_construction.json').read_text());assert construction['complete']
results=[]
for row in construction['results']:
 assert hashlib.sha256(Path(row['masks_file']).read_bytes()).hexdigest()==row['masks_sha256']
 with np.load(row['masks_file'],allow_pickle=False) as f:masks=f['masks'].copy();scores=f['scores'].copy()
 if row['dataset']=='scannet':_,gt,_,_=ref.ground_truth(row['scene'])
 else:
  with np.load(ROOT/'esam_hypersim_work/evaluation_GT.npz',allow_pickle=False) as f:gt=f['gt'].copy()
 assert masks.shape==(len(scores),len(gt))
 pred={str(i):dict(mask=m,label_id=3,conf=float(s)) for i,(m,s) in enumerate(zip(masks,scores))}
 metrics=scannet_eval([pred],[gt],None,(3,),('object',),{3:'object'})
 results.append(dict(scene=row['scene'],dataset=row['dataset'],metrics=metrics))
 print('FOUR_DOMAIN_IMAGE_CONFIRMATION_AP',row['scene'],json.dumps(metrics),flush=True)
(ROOT/'v20_work/image_confirmed_evaluation.json').write_text(json.dumps(dict(complete=True,full_pipeline_replay=False,GT_in_construction=False,results=results,hypersim_strict_box_pending=True),indent=2)+'\n')

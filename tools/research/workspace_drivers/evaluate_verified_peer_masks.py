"""Independent evaluation; these are mask adapters, not full pipeline results."""
import argparse,importlib.util,json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'v20_work/research'),'/home/lewisliu/PartAware-SG/tools/research','/home/lewisliu/PartAware-SG/scannet/script']
from proposal_ensemble import propose as weighted
from core_ownership import propose as core
from source_agreement import smooth
from peer_quality_ranking import rank,consistency
from purity_core import propose as purity
parser=argparse.ArgumentParser();parser.add_argument('--prefix',default='');parser.add_argument('--purity-core',action='store_true');group=parser.add_mutually_exclusive_group();group.add_argument('--peer-quality-ranking',action='store_true');group.add_argument('--cross-consistency-ranking',action='store_true');group.add_argument('--ownership-retention-ranking',action='store_true');args=parser.parse_args()
spec=importlib.util.spec_from_file_location('native',ROOT/'scan200_work/evaluate_scan200.py');ref=importlib.util.module_from_spec(spec);spec.loader.exec_module(ref)
reg=json.loads((ROOT/'scan200_work/evaluation_results.json').read_text());rows=[]
for mode in (['purity_core'] if args.purity_core else ['peer_only','exclusive_weighted','exclusive_core']):
 results=[];preds=[];truth=[]
 for scene in reg['registration']['scenes']:
  target=Path('/home/lewisliu/datasets/scannet-sg-processed/research_cycle/projection_verification/scannet')/scene
  assert json.loads((target/f'verified_{args.prefix}peer_receipt.json').read_text())['complete']
  with np.load(target/f'verified_{args.prefix}peer_masks.npz',allow_pickle=False) as f:b=f['masks'].copy();sb=f['scores'].copy()
  source=Path(next(r for r in reg['results'] if r['scene_id']==scene and r['profile']=='v19')['processed_scene'])
  _,original,gt=ref.result_for(scene,source);xyz,_,_,_=ref.ground_truth(scene)
  if mode=='peer_only':masks,scores=b,sb
  else:
   a=np.array([v['mask'] for v in original.values()]);sa=np.array([v['conf'] for v in original.values()])
   fn=purity if mode=='purity_core' else (weighted if mode=='exclusive_weighted' else core)
   masks,scores,audit=fn(a,sa,b,sb,mode,points=xyz)
   if args.peer_quality_ranking:scores=rank(a,b,sb,audit['kept'])
   if args.cross_consistency_ranking:scores=consistency(a,b,audit['kept'],scores)
   # Match the real exclusive publication rule before evaluating the adapter.
   segments=np.array(json.loads((Path('/home/lewisliu/datasets/scannet/scans')/scene/f'{scene}_vh_clean_2.0.010000.segs.json').read_text())['segIndices'])
   ownership=np.full(len(xyz),-1,int)
   for i,mask in enumerate(masks):ownership[mask]=i
   ownership=smooth(ownership,segments)
   masks=np.array([ownership==i for i in range(len(scores))])
   if args.ownership_retention_ranking:
    original_mass=np.concatenate([a,b])[audit['kept']].sum(1)
    scores=scores*np.sqrt(np.minimum(masks.sum(1)/np.maximum(original_mass,1),1))
  pred={str(i):dict(mask=m,label_id=3,conf=float(s)) for i,(m,s) in enumerate(zip(masks,scores))}
  metrics=ref.scannet_eval([pred],[gt],None,(3,),('object',),{3:'object'})
  results.append(dict(scene=scene,metrics=metrics));preds.append(pred);truth.append(gt)
  print('VERIFIED_PEER_SCREEN',mode,scene,json.dumps(metrics),flush=True)
 rows.append(dict(mode=mode,results=results,aggregate=ref.scannet_eval(preds,truth,None,(3,),('object',),{3:'object'})))
suffix='ranked_' if args.peer_quality_ranking else ('consistency_' if args.cross_consistency_ranking else '')
if args.ownership_retention_ranking:suffix+='retention_'
if args.purity_core:suffix+='purity_'
(ROOT/f'v20_work/verified_{args.prefix}{suffix}peer_publication_trials.json').write_text(json.dumps(dict(complete=True,GT_in_construction=False,full_geometry_replay=False,peer_quality_ranking=args.peer_quality_ranking,cross_consistency_ranking=args.cross_consistency_ranking,publication_superpoint_smoothing=True,results=rows),indent=2)+'\n')

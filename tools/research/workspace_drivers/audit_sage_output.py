"""Independent post-hoc geometry and graph audit; no candidate construction."""
import importlib.util,json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'v20_work/research'),'/home/lewisliu/PartAware-SG/tools/research']
from proposal_ensemble import propose
from mask_proposals import overlap
spec=importlib.util.spec_from_file_location('native',ROOT/'scan200_work/evaluate_scan200.py');ref=importlib.util.module_from_spec(spec);spec.loader.exec_module(ref)
reg=json.loads((ROOT/'scan200_work/evaluation_results.json').read_text());peers=json.loads((ROOT/'scan200_work/esam_results.json').read_text());out=[]
for scene in reg['registration']['scenes']:
 target=Path('/home/lewisliu/datasets/scannet-sg-processed/research_cycle/sage_exclusive_weighted/scannet')/scene
 if not (target/'research_completion.json').exists():continue
 source=Path(next(r for r in reg['results'] if r['scene_id']==scene and r['profile']=='v19')['processed_scene'])
 old,p,gt=ref.result_for(scene,source);actual,c,_=ref.result_for(scene,target)
 _,e,_=ref.esam_result_for(next(r for r in peers['results'] if r['scene']==scene))
 a=np.array([v['mask'] for v in p.values()]);b=np.array([v['mask'] for v in e.values()]);sa=np.array([v['conf'] for v in p.values()]);sb=np.array([v['conf'] for v in e.values()])
 masks,scores,screen=propose(a,sa,b,sb,'exclusive_weighted');real=np.array([v['mask'] for v in c.values()]);agreement=overlap(real,masks)
 differences=[]
 for i,(key,info) in enumerate(c.items()):
  if info['mask'].sum()<100:continue
  j=int(agreement[i].argmax());differences.append(dict(node=key,name=actual['predictions'][i]['label'],confidence=info['conf'],mesh_points=int(info['mask'].sum()),screen_nearest_IoU=float(agreement[i,j]),screen_points=int(masks[j].sum()),screen_confidence=float(scores[j])))
 differences.sort(key=lambda r:r['screen_nearest_IoU'])
 graph=json.loads((target/'topology_map.json').read_text());nodes=graph['object_nodes']['nodes']
 invalid_parents=[str(k) for k,v in graph.get('part_nodes',{}).items() if v.get('status')=='confirmed' and str(v.get('parent_id')) not in nodes]
 entries=[]
 for g in actual['GT']:
  if not g['eligible']:continue
  oldg=next(v for v in old['GT'] if v['gt_id']==g['gt_id'])
  entries.append(dict(gt_id=g['gt_id'],label=g['raw_label'],old=oldg['best_prediction'],new=g['best_prediction']))
 out.append(dict(scene=scene,GT_audit_only=True,real_metrics=actual['metrics'],mask_space_metrics=ref.scannet_eval([{str(i):dict(mask=m,label_id=3,conf=float(s)) for i,(m,s) in enumerate(zip(masks,scores))}],[gt],None,(3,),('object',),{3:'object'}),largest_domain_differences=differences[:12],confirmed_parts_missing_parent=invalid_parents,GT=entries))
 print('SAGE_OUTPUT_AUDIT',scene,'MISSING_CONFIRMED_PARENTS',len(invalid_parents),'DOMAIN_DIFFERENCES',json.dumps(differences[:5]),flush=True)
(ROOT/'v20_work/sage_output_audit.json').write_text(json.dumps(out,indent=2)+'\n')

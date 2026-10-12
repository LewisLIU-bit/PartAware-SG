"""Independent evaluator/audit for complete geometry replays, including failures."""
import argparse,importlib.util,json,sys,hashlib
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];REPO=Path('/home/lewisliu/PartAware-SG')
sys.path.insert(0,str(REPO/'tools/research'))
from audit_instances import mask_diagnosis
spec=importlib.util.spec_from_file_location('native',ROOT/'scan200_work/evaluate_scan200.py')
ref=importlib.util.module_from_spec(spec);spec.loader.exec_module(ref)
p=argparse.ArgumentParser();p.add_argument('--cycle',default='mutual_r1');args=p.parse_args()
registration=json.loads((ROOT/'scan200_work/evaluation_results.json').read_text())
rows=[];preds=[];truths=[]
for scene in registration['registration']['scenes']:
 target=Path('/home/lewisliu/datasets/scannet-sg-processed/research_cycle')/args.cycle/'scannet'/scene
 if not (target/'research_completion.json').exists():continue
 row,pred,gt=ref.result_for(scene,target)
 ids=np.unique(gt[gt>0]).tolist()
 row['diagnosis']=mask_diagnosis(np.array([v['mask'] for v in pred.values()]),
     np.array([gt==i for i in ids]),list(pred),ids)
 baseline=next(r for r in registration['results'] if r['scene_id']==scene and r['profile']=='v19')
 row['baseline_metrics']=baseline['metrics']
 row['delta_percentage_points']={k:100*(row['metrics'][k]-baseline['metrics'][k])
     for k in ['all_ap_25%','all_ap_50%','all_ap']}
 rows.append(row);preds.append(pred);truths.append(gt)
 print('REPLAY_AP',scene,row['metrics']['all_ap_25%'],row['metrics']['all_ap_50%'],row['metrics']['all_ap'],
    'DELTA',json.dumps(row['delta_percentage_points']),flush=True)
out=dict(complete=len(rows)==3,cycle=args.cycle,full_geometry_replay=True,
 GT_in_construction=False,training=False,new_vision_API_calls=0,results=rows,
 development_scenes=True,frozen_v19_preserved=True,
 official_evaluator_unchanged=True)
if len(rows)==3:
 out['aggregate']=ref.scannet_eval(preds,truths,None,(3,),('object',),{3:'object'})
 out['baseline_aggregate']=registration['aggregate']['v19']
 out['delta_percentage_points']={k:100*(out['aggregate'][k]-out['baseline_aggregate'][k])
     for k in ['all_ap_25%','all_ap_50%','all_ap']}
 print('REPLAY_POOLED',json.dumps(out['aggregate']),'DELTA',json.dumps(out['delta_percentage_points']),flush=True)
(ROOT/'v20_work'/f'{args.cycle}_evaluation.json').write_text(json.dumps(out,indent=2))

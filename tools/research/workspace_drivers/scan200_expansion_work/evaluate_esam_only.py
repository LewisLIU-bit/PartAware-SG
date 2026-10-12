"""Independent frozen baseline evaluation, while GPT acquisition recovers."""
import json
from pathlib import Path
from evaluate_scan200 import verify_evaluator,esam_result_for,scannet_eval
WORK=Path(__file__).resolve().parent;verify_evaluator()
baseline=json.loads((WORK/'esam_results.json').read_text());assert baseline['complete']
results=[];predictions=[];truths=[]
for item in baseline['results']:
 row,pred,gt=esam_result_for(item);results.append(row);predictions.append(pred);truths.append(gt)
 print('NEW_SCAN_FROZEN_ESAM_AP',item['scene'],json.dumps(row['metrics']),flush=True)
(WORK/'esam_only_evaluation.json').write_text(json.dumps(dict(complete=True,results=results,
 aggregate=scannet_eval(predictions,truths,None,(3,),('object',),{3:'object'}),
 fixed_preregistered_scenes=True,comparison_to_ours_pending=True,training=False),indent=2)+'\n')

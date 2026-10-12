"""Copy the frozen strategy's workspace adapters for five new input records.

Registration paths, output receipts and five-scan completion counts differ;
algorithm modules, weights, thresholds, transforms and evaluation do not.
"""
import hashlib,json,py_compile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OLD=ROOT/'v20_work';WORK=ROOT/'scan200_expansion_work'
files=['prepare_projection_verification.py','add_projection_text.py','run_projection_worker.py','build_verified_peer_masks.py','replay_source_verified.py','evaluate_replay.py']
rows=[]
for name in files:
 text=(OLD/name).read_text()
 text=text.replace("ROOT/'scan200_work/evaluation_results.json'","ROOT/'scan200_expansion_work/evaluation_results.json'")
 text=text.replace("ROOT/'scan200_work/esam_results.json'","ROOT/'scan200_expansion_work/esam_results.json'")
 text=text.replace("ROOT/'scan200_work'/f'{scene}_input.json'","ROOT/'scan200_expansion_work'/f'{scene}_input.json'")
 text=text.replace("ROOT/'v20_work/projection_preparation.json'","ROOT/'scan200_expansion_work/projection_preparation.json'")
 if name=='run_projection_worker.py':text=text.replace("ROOT/'v20_work'/('projection_text_inference.json'","ROOT/'scan200_expansion_work'/('projection_text_inference.json'")
 if name=='build_verified_peer_masks.py':text=text.replace("ROOT/f'v20_work/verified_","ROOT/f'scan200_expansion_work/verified_")
 if name=='replay_source_verified.py':text=text.replace("ROOT/'v20_work'/f'{scene}_{cycle}.log'","ROOT/'scan200_expansion_work'/f'{scene}_{cycle}.log'").replace('ALL_THREE_SAGE_VERIFIED_REPLAYS_COMPLETE','ALL_FIVE_SAGE_VERIFIED_REPLAYS_COMPLETE')
 if name=='evaluate_replay.py':
  text=text.replace('complete=len(rows)==3','complete=len(rows)==len(registration["registration"]["scenes"])').replace('if len(rows)==3:','if len(rows)==len(registration["registration"]["scenes"]):')
  text=text.replace("ROOT/'v20_work'/f'{args.cycle}_evaluation.json'","ROOT/'scan200_expansion_work'/f'{args.cycle}_evaluation.json'")
 p=WORK/name;p.write_text(text);py_compile.compile(str(p),doraise=True)
 rows.append(dict(file=name,source_sha256=hashlib.sha256((OLD/name).read_bytes()).hexdigest(),adapter_sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
for p in WORK.glob('*.py'):py_compile.compile(str(p),doraise=True)
(WORK/'candidate_adapter_receipt.json').write_text(json.dumps(dict(complete=True,syntax_checked=True,actual_pipeline_results_pending=True,GT_in_construction=False,parameter_changes=False,algorithm_module_paths='workspace v20_work/research, unchanged frozen CORE-V',results=rows),indent=2)+'\n')
print('FROZEN_FIVE_SCAN_ADAPTERS_PREPARED',flush=True)

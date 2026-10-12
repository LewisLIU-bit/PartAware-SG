"""Third full all-scan research replay, using one declared policy and cache."""
import argparse,copy,hashlib,json,os,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];REPO=Path('/home/lewisliu/PartAware-SG');PY='/home/lewisliu/miniconda3/envs/scannet-sg/bin/python'
sys.path[:0]=[str(REPO/'scannet/script'),str(ROOT/'v20_work/research'),str(REPO/'tools/research')]
parser=argparse.ArgumentParser();parser.add_argument('--worker');args=parser.parse_args()
reg=json.loads((ROOT/'scan200_work/evaluation_results.json').read_text());peers=json.loads((ROOT/'scan200_work/esam_results.json').read_text());cycle='sage_verified_core';outroot=Path('/home/lewisliu/datasets/scannet-sg-processed/research_cycle')/cycle
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
if args.worker:
 scene=args.worker;source=Path(next(r for r in reg['results'] if r['scene_id']==scene and r['profile']=='v19')['processed_scene']);info=json.loads((ROOT/'scan200_work'/f'{scene}_input.json').read_text());peer=next(r for r in peers['results'] if r['scene']==scene);target=outroot/'scannet'/scene
 evidence_dir=outroot.parent/'projection_verification/scannet'/scene;evidence_path=evidence_dir/'verified_text_positive_peer_receipt.json';evidence=json.loads(evidence_path.read_text());mask_path=evidence_dir/'verified_text_positive_peer_masks.npz'
 assert evidence['complete'] and evidence['input_peer_sha256']==peer['prediction_sha256'] and sha(mask_path)==evidence['output_sha256']
 os.environ.update(PARTAWARE_COMPARISON_PROFILE='v19',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',DISPLAY=':0',XDG_SESSION_TYPE='x11',LIBGL_ALWAYS_SOFTWARE='true');os.environ.pop('WAYLAND_DISPLAY',None)
 import run_pipeline,source_refined_core,observed_domain_guard
 from run_gpt_comparison import configure_profile,construction_fingerprint
 configure_profile('v19',run_pipeline.components);original=run_pipeline.Context
 class Context(original):
  def __init__(self,options):
   super().__init__(options);raw=Path('/home/lewisliu/datasets/scannet/scans')/scene
   self.research_mesh=raw/f'{scene}_vh_clean_2.ply';self.research_segments=raw/f'{scene}_vh_clean_2.0.010000.segs.json';self.research_peer=mask_path;self.research_peer_receipt=copy.deepcopy(peer);self.research_peer_receipt['prediction_sha256']=evidence['output_sha256'];self.research_peer_receipt['original_prediction_sha256']=peer['prediction_sha256'];self.research_refinement_receipt=evidence_path;self.research_ensemble_mode='exclusive_core'
 run_pipeline.Context=Context;run_pipeline.components.FINAL_GEOMETRY=list(run_pipeline.components.FINAL_GEOMETRY)+[source_refined_core,observed_domain_guard];run_pipeline.components.VERSION='research_'+cycle
 reuse=['--reuse-scene',str(source)]
 if (target/'refined_instance').exists():
  for fid in json.loads((source/'recognition_provenance.json').read_text())['expected_frame_ids']:
   for suffix in ['.png','_instance.json']:
    assert sha(source/'refined_instance'/f'{fid}{suffix}')==sha(target/'refined_instance'/f'{fid}{suffix}')
  reuse=[]
 sys.argv=[str(REPO/'scannet/script/run_pipeline.py'),'--manifest',info['manifest'],'--processed-scene',str(target),*reuse,'--start-stage','fusion'];run_pipeline.main()
 receipt=dict(complete=True,cycle=cycle,full_pipeline_from_cached_masks=True,training=False,GT_in_construction=False,new_API_calls=0,source=str(source),manifest_sha256=sha(info['manifest']),frozen_core_fingerprint=construction_fingerprint(),refinement_receipt_sha256=sha(evidence_path),components={n:sha(ROOT/'v20_work/research'/n) for n in ['source_agreement.py','source_agreement_core.py','source_refined_core.py','core_ownership.py','observed_domain_guard.py']})
 (target/'research_completion.json').write_text(json.dumps(receipt,indent=2)+'\n');print('FULL_SAGE_VERIFIED_REPLAY_COMPLETE',scene,flush=True);exit()
environment=os.environ.copy();environment.update(PYTHONUNBUFFERED='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
for scene in reg['registration']['scenes']:
 if (outroot/'scannet'/scene/'research_completion.json').exists():continue
 print('FULL_SAGE_VERIFIED_BEGIN',scene,flush=True)
 with (ROOT/'v20_work'/f'{scene}_{cycle}.log').open('w') as log:r=subprocess.run([PY,str(Path(__file__).resolve()),'--worker',scene],cwd=REPO,env=environment,stdout=log,stderr=subprocess.STDOUT)
 if r.returncode:raise RuntimeError(f'{scene} full replay failed; inspect saved log')
print('ALL_THREE_SAGE_VERIFIED_REPLAYS_COMPLETE',flush=True)

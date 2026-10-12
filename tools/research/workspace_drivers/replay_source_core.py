"""Second full three-scan replay, declared CORE policy."""
import argparse,hashlib,json,os,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];REPO=Path('/home/lewisliu/PartAware-SG');PY='/home/lewisliu/miniconda3/envs/scannet-sg/bin/python'
sys.path[:0]=[str(REPO/'scannet/script'),str(ROOT/'v20_work/research'),str(REPO/'tools/research')]
p=argparse.ArgumentParser();p.add_argument('--worker');p.add_argument('--mode',default='exclusive_core',choices=['exclusive_core']);args=p.parse_args()
reg=json.loads((ROOT/'scan200_work/evaluation_results.json').read_text());peers=json.loads((ROOT/'scan200_work/esam_results.json').read_text())
cycle='sage_'+args.mode;outroot=Path('/home/lewisliu/datasets/scannet-sg-processed/research_cycle')/cycle
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
if args.worker:
 scene=args.worker;src=next(r for r in reg['results'] if r['scene_id']==scene and r['profile']=='v19');info=json.loads((ROOT/'scan200_work'/f'{scene}_input.json').read_text());peer=next(r for r in peers['results'] if r['scene']==scene)
 source=Path(src['processed_scene']);target=outroot/'scannet'/scene
 os.environ.update(PARTAWARE_COMPARISON_PROFILE='v19',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',DISPLAY=':0',XDG_SESSION_TYPE='x11',LIBGL_ALWAYS_SOFTWARE='true');os.environ.pop('WAYLAND_DISPLAY',None)
 import run_pipeline,source_agreement_core as source_agreement
 from run_gpt_comparison import configure_profile,construction_fingerprint
 configure_profile('v19',run_pipeline.components)
 original=run_pipeline.Context
 class Context(original):
  def __init__(self,options):
   super().__init__(options);raw=Path('/home/lewisliu/datasets/scannet/scans')/scene
   self.research_mesh=raw/f'{scene}_vh_clean_2.ply';self.research_segments=raw/f'{scene}_vh_clean_2.0.010000.segs.json';self.research_peer=Path(peer['output'])/'predictions.npz';self.research_peer_receipt=peer;self.research_ensemble_mode=args.mode
 run_pipeline.Context=Context
 run_pipeline.components.FINAL_GEOMETRY=list(run_pipeline.components.FINAL_GEOMETRY)+[source_agreement]
 run_pipeline.components.VERSION='research_'+cycle
 reuse=['--reuse-scene',str(source)]
 if (target/'refined_instance').exists():
  for fid in json.loads((source/'recognition_provenance.json').read_text())['expected_frame_ids']:
   for suffix in ['.png','_instance.json']:
    if sha(source/'refined_instance'/f'{fid}{suffix}')!=sha(target/'refined_instance'/f'{fid}{suffix}'):raise ValueError('Copied frontend differs')
  reuse=[]
 sys.argv=[str(REPO/'scannet/script/run_pipeline.py'),'--manifest',info['manifest'],'--processed-scene',str(target),*reuse,'--start-stage','fusion']
 run_pipeline.main()
 receipt=dict(complete=True,full_pipeline_from_cached_masks=True,cycle=cycle,training=False,GT_in_construction=False,vision_API_calls=0,source=str(source),manifest_sha256=sha(info['manifest']),frozen_core_fingerprint=construction_fingerprint(),components={n:sha(ROOT/'v20_work/research'/n) for n in ['source_agreement.py','proposal_ensemble.py','source_agreement_core.py','core_ownership.py']})
 (target/'research_completion.json').write_text(json.dumps(receipt,indent=2)+'\n');print('FULL_SAGE_REPLAY_COMPLETE',scene,flush=True);exit()
environment=os.environ.copy();environment.update(PYTHONUNBUFFERED='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
for scene in reg['registration']['scenes']:
 target=outroot/'scannet'/scene
 if (target/'research_completion.json').exists():continue
 print('FULL_SAGE_REPLAY_BEGIN',scene,flush=True)
 with (ROOT/'v20_work'/f'{scene}_{cycle}.log').open('w') as log:
  result=subprocess.run([PY,str(Path(__file__).resolve()),'--mode',args.mode,'--worker',scene],cwd=REPO,env=environment,stdout=log,stderr=subprocess.STDOUT)
 if result.returncode:raise RuntimeError(f'{scene} replay failed; inspect saved log')
print('ALL_THREE_SAGE_REPLAYS_COMPLETE',flush=True)

"""Durable ordered expansion: data, shared cache, baselines, frozen candidate.

One documented strategy across all scenes. Errors stop dependent stages; each
completed stage remains restartable and has a saved return code and log.
"""
import concurrent.futures,json,os,subprocess,sys,time
from pathlib import Path
WORK=Path(__file__).resolve().parent;ROOT=WORK.parent;REPO=Path('/home/lewisliu/PartAware-SG')
PY='/home/lewisliu/miniconda3/envs/scannet-sg/bin/python';ESAM='/home/lewisliu/baselines/esam-env/bin/python';SAM3='/home/lewisliu/miniconda3/envs/sg-sam3/bin/python'
env=os.environ.copy();env.update(PYTHONUNBUFFERED='1',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',DISPLAY=':0',XDG_SESSION_TYPE='x11',LIBGL_ALWAYS_SOFTWARE='true');env.pop('WAYLAND_DISPLAY',None)
def run(name,script,python=PY,extra=()):
 print('EXPANSION_STAGE_BEGIN',name,flush=True)
 with (WORK/(name+'.log')).open('a') as log:r=subprocess.run([python,str(WORK/script),*extra],cwd=REPO,env=env,stdout=log,stderr=subprocess.STDOUT)
 (WORK/(name+'.exit')).write_text(str(r.returncode)+'\n')
 if r.returncode:raise RuntimeError(f'{name} failed with exit {r.returncode}; saved partials retained')
 print('EXPANSION_STAGE_COMPLETE',name,flush=True)
try:
 run('download','download_inputs.py')
 # Frontend API uses one shared cache; ESAM's frozen SAM preparation runs in
 # isolation and the global GPU lock. These two reads are independent.
 with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
  jobs=[pool.submit(run,'recognition','acquire_cache.py'),pool.submit(run,'esam_prepare','prepare_esam.py')]
  for job in jobs:job.result()
 with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
  jobs=[pool.submit(run,'construction','run_scan200.py'),pool.submit(run,'esam_inference','run_esam.py',ESAM)]
  for job in jobs:job.result()
 run('baseline_evaluation','evaluate_scan200.py',extra=('--construction-record',str(WORK/'construction_results.json'),'--esam-record',str(WORK/'esam_results.json')))
 run('projection_prepare','prepare_projection_verification.py')
 run('projection_names','add_projection_text.py')
 run('projection_inference','run_projection_worker.py',SAM3,('--text',))
 run('verified_geometry','build_verified_peer_masks.py',extra=('--text','--positive-only'))
 run('candidate_replay','replay_source_verified.py')
 run('candidate_evaluation','evaluate_replay.py',extra=('--cycle','sage_verified_core'))
 run('validation','validate_expansion.py')
 (WORK/'suite_completion.json').write_text(json.dumps(dict(complete=True,training=False,GT_in_construction=False,formal_default='v19',candidate='frozen SAGE-CORE-V',all_scene_goal_achieved=False,source='five fixed preregistered scans; results must be compared individually and against ai_001_010'),indent=2)+'\n')
except Exception as error:
 (WORK/'suite_failure.json').write_text(json.dumps(dict(complete=False,error=str(error)),indent=2)+'\n');raise
print('FIVE_SCAN_FULL_COMPARISON_COMPLETE',flush=True)

"""Resume cached recognition and dependent stages; reuse independent ESAM job."""
import json,os,subprocess,time
from pathlib import Path
WORK=Path(__file__).resolve().parent;REPO=Path('/home/lewisliu/PartAware-SG')
PY='/home/lewisliu/miniconda3/envs/scannet-sg/bin/python';SAM3='/home/lewisliu/miniconda3/envs/sg-sam3/bin/python'
environment=os.environ.copy();environment.update(PYTHONUNBUFFERED='1',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',
 HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HTTP_PROXY='http://127.0.0.1:17890',HTTPS_PROXY='http://127.0.0.1:17890',
 DISPLAY=':0',XDG_SESSION_TYPE='x11',LIBGL_ALWAYS_SOFTWARE='true');environment.pop('WAYLAND_DISPLAY',None)
def run(name,script,python=PY,extra=()):
 print('EXPANSION_RESUME_BEGIN',name,flush=True)
 with (WORK/(name+'.log')).open('a') as log:r=subprocess.run([python,str(WORK/script),*extra],cwd=REPO,env=environment,stdout=log,stderr=subprocess.STDOUT)
 (WORK/(name+'.exit')).write_text(str(r.returncode)+'\n')
 if r.returncode:raise RuntimeError(f'{name} failed; successful caches retained')
 print('EXPANSION_RESUME_COMPLETE',name,flush=True)
try:
 run('recognition','acquire_cache.py')
 run('construction','run_scan200.py')
 print('WAIT_EXISTING_ESAM_INFERENCE_NO_DUPLICATE',flush=True)
 while True:
  path=WORK/'esam_results.json'
  if path.exists() and json.loads(path.read_text()).get('complete'):break
  failure=WORK/'esam_only.exit'
  if failure.exists() and failure.read_text().strip()!='0':raise RuntimeError('Independent ESAM job failed; saved receipt requires recovery')
  time.sleep(5)
 run('baseline_evaluation','evaluate_scan200.py',extra=('--construction-record',str(WORK/'construction_results.json'),'--esam-record',str(WORK/'esam_results.json')))
 run('projection_prepare','prepare_projection_verification.py');run('projection_names','add_projection_text.py')
 run('projection_inference','run_projection_worker.py',SAM3,('--text',))
 run('verified_geometry','build_verified_peer_masks.py',extra=('--text','--positive-only'))
 run('candidate_replay','replay_source_verified.py');run('candidate_evaluation','evaluate_replay.py',extra=('--cycle','sage_verified_core'))
 run('validation','validate_expansion.py')
 (WORK/'suite_completion.json').write_text(json.dumps(dict(complete=True,training=False,formal_default='v19',candidate='frozen SAGE-CORE-V',all_scene_goal_achieved=False,recovered_transport=True),indent=2)+'\n')
except Exception as error:
 (WORK/'resume_failure.json').write_text(json.dumps(dict(complete=False,error=str(error)),indent=2)+'\n');raise
print('FIVE_SCAN_RECOVERED_COMPARISON_COMPLETE',flush=True)

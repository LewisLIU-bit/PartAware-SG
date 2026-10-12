"""Run frozen full profiles on the three registered ScanNet validation scans."""
import argparse,fcntl,hashlib,json,os,subprocess,sys,time
from pathlib import Path
WORK=Path(__file__).resolve().parent
REPO=Path('/home/lewisliu/PartAware-SG')
ROOT=Path('/home/lewisliu/datasets/scannet-sg-processed')
PY='/home/lewisliu/miniconda3/envs/scannet-sg/bin/python'
sys.path.insert(0,str(REPO/'scannet/script'))
from run_gpt_comparison import construction_fingerprint,scene_directory
from vision_api import attach_cache,atomic_json
from validate_recovered_cache import validate_complete_cache
from cached_relay_recovery import vision_api as recovery_api
registration=json.loads((WORK/'preregistration.json').read_text())
parser=argparse.ArgumentParser();parser.add_argument('--scene');arguments=parser.parse_args()
if arguments.scene:assert arguments.scene in registration['scenes']
selected_scenes=[arguments.scene] if arguments.scene else registration['scenes']
result_file=WORK/(arguments.scene+'_construction_results.json' if arguments.scene else 'construction_results.json')
# Serialize complete/check/run/mark transactions across an early per-scene job
# and the later suite. GPU locking alone cannot prevent duplicate late checks.
construction_mutex=(WORK/'construction.lock').open('a')
fcntl.flock(construction_mutex,fcntl.LOCK_EX)
environment=os.environ.copy();environment.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',OMP_NUM_THREADS='2',
    PYTHONPATH=str(REPO/'scannet/script'),DISPLAY=':0',XDG_SESSION_TYPE='x11',LIBGL_ALWAYS_SOFTWARE='true')
environment.pop('WAYLAND_DISPLAY',None);environment.pop('WAYLAND_SOCKET',None)
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def execute(command,log):
    print('START',log.name,flush=True)
    with (WORK/'gpu.lock').open('a') as mutex:
        fcntl.flock(mutex,fcntl.LOCK_EX)
        with log.open('a') as output:result=subprocess.run([str(c) for c in command],cwd=REPO,env=environment,stdout=output,stderr=subprocess.STDOUT)
    if result.returncode:raise RuntimeError(f'{log.name} failed with code {result.returncode}')
    print('COMPLETE',log.name,flush=True)
def wait_for(path,check=lambda _:True):
    while True:
        if path.is_file():
            data=json.loads(path.read_text())
            if check(data):return data
        time.sleep(5)
print('WAIT_FOR_EXISTING_HYPERSIM_GPU_JOB',flush=True)
wait_for(WORK.parent/'v19_holdout_work/comparison_result.json',lambda r:r['complete'])
rows=[]
for scene in selected_scenes:
    inputs=wait_for(WORK/(scene+'_input.json'))
    recognition=wait_for(WORK/(scene+'_recognition_complete.json'))
    manifest=Path(inputs['manifest']);cache=Path(recognition['cache'])
    cache_validation=validate_complete_cache(cache,recovery_api.Settings.load(REPO/'.gpt-vision.local.json'))
    atomic_json(WORK/(scene+'_recovery_cache_validation.json'),cache_validation)
    for profile in ['original','v19']:
        assert construction_fingerprint()==registration['construction_sha256']
        target=scene_directory(ROOT,profile,'scannet',scene)
        identity=dict(scene=scene,profile=profile,construction_sha256=registration['construction_sha256'],
            manifest_sha256=sha(manifest),recognition_fingerprint=recognition['request_fingerprint'],GT_in_construction=False)
        marker=target/'scan200_identity.json';completed=target/'scan200_completed.json'
        if marker.is_file():assert json.loads(marker.read_text())==identity
        else:
            assert not (target/'topology_map.json').exists(),'Refuse replacing an unrelated result'
            target.mkdir(parents=True,exist_ok=True);atomic_json(marker,identity)
        attach_cache(cache,target,manifest)
        if not completed.exists():
            execute([PY,REPO/'scannet/script/run_gpt_comparison.py','--worker',profile,'--manifest',manifest,
                '--processed-scene',target],WORK/(scene+'_'+profile+'.log'))
            assert construction_fingerprint()==registration['construction_sha256']
            atomic_json(completed,dict(graph_sha256=sha(target/'topology_map.json'),construction_unchanged=True,consumer_API_calls=0))
        record=dict(scene=scene,profile=profile,output=str(target),manifest=str(manifest),frames=inputs['frames'],
            graph_sha256=sha(target/'topology_map.json'),GT_in_construction=False,consumer_API_calls=0)
        rows.append(record);atomic_json(result_file,dict(registration=registration,results=rows,complete=len(rows)==2*len(selected_scenes)))
print('ALL_SCAN200_CONSTRUCTIONS_COMPLETE',flush=True)

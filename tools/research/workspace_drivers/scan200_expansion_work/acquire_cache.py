"""Acquire one reusable GPT cache per scan, without any annotation access."""
import json,os,subprocess,time
from pathlib import Path
WORK=Path(__file__).resolve().parent
REPO=Path('/home/lewisliu/PartAware-SG')
ROOT=Path('/home/lewisliu/datasets/scannet-sg-processed')
PY='/home/lewisliu/miniconda3/envs/scannet-sg/bin/python'
from validate_recovered_cache import validate_complete_cache
from cached_relay_recovery import vision_api as recovery_api
registration=json.loads((WORK/'preregistration.json').read_text())
completed=set()
while len(completed)<len(registration['scenes']):
    ready=[s for s in registration['scenes'] if s not in completed and (WORK/(s+'_input.json')).is_file()]
    if not ready:time.sleep(5);continue
    for scene in ready:
        record=json.loads((WORK/(scene+'_input.json')).read_text())
        print('RECOGNITION_START',scene,record['frames'],flush=True)
        # Explicit finite recovery. Successes are never repeated; persisted
        # transient stream errors are archived by the unchanged API contract.
        for attempt in range(2):
            with (WORK/(scene+'_recognition.log')).open('a') as output:
                result=subprocess.run([PY,str(WORK/'cached_relay_recovery.py'),'--config',str(REPO/'.gpt-vision.local.json'),
                    '--manifest',record['manifest'],'--cache-root',str(ROOT/'gpt_vision_cache'),'--workers','1',
                    '--retry-transport-failures','--recover-non-stream'],
                    cwd=REPO,env=os.environ.copy(),stdout=output,stderr=subprocess.STDOUT)
            if not result.returncode:break
            print('TRANSPORT_RETRY',scene,attempt+1,'successful caches retained; failed billing unknown',flush=True)
            time.sleep(5)
        if result.returncode:raise RuntimeError(f'Recognition failed for {scene}; preserve cache and inspect transport log')
        provenance=json.loads((ROOT/'gpt_vision_cache/scannet'/scene/'recognition_provenance.json').read_text())
        assert provenance['complete'] and not provenance['ground_truth_used']
        validated=validate_complete_cache(ROOT/'gpt_vision_cache/scannet'/scene,recovery_api.Settings.load(REPO/'.gpt-vision.local.json'))
        (WORK/(scene+'_recovery_cache_validation.json')).write_text(json.dumps(validated,indent=2)+'\n')
        (WORK/(scene+'_recognition_complete.json')).write_text(json.dumps(dict(scene=scene,cache=str(ROOT/'gpt_vision_cache/scannet'/scene),
            frames=record['frames'],request_fingerprint=provenance['request_fingerprint']),indent=2)+'\n')
        completed.add(scene);print('RECOGNITION_COMPLETE',scene,flush=True)
print('ALL_CACHES_COMPLETE',flush=True)

"""Restartable download/export of the five preregistered validation scans."""
import concurrent.futures,json,sys
from pathlib import Path
WORK=Path(__file__).resolve().parent;sys.path.insert(0,str(WORK))
from download_helpers import sha,download,export
registration=json.loads((WORK/'preregistration.json').read_text())
ROOT=Path('/home/lewisliu/datasets');BASE='https://kaldir.vc.cit.tum.de/scannet/'
def prepare(scene):
 raw=ROOT/'scannet/scans'/scene
 files=[('.sens','v1/scans'),('.txt','v2/scans'),('_vh_clean_2.ply','v2/scans'),('.aggregation.json','v2/scans'),('_vh_clean_2.0.010000.segs.json','v2/scans')]
 for suffix,release in files:download(BASE+release+'/'+scene+'/'+scene+suffix,raw/(scene+suffix))
 manifest=ROOT/'scannet-sg-input/scannet'/scene/'manifest.json'
 metadata=export(raw/(scene+'.sens'),manifest.parent,scene)
 result=dict(scene=scene,manifest=str(manifest),manifest_sha256=sha(manifest),frames=len(metadata['frames']),split='val',files=[dict(path=str(raw/(scene+s)),sha256=sha(raw/(scene+s))) for s,_ in files])
 partial=WORK/(scene+'_input.partial.json');partial.write_text(json.dumps(result,indent=2)+'\n');partial.replace(WORK/(scene+'_input.json'))
 return result
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:results=list(pool.map(prepare,registration['scenes']))
(WORK/'inputs.json').write_text(json.dumps(dict(preregistration=registration,results=results,complete=True),indent=2)+'\n')
print('FIVE_SCAN_INPUTS_COMPLETE',[(r['scene'],r['frames']) for r in results],flush=True)

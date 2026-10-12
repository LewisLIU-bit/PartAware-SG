"""Independent cache and published-artifact audit for the expanded comparison."""
import hashlib,json,sys
from pathlib import Path
WORK=Path(__file__).resolve().parent;ROOT=WORK.parent;REPO=Path('/home/lewisliu/PartAware-SG')
sys.path.insert(0,str(REPO/'scannet/script'))
from run_gpt_comparison import construction_fingerprint
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
registration=json.loads((WORK/'preregistration.json').read_text());assert construction_fingerprint()==registration['construction_sha256']
base=json.loads((WORK/'evaluation_results.json').read_text());candidate=json.loads((WORK/'sage_verified_core_evaluation.json').read_text());assert base['complete'] and candidate['complete']
assert len(base['results'])==15 and len(candidate['results'])==5
rows=[]
for scene in registration['scenes']:
 inputs=json.loads((WORK/(scene+'_input.json')).read_text());assert sha(inputs['manifest'])==inputs['manifest_sha256']
 sources=[Path(r['processed_scene']) for r in base['results'] if r['scene_id']==scene and r['profile'] in ['original','v19']]
 target=Path(next(r['processed_scene'] for r in candidate['results'] if r['scene_id']==scene));assert json.loads((target/'research_completion.json').read_text())['complete']
 provenance=[json.loads((p/'recognition_provenance.json').read_text()) for p in sources+[target]]
 assert all(p['complete'] and p['consumer_api_calls']==0 for p in provenance)
 assert len({p['request_fingerprint'] for p in provenance})==1
 v19=next(p for p in sources if 'partaware_v19' in str(p))
 for fid in provenance[-1]['expected_frame_ids']:
  for suffix in ['.png','_instance.json']:assert sha(v19/'refined_instance'/f'{fid}{suffix}')==sha(target/'refined_instance'/f'{fid}{suffix}')
 graph=json.loads((target/'topology_map.json').read_text());geometry=target/graph['geometry_provenance']['source'];assert geometry.name=='instance_cloud_observed_domain.ply'
 rows.append(dict(scene=scene,frames=inputs['frames'],manifest_sha256=inputs['manifest_sha256'],request_fingerprint=provenance[-1]['request_fingerprint'],consumer_API_calls=0,source_masks_features_byte_identical=True,graph_sha256=sha(target/'topology_map.json'),geometry_sha256=sha(geometry)))
(WORK/'validation.json').write_text(json.dumps(dict(complete=True,results=rows,training=False,GT_in_construction=False,formal_default='v19',new_image_API_calls='initial shared cache only; consumer calls 0; retries/cost recorded by cache acquisition',same_frozen_strategy_all_five=True,selection_before_results=True,full_312_benchmark=False),indent=2)+'\n')
print('FIVE_SCAN_REAL_OUTPUTS_VALIDATED',flush=True)

"""Independent publication/cache audit for completed research replays."""
import hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];REPO=Path('/home/lewisliu/PartAware-SG')
sys.path.insert(0,str(REPO/'scannet/script'))
from run_gpt_comparison import construction_fingerprint
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
frozen='250368fb6d5d99eb5a9129ef3d51ed44be10176789d8d7ea4c8d9bd882b2f894';assert construction_fingerprint()==frozen
reg=json.loads((ROOT/'scan200_work/evaluation_results.json').read_text());runs=[]
for cycle in ['sage_exclusive_weighted','sage_exclusive_core','sage_verified_core']:
 path=ROOT/'v20_work'/f'{cycle}_evaluation.json'
 if not path.exists():continue
 evaluation=json.loads(path.read_text())
 if not evaluation['complete']:continue
 rows=[]
 for result in evaluation['results']:
  scene=result['scene_id'];target=Path(result['processed_scene']);base=Path(next(r for r in reg['results'] if r['scene_id']==scene and r['profile']=='v19')['processed_scene'])
  assert json.loads((target/'research_completion.json').read_text())['complete']
  a=json.loads((base/'recognition_provenance.json').read_text());b=json.loads((target/'recognition_provenance.json').read_text());assert a['request_fingerprint']==b['request_fingerprint'] and b['complete'] and b['consumer_api_calls']==0 and a['expected_frame_ids']==b['expected_frame_ids']
  for fid in a['expected_frame_ids']:
   for suffix in ['.png','_instance.json']:assert sha(base/'refined_instance'/f'{fid}{suffix}')==sha(target/'refined_instance'/f'{fid}{suffix}')
  graph=json.loads((target/'topology_map.json').read_text());ply=target/graph['geometry_provenance']['source'];assert ply.name=='instance_cloud_observed_domain.ply'
  audit=json.loads((target/'source_agreement_audit.json').read_text());guard=json.loads((target/'observed_domain_guard_audit.json').read_text());assert not audit['GT_in_construction'] and guard['existing_points_and_labels_unchanged'] and guard['generated_points']==0
  assert guard['code_sha256']==sha(ROOT/'v20_work/research/observed_domain_guard.py')
  assert audit['component_sha256']==sha(ROOT/'v20_work/research/source_agreement.py')
  missing=[p for p in graph.get('part_nodes',{}).values() if str(p.get('parent_id')) not in graph['object_nodes']['nodes'] and p.get('status')!='provisional'];assert not missing
  rows.append(dict(scene=scene,frames=len(a['expected_frame_ids']),original_GPT_request_fingerprint=a['request_fingerprint'],source_masks_and_features_byte_identical=True,new_API_calls=0,graph_sha256=sha(target/'topology_map.json'),geometry_sha256=sha(ply),source_agreement_audit_sha256=sha(target/'source_agreement_audit.json'),observed_domain_guard_audit_sha256=sha(target/'observed_domain_guard_audit.json'),part_parent_links_valid=True,unlabeled_objects=sum(n.get('name')=='unlabeled object' for n in graph['object_nodes']['nodes'].values())))
 runs.append(dict(cycle=cycle,complete=True,evaluation_sha256=sha(path),full_replay=True,results=rows))
receipt=dict(complete=len(runs)>=2,frozen_v19_fingerprint=frozen,official_evaluator_sha256=sha(ROOT/'scan200_work/evaluate_scan200.py'),training=False,new_API_calls=0,GT_in_construction=False,formal_default='v19',all_scene_target_achieved=False,runs=runs)
(ROOT/'v20_work/sage_full_replay_validation.json').write_text(json.dumps(receipt,indent=2)+'\n');print('FULL_SAGE_VALIDATION',len(runs),'cycles',sum(r['frames'] for run in runs for r in run['results']),'frames',flush=True)

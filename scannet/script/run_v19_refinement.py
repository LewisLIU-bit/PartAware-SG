"""Replay generic v19 stages on an immutable, verified v18 final product.

This incremental protocol is explicit: it does not claim a fresh raw-fusion
rerun. The full v19 profile remains available through run_gpt_comparison.py.
Never change a base result, call a vision API, load an annotation in construction,
or promote a hand-edited trial folder.
"""
import argparse,hashlib,json,os,shutil,subprocess,sys
from pathlib import Path
from types import SimpleNamespace
os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
REPO=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(REPO/'scannet/script'))
from run_gpt_comparison import configure_profile,construction_fingerprint
from pipeline_components import consensus_ownership,support_residuals,face_boxes,scene_review,flow
import pipeline_components as registry


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def publish_profile_provenance(context,source):
    path=context.scene/'topology_map.json';graph=json.loads(path.read_text())
    recognition=json.loads((context.scene/'recognition_provenance.json').read_text())
    prior=graph['pipeline_provenance']
    if 'base_pipeline_provenance' not in graph:graph['base_pipeline_provenance']=dict(prior)
    tree=flow.process_tree(registry,recognition)
    prior.update(version='v19_gpt',functional_tree=tree,
        start_stage='verified_v18_final',construction_start_stage='verified_v18_final',
        final_geometry=[module.__name__ for module in registry.FINAL_GEOMETRY])
    graph['process_tree']=tree
    graph['refinement_protocol']='v19_new_stages_on_immutable_v18_final_product'
    graph['refinement_provenance']=dict(base_scene=str(source),
        stages=[consensus_ownership.__name__,support_residuals.__name__],
        raw_fusion_rerun=False,vision_api_calls=0,ground_truth_used_in_construction=False)
    path.write_text(json.dumps(graph,indent=2)+'\n')


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--base-scene',required=True);parser.add_argument('--processed-scene',required=True);parser.add_argument('--manifest',required=True);args=parser.parse_args()
    source=Path(args.base_scene).resolve();scene=Path(args.processed_scene).resolve();manifest=Path(args.manifest).resolve()
    if source==scene or scene.exists():raise ValueError('Refuse base mutation or overwrite of an existing final scene')
    recognition=json.loads((source/'recognition_provenance.json').read_text())
    if recognition.get('provider')!='relay_gpt' or not recognition.get('complete'):raise ValueError('Requires complete existing GPT provenance')
    base_files={str(p.relative_to(source)):sha(p) for p in source.rglob('*') if p.is_file()}
    shutil.copytree(source,scene);code_hash=construction_fingerprint();environment=os.environ.copy();environment.update(PYTHONPATH=str(REPO/'scannet/script'),OMP_NUM_THREADS='2',PYTHONUNBUFFERED='1')
    configure_profile('v19',registry);registry.BOX_FITTING=face_boxes
    graph=json.loads((scene/'topology_map.json').read_text());geometry=scene/graph['geometry_provenance']['source'];events=[]
    def event(message,**kwargs):
        events.append(dict(message=message,**kwargs));print(message,kwargs,flush=True)
    def execute(command,message):
        event(message);result=subprocess.run(command,cwd=REPO,env=environment)
        if result.returncode:raise RuntimeError(f'Stage failed: {message}')
    context=SimpleNamespace(repo=REPO,scene=scene,manifest=manifest,image_dir=None,graph_geometry=geometry,canonical_geometry_input=scene/'topology_map.json',edge_threshold=.5,environment=environment,event=event,execute=execute)
    consensus_ownership.construct(context);support_residuals.construct(context)
    publish_profile_provenance(context,source)
    scene_review.construct(context)
    assert construction_fingerprint()==code_hash,'Construction code changed during build'
    assert all(sha(source/name)==value for name,value in base_files.items()),'Base scene changed'
    receipt=dict(version='v19',protocol='incremental_generic_refinement_of_verified_v18_final',base_scene=str(source),base_graph_sha256=base_files['topology_map.json'],base_files_sha256=base_files,construction_sha256=code_hash,runner_sha256=sha(Path(__file__)),vision_api_calls=0,ground_truth_used_in_construction=False,raw_fusion_rerun=False,base_unchanged=True,graph_sha256=sha(scene/'topology_map.json'),geometry_sha256=sha(context.graph_geometry),events=events)
    (scene/'v19_build_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print('V19_FINAL_VERIFIED',scene,flush=True)


if __name__=='__main__':main()

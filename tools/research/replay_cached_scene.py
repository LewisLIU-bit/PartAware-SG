"""Replay MICA-VSA from signed v19 observations, including interrupted runs.

This research entry point leaves the production registry/default unchanged.
RGB-D/manifest paths come from the caller; no GT is loaded here.
"""
import argparse,hashlib,json,os,sys
from pathlib import Path

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',required=True)
    parser.add_argument('--source',required=True,help='Frozen v19 scene with signed observations')
    parser.add_argument('--target',required=True,help='Separate research output, dataset/scene layout')
    parser.add_argument('--validate-only',action='store_true',help='Check input/cache contracts without replay')
    args=parser.parse_args()
    repo=Path(__file__).resolve().parents[2]
    source=Path(args.source).expanduser().resolve()
    target=Path(args.target).expanduser().resolve()
    if source==target or source.name!=target.name or source.parent.name!=target.parent.name:
        parser.error('Source/target must differ and share dataset/scene identity')
    provenance=json.loads((source/'recognition_provenance.json').read_text())
    if not provenance.get('complete'):
        parser.error('Incomplete source recognition provenance')
    sys.path[:0]=[str(repo/'scannet/script'),str(repo/'tools/research')]
    from partaware.geometry import load_capture
    manifest,jobs,_,_,_=load_capture(Path(args.manifest).expanduser().resolve())
    if manifest['dataset']!=source.parent.name or manifest['scene_id']!=source.name:
        parser.error('Manifest dataset/scene differs from source')
    if [job['frame_id'] for job in jobs]!=provenance['expected_frame_ids']:
        parser.error('Manifest frame sequence differs from signed observations')
    for fid in provenance['expected_frame_ids']:
        for suffix in ['.png','_instance.json']:
            if not (source/'refined_instance'/f'{fid}{suffix}').is_file():
                parser.error(f'Missing source observation: {fid}{suffix}')
    reuse=['--reuse-scene',str(source)]
    if (target/'refined_instance').exists():
        copied=json.loads((target/'recognition_provenance.json').read_text())
        for key in ['request_fingerprint','expected_frame_ids']:
            if copied[key]!=provenance[key]:parser.error('Copied recognition provenance differs')
        for fid in provenance['expected_frame_ids']:
            for suffix in ['.png','_instance.json']:
                if sha(source/'refined_instance'/f'{fid}{suffix}')!=sha(target/'refined_instance'/f'{fid}{suffix}'):
                    parser.error(f'Copied frontend differs: {fid}{suffix}')
        reuse=[]
    if args.validate_only:
        print(json.dumps(dict(validated=True,frames=len(jobs),resume=not reuse,
            source=str(source),target=str(target),manifest_sha256=sha(args.manifest),
            full_geometry_replay=False,vision_API_calls=0)))
        return
    os.environ.update(PARTAWARE_COMPARISON_PROFILE='v19',HF_HUB_OFFLINE='1',
        TRANSFORMERS_OFFLINE='1',DISPLAY=':0',XDG_SESSION_TYPE='x11',LIBGL_ALWAYS_SOFTWARE='true')
    os.environ.pop('WAYLAND_DISPLAY',None)
    import run_pipeline
    from run_gpt_comparison import configure_profile,construction_fingerprint
    import soft_mutual_association
    configure_profile('v19',run_pipeline.components)
    run_pipeline.components.ASSOCIATION=soft_mutual_association
    run_pipeline.components.VERSION='research_mutual_r2'
    sys.argv=[str(repo/'scannet/script/run_pipeline.py'),'--manifest',args.manifest,
        '--processed-scene',str(target),*reuse,'--start-stage','fusion']
    run_pipeline.main()
    # A marker is written only after all actual stages return successfully.
    receipt=dict(full_pipeline_from_cached_masks=True,algorithm='soft_mutual_association',
        training=False,vision_API_calls=0,GT_in_construction=False,frozen_v19_preserved=True,
        source=str(source),target=str(target),manifest_sha256=sha(args.manifest),
        recognition_request_fingerprint=provenance['request_fingerprint'],
        frozen_core_fingerprint=construction_fingerprint(),
        research_source_sha256={p.name:sha(p) for p in [Path(__file__),
            repo/'tools/research/mutual_association.py',repo/'tools/research/soft_mutual_association.py']})
    (target/'research_completion.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print('MICA_VSA_FULL_REPLAY_COMPLETE',target,flush=True)

if __name__=='__main__':main()

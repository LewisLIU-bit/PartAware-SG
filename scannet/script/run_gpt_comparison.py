"""Construct v14 using one existing GPT cache per scene; keep historical profiles."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

from qwen_tools.manifest_io import load_manifest_images
from vision_api import attach_cache, atomic_json

REPO = Path(__file__).resolve().parents[2]
DEFAULT_DATA = Path.home() / 'datasets'
VERSIONS = ('original', 'v11', 'v12', 'v13', 'v14')
ACTIVE_VERSIONS = ('v14',)


def configure_profile(version, registry):
    """Research comparison profiles run in separate processes, leaving defaults intact."""
    if version not in VERSIONS:
        raise ValueError('Unknown comparison version')
    registry.VERSION = version + '_gpt'
    registry.WHOLE_OBJECT_VALIDATION = None
    registry.SURFACE_ASSEMBLY = None
    registry.HIERARCHY_VALIDATION = None
    registry.BOX_FITTING = None
    registry.AXIAL_VALIDATION = None
    registry.BODY_CONTINUITY = None
    from pipeline_components import suspension_geometry
    registry.FINAL_GEOMETRY = [suspension_geometry if entry.__name__.endswith('.verified_suspension') else entry
        for entry in getattr(registry, 'FINAL_GEOMETRY', []) if entry.__name__.rsplit('.', 1)[-1] not in ('verified_cuboids', 'enclosure_continuity')]
    if any(entry.__name__.endswith('.seeded_surfaces') for entry in getattr(registry, 'GEOMETRY_COMPONENTS', [])):
        from pipeline_components import structural_surfaces, backed_cuboid
        registry.GEOMETRY_COMPONENTS = [structural_surfaces if entry.__name__.endswith('.seeded_surfaces') else entry
            for entry in registry.GEOMETRY_COMPONENTS]
        if backed_cuboid not in registry.GEOMETRY_COMPONENTS:
            registry.GEOMETRY_COMPONENTS.append(backed_cuboid)
    registry.MEASURED_REFINEMENT = [entry for entry in getattr(registry, 'MEASURED_REFINEMENT', [])
        if entry.__name__.rsplit('.', 1)[-1] not in ('part_body_assembly', 'repeated_instances', 'visual_part_anchoring')]
    if version != 'original':
        from pipeline_components import visibility_ownership
        registry.OWNERSHIP_VALIDATION = visibility_ownership
    if version == 'original':
        registry.GROUNDING_BACKEND = 'dino'
        for key in ('FRONTEND', 'FUSION', 'ASSOCIATION', 'INSTANCE_REFINEMENT',
                    'GEOMETRY_OUTPUT', 'OBJECT_VALIDATION', 'OBSERVED_VALIDATION',
                    'BACKGROUND_VALIDATION', 'IDENTITY_VALIDATION', 'SURFACE_VALIDATION',
                    'PART_GEOMETRY', 'OWNERSHIP_VALIDATION'):
            setattr(registry, key, None)
        for key in ('GRAPH_COMPONENTS', 'GEOMETRY_COMPONENTS',
                    'MEASURED_REFINEMENT', 'FINAL_GEOMETRY'):
            setattr(registry, key, [])
    elif version in ('v12', 'v13', 'v14'):
        from pipeline_components import sam3_frontend, whole_object_consensus, native_assembly, part_body_assembly
        registry.FRONTEND = sam3_frontend
        registry.WHOLE_OBJECT_VALIDATION = whole_object_consensus
        registry.SURFACE_ASSEMBLY = native_assembly
        registry.MEASURED_REFINEMENT.insert(1, part_body_assembly)
        if version in ('v13', 'v14'):
            from pipeline_components import hierarchical_masks, plane_boxes, repeated_instances, visual_part_anchoring
            registry.HIERARCHY_VALIDATION = hierarchical_masks
            registry.BOX_FITTING = plane_boxes
            registry.MEASURED_REFINEMENT.insert(1, repeated_instances)
            registry.MEASURED_REFINEMENT.insert(3, visual_part_anchoring)
        if version == 'v14':
            from pipeline_components import seeded_surfaces, verified_cuboids, verified_suspension, attachment_identity, enclosure_continuity
            registry.GEOMETRY_COMPONENTS = [seeded_surfaces if entry.__name__.endswith('.structural_surfaces') else entry
                for entry in registry.GEOMETRY_COMPONENTS if not entry.__name__.endswith('.backed_cuboid')]
            registry.AXIAL_VALIDATION = attachment_identity
            registry.BODY_CONTINUITY = enclosure_continuity
            registry.FINAL_GEOMETRY = [verified_suspension if entry.__name__.endswith('.suspension_geometry') else entry
                for entry in registry.FINAL_GEOMETRY]+[enclosure_continuity, verified_cuboids]


def run_worker(args):
    # Model subprocesses must inherit the same code-defined experimental registry.
    os.environ['PARTAWARE_COMPARISON_PROFILE'] = args.worker
    import run_pipeline
    source = Path(args.reuse_scene or args.processed_scene)
    provenance = json.loads((source/'recognition_provenance.json').read_text())
    if provenance.get('provider') != 'relay_gpt' or not provenance.get('complete'):
        raise ValueError('GPT comparison requires complete shared GPT recognition provenance')
    configure_profile(args.worker, run_pipeline.components)
    sys.argv = [str(REPO/'scannet/script/run_pipeline.py'),
                '--manifest', args.manifest[0], '--processed-scene', args.processed_scene,
                '--start-stage', args.start_stage]
    if args.reuse_scene:
        sys.argv += ['--reuse-scene', args.reuse_scene]
    run_pipeline.main()


def execute(command, logfile):
    logfile.parent.mkdir(parents=True, exist_ok=True)
    print('开始执行对比阶段', {'command': command[1:3], 'log': str(logfile)}, flush=True)
    environment = os.environ.copy()
    environment.update(PYTHONPATH=str(REPO/'scannet/script'), PYTHONUNBUFFERED='1',
                       HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    with logfile.open('a') as output:
        result = subprocess.run(command, cwd=REPO, env=environment,
                                stdout=output, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f'Comparison stage failed ({result.returncode}); inspect {logfile}')


def scene_directory(root, version, dataset, scene_id):
    experiment = ('gpt_original_v1' if version == 'original'
                  else f'partaware_{version}' if version in ('v12', 'v13', 'v14') else 'partaware_gpt_v11')
    return root/experiment/dataset/scene_id


def construction_fingerprint():
    script = REPO/'scannet/script'
    files = [Path(__file__), script/'run_pipeline.py', script/'vision_api.py',
             script/'grounded_sam/scannet_process/get_seg_openset.py',
             *sorted((script/'grounded_sam/grounded_sam').glob('*.py')),
             *sorted((script/'qwen_tools').glob('*.py')),
             *sorted((script/'pipeline_components').glob('*.py')),
             *sorted((script/'partaware').glob('*.py')),
             REPO/'scannet/build-partaware/openset_ply_map',
             REPO/'scannet/build-partaware/generate_json']
    digest = hashlib.sha256()
    for file in files:
        digest.update(str(file.relative_to(REPO)).encode())
        digest.update(file.read_bytes())
    return digest.hexdigest()


def summarize(report):
    mvo = report['maximum_volume_overlap']
    return {key: report[key] for key in (
        'gt_objects', 'predicted_objects', 'geometry_only_box_AP25',
        'geometry_only_box_AP50', 'geometry_only_box_AP75', 'object_count_consistency')
    } | {'MVO25': mvo['AP25'], 'MVO50': mvo['AP50']}


def compare(args):
    manifests = args.manifest or [str(DEFAULT_DATA/'scannet-sg-input/hypersim'/name/'manifest.json')
                                  for name in ('ai_001_002_v3', 'ai_001_010_v3')]
    root = Path(args.output_root).expanduser().resolve()
    records = []
    for manifest in manifests:
        dataset, scene_id, frames = load_manifest_images(manifest)
        # Construction never makes a new VLM request, including incomplete caches.
        cache = Path(args.cache_root).expanduser().resolve()/dataset/scene_id
        provenance_file = cache/'recognition_provenance.json'
        if not provenance_file.is_file():
            raise FileNotFoundError('Acquire the shared GPT cache explicitly before construction')
        provenance = json.loads(provenance_file.read_text())
        if provenance.get('provider') != 'relay_gpt' or not provenance.get('complete'):
            raise ValueError('Shared GPT cache is incomplete; no image API request was made')
        expected = {'manifest': str(Path(manifest).resolve()),
                    'manifest_sha256': hashlib.sha256(Path(manifest).read_bytes()).hexdigest(),
                    'recognition_fingerprint': json.loads((cache/'recognition_provenance.json').read_text())['request_fingerprint'],
                    'driver_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    'construction_sha256': construction_fingerprint()}
        print('复用同场景既有 GPT 识图缓存', {'scene': scene_id, 'frames': len(frames)}, flush=True)
        for version in ACTIVE_VERSIONS:
            target = scene_directory(root, version, dataset, scene_id)
            target.mkdir(parents=True, exist_ok=True)
            marker = target/'gpt_comparison.json'
            identity = {**expected, 'version': version}
            if marker.exists() and json.loads(marker.read_text()) != identity:
                raise ValueError(f'Comparison inputs or driver changed: {target}; select a fresh output root')
            if not marker.exists() and (target/'topology_map.json').exists():
                raise ValueError(f'Refusing to overwrite an unrelated existing graph: {target}')
            atomic_json(marker, identity)
            completed = target/'comparison_completed.json'
            if completed.exists():
                saved = json.loads(completed.read_text())
                graph = target/'topology_map.json'
                if saved['graph_sha256'] != hashlib.sha256(graph.read_bytes()).hexdigest():
                    raise ValueError(f'Completed graph changed: {graph}')
                print('复用已完成的对比产物', {'scene': scene_id, 'version': version}, flush=True)
            else:
                command = [sys.executable, str(Path(__file__)), '--worker', version,
                           '--manifest', str(manifest), '--processed-scene', str(target)]
                attach_cache(cache, target, manifest)
                execute(command, target/'comparison_steps.log')
                atomic_json(completed, {'graph_sha256': hashlib.sha256((target/'topology_map.json').read_bytes()).hexdigest(),
                                        'recognition_cache': str(cache), 'consumer_api_calls': 0})
            evaluation = target/'evaluation.json'
            if not evaluation.exists():
                execute([sys.executable, str(REPO/'scannet/script/evaluate_hypersim.py'),
                         '--manifest', str(manifest), '--processed-scene', str(target),
                         '--output', str(evaluation)], target/'comparison_steps.log')
            report = json.loads(evaluation.read_text())
            if report['graph_sha256'] != hashlib.sha256((target/'topology_map.json').read_bytes()).hexdigest():
                raise ValueError('Evaluation refers to a different graph')
            if report['prediction_ply_sha256'] != hashlib.sha256(Path(report['prediction_ply']).read_bytes()).hexdigest():
                raise ValueError('Evaluation refers to different point geometry')
            records.append({'scene_id': scene_id, 'version': version, 'output': str(target),
                            'evaluation': str(evaluation), 'metrics': summarize(report),
                            'shared_recognition_cache': str(cache), 'consumer_api_calls': 0})
            summary_path = root/'gpt_comparison_summary.json'
            previous = json.loads(summary_path.read_text()).get('results', []) if summary_path.is_file() else []
            combined = {(r['scene_id'], r['version']): r for r in previous}
            combined.update({(r['scene_id'], r['version']): r for r in records})
            atomic_json(summary_path, {
                'recognition_scope': 'v13 and v12 use one existing shared GPT cache per scene; historical comparisons remain unchanged',
                'qwen_api_calls': 0, 'results': list(combined.values()),
                'evaluation_protocol': 'observed_hypersim_adaptation_not_official_benchmark',
                'learned_universal_completion_solved': False})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config')
    parser.add_argument('--manifest', action='append')
    parser.add_argument('--cache-root', default=str(DEFAULT_DATA/'scannet-sg-processed/gpt_vision_cache'))
    parser.add_argument('--output-root', default=str(DEFAULT_DATA/'scannet-sg-processed'))
    parser.add_argument('--recognition-workers', type=int, default=2)
    parser.add_argument('--worker', choices=VERSIONS, help=argparse.SUPPRESS)
    parser.add_argument('--processed-scene', help=argparse.SUPPRESS)
    parser.add_argument('--reuse-scene', help=argparse.SUPPRESS)
    parser.add_argument('--start-stage', default='tags', help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        if not args.processed_scene or not args.manifest or len(args.manifest) != 1:
            parser.error('A construction worker requires one manifest and a processed scene')
        run_worker(args)
    else:
        compare(args)


if __name__ == '__main__':
    main()

"""Default removable-component pipeline for existing ScanNet and Hypersim inputs."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import shutil
import sys
import pipeline_components as components
from partaware.geometry import load_capture

REPO = Path(__file__).resolve().parents[2]


class Context:
    def __init__(self, args):
        self.repo = REPO
        self.manifest = Path(args.manifest).expanduser().resolve() if args.manifest else None
        self.image_dir = Path(args.image_dir).expanduser().resolve() if args.image_dir else None
        self.scene = Path(args.processed_scene).expanduser().resolve()
        self.scene.mkdir(parents=True, exist_ok=True)
        self.max_depth, self.stride, self.edge_threshold = args.max_depth, args.stride, args.edge_threshold
        self.log = (self.scene / 'pipeline_zh.jsonl').open('a', encoding='utf-8')
        self.environment = os.environ.copy()
        self.environment.update(PYTHONPATH=str(REPO / 'scannet/script'), PYTHONUNBUFFERED='1',
                                HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')

    def event(self, message, **fields):
        record = {'time': datetime.now(timezone.utc).isoformat(), 'message': message, **fields}
        self.log.write(json.dumps(record, ensure_ascii=False) + '\n')
        self.log.flush()
        print(message, fields, flush=True)

    def execute(self, command, title):
        self.event('开始执行阶段', stage=title)
        with (self.scene / 'pipeline_steps.log').open('a') as file:
            result = subprocess.run(command, cwd=self.repo, env=self.environment, stdout=file, stderr=subprocess.STDOUT)
        if result.returncode:
            raise RuntimeError(f'{title} failed ({result.returncode}); inspect pipeline_steps.log')
        self.event('阶段执行完成', stage=title)


def legacy_fusion(context):
    build = context.repo / 'scannet/build-partaware/openset_ply_map'
    if context.manifest:
        command = [str(build), '--manifest', str(context.manifest), str(context.scene), str(context.max_depth), str(context.stride)]
        capture = json.loads(context.manifest.read_text())
        if (capture.get('dataset') == 'hypersim' and capture.get('world_frame') == 'hypersim_world_z_up'
                and capture.get('length_unit') == 'meter'):
            command.append('--filter_floor')
        context.execute(command, 'ScanNet-SG 原始物体融合')
        return
    command = [str(build), context.scene.name, '0', str(context.scene.parent),
               str(context.image_dir.parent) if context.image_dir else '-', str(context.max_depth), str(context.stride)]
    context.execute(command, 'ScanNet-SG 原始物体融合')


def build_legacy_graph(context):
    context.execute([str(context.repo / 'scannet/build-partaware/generate_json'),
                     str(context.scene / 'instance_cloud.ply'), '0', '1', str(context.edge_threshold)], 'ScanNet-SG 原始空间图构建')
    context.execute([sys.executable, str(context.repo / 'scannet/script/map_ply_post_filter.py'),
                     str(context.scene), '--openset'], 'ScanNet-SG 原始点云与包围盒精修')


def build_graph(context):
    build_legacy_graph(context)
    validator = getattr(components, 'OBJECT_VALIDATION', None)
    if validator is not None:
        validator.construct(context)
    context.graph_geometry = context.scene/'instance_cloud_cleaned.ply'
    for component in getattr(components, 'GEOMETRY_COMPONENTS', []):
        component.construct(context)
    publisher = getattr(components, 'GEOMETRY_OUTPUT', None)
    if publisher is not None:
        publisher.publish(context)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument('--manifest')
    inputs.add_argument('--image-dir')
    parser.add_argument('--processed-scene', required=True)
    parser.add_argument('--reuse-scene', help='Existing observations and cached tags from the same scene')
    parser.add_argument('--qwen-model', default='qwen3-vl-plus', help='Cached tag provenance only; no API call')
    parser.add_argument('--max-depth', type=float, default=0)
    parser.add_argument('--stride', type=int, default=2)
    parser.add_argument('--edge-threshold', type=float, default=2)
    # Stages support recovery, not removal of individual construction components.
    parser.add_argument('--start-stage', choices=['tags', 'segmentation', 'fusion', 'graph', 'parts'], default='tags')
    args = parser.parse_args()
    if args.max_depth < 0 or args.stride < 1 or args.edge_threshold <= 0:
        parser.error('Invalid geometry settings')
    context = Context(args)
    try:
        if args.reuse_scene:
            source = Path(args.reuse_scene).expanduser().resolve()
            if source.name != context.scene.name or source.parent.name != context.scene.parent.name:
                raise ValueError('Observation reuse requires the same dataset and scene identity')
            if source == context.scene:
                raise ValueError('Reuse source and target must differ')
            destination = context.scene/'refined_instance'
            if destination.exists():
                raise FileExistsError('Reuse target already has observations; resume with --start-stage')
            shutil.copytree(source/'refined_instance', destination)
            if (source/'frontend_cache').exists():
                shutil.copytree(source/'frontend_cache', context.scene/'frontend_cache')
            if (source/'frontend_provenance.json').is_file():
                shutil.copyfile(source/'frontend_provenance.json', context.scene/'frontend_provenance.json')
            if args.start_stage == 'graph':
                for name in ['instance_cloud.ply', 'instance_cloud_colored.ply', 'instance_cloud_with_background.ply',
                             'instance_name_map.csv', 'averaged_instance_features.json', 'instance_bert_embeddings.json',
                             'object_tracks.json', 'floor_filter.json', 'object_association_zh.jsonl']:
                    if (source/name).is_file():
                        shutil.copyfile(source/name, context.scene/name)
            hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in destination.glob('*.json')
                      if not p.name.endswith(('_instance.json', '_updated_instance.json'))}
            (context.scene/'cache_reuse.json').write_text(json.dumps({'source': str(source),
                'cached_category_sha256': hashes, 'qwen_api_calls': 0}, indent=2)+'\n')
            context.event('复制并校验同场景观测缓存', source=str(source), cached_categories=len(hashes))
        stages = ['tags', 'segmentation', 'fusion', 'graph', 'parts']
        first = stages.index(args.start_stage)
        if first <= 1 and not context.manifest:
            raise ValueError('Legacy ScanNet frontend remains available through run_scannet_sg.sh; use fusion with its outputs')
        if first == 0:
            data, jobs, _, _, _ = load_capture(context.manifest)
            if context.scene.name != data['scene_id'] or context.scene.parent.name != data['dataset']:
                raise ValueError('Processed scene must have the existing output_root/dataset/scene_id layout')
            missing = [j['frame_id'] for j in jobs if not (context.scene/'refined_instance'/f"{j['frame_id']}.json").is_file()]
            if missing:
                vocabulary = json.loads((REPO/'scannet/script/ram/hypersim_indoor_57.json').read_text())
                objects = [{'name': name, 'description': descriptions[0]} for entry in vocabulary for name, descriptions in entry.items()]
                for fid in missing:
                    path = context.scene/'refined_instance'/f'{fid}.json'
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text(json.dumps({'objects': objects, 'source': 'local_indoor_vocabulary_no_qwen'}, indent=2)+'\n')
                context.event('缺少语义缓存的帧采用固定室内词表，不调用千问', frames=len(missing))
            context.event('复用已有类别与描述，禁止再次调用千问', frames=len(jobs))
        if first <= 1:
            _, jobs, _, _, _ = load_capture(context.manifest, context.image_dir, context.scene/'refined_instance')
            complete = all((context.scene/'refined_instance'/f"{j['frame_id']}_instance.json").exists()
                           and (context.scene/'refined_instance'/f"{j['frame_id']}.png").exists() for j in jobs)
            if not complete:
                context.execute([sys.executable, str(REPO / 'scannet/script/grounded_sam/scannet_process/get_seg_openset.py'),
                    '--manifest', str(context.manifest), '--json_folder', str(context.scene/'refined_instance'),
                    '--grounding_backend', 'florence', '--florence_model_dir',
                    str(Path.home()/'models/vision/Florence-2-large-ft'), '--visualize'], 'Florence 物体定位与 SAM 分割')
            frontend = getattr(components, 'FRONTEND', None)
            if frontend is not None:
                frontend.segment(context)
        if first <= 2:
            fusion = getattr(components, 'FUSION', None)
            if fusion is None:
                legacy_fusion(context)
            else:
                fusion.fuse(context, getattr(components, 'ASSOCIATION', None))
        if first <= 3:
            build_graph(context)
        if first <= 4:
            for component in getattr(components, 'GRAPH_COMPONENTS', []):
                component.construct(context)
        graph_path = context.scene / 'topology_map.json'
        graph = json.loads(graph_path.read_text())
        registry = REPO / 'scannet/script/pipeline_components/__init__.py'
        graph['pipeline_provenance'] = {'registry_sha256': hashlib.sha256(registry.read_bytes()).hexdigest(),
                                      'registry_modules': [x.__name__ for x in getattr(components, 'GRAPH_COMPONENTS', [])],
                                      'fusion_module': getattr(getattr(components, 'FUSION', None), '__name__', 'legacy_cpp'),
                                      'association_module': getattr(getattr(components, 'ASSOCIATION', None), '__name__', None),
                                      'geometry_modules': [x.__name__ for x in getattr(components, 'GEOMETRY_COMPONENTS', [])],
                                      'frontend_module': getattr(getattr(components, 'FRONTEND', None), '__name__', None),
                                      'instance_refinement': getattr(getattr(components, 'INSTANCE_REFINEMENT', None), '__name__', None),
                                      'input_manifest': str(context.manifest), 'start_stage': args.start_stage,
                                      'qwen_model': args.qwen_model, 'qwen_api_calls': 0, 'version': 'v5',
                                      'object_validation': getattr(getattr(components, 'OBJECT_VALIDATION', None), '__name__', None)}
        graph_path.write_text(json.dumps(graph, indent=2) + '\n')
        context.event('完整主流程完成', output=str(graph_path))
    except Exception as error:
        context.event('主流程失败', error=str(error))
        raise
    finally:
        context.log.close()


if __name__ == '__main__':
    main()

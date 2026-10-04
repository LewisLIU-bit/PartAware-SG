"""Validate canonical graph contracts and exercise both ScanNet fusion paths."""
import argparse
import json
from pathlib import Path
import shutil
import sys
import tempfile
from types import SimpleNamespace
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / 'script/include'))
from topology_map import TopologyMap


def validate_scene(scene):
    scene = Path(scene).expanduser().resolve()
    graph = json.loads((scene / 'topology_map.json').read_text())
    loader = TopologyMap()
    loader.read_from_json(json.dumps(graph))
    nodes = graph['object_nodes']['nodes'] or {}
    for key, node in nodes.items():
        if np.asarray(node['visual_embedding']).shape != (256,) or np.asarray(node['text_embedding']).shape != (384,):
            raise ValueError(f'Object feature interface changed: {key}')
    for key, part in (graph.get('part_nodes') or {}).items():
        if np.asarray(part['semantic_embedding']).shape != (1024,):
            raise ValueError(f'Part feature space changed: {key}')
        if part['parent_id'] is not None and str(part['parent_id']) not in nodes:
            raise ValueError(f'Part references a removed parent: {key}')
        if loader.get_entity(key) is None:
            raise ValueError('Unified entity query lost a part')
    for edge in graph.get('part_relations') or []:
        part = graph['part_nodes'][edge['source_id']]
        if part['status'] != 'confirmed' or part['parent_id'] != edge['target_id'] or part.get('parent_evidence_frames', 0) < 2:
            raise ValueError('Hierarchy edge lacks stable multi-view ownership')
        if edge not in graph['scene_graph']['edges']:
            raise ValueError('Hierarchy relation is only a visualization overlay')
    for fid in json.loads((scene / 'object_tracks.json').read_text()).values():
        frames = fid['observed_frames']
        if len(frames) != len(set(frames)) or len(frames) < 2:
            raise ValueError('Object confirmation does not use distinct frames')
    return {'message': '正式图接口验证通过', 'scene': str(scene), 'objects': len(nodes),
            'confirmed_parts': sum(p['status'] == 'confirmed' for p in loader.part_nodes.values()),
            'part_of_edges': len(loader.part_relations)}


def scannet_smoke():
    from run_pipeline import Context, legacy_fusion, build_graph
    from pipeline_components import FUSION, ASSOCIATION
    source = Path.home() / 'datasets/scannet/processed/baseline30/openset_scans/scene0000_00/refined_instance'
    images = Path.home() / 'datasets/scannet/images/scans/scene0000_00'
    folder = Path.home() / 'datasets/scannet-sg-processed'
    if not source.is_dir() or not images.is_dir():
        raise FileNotFoundError('The existing ScanNet validation input is unavailable')
    records = []
    # The temporary outputs are deleted by the context manager; source data is read-only.
    with tempfile.TemporaryDirectory(prefix='partaware-contract-check-', dir=folder) as temporary:
        for title in ['legacy_cpp', 'default_components']:
            scene = Path(temporary) / title / 'scene0000_00'
            refined = scene / 'refined_instance'
            refined.mkdir(parents=True)
            for fid in ['0', '3', '6']:
                for suffix in ['.png', '_instance.json']:
                    shutil.copyfile(source / (fid+suffix), refined / (fid+suffix))
            args = SimpleNamespace(manifest=None, image_dir=str(images), processed_scene=str(scene),
                                   max_depth=0, stride=2, edge_threshold=2)
            context = Context(args)
            try:
                if title == 'legacy_cpp':
                    legacy_fusion(context)
                else:
                    FUSION.fuse(context, ASSOCIATION)
                build_graph(context)
                graph = json.loads((scene / 'topology_map.json').read_text())
                loader = TopologyMap()
                loader.read_from_json(json.dumps(graph))
                if not loader.object_nodes.nodes or not (scene / 'instance_cloud_cleaned.ply').is_file():
                    raise RuntimeError('ScanNet base construction failed')
                records.append({'message': 'ScanNet 基础输入建图验证通过', 'path': title,
                                'frames': 3, 'objects': len(loader.object_nodes.nodes), 'temporary_outputs_removed': True})
            finally:
                context.log.close()
    return records


def build_legacy_control(source_scene, manifest, control_scene):
    """Hold the exact frontend masks and features fixed for fusion comparison."""
    import hashlib
    from run_pipeline import Context, legacy_fusion, build_graph
    source, manifest, target = Path(source_scene).resolve(), Path(manifest).resolve(), Path(control_scene).resolve()
    if target.exists() and any(target.iterdir()):
        raise FileExistsError('The control output must be fresh')
    refined = target / 'refined_instance'
    refined.mkdir(parents=True)
    inputs = json.loads(manifest.read_text())
    checksums = {}
    for frame in inputs['frames']:
        for suffix in ['.png', '_instance.json']:
            name = frame['frame_id'] + suffix
            path = source / 'refined_instance' / name
            shutil.copyfile(path, refined / name)
            checksums[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    context = Context(SimpleNamespace(manifest=str(manifest), image_dir=None, processed_scene=str(target),
                                      max_depth=0, stride=2, edge_threshold=2))
    try:
        legacy_fusion(context)
        build_graph(context)
    finally:
        context.log.close()
    record = {'message': '同前端原始融合对照完成', 'frontend_source': str(source), 'control_output': str(target),
              'frames': len(inputs['frames']), 'frontend_sha256': checksums, 'fusion': 'original_cpp_legacy'}
    (target / 'control_record.json').write_text(json.dumps(record, indent=2, ensure_ascii=False)+'\n')
    return {key: value for key, value in record.items() if key != 'frontend_sha256'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--processed-scene', action='append', default=[])
    parser.add_argument('--scannet-smoke', action='store_true')
    parser.add_argument('--output', required=True)
    parser.add_argument('--legacy-control-from')
    parser.add_argument('--manifest')
    parser.add_argument('--control-scene')
    args = parser.parse_args()
    records = [validate_scene(scene) for scene in args.processed_scene]
    if args.scannet_smoke:
        records.extend(scannet_smoke())
    if args.legacy_control_from:
        if not args.manifest or not args.control_scene:
            parser.error('A legacy control requires manifest and control-scene')
        records.append(build_legacy_control(args.legacy_control_from, args.manifest, args.control_scene))
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({'records': records}, indent=2, ensure_ascii=False)+'\n')
    print(json.dumps(records, ensure_ascii=False), flush=True)

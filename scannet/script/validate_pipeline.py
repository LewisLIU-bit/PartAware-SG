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
        geometry_only = part.get('semantic_feature_space') == 'geometry_only_no_visual_embedding'
        if geometry_only and (part['semantic_embedding'] is not None
                or len(set(part.get('observed_frames', []))) < 2):
            raise ValueError(f'Geometry-only part has invalid feature or frame evidence: {key}')
        if not geometry_only and np.asarray(part['semantic_embedding']).shape != (1024,):
            raise ValueError(f'Part feature space changed: {key}')
        if part['parent_id'] is not None and str(part['parent_id']) not in nodes:
            raise ValueError(f'Part references a removed parent: {key}')
        if loader.get_entity(key) is None:
            raise ValueError('Unified entity query lost a part')
    for edge in graph.get('part_relations') or []:
        part = graph['part_nodes'][edge['source_id']]
        geometry_only = part.get('semantic_feature_space') == 'geometry_only_no_visual_embedding'
        evidence = (min(len(set(part.get('observed_frames', []))), edge.get('evidence_frames', 0))
                    if geometry_only else part.get('parent_evidence_frames', 0))
        if part['status'] != 'confirmed' or part['parent_id'] != edge['target_id'] or evidence < 2:
            raise ValueError('Hierarchy edge lacks stable multi-view ownership')
        if edge not in graph['scene_graph']['edges']:
            raise ValueError('Hierarchy relation is only a visualization overlay')
    provenance = graph.get('geometry_provenance')
    if provenance:
        import open3d as o3d
        from scipy.spatial.transform import Rotation
        geometry = o3d.io.read_point_cloud(str(scene/provenance['source']))
        points = np.asarray(geometry.points)
        colors = np.rint(np.asarray(geometry.colors)*255).astype(int)
        encoded = colors[:, 0]+255*colors[:, 1]+255**2*colors[:, 2]
        if set(str(x) for x in np.unique(encoded) if x > 0) != set(nodes):
            raise ValueError('Published geometry and canonical object IDs disagree')
        for gid, node in nodes.items():
            shape = node['shape']
            rotation = Rotation.from_quat([shape['orientation'][k] for k in ['x', 'y', 'z', 'w']]).as_matrix()
            local = (points[encoded == int(gid)]-np.asarray(node['position'])) @ rotation
            extent = np.array([shape[k] for k in ['length','width','height']])
            if not np.all(np.abs(local) <= extent/2+1e-5):
                raise ValueError(f'Published box does not contain its geometry: {gid}')
        for hypothesis in (graph.get('edge_hypotheses') or {}).values():
            for edge in (hypothesis.get('edges') or {}).values():
                a, b = edge['source_id'], edge['target_id']
                delta = np.asarray(nodes[b]['position'])-nodes[a]['position']
                distance = np.linalg.norm(delta)
                if (not np.isclose(edge['distance'], distance, atol=1e-5)
                        or not np.allclose(edge['direction'], delta/distance, atol=1e-5)):
                    raise ValueError('Spatial edges do not describe current box centers')
    tracks_path = scene/'validated_object_tracks.json'
    if not tracks_path.exists():
        tracks_path = scene/'object_tracks.json'
    fusion_module = graph.get('pipeline_provenance', {}).get('fusion_module')
    if fusion_module and fusion_module != 'legacy_cpp' and not tracks_path.exists():
        raise FileNotFoundError('Default component graph is missing object track provenance')
    tracks = json.loads(tracks_path.read_text()) if tracks_path.exists() else {}
    for fid in tracks.values():
        frames = fid['observed_frames']
        if len(frames) != len(set(frames)) or len(frames) < 2:
            raise ValueError('Object confirmation does not use distinct frames')
    return {'message': '正式图接口验证通过', 'scene': str(scene), 'objects': len(nodes),
            'confirmed_parts': sum(p['status'] == 'confirmed' for p in loader.part_nodes.values()),
            'part_of_edges': len(loader.part_relations)}


def scannet_smoke():
    from run_pipeline import Context, legacy_fusion, build_graph, build_legacy_graph
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
                if title == 'legacy_cpp':
                    build_legacy_graph(context)
                else:
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
    from run_pipeline import Context, legacy_fusion, build_graph, build_legacy_graph
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
        build_legacy_graph(context)
    finally:
        context.log.close()
    record = {'message': '同前端原始融合对照完成', 'frontend_source': str(source), 'control_output': str(target),
              'frames': len(inputs['frames']), 'frontend_sha256': checksums, 'fusion': 'original_cpp_legacy'}
    (target / 'control_record.json').write_text(json.dumps(record, indent=2, ensure_ascii=False)+'\n')
    return {key: value for key, value in record.items() if key != 'frontend_sha256'}


def build_frontend_control(manifest, scene_path, tag_source, reuse_from=None, start_stage='tags'):
    """Compare tag sources with fresh DINO/SAM and the same original C++ graph path."""
    import hashlib
    from partaware.geometry import load_capture
    from run_pipeline import Context, legacy_fusion, build_graph, build_legacy_graph
    manifest = Path(manifest).expanduser().resolve()
    scene = Path(scene_path).expanduser().resolve()
    data, jobs, _, _, _ = load_capture(manifest)
    if scene.name != data['scene_id'] or scene.parent.name != data['dataset']:
        raise ValueError('Control scene must preserve output_root/dataset/scene_id')
    if start_stage == 'tags' and scene.exists() and any(scene.iterdir()):
        raise FileExistsError('Use a fresh control scene or resume a later stage')
    args = SimpleNamespace(manifest=str(manifest), image_dir=None, processed_scene=str(scene),
                           max_depth=0, stride=2, edge_threshold=2)
    context = Context(args)
    phases = ['tags', 'segmentation', 'fusion', 'graph']
    first = phases.index(start_stage)
    record_path = scene/'control_record.json'
    record = json.loads(record_path.read_text()) if record_path.exists() else {
        'message': '原始流程前端对照', 'tag_source': tag_source, 'frames': len(jobs),
        'manifest_sha256': hashlib.sha256(manifest.read_bytes()).hexdigest(),
        'grounding_backend': 'dino', 'fusion_backend': 'legacy_cpp',
        'structural_background_policy': 'same segmentation exclusions and Hypersim floor guards',
        'reuse_from': str(reuse_from) if reuse_from else None, 'tag_sha256': {}}
    if record['tag_source'] != tag_source or record['manifest_sha256'] != hashlib.sha256(manifest.read_bytes()).hexdigest():
        context.log.close()
        raise ValueError('Resume inputs do not match the saved control record')
    try:
        context.event('开始原始流程前端对照', source=tag_source, frames=len(jobs), start_stage=start_stage)
        refined = scene/'refined_instance'
        refined.mkdir(exist_ok=True)
        if first == 0:
            if tag_source == 'ram_native':
                context.execute([sys.executable, str(context.repo/'scannet/script/ram/inference_ram_given_folders.py'),
                                 '--manifest', str(manifest), '--output_root', str(scene.parent.parent),
                                 '--pretrained', str(Path.home()/'models/ram/ram_plus_swin_large_14m.pth'),
                                 '--native-vocabulary'], 'RAM++ 原生词表逐帧类别')
                record['vocabulary'] = 'RAM++ native 4585 classes and checkpoint thresholds'
            elif tag_source == 'qwen_reuse':
                if reuse_from is None:
                    raise ValueError('Qwen reuse requires the completed source scene')
                original_scene = Path(reuse_from).expanduser().resolve()
                original_graph = json.loads((original_scene/'topology_map.json').read_text())
                source_manifest = original_graph.get('pipeline_provenance', {}).get('input_manifest')
                if not source_manifest or hashlib.sha256(Path(source_manifest).read_bytes()).hexdigest() != record['manifest_sha256']:
                    raise ValueError('Reused Qwen tags do not belong to the same input manifest')
                original = original_scene/'refined_instance'
                for job in jobs:
                    source = original/f"{job['frame_id']}.json"
                    destination = refined/source.name
                    payload = source.read_bytes()
                    if not isinstance(json.loads(payload).get('objects'), list):
                        raise ValueError(f'Invalid reused tag JSON: {source}')
                    destination.write_bytes(payload)
                    record['tag_sha256'][job['frame_id']] = hashlib.sha256(payload).hexdigest()
                context.event('逐文件复用千问类别，未调用 API', files=len(record['tag_sha256']))
            else:
                raise ValueError('Unknown control tag source')
            record_path.write_text(json.dumps(record, ensure_ascii=False, indent=2)+'\n')
        if first <= 1:
            context.execute([sys.executable, str(context.repo/'scannet/script/grounded_sam/scannet_process/get_seg_openset.py'),
                             '--manifest', str(manifest), '--json_folder', str(refined),
                             '--grounding_backend', 'dino', '--confidence_threshold', '.4', '--visualize'],
                            '原始 DINO 候选与 SAM 实例分割')
        if first <= 2:
            legacy_fusion(context)
        if first <= 3:
            build_legacy_graph(context)
        graph_path = scene/'topology_map.json'
        graph = json.loads(graph_path.read_text())
        graph['benchmark_provenance'] = {key:value for key,value in record.items() if key != 'tag_sha256'}
        graph_path.write_text(json.dumps(graph, indent=2)+'\n')
        record['objects'] = len(graph['object_nodes']['nodes'] or {})
        record['status'] = 'complete'
        record_path.write_text(json.dumps(record, ensure_ascii=False, indent=2)+'\n')
        context.event('原始流程前端对照完成', objects=record['objects'])
        return {key:value for key,value in record.items() if key != 'tag_sha256'}
    except Exception as error:
        context.event('原始流程前端对照失败', error=str(error))
        raise
    finally:
        context.log.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--processed-scene', action='append', default=[])
    parser.add_argument('--scannet-smoke', action='store_true')
    parser.add_argument('--output', required=True)
    parser.add_argument('--legacy-control-from')
    parser.add_argument('--manifest')
    parser.add_argument('--control-scene')
    parser.add_argument('--frontend-control', choices=['ram_native', 'qwen_reuse'])
    parser.add_argument('--reuse-tags-from')
    parser.add_argument('--control-start-stage', choices=['tags', 'segmentation', 'fusion', 'graph'], default='tags')
    args = parser.parse_args()
    if args.frontend_control and (not args.manifest or not args.control_scene or args.legacy_control_from):
        parser.error('Frontend control requires manifest/control-scene and cannot be a reused-mask control')
    records = [validate_scene(scene) for scene in args.processed_scene]
    if args.scannet_smoke:
        records.extend(scannet_smoke())
    if args.legacy_control_from:
        if not args.manifest or not args.control_scene:
            parser.error('A legacy control requires manifest and control-scene')
        records.append(build_legacy_control(args.legacy_control_from, args.manifest, args.control_scene))
    if args.frontend_control:
        records.append(build_frontend_control(args.manifest, args.control_scene, args.frontend_control,
                                             args.reuse_tags_from, args.control_start_stage))
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({'records': records}, indent=2, ensure_ascii=False)+'\n')
    print(json.dumps(records, ensure_ascii=False), flush=True)

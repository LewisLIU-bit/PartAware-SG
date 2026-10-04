"""Construct OP3DSG/VLPart parts using the existing RGB-D and object interfaces."""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
import cv2
import numpy as np
from partaware.geometry import load_capture, project_mask, denoise_largest_cluster, subtract_contained_masks
from partaware.fusion import PartFusion, color_histogram

ROOT = Path(__file__).resolve().parents[2]


def canonical(name):
    return ' '.join(name.casefold().replace('_', ' ').strip().split())


def knowledge_map(path, overrides=None):
    data = json.loads(Path(path).read_text())['ram_add_part']
    mapping = {canonical(k): sorted(set(p.strip().strip(',') for p in v if p.strip())) for k, v in data.items()}
    if overrides:
        for name, parts in json.loads(Path(overrides).read_text()).items():
            mapping[canonical(name)] = sorted(set(parts))
    return mapping


def select_parent(part_mask, label, parent_mask, parents, graph_ids, min_coverage=0.2, margin=0.1):
    parent_name = label.split(':', 1)[0]
    choices = []
    area = int(part_mask.sum())
    if not area:
        return None
    for parent in parents:
        gid = str(parent.get('instance_id', -1))
        if canonical(parent['object_name']) != parent_name or gid not in graph_ids:
            continue
        region = parent_mask == int(parent['frame_instance_id'])
        coverage = float(np.count_nonzero(part_mask & region) / area)
        choices.append((coverage, gid))
    # Ambiguous geometry leaves the part unattached rather than inventing ownership.
    scores = {}
    for coverage, gid in choices:
        scores[gid] = max(coverage, scores.get(gid, 0))
    choices = sorted(((v, k) for k, v in scores.items()), reverse=True)
    if not choices or choices[0][0] < min_coverage:
        return None
    if len(choices) > 1 and choices[0][0] - choices[1][0] < margin:
        return None
    return choices[0][1]


def run(args):
    scene = Path(args.processed_scene).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    if output == scene or scene.is_relative_to(output):
        raise ValueError('Part output must not be the baseline output directory or its ancestor')
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f'Use a fresh output directory: {output}')
    graph_path = scene / 'topology_map.json'
    graph = json.loads(graph_path.read_text())
    node_group = graph.get('object_nodes', {})
    graph_ids = set(map(str, node_group.get('nodes', node_group)))
    data, jobs, kd, kc, scale = load_capture(args.manifest, args.image_dir, scene / 'refined_instance')
    if args.limit:
        jobs = jobs[:args.limit]
    for job in jobs:
        fid = job['frame_id']
        for name in (f'{fid}.png', f'{fid}_updated_instance.json'):
            if not (scene / 'refined_instance' / name).is_file():
                raise FileNotFoundError(scene / 'refined_instance' / name)
    mapping = knowledge_map(args.knowledge, args.part_overrides)
    output.mkdir(parents=True, exist_ok=True)
    masks_dir = output / 'frame_parts'
    masks_dir.mkdir()
    log = (output / 'run_zh.jsonl').open('w', encoding='utf-8')
    def event(message, **fields):
        log.write(json.dumps({'message': message, **fields}, ensure_ascii=False) + '\n')
        log.flush()
        print(message, fields, flush=True)
    # Import large models only after CLI and sensor validation.
    backend = None
    import pipeline_components
    association = getattr(pipeline_components, 'ASSOCIATION', None)
    fusion_type = getattr(association, 'PartFusion', PartFusion)
    fusion = fusion_type(voxel=args.voxel, radius=args.radius, min_frames=args.min_frames,
                        color_weight=args.color_weight, threshold=args.fusion_threshold)
    provenance = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    provenance.update(dataset=data['dataset'], scene_id=data['scene_id'],
                      base_graph_sha256=hashlib.sha256(graph_path.read_bytes()).hexdigest())
    (output / 'run_config.json').write_text(json.dumps(provenance, indent=2), encoding='utf-8')
    sam = None
    try:
        if args.sam_checkpoint:
            sam_root = ROOT / 'scannet/script/thirdparty/Grounded-Segment-Anything/segment_anything'
            sys.path.insert(0, str(sam_root))
            from segment_anything import SamPredictor, sam_model_registry
            sam = SamPredictor(sam_model_registry[args.sam_model](checkpoint=args.sam_checkpoint).to(args.device))
        for job in jobs:
            fid = job['frame_id']
            parents = json.loads((scene / 'refined_instance' / f'{fid}_updated_instance.json').read_text())
            labels = sorted({f'{canonical(p["object_name"])}: {part}' for p in parents
                             if str(p.get('instance_id', -1)) in graph_ids
                             for part in mapping.get(canonical(p['object_name']), [])})
            rgb = cv2.imread(str(job['rgb']))
            depth = cv2.imread(str(job['depth']), cv2.IMREAD_UNCHANGED)
            parent_mask = cv2.imread(str(scene / 'refined_instance' / f'{fid}.png'), cv2.IMREAD_UNCHANGED)
            pose = np.loadtxt(job['pose'])
            if rgb is None or depth is None or parent_mask is None:
                raise ValueError(f'Unreadable input image: {fid}')
            if parent_mask.shape != rgb.shape[:2]:
                raise ValueError('Parent mask resolution does not match RGB')
            if labels and backend is None:
                from partaware.vlpart_backend import VLPartBackend
                backend = VLPartBackend(args.vlpart_root, args.config, args.checkpoint,
                                        args.device, args.score_threshold, args.image_size)
            detections = backend.predict(rgb, labels) if labels else []
            if sam:
                sam.set_image(cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB))
                for detection in detections:
                    candidates, scores, _ = sam.predict(box=detection['box'], multimask_output=True)
                    detection['mask'] = candidates[int(np.argmax(scores))]
            if args.subtract_contained:
                parent_ids = [select_parent(d['mask'], d['label'], parent_mask, parents, graph_ids)
                              for d in detections]
                refined = subtract_contained_masks([d['mask'] for d in detections],
                                                   [d['label'] for d in detections], parent_ids)
                for detection, mask in zip(detections, refined):
                    detection['mask'] = mask
            records, masks = [], []
            canvas = rgb.copy()
            for index, detection in enumerate(detections):
                mask = detection['mask']
                if args.erode_pixels:
                    size = 2 * args.erode_pixels + 1
                    mask = cv2.erode(mask.astype(np.uint8), np.ones((size, size), np.uint8)).astype(bool)
                if mask.sum() < args.min_pixels:
                    continue
                parent_id = select_parent(mask, detection['label'], parent_mask, parents, graph_ids)
                points, colors = project_mask(mask, rgb, depth, pose, kd, kc, scale,
                                               args.stride, args.max_depth)
                if len(points) < args.min_points:
                    continue
                if args.dbscan_eps:
                    before = len(points)
                    points, colors = denoise_largest_cluster(points, colors, args.dbscan_eps)
                    event('部件点云聚类去噪完成', frame_id=fid, detection_id=index,
                          before_points=before, after_points=len(points))
                    if len(points) < args.min_points:
                        continue
                tid, action = fusion.add(fid, index, detection['label'], parent_id, points,
                                         detection['feature'], color_histogram(colors), detection['score'])
                records.append({'detection_id': index, 'track_id': tid, 'label': detection['label'],
                                'parent_id': parent_id, 'score': detection['score'],
                                'mask_index': len(masks), 'points': len(points)})
                masks.append(mask)
                color = np.array([80, 200, 255], dtype=np.uint8)
                canvas[mask] = (0.6 * canvas[mask] + 0.4 * color).astype(np.uint8)
                x1, y1, x2, y2 = detection['box'].astype(int)
                cv2.rectangle(canvas, (x1, y1), (x2, y2), (80, 200, 255), 1)
                cv2.putText(canvas, detection['label'], (max(x1, 0), max(y1, 15)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (80, 200, 255), 1)
                event('部件观测已处理', frame_id=fid, track_id=tid, action=action,
                      parent_id=parent_id, label=detection['label'], point_count=len(points))
            np.savez_compressed(masks_dir / f'{fid}.npz',
                                masks=np.asarray(masks, dtype=bool).reshape((-1, *rgb.shape[:2])))
            (masks_dir / f'{fid}.json').write_text(json.dumps(records, indent=2), encoding='utf-8')
            if args.visualize and not cv2.imwrite(str(masks_dir / f'{fid}.jpg'), canvas):
                raise RuntimeError('Visualization write failed')
            event('帧处理完成', frame_id=fid, candidate_count=len(labels), retained_parts=len(records))
        nodes, edges = fusion.export()
        result = deepcopy(graph)
        result.update(partaware_schema_version=1, part_nodes=nodes, part_relations=edges,
                      partaware_provenance=provenance)
        for track in fusion.tracks:
            np.save(output / f'{track.id}.points.npy', track.points)
        (output / 'partaware_graph.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
        event('部件图生成完成', tracks=len(nodes), confirmed=sum(n['status'] == 'confirmed' for n in nodes.values()),
              part_of_edges=len(edges), baseline_graph_unchanged=True)
    except Exception as error:
        event('运行失败', error_type=type(error).__name__, error=str(error))
        raise
    finally:
        log.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument('--manifest', help='Existing ScanNet-SG RGB-D manifest, including Hypersim')
    inputs.add_argument('--image-dir', help='Legacy ScanNet image folder containing _info.txt')
    parser.add_argument('--processed-scene', required=True, help='Existing scene with refined_instance and topology_map.json')
    parser.add_argument('--output', required=True, help='Fresh part output directory')
    parser.add_argument('--vlpart-root', default=str(ROOT / 'scannet/script/thirdparty/VLPart'))
    parser.add_argument('--config', default=str(ROOT / 'scannet/script/thirdparty/VLPart/configs/joint/swinbase_cascade_lvis_paco.yaml'))
    parser.add_argument('--checkpoint', default=str(ROOT / 'checkpoints/vlpart/swinbase_cascade_lvis_paco.pth'))
    parser.add_argument('--knowledge', default=str(ROOT / 'scannet/script/partaware/object_part_knowledge.json'))
    parser.add_argument('--part-overrides', help='Optional JSON object-to-parts overrides')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--score-threshold', type=float, default=0.2)
    parser.add_argument('--image-size', type=int, default=640)
    parser.add_argument('--min-pixels', type=int, default=25)
    parser.add_argument('--min-points', type=int, default=16)
    parser.add_argument('--stride', type=int, default=2)
    parser.add_argument('--max-depth', type=float, default=0)
    parser.add_argument('--voxel', type=float, default=0.01)
    parser.add_argument('--radius', type=float, default=0.03)
    parser.add_argument('--min-frames', type=int, default=2)
    parser.add_argument('--fusion-threshold', type=float, default=1.5)
    parser.add_argument('--color-weight', type=float, default=0.6)
    parser.add_argument('--erode-pixels', type=int, default=0)
    parser.add_argument('--dbscan-eps', type=float, default=0, help='Optional largest-cluster denoising radius in meters; 0 disables it')
    parser.add_argument('--subtract-contained', action='store_true', help='Optional containment-based sibling mask cleanup')
    parser.add_argument('--sam-checkpoint', help='Optional SAM refinement checkpoint')
    parser.add_argument('--sam-model', default='vit_h')
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--visualize', action='store_true')
    args = parser.parse_args()
    if args.limit < 0 or args.erode_pixels < 0 or args.dbscan_eps < 0 or args.min_pixels < 1 or args.min_points < 1:
        parser.error('Invalid sampling or filtering parameters')
    if not np.isfinite([args.score_threshold, args.dbscan_eps, args.max_depth]).all():
        parser.error('Parameters must be finite')
    if not 0 <= args.score_threshold <= 1 or args.image_size < 32 or args.stride < 1 or args.max_depth < 0:
        parser.error('Invalid detector or projection parameters')
    import fcntl
    gpu_lock_dir = Path.home() / '.cache' / 'partaware-sg'
    gpu_lock_dir.mkdir(parents=True, exist_ok=True)
    gpu_lock = (gpu_lock_dir / 'gpu.lock').open('a')
    fcntl.flock(gpu_lock.fileno(), fcntl.LOCK_EX)
    run(args)


if __name__ == '__main__':
    main()

"""Removable SAM3 observation backend preserving the ScanNet-SG public schema.

Noun queries come from existing GPT JSON and the existing indoor vocabulary.
This adapter acquires no VLM response and never opens ground-truth annotations.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys

import cv2
import numpy as np

OWNS_COARSE_SEGMENTATION = True
SOURCE_REVISION = '0570b3a5be9c4e694f23d85232fb55f4a6f1f7fc'


def segment(context):
    command = [sys.executable, str(Path(__file__)), '--processed-scene', str(context.scene)]
    command += ['--manifest', str(context.manifest)] if context.manifest else ['--image-dir', str(context.image_dir)]
    context.execute(command+['--prepare'], 'SAM3：读取共享 GPT 缓存并准备通用物体查询')
    python = Path.home()/'miniconda3/envs/sg-sam3/bin/python'
    context.execute([str(python), str(Path(__file__).with_name('sam3_worker.py')),
        '--jobs', str(context.scene/'sam3_jobs.json'), '--cache-dir', str(context.scene/'sam3_cache')],
        'SAM3：本地概念实例分割与局部放大，不调用 GPT 或千问')
    context.execute(command+['--attach'], 'SAM3：实测尺度核验及原 DINO/SBERT 特征接口')


def names_for_frame(tags, vocabulary):
    names = {key.strip().lower() for entry in vocabulary for key in entry}
    names.update(obj['name'].strip().lower() for obj in tags['objects']
                 if isinstance(obj.get('name'), str))
    return sorted(name for name in names if name and name not in {'wall', 'floor', 'ceiling'})


def zoom_regions(records, shape, queries, maximum=4):
    """Cover small observed regions without inventing a scene-specific prompt."""
    height, width = shape
    regions = []
    boxes = []
    for record in records:
        box = record.get('segmentation_box')
        if box is None or len(box) != 4 or not np.isfinite(box).all():
            continue
        x0, y0, x1, y1 = box
        area = max(0, x1-x0)*max(0, y1-y0)
        if area < 32 or area > .12*height*width:
            continue
        side = max(256, min(512, 2*max(x1-x0, y1-y0)))
        xa = max(0, min(width-int(side), int((x0+x1-side)/2)))
        ya = max(0, min(height-int(side), int((y0+y1-side)/2)))
        boxes.append((area, [xa, ya, min(width, xa+int(side)), min(height, ya+int(side))]))
    for _, box in sorted(boxes, key=lambda value: value[0]):
        if any(box_iou(box, prior['bounds']) >= .5 for prior in regions):
            continue
        regions.append({'bounds': box, 'queries': queries})
        if len(regions) == maximum:
            break
    return regions


def box_iou(left, right):
    intersection = max(0, min(left[2], right[2])-max(left[0], right[0]))*max(0, min(left[3], right[3])-max(left[1], right[1]))
    a = max(0, left[2]-left[0])*max(0, left[3]-left[1])
    b = max(0, right[2]-right[0])*max(0, right[3]-right[1])
    return intersection/max(a+b-intersection, 1)


def deduplicate(proposals):
    """Remove same-surface aliases while retaining spatially disjoint instances."""
    selected = []
    for record, mask in sorted(proposals, key=lambda value: -value[0]['confidence']):
        area = int(mask.sum())
        if area < 32:
            continue
        duplicate = False
        for prior, other in selected:
            intersection = int(np.count_nonzero(mask & other))
            iou = intersection/max(area+int(other.sum())-intersection, 1)
            if iou >= .65:
                prior.setdefault('query_aliases', []).append(record['object_name'])
                duplicate = True
                break
        if not duplicate:
            selected.append((copy.deepcopy(record), mask))
    return selected


def partition(proposals, shape):
    """Keep the uint8 local-label interface; small instances own nested pixels."""
    if len(proposals) > 255:
        raise ValueError('SAM3 observations exceed the retained uint8 instance interface')
    labels = np.zeros(shape, np.uint8)
    for index in sorted(range(len(proposals)), key=lambda i: (-int(proposals[i][1].sum()), i)):
        labels[proposals[index][1]] = index+1
    result = []
    for index, (raw, _) in enumerate(proposals):
        ys, xs = np.where(labels == index+1)
        if len(xs) < 32:
            labels[labels == index+1] = 0
            continue
        record = copy.deepcopy(raw)
        record.update(instance_id=-1, frame_instance_id=index+1, saved_pixels=len(xs),
            segmentation_box=[int(xs.min()), int(ys.min()), int(xs.max()+1), int(ys.max()+1)])
        result.append(record)
    return result, labels


def prepare(repo, scene, jobs):
    provenance = json.loads((scene/'recognition_provenance.json').read_text(encoding='utf-8-sig'))
    if provenance.get('provider') != 'relay_gpt' or not provenance.get('complete'):
        raise ValueError('This v12 SAM3 experiment requires a complete existing GPT cache')
    folder = scene/'refined_instance'
    vocabulary = json.loads((repo/'scannet/script/ram/hypersim_indoor_57.json').read_text())
    prepared = []
    for job in jobs:
        fid = job['frame_id']
        tags = json.loads((folder/f'{fid}.json').read_text())
        queries = names_for_frame(tags, vocabulary)
        image = cv2.imread(str(job['rgb']))
        if image is None:
            raise IOError(job['rgb'])
        source_path = folder/f'{fid}_instance.json'
        records = json.loads(source_path.read_text()) if source_path.is_file() else []
        # A preserved source snapshot makes an interrupted run deterministic.
        snapshot = scene/'sam3_source_evidence'/f'{fid}.json'
        snapshot.parent.mkdir(exist_ok=True)
        if snapshot.is_file():
            records = json.loads(snapshot.read_text())
        else:
            snapshot.write_text(json.dumps(records, indent=2)+'\n')
        prepared.append({'frame_id': fid, 'rgb': str(job['rgb']), 'queries': queries,
                         'regions': zoom_regions(records, image.shape[:2], queries)})
    (scene/'sam3_jobs.json').write_text(json.dumps({'source_revision': SOURCE_REVISION,
        'qwen_api_calls': 0, 'gpt_api_calls': 0, 'jobs': prepared}, indent=2)+'\n')


def attach(repo, scene, jobs, kd, kc, scale):
    from partaware.geometry import project_mask
    from pipeline_components.fovea import independent_proposals
    from pipeline_components.yoloe_frontend import roi_features
    from sentence_transformers import SentenceTransformer
    from groundingdino.util.inference import load_model
    import fcntl
    import torch
    audit = json.loads((scene/'sam3_inference_audit.json').read_text())
    expected = {record['frame_id']: record for record in audit['frames']}
    names = sorted({name for job in json.loads((scene/'sam3_jobs.json').read_text())['jobs'] for name in job['queries']})
    bert = SentenceTransformer(str(Path.home()/'.cache/torch/sentence_transformers/sentence-transformers_all-MiniLM-L6-v2'), device='cpu')
    semantic = dict(zip(names, bert.encode(names)))
    root = repo/'scannet/script/thirdparty/Grounded-Segment-Anything'
    lock_path = Path.home()/'.cache/partaware-sg/gpu.lock'
    frame_audit = []
    with lock_path.open('a') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        dino = load_model(str(root/'GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py'),
            str(root/'groundingdino_swint_ogc.pth'), device='cpu').to('cuda').eval()
        for job in jobs:
            fid = job['frame_id']; cp = scene/'sam3_cache'/f'{fid}.npz'
            with np.load(cp, allow_pickle=False) as saved:
                if str(saved['signature']) != expected[fid]['signature']:
                    raise ValueError(f'SAM3 inference receipt disagrees with cache: {fid}')
                shape = saved['shape'].tolist()
                masks = np.unpackbits(saved['packed_masks'], axis=-1, count=shape[-1]).astype(bool)
                records = json.loads(str(saved['records']))
            image = cv2.imread(str(job['rgb']))
            if hashlib.sha256(Path(job['rgb']).read_bytes()).hexdigest() != expected[fid]['rgb_sha256']:
                raise ValueError(f'RGB changed after SAM3 inference: {fid}')
            full = deduplicate([(r, m) for r, m in zip(records, masks) if not r['is_zoom_proposal']])
            coarse_records, coarse_labels = partition(full, image.shape[:2])
            depth, pose = cv2.imread(str(job['depth']), -1), np.loadtxt(job['pose'])
            fine = []
            for record, mask in deduplicate([(r, m) for r, m in zip(records, masks) if r['is_zoom_proposal']]):
                if any(np.count_nonzero(mask & m)/max(np.count_nonzero(mask | m), 1) >= .55 for _, m in full):
                    continue
                points, _ = project_mask(mask, image, depth, pose, kd, kc, scale, stride=2, max_depth=0)
                if len(points) < 32 or np.diff(np.quantile(points, [.02, .98], axis=0), axis=0).max() > .35:
                    continue
                fine.append((record, mask))
            retained = independent_proposals(fine, coarse_labels)
            combined = full+retained
            final, labels = partition(combined, image.shape[:2])
            for record in final:
                record['bert_embedding'] = semantic[record['object_name']].tolist()
                record['object_description'] = f'Local SAM3 instance of {record["object_name"]}.'
            roi_features(dino, image, final)
            folder = scene/'refined_instance'
            (folder/f'{fid}_instance.json').write_text(json.dumps(final, indent=2)+'\n')
            if not cv2.imwrite(str(folder/f'{fid}.png'), labels):
                raise IOError(fid)
            item = {'frame_id': fid, 'raw': len(records), 'full_instances': len(full),
                    'scale_confirmed_zoom': len(fine), 'independent_zoom': len(retained), 'final_instances': len(final)}
            frame_audit.append(item)
            print('SAM3 观测写入原始接口', item, flush=True)
        del dino
        torch.cuda.empty_cache()
    (scene/'frontend_provenance.json').write_text(json.dumps({
        'algorithm': 'SAM3_FOVEA_v12', 'backend_module': __name__,
        'checkpoint_sha256': audit['checkpoint_sha256'], 'source_revision': SOURCE_REVISION,
        'object_visual_dimensions': 256, 'object_semantic_dimensions': 384,
        'gpt_api_calls': 0, 'qwen_api_calls': 0, 'ground_truth_used': False,
        'uses_cached_gpt_nouns': True, 'fine_instances_have_metric_scale_and_sibling_checks': True,
        'creates_unobserved_3d_points': False, 'frames': frame_audit}, indent=2)+'\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--processed-scene', required=True)
    parser.add_argument('--manifest'); parser.add_argument('--image-dir')
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument('--prepare', action='store_true'); modes.add_argument('--attach', action='store_true')
    parser.add_argument('--limit', type=int, help='Leading-frame smoke test only; never used by the main pipeline')
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(repo/'scannet/script'))
    from partaware.geometry import load_capture
    scene = Path(args.processed_scene).resolve()
    _, jobs, kd, kc, scale = load_capture(args.manifest, args.image_dir, scene/'refined_instance')
    if args.limit is not None:
        if args.limit <= 0: raise ValueError('Smoke-test frame limit must be positive')
        jobs = jobs[:args.limit]
    if args.prepare: prepare(repo, scene, jobs)
    else: attach(repo, scene, jobs, kd, kc, scale)


if __name__ == '__main__':
    main()

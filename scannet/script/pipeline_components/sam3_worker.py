"""Offline SAM3 image inference in the isolated sg-sam3 environment."""
import argparse
from contextlib import redirect_stdout
import fcntl
import hashlib
import io
import json
import os
from pathlib import Path
import time

import numpy as np
from PIL import Image

CHECKPOINT_SHA256 = '9999e2341ceef5e136daa386eecb55cb414446a00ac2b55eb2dfd2f7c3cf8c9e'


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def cache_identity(job, implementation):
    return hashlib.sha256(json.dumps({
        'implementation': implementation, 'checkpoint': CHECKPOINT_SHA256,
        'rgb': sha256(job['rgb']), 'queries': job['queries'],
        'regions': job['regions'], 'threshold': .5, 'resolution': 1008,
    }, sort_keys=True).encode()).hexdigest()


def infer_region(processor, image, names, bounds, full_shape):
    """Encode a region once and return separate instances for each noun query."""
    import torch
    x0, y0, x1, y1 = bounds
    state = processor.set_image(image.crop(bounds))
    result = []
    for name in names:
        processor.reset_all_prompts(state)
        output = processor.set_text_prompt(prompt=name, state=state)
        masks = output['masks'][:, 0].cpu().numpy().astype(bool)
        scores = output['scores'].float().cpu().numpy()
        for mask, score in zip(masks, scores):
            area = int(mask.sum())
            if area < 32 or area > .8 * mask.size:
                continue
            ys, xs = np.where(mask)
            is_crop = bounds != [0, 0, image.width, image.height]
            # A cut object at a crop boundary is not a new physical instance.
            if is_crop and ((x0 > 0 and xs.min() < 3) or (y0 > 0 and ys.min() < 3)
                    or (x1 < image.width and xs.max() >= x1-x0-3)
                    or (y1 < image.height and ys.max() >= y1-y0-3)):
                continue
            full = np.zeros(full_shape, bool)
            full[y0:y1, x0:x1] = mask
            result.append(({
                'object_name': name, 'confidence': float(score),
                'confidence_type': 'uncalibrated_sam3_presence_times_detection',
                'sam_quality_score': None,
                'segmentation_box': [int(xs.min()+x0), int(ys.min()+y0),
                                     int(xs.max()+x0+1), int(ys.max()+y0+1)],
                'proposal_sources': ['sam3_zoom' if is_crop else 'sam3_concept'],
                'inference_crop': bounds, 'is_zoom_proposal': is_crop,
            }, full))
        del output, masks, scores
    del state
    torch.cuda.empty_cache()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--jobs', required=True)
    parser.add_argument('--cache-dir', required=True)
    parser.add_argument('--limit', type=int, help='Leading-frame diagnostic only; never used by segment(context)')
    args = parser.parse_args()
    os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    jobs_path = Path(args.jobs)
    config = json.loads(jobs_path.read_text())
    repo = Path(__file__).resolve().parents[3]
    weights = repo/'checkpoints/sam3/sam3.pt'
    if not weights.is_file() or sha256(weights) != CHECKPOINT_SHA256:
        raise ValueError('SAM3 checkpoint is missing or fails the pinned ModelScope SHA256')
    import torch
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor
    if not torch.cuda.is_available():
        raise RuntimeError('The managed SAM3 image backend requires CUDA')
    source = repo/'scannet/script/thirdparty/SAM3'
    bpe = source/'sam3/assets/bpe_simple_vocab_16e6.txt.gz'
    implementation = hashlib.sha256(Path(__file__).read_bytes()
        + Path(__file__).with_name('sam3_frontend.py').read_bytes()
        + (source/'sam3/model_builder.py').read_bytes()
        + (source/'sam3/model/sam3_image_processor.py').read_bytes()
        + bpe.read_bytes()).hexdigest()
    cache = Path(args.cache_dir); cache.mkdir(parents=True, exist_ok=True)
    lock_path = Path.home()/'.cache/partaware-sg/gpu.lock'
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    receipts = []
    with lock_path.open('a') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        loader_log = io.StringIO()
        with redirect_stdout(loader_log):
            model = build_sam3_image_model(checkpoint_path=str(weights),
                bpe_path=str(bpe), load_from_HF=False, device='cuda',
                enable_inst_interactivity=False)
        if 'missing_keys=' in loader_log.getvalue():
            raise RuntimeError('SAM3 image checkpoint did not populate every required model parameter')
        processor = Sam3Processor(model, device='cuda', confidence_threshold=.5)
        if args.limit is not None and args.limit <= 0:
            raise ValueError('Diagnostic frame limit must be positive')
        jobs = config['jobs'] if args.limit is None else config['jobs'][:args.limit]
        for job in jobs:
            fid = job['frame_id']; cp = cache/f'{fid}.npz'
            identity = cache_identity(job, implementation)
            if cp.is_file():
                with np.load(cp, allow_pickle=False) as saved:
                    if str(saved['signature']) != identity:
                        raise ValueError(f'SAM3 cache inputs changed: {fid}; use a fresh output directory')
                    receipts.append(json.loads(str(saved['audit'])))
                print('复用本地 SAM3 实例缓存', fid, flush=True)
                continue
            start = time.monotonic()
            with Image.open(job['rgb']) as raw:
                image = raw.convert('RGB')
            torch.cuda.reset_peak_memory_stats()
            with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
                proposals = infer_region(processor, image, job['queries'],
                    [0, 0, image.width, image.height], (image.height, image.width))
                from sam3_frontend import zoom_regions, box_iou
                regions = zoom_regions([r for r, _ in proposals], (image.height, image.width), job['queries'])
                for region in job['regions']:
                    if len(regions) >= 4:
                        break
                    if not any(box_iou(region['bounds'], other['bounds']) >= .5 for other in regions):
                        regions.append(region)
                for region in regions:
                    proposals.extend(infer_region(processor, image, region['queries'],
                        region['bounds'], (image.height, image.width)))
            records = [record for record, _ in proposals]
            masks = np.stack([mask for _, mask in proposals]) if proposals else np.empty((0, image.height, image.width), bool)
            item = {'frame_id': fid, 'queries': len(job['queries']), 'zoom_regions': len(regions),
                'effective_regions': regions, 'region_source': 'sam3_detection_then_cached_evidence',
                'raw_proposals': len(records), 'seconds': time.monotonic()-start,
                'gpu_peak_allocated_bytes': torch.cuda.max_memory_allocated(),
                'rgb_sha256': sha256(job['rgb']), 'signature': identity}
            temporary = cp.with_suffix('.npz.partial')
            with temporary.open('wb') as stream:
                np.savez_compressed(stream, records=json.dumps(records),
                    packed_masks=np.packbits(masks, axis=-1), shape=np.array(masks.shape),
                    signature=identity, audit=json.dumps(item))
            temporary.replace(cp)
            receipts.append(item)
            print('SAM3 实例推理完成', item, flush=True)
        del processor, model
        torch.cuda.empty_cache()
    audit = {'algorithm': 'SAM3_concept_and_adaptive_zoom_v12',
        'checkpoint_source': 'https://modelscope.cn/models/facebook/sam3',
        'checkpoint_sha256': CHECKPOINT_SHA256, 'implementation_sha256': implementation,
        'source_revision': config['source_revision'], 'trained_weights_loaded': True,
        'image_backbone_reused_between_queries': True, 'qwen_api_calls': 0,
        'gpt_api_calls': 0, 'ground_truth_used': False, 'frames': receipts}
    audit['complete'] = len(receipts) == len(config['jobs'])
    (cache.parent/'sam3_inference_audit.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2)+'\n')


if __name__ == '__main__':
    main()

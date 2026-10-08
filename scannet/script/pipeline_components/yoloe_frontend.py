"""Local YOLOE masks augment cached Florence/DINO/SAM observations.

All category requests are cached JSON or a fixed indoor vocabulary. This module
never imports or calls a Qwen image API. Model stages share the existing GPU lock.
"""
import argparse
import copy
import fcntl
import hashlib
import json
from pathlib import Path
import sys
import cv2
import numpy as np


def segment(context):
    python = context.repo/'.venv-yoloe/bin/python'
    command = [str(python), str(Path(__file__)), '--processed-scene', str(context.scene)]
    command += ['--manifest', str(context.manifest)] if context.manifest else ['--image-dir', str(context.image_dir)]
    context.execute(command, 'YOLOE 原生分割融合既有 Florence 缓存，不调用千问')


def mask_iou(a, b):
    intersection = np.count_nonzero(a & b)
    return intersection/max(np.count_nonzero(a | b), 1)


def union_masks(records, old_mask, additions):
    """Preserve unsupported old observations and add independent new proposals."""
    candidates = [(copy.deepcopy(r), old_mask == int(r['frame_instance_id'])) for r in records]
    for record, mask in additions:
        overlaps = [mask_iou(mask, m) for _, m in candidates]
        best = int(np.argmax(overlaps)) if overlaps else None
        if best is not None and overlaps[best] >= .6:
            prior, prior_mask = candidates[best]
            prior['yoloe_confidence'] = record['confidence']
            prior['proposal_sources'] = ['cached_florence_dino_sam', 'yoloe']
            prior['confidence'] = float(np.sqrt(prior['confidence']*record['confidence']))
            # Reject gross expansion; a contained part cannot replace a whole mask.
            if .65 <= mask.sum()/max(prior_mask.sum(), 1) <= 1.5:
                candidates[best] = (prior, mask)
            else:
                candidates[best] = (prior, prior_mask)
        elif record['confidence'] >= .3:
            candidates.append((record, mask))
    if len(candidates) > 255:
        raise RuntimeError('The original uint8 label interface supports at most 255 instances')
    labels = np.zeros(old_mask.shape, np.uint8)
    # Small genuine objects retain ownership inside larger supporting furniture.
    order = sorted(range(len(candidates)), key=lambda i: (-int(candidates[i][1].sum()), i))
    for i in order:
        labels[candidates[i][1]] = i+1
    result = []
    for i, (record, _) in enumerate(candidates):
        pixels = int(np.count_nonzero(labels == i+1))
        if pixels < 32:
            labels[labels == i+1] = 0
            continue
        record.update(instance_id=-1, frame_instance_id=i+1, saved_pixels=pixels)
        result.append(record)
    return result, labels


def roi_features(model, image, records):
    """Pool the official DINO projected backbone into consistent 256D features."""
    import torch
    from PIL import Image
    import groundingdino.datasets.transforms as transforms
    from groundingdino.util.misc import nested_tensor_from_tensor_list
    transform = transforms.Compose([transforms.RandomResize([800], max_size=1333),
        transforms.ToTensor(), transforms.Normalize([.485, .456, .406], [.229, .224, .225])])
    tensor, _ = transform(Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)), None)
    device = next(model.parameters()).device
    with torch.inference_mode():
        features, _ = model.backbone(nested_tensor_from_tensor_list([tensor.to(device)]))
        grid = model.input_proj[-2](features[-1].tensors)[0]
    gh, gw = grid.shape[1:]
    h, w = image.shape[:2]
    for record in records:
        x1, y1, x2, y2 = record['segmentation_box']
        xa, ya = max(0, int(x1/w*gw)), max(0, int(y1/h*gh))
        xb, yb = min(gw, max(xa+1, int(np.ceil(x2/w*gw)))), min(gh, max(ya+1, int(np.ceil(y2/h*gh))))
        vector = grid[:, ya:yb, xa:xb].mean((1, 2)).float().cpu().numpy()
        if vector.shape != (256,) or not np.isfinite(vector).all():
            raise ValueError('Invalid projected DINO ROI feature')
        record['feature'] = vector.tolist()
        record['feature_source'] = 'groundingdino_projected_backbone_roi_256_v4'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest')
    parser.add_argument('--image-dir')
    parser.add_argument('--processed-scene', required=True)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(repo/'scannet/script'))
    from partaware.geometry import load_capture
    from ultralytics import YOLOE
    import torch
    from sentence_transformers import SentenceTransformer
    from groundingdino.util.inference import load_model
    lock_dir = Path.home()/'.cache/partaware-sg'
    lock_dir.mkdir(parents=True, exist_ok=True)
    with (lock_dir/'gpu.lock').open('a') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        scene = Path(args.processed_scene).resolve()
        _, jobs, _, _, _ = load_capture(args.manifest, args.image_dir, scene/'refined_instance')
        folder = scene/'refined_instance'
        cache_dir = scene/'frontend_cache'
        cache_dir.mkdir(exist_ok=True)
        model_path = repo/'checkpoints/yoloe/yoloe-v8s-seg.pt'
        text_path = repo/'checkpoints/yoloe/mobileclip_blt.ts'
        if not model_path.is_file() or not text_path.is_file():
            raise FileNotFoundError('Install the pinned YOLOE and MobileCLIP weights first')
        # Load the pinned text encoder directly: no automatic network download.
        from ultralytics.nn.text_model import TextModel
        import clip
        class LocalText(TextModel):
            def __init__(self):
                super().__init__()
                self.encoder = torch.jit.load(str(text_path), map_location='cpu').eval()
            def tokenize(self, texts):
                return clip.tokenize(texts, truncate=True)
            def encode_text(self, tokens):
                return self.encoder(tokens).float()
        model = YOLOE(str(model_path))
        model.model.clip_model = LocalText()
        names = set()
        for job in jobs:
            tag_path = folder/f"{job['frame_id']}.json"
            tags = json.loads(tag_path.read_text())
            names.update(o['name'].strip().lower() for o in tags['objects'] if isinstance(o.get('name'), str))
        vocabulary = json.loads((repo/'scannet/script/ram/hypersim_indoor_57.json').read_text())
        names.update(key for entry in vocabulary for key in entry)
        names = sorted(x for x in names if x and x not in {'wall', 'floor', 'ceiling'})
        with torch.inference_mode():
            embedding = model.model.get_text_pe(names, cache_clip_model=True)
        model.set_classes(names, embedding)
        del model.model.clip_model
        bert = SentenceTransformer('/home/lewisliu/.cache/torch/sentence_transformers/sentence-transformers_all-MiniLM-L6-v2', device='cpu')
        bert_features = dict(zip(names, bert.encode(names)))
        dino_root = repo/'scannet/script/thirdparty/Grounded-Segment-Anything'
        dino = load_model(str(dino_root/'GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py'),
                          str(dino_root/'groundingdino_swint_ogc.pth'), device='cpu')
        dino.to('cuda' if torch.cuda.is_available() else 'cpu').eval()
        prior_provenance = scene/'frontend_provenance.json'
        model_sha = hashlib.sha256(model_path.read_bytes()).hexdigest()
        text_sha = hashlib.sha256(text_path.read_bytes()).hexdigest()
        signature = hashlib.sha256(json.dumps({'algorithm': 'yoloe_union_roi_v4', 'model': model_sha,
            'text_encoder': text_sha, 'vocabulary': names, 'conf': .22, 'union_conf': .3, 'iou': .6}, sort_keys=True).encode()).hexdigest()
        prior = json.loads(prior_provenance.read_text()) if prior_provenance.is_file() else {}
        if prior.get('cache_signature') and prior['cache_signature'] != signature:
            raise ValueError('Frontend weights or vocabulary changed; rebuild into a fresh experiment')
        if prior.get('model_sha256') and prior['model_sha256'] != model_sha:
            raise ValueError('Frontend model changed; do not silently reuse old observations')
        total_new, total_old = 0, 0
        for job in jobs:
            fid = job['frame_id']
            cache = cache_dir/f'{fid}.npz'
            record_path = folder/f'{fid}_instance.json'
            mask_path = folder/f'{fid}.png'
            if cache.exists():
                with np.load(cache) as saved:
                    expected = str(saved['rgb_sha256'])
                    original_records = json.loads(str(saved['source_records']))
                if hashlib.sha256(Path(job['rgb']).read_bytes()).hexdigest() != expected:
                    raise ValueError(f'Cached frontend RGB changed: {fid}')
                records = json.loads(record_path.read_text())
                labels = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
                if (labels is None or any(r.get('feature_source') != 'groundingdino_projected_backbone_roi_256_v4'
                        or len(r.get('feature', [])) != 256 for r in records)
                        or any(not np.any(labels == int(r['frame_instance_id'])) for r in records)):
                    raise ValueError(f'Incomplete frontend cache: {fid}')
                total_old += len(original_records)
                total_new += sum(r.get('proposal_sources') == ['yoloe'] for r in records)
                print('复用已完成的本地前端', fid, flush=True)
                continue
            image = cv2.imread(str(job['rgb']))
            old_records = json.loads(record_path.read_text())
            old_mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
            result = model.predict(image, conf=.22, iou=.6, imgsz=640,
                                   device=0 if torch.cuda.is_available() else 'cpu', retina_masks=True,
                                   save=False, verbose=False)[0]
            additions = []
            if result.masks is not None:
                for mask, box in zip(result.masks.data.cpu().numpy(), result.boxes):
                    mask = mask > .5
                    if mask.shape != image.shape[:2]:
                        mask = cv2.resize(mask.astype(np.uint8), image.shape[1::-1], interpolation=cv2.INTER_NEAREST).astype(bool)
                    if mask.sum() < 64 or mask.sum() > .8*mask.size:
                        continue
                    name = names[int(box.cls.item())]
                    record = {'object_name': name, 'object_description': f'Local YOLOE detection of {name}.',
                              'confidence': float(box.conf.item()), 'bert_embedding': bert_features[name].tolist(),
                              'segmentation_box': box.xyxy[0].cpu().tolist(), 'sam_quality_score': None,
                              'proposal_sources': ['yoloe'], 'confidence_type': 'uncalibrated_yoloe_score'}
                    additions.append((record, mask))
            records, labels = union_masks(old_records, old_mask, additions)
            for record in records:
                if 'segmentation_box' not in record:
                    ys, xs = np.where(labels == record['frame_instance_id'])
                    record['segmentation_box'] = [int(xs.min()), int(ys.min()), int(xs.max()+1), int(ys.max()+1)]
            roi_features(dino, image, records)
            total_new += sum(r.get('proposal_sources') == ['yoloe'] for r in records)
            total_old += len(old_records)
            # The complete original observations make interrupted inference resumable.
            record_path.write_text(json.dumps(records, indent=2)+'\n')
            if not cv2.imwrite(str(mask_path), labels):
                raise IOError(mask_path)
            np.savez_compressed(cache, source_mask=old_mask, source_records=json.dumps(old_records),
                                rgb_sha256=hashlib.sha256(Path(job['rgb']).read_bytes()).hexdigest())
            print('本地 YOLOE 掩码融合完成', fid, '原候选', len(old_records), '新总数', len(records), flush=True)
        (scene/'frontend_provenance.json').write_text(json.dumps({'algorithm': 'yoloe_union_roi_v4',
            'qwen_api_calls': 0, 'vocabulary': names, 'model_sha256': model_sha, 'text_encoder_sha256': text_sha, 'cache_signature': signature,
            'feature_space': 'groundingdino_projected_backbone_roi_256',
            'new_uncorroborated_observations': total_new, 'original_observations': total_old}, indent=2)+'\n')


if __name__ == '__main__':
    main()

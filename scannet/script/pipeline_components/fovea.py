"""Fine Object and Visual Evidence Acquisition, preserving cached coarse evidence."""
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
    from pipeline_components import yoloe_frontend
    yoloe_frontend.segment(context)
    command = [str(context.repo/'.venv-yoloe/bin/python'), str(Path(__file__)),
               '--processed-scene', str(context.scene)]
    command += ['--manifest', str(context.manifest)] if context.manifest else ['--image-dir', str(context.image_dir)]
    context.execute(command, 'FOVEA：保留既有观测并核验放大检测的独立小实例')


def independent_proposals(proposals, labels):
    """A nested region needs disjoint siblings; a solitary part is insufficient."""
    candidates = []
    for record, mask in proposals:
        nonzero, counts = np.unique(labels[mask], return_counts=True)
        parent, fraction = max(zip(nonzero, counts/mask.sum()), key=lambda value: value[1])
        record = {**record, 'coarse_parent_local_id': int(parent) if fraction >= .7 else 0}
        candidates.append((record, mask))
    result = []
    for i, (record, mask) in enumerate(candidates):
        parent = record['coarse_parent_local_id']
        siblings = [j for j, (other, region) in enumerate(candidates)
            if i != j and other['coarse_parent_local_id'] == parent
            and other['object_name'] == record['object_name'] and .25 <= region.sum()/mask.sum() <= 4
            and np.count_nonzero(mask & region)/max(min(mask.sum(), region.sum()), 1) < .1]
        if parent and not siblings:
            continue
        result.append(({**record, 'independent_sibling_count': len(siblings), 'fine_scale_instance': True}, mask))
    return result


def append_instances(source, original_mask, additions):
    """Keep every coarse local ID and feature; append fine IDs without reindexing."""
    records = copy.deepcopy(source); labels = original_mask.copy()
    next_id = max([int(labels.max()), *[int(r['frame_instance_id']) for r in records]], default=0)+1
    for record, mask in additions:
        if next_id > 255: raise ValueError('The retained uint8 mask interface is full')
        record = copy.deepcopy(record)
        record.update(instance_id=-1, frame_instance_id=next_id, saved_pixels=int(mask.sum()))
        labels[mask] = next_id; records.append(record); next_id += 1
    return records, labels


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--processed-scene', required=True)
    parser.add_argument('--manifest'); parser.add_argument('--image-dir')
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(repo/'scannet/script'))
    from partaware.geometry import load_capture, project_mask
    from pipeline_components.fine_instances import proposals
    from pipeline_components.yoloe_frontend import union_masks, mask_iou, roi_features
    import torch
    import clip
    from ultralytics import YOLOE
    from ultralytics.nn.text_model import TextModel
    from sentence_transformers import SentenceTransformer
    from groundingdino.util.inference import load_model
    scene = Path(args.processed_scene).resolve(); folder = scene/'refined_instance'
    _, jobs, kd, kc, scale = load_capture(args.manifest, args.image_dir, folder)
    names = json.loads((scene/'frontend_provenance.json').read_text())['vocabulary']
    cache = scene/'fine_frontend_cache'; cache.mkdir(exist_ok=True)
    weights = repo/'checkpoints/yoloe/yoloe-v8l-seg.pt'
    implementation = hashlib.sha256(Path(__file__).read_bytes()+Path(__file__).with_name('fine_instances.py').read_bytes()).hexdigest()
    signature = hashlib.sha256((hashlib.sha256(weights.read_bytes()).hexdigest()+implementation+json.dumps(names)).encode()).hexdigest()
    audit = {'algorithm': 'FOVEA_coarse_evidence_preserving_v12', 'component_sha256': implementation,
             'cache_signature': signature, 'qwen_api_calls': 0, 'ground_truth_used': False, 'frames': []}
    lock_path = Path.home()/'.cache/partaware-sg/gpu.lock'; lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open('a') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        model = YOLOE(str(weights))
        class LocalText(TextModel):
            def __init__(self):
                super().__init__(); self.encoder = torch.jit.load(str(repo/'checkpoints/yoloe/mobileclip_blt.ts'), map_location='cpu').eval()
            def tokenize(self, texts): return clip.tokenize(texts, truncate=True)
            def encode_text(self, tokens): return self.encoder(tokens).float()
        model.model.clip_model = LocalText()
        model.set_classes(names, model.model.get_text_pe(names, cache_clip_model=True))
        del model.model.clip_model
        bert = SentenceTransformer('/home/lewisliu/.cache/torch/sentence_transformers/sentence-transformers_all-MiniLM-L6-v2', device='cpu')
        semantic = dict(zip(names, bert.encode(names)))
        root = repo/'scannet/script/thirdparty/Grounded-Segment-Anything'
        dino = load_model(str(root/'GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py'),
                          str(root/'groundingdino_swint_ogc.pth'), device='cpu').to('cuda' if torch.cuda.is_available() else 'cpu').eval()
        for job in jobs:
            fid = job['frame_id']; cp = cache/f'{fid}.npz'
            rp, mp = folder/f'{fid}_instance.json', folder/f'{fid}.png'
            rgb_sha = hashlib.sha256(Path(job['rgb']).read_bytes()).hexdigest()
            source, labels = json.loads(rp.read_text()), cv2.imread(str(mp), -1)
            if cp.is_file():
                with np.load(cp) as saved:
                    if str(saved['rgb_sha256']) != rgb_sha: raise ValueError(f'Fine frontend RGB changed: {fid}')
                    if str(saved['signature']) == signature:
                        if 'output_mask' in saved:
                            rp.write_text(str(saved['output_records'])+'\n')
                            if not cv2.imwrite(str(mp), saved['output_mask']): raise IOError(mp)
                            audit['frames'].append(json.loads(str(saved['audit']))); continue
                    source, labels = json.loads(str(saved['source_records'])), saved['source_mask']
            original_mask = labels.copy(); image = cv2.imread(str(job['rgb']))
            depth, pose = cv2.imread(str(job['depth']), -1), np.loadtxt(job['pose'])
            old_regions = [labels == int(record['frame_instance_id']) for record in source]
            fine = []; raw = proposals(model, image, names, 0 if torch.cuda.is_available() else 'cpu')
            for record, region in raw:
                if any(mask_iou(region, existing) >= .55 for existing in old_regions): continue
                points, _ = project_mask(region, image, depth, pose, kd, kc, scale, stride=2, max_depth=0)
                if len(points) < 32 or np.diff(np.quantile(points, [.02, .98], axis=0), axis=0).max() > .35: continue
                fine.append((record, region))
            retained = independent_proposals(fine, labels)
            for record, _ in retained:
                record['bert_embedding'] = semantic[record['object_name']].tolist()
                record['object_description'] = f'Independent fine-scale local detection of {record["object_name"]}.'
            records, labels = append_instances(source, labels, retained)
            new = [record for record in records if record.get('fine_scale_instance')]
            if new: roi_features(dino, image, new)
            item = {'frame_id': fid, 'coarse_instances': len(source), 'raw_fine_proposals': len(raw),
                    'scale_confirmed': len(fine), 'independent_retained': len(retained), 'final_instances': len(records)}
            rp.write_text(json.dumps(records, indent=2)+'\n')
            if not cv2.imwrite(str(mp), labels): raise IOError(mp)
            np.savez_compressed(cp, source_records=json.dumps(source), source_mask=original_mask,
                                output_records=json.dumps(records), output_mask=labels,
                                rgb_sha256=rgb_sha, signature=signature, audit=json.dumps(item))
            audit['frames'].append(item); print('FOVEA独立小实例审核', item, flush=True)
    (scene/'fine_frontend_audit.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2)+'\n')


if __name__ == '__main__': main()

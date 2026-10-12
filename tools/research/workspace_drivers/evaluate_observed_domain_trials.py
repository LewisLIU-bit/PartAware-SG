"""Independent Hypersim screen; immutable registered GT and AP protocol."""
import hashlib, json, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]; WORK = ROOT/'esam_hypersim_work'
sys.path[:0] = [str(ROOT/'scan200_work'), '/home/lewisliu/PartAware-SG/scannet/script']
from evaluate_scan200 import verify_evaluator
from scannet_utils.evaluate_semantic_instance import scannet_eval

target = Path('/home/lewisliu/datasets/scannet-sg-processed/research_cycle/sage_domain_transfer/hypersim/ai_001_010')
receipt = json.loads((target/'candidate_receipt.json').read_text()); assert receipt['complete']
comparison = json.loads((WORK/'comparison.json').read_text()); assert comparison['complete']
assert receipt['inputs']['manifest_sha256'] == comparison['registration']['manifest_sha256']
verify_evaluator()
with np.load(WORK/'evaluation_GT.npz', allow_pickle=False) as f: gt = f['gt'].copy()
results = []
for item in receipt['results']:
    path = Path(item['masks_file']); assert hashlib.sha256(path.read_bytes()).hexdigest() == item['masks_sha256']
    with np.load(path, allow_pickle=False) as f: masks = f['masks'].copy(); scores = f['scores'].copy()
    assert masks.shape == (len(scores), len(gt))
    pred = {str(i):dict(mask=m, label_id=3, conf=float(s)) for i, (m, s) in enumerate(zip(masks, scores))}
    metrics = scannet_eval([pred], [gt], None, (3,), ('object',), {3:'object'})
    results.append(dict(mode=item['mode'], metrics=metrics, predictions=len(pred)))
    print('010_SAME_PROTOCOL_SCREEN', item['mode'], json.dumps(metrics), flush=True)
baseline = next(r['point_metrics'] for r in comparison['results'] if r['profile']=='GPT v19 saved')
(ROOT/'v20_work/010_sage_domain_screen.json').write_text(json.dumps(dict(complete=True, scene='ai_001_010',
    candidate_screen_only=True, full_pipeline_replay=False, GT_in_construction=False,
    baseline_point_metrics=baseline, historical_strict_box_metrics_unchanged=True,
    box_evaluation_pending=True, results=results, construction_receipt_sha256=hashlib.sha256((target/'candidate_receipt.json').read_bytes()).hexdigest()), indent=2)+'\n')

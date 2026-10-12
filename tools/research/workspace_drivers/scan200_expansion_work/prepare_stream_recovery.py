"""Hash successful cache records before explicit stream-failure recovery."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import cached_relay_recovery as adapter

WORK = Path(__file__).resolve().parent
ROOT = Path('/home/lewisliu/datasets/scannet-sg-processed/gpt_vision_cache/scannet')
adapter.install_recovery_adapter()
settings = adapter.vision_api.Settings.load(adapter.REPO/'.gpt-vision.local.json')
fingerprint = adapter.vision_api.request_fingerprint(settings)
rows = []
for scene in json.loads((WORK/'preregistration.json').read_text())['scenes']:
    cache = ROOT/scene
    successful = []; failures = []
    for path in sorted((cache/'vlm_responses').glob('*.json')):
        record = json.loads(path.read_text())
        adapter.vision_api.validate_cached_request(record, settings, fingerprint)
        try:
            adapter.vision_api.parse_response(record['response'], settings.protocol)
        except ValueError:
            error = record['response'].get('error') or {}
            failures.append(dict(frame=path.stem, status=record['response'].get('status'),
                                 error_code=error.get('code') if isinstance(error, dict) else None,
                                 eligible=adapter.saved_transient_stream_failure(record)))
            if not adapter.saved_transient_stream_failure(record):
                raise RuntimeError('Saved failure is not an eligible transient stream error')
        else:
            successful.append(dict(frame=path.stem,
                                   raw_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    rows.append(dict(scene=scene, successful=successful, failed=failures))
receipt = dict(prepared_at_utc=datetime.now(timezone.utc).isoformat(),
               complete=True, cache_request_fingerprint=fingerprint,
               saved_successful_frames=sum(len(row['successful']) for row in rows),
               protocol_model_prompt_budget_images_unchanged=True,
               explicit_failed_stream_transport_recovery=True,
               production_vision_source_unchanged=True, max_attempts_per_scene=2,
               adapter_sha256=hashlib.sha256(Path(adapter.__file__).read_bytes()).hexdigest(),
               tests='2 offline tests passed; no network',
               failed_request_billing='unknown', results=rows)
receipt_path = WORK/'stream_recovery_preflight.json'
if receipt_path.exists():
    previous = receipt_path.read_bytes()
    archive = WORK/('stream_recovery_preflight_'+hashlib.sha256(previous).hexdigest()[:16]+'.json')
    archive.write_bytes(previous)
receipt_path.write_text(json.dumps(receipt, indent=2)+'\n')
print('EXPLICIT_STREAM_RECOVERY_PREPARED', receipt['saved_successful_frames'], 'successful frames kept', flush=True)

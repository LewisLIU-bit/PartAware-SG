"""Validate explicit archived recovery chains before attaching a full cache."""
import hashlib
import json
from pathlib import Path
import cached_relay_recovery as adapter


def validate_complete_cache(cache, settings):
    adapter.install_recovery_adapter()
    cache = Path(cache)
    provenance = json.loads((cache/'recognition_provenance.json').read_text())
    fingerprint = adapter.vision_api.request_fingerprint(settings)
    if not provenance.get('complete') or provenance['request_fingerprint'] != fingerprint:
        raise ValueError('An unchanged complete shared cache is required')
    expected = provenance['expected_frame_ids']
    if [frame['frame_id'] for frame in provenance['frames']] != expected:
        raise ValueError('Provenance does not enumerate the full original manifest')
    recovered = []
    for frame in provenance['frames']:
        raw = cache/'vlm_responses'/(frame['frame_id']+'.json')
        if hashlib.sha256(raw.read_bytes()).hexdigest() != frame['raw_response_sha256']:
            raise ValueError('Completed raw response signature changed')
        record = json.loads(raw.read_text())
        adapter.vision_api.validate_cached_request(record, settings, fingerprint)
        adapter.vision_api.parse_response(record['response'], settings.protocol)
        if record.get('budget_recovery'):
            recovered.append(frame['frame_id'])
    return dict(complete=True, frames=len(expected), recovered_frames=recovered,
                cache_request_fingerprint=fingerprint,
                recovery_archives_validated=True, new_API_calls=0,
                immutable_production_validator_alone_compatible=not bool(recovered))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--cache', required=True)
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    print(json.dumps(validate_complete_cache(args.cache, adapter.vision_api.Settings.load(args.config))))

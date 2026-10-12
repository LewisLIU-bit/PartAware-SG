"""Explicit, bounded recovery of saved transient upstream stream failures.

This workspace adapter leaves the frozen production source and request prompt,
model, budget, images and shared-cache fingerprint unchanged. The existing API
implementation archives a failed response and records any non-stream recovery.
Successful responses, refusals, authentication failures and malformed content
are never eligible for replacement here.
"""
import sys
from pathlib import Path

REPO = Path('/home/lewisliu/PartAware-SG')
sys.path.insert(0, str(REPO/'scannet/script'))
import vision_api

_original_transport_failed = vision_api.transport_failed
TRANSIENT_STREAM_CODES = frozenset({'upstream_http2_stream_error', 'upstream_error'})


def saved_transient_stream_failure(record):
    if _original_transport_failed(record):
        return True
    response = record.get('response', {})
    error = response.get('error') or {}
    return (record.get('request_settings', {}).get('stream') is True
            and response.get('status') == 'failed'
            and isinstance(error, dict)
            and error.get('type') in (None, 'upstream_error')
            and error.get('code') in TRANSIENT_STREAM_CODES)


def install_recovery_adapter():
    vision_api.transport_failed = saved_transient_stream_failure


if __name__ == '__main__':
    install_recovery_adapter()
    vision_api.main()

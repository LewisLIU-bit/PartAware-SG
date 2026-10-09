"""Cached relay-GPT image categories using the retained ScanNet-SG tag contract."""
import argparse
import base64
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
import fcntl
import hashlib
from io import BytesIO
import json
import os
from pathlib import Path
import tempfile
from urllib.parse import urlsplit

from PIL import Image

from qwen_tools.manifest_io import load_manifest_images
from qwen_tools.response_parser import decode_model_json, validate_categories

PROMPT_VERSION = 'gpt_visible_categories_v1'
CATEGORY_POLICY = 'envelope_noun_exclusion_v2'
PROMPT = '''Identify visible physical object categories in this image for an
open-vocabulary RGB-D object mapping system. Discover categories from the image;
do not restrict the answer to a predefined vocabulary. Include recognizable
small objects and partially visible objects when supported by visual evidence.
Use short common English category names and brief descriptions of visible
appearance. Return each category once; mention visible variation in its
description. Prefer a whole physical object over its attached components.
Do not invent hidden objects, dimensions, materials, brands or identities.
Reflections, shadows, illumination, textures and objects depicted in printed
images are not separate physical objects. Put walls, floors and ceilings in
surface_categories. Furniture and other independent physical structures remain
objects even when their surfaces are planar. Return only a valid JSON object:
{"objects":[{"name":"category","description":"visible appearance"}],
"surface_categories":[{"name":"category","description":"visible appearance"}]}
Use empty arrays when no suitable categories are visible.'''

ITEM_SCHEMA = {'type': 'object', 'properties': {
    'name': {'type': 'string'}, 'description': {'type': 'string'}},
    'required': ['name', 'description'], 'additionalProperties': False}
SCHEMA = {'type': 'object', 'properties': {
    name: {'type': 'array', 'items': ITEM_SCHEMA}
    for name in ('objects', 'surface_categories')},
    'required': ['objects', 'surface_categories'], 'additionalProperties': False}


@dataclass(frozen=True)
class Settings:
    base_url: str
    model: str
    api_key: str = field(repr=False)
    protocol: str = 'chat_completions'
    response_format: str = 'prompt_json'
    image_detail: str = 'high'
    max_output_tokens: int = 4096
    chat_token_parameter: str = 'max_completion_tokens'
    timeout_seconds: float = 180.
    stream: bool = False

    @classmethod
    def load(cls, path, require_model=True):
        path = Path(path).expanduser().resolve()
        data = json.loads(path.read_text(encoding='utf-8-sig'))
        if not isinstance(data, dict):
            raise ValueError('GPT configuration must be a JSON object')
        key_env = data.get('api_key_env', 'PARTAWARE_GPT_API_KEY')
        if not isinstance(key_env, str) or not key_env.strip():
            raise ValueError('api_key_env must be a nonempty environment variable name')
        key = os.environ.get(key_env, '').strip()
        if not key and data.get('api_key_file'):
            key_path = Path(data['api_key_file']).expanduser()
            if not key_path.is_absolute():
                key_path = path.parent / key_path
            key = key_path.read_text(encoding='utf-8-sig').strip()
        key = key or str(data.get('api_key', '')).strip()
        values = {name: data[name] for name in cls.__dataclass_fields__
                  if name != 'api_key' and name in data}
        values.setdefault('model', '')
        if 'base_url' not in values:
            raise ValueError('GPT configuration requires base_url')
        if not key or 'YOUR_' in key:
            raise ValueError('Supply the relay key through api_key_env or api_key_file')
        settings = cls(api_key=key, **values)
        settings.validate(require_model=require_model)
        return settings

    def validate(self, require_model=True):
        if not isinstance(self.api_key, str) or not self.api_key or any(c.isspace() for c in self.api_key):
            raise ValueError('API key must contain exactly one nonempty credential without whitespace')
        if not isinstance(self.base_url, str) or not isinstance(self.model, str):
            raise ValueError('Base URL and model must be strings')
        url = urlsplit(self.base_url)
        if url.scheme not in ('https', 'http') or not url.hostname:
            raise ValueError('Supply the actual relay Base URL, including its API prefix')
        if url.username or url.password or url.query or url.fragment or 'YOUR_' in self.base_url:
            raise ValueError('Base URL must not contain embedded credentials or placeholders')
        if require_model and (not self.model.strip() or 'YOUR_' in self.model):
            raise ValueError('Supply a vision model ID returned by your relay')
        if self.protocol not in ('chat_completions', 'responses'):
            raise ValueError('Unknown GPT protocol')
        if self.response_format not in ('prompt_json', 'json_object', 'json_schema'):
            raise ValueError('Unknown GPT response format')
        if self.image_detail not in ('auto', 'low', 'high'):
            raise ValueError('Unknown image detail')
        if self.chat_token_parameter not in ('max_tokens', 'max_completion_tokens'):
            raise ValueError('Unknown Chat Completions token parameter')
        if not 128 <= self.max_output_tokens <= 32768 or not 1 <= self.timeout_seconds <= 1800:
            raise ValueError('Invalid GPT request limits')
        if not isinstance(self.stream, bool):
            raise ValueError('stream must be a boolean')
        if self.stream and self.protocol != 'responses':
            raise ValueError('Streaming currently requires the Responses protocol')

    def public(self):
        return {name: getattr(self, name) for name in self.__dataclass_fields__
                if name != 'api_key'}

    def redact(self, value):
        return value.replace(self.api_key, '[REDACTED]')

    def client(self):
        from openai import OpenAI
        return OpenAI(api_key=self.api_key, base_url=self.base_url,
                      timeout=self.timeout_seconds, max_retries=0)


def atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix='.' + path.name, dir=path.parent)
    try:
        with os.fdopen(descriptor, 'w', encoding='utf-8') as output:
            json.dump(payload, output, indent=2, ensure_ascii=False, allow_nan=False)
            output.write('\n')
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def parse_response(response, protocol):
    if protocol == 'chat_completions':
        choices = response.get('choices', [])
        if len(choices) != 1 or choices[0].get('finish_reason') != 'stop':
            raise ValueError('GPT image response is incomplete or has multiple choices')
        message = choices[0].get('message', {})
        if message.get('refusal'):
            raise ValueError('GPT refused this image request')
        content = message.get('content')
    else:
        if response.get('status') != 'completed':
            raise ValueError('GPT Responses request is incomplete')
        parts = []
        for item in response.get('output', []):
            if item.get('type') != 'message':
                continue
            for part in item.get('content', []):
                if part.get('type') == 'refusal':
                    raise ValueError('GPT refused this image request')
                if part.get('type') == 'output_text':
                    parts.append(part.get('text', ''))
        content = ''.join(parts)
    if not isinstance(content, str) or not content.strip():
        raise ValueError('GPT image response has no text')
    payload = decode_model_json(content)
    if not isinstance(payload, dict):
        raise ValueError('GPT categories must be a JSON object')
    if 'objects' not in payload:
        raise ValueError('GPT categories must include the objects array')
    categories = {name: validate_categories(payload.get(name, []), name)
                  for name in ('objects', 'surface_categories')}
    # Classify the head noun in both arrays, retaining furniture with planar surfaces.
    # "Tiled wall" is a boundary; "wall-mounted cabinet" and "floor lamp" are not.
    envelope = {'wall', 'walls', 'floor', 'floors', 'flooring', 'ceiling', 'ceilings'}
    objects, surfaces = {}, {}
    for item in categories['objects'] + categories['surface_categories']:
        words = item['name'].replace('-', ' ').split()
        target = surfaces if words and words[-1] in envelope else objects
        target.setdefault(item['name'], item)
    return {'objects': list(objects.values()), 'surface_categories': list(surfaces.values())}


def read_sse_response(lines):
    """Keep relay events without assuming every SDK event has a known model type."""
    events, buffer, final = [], [], None
    def consume():
        nonlocal final
        data = '\n'.join(buffer)
        buffer.clear()
        if not data or data == '[DONE]':
            return
        event = json.loads(data)
        events.append(event)
        if event.get('type') in ('response.completed', 'response.failed', 'response.incomplete'):
            final = event.get('response')
    for line in lines:
        if line.startswith('data:'):
            buffer.append(line[5:].lstrip())
        elif not line:
            consume()
    consume()
    if not isinstance(final, dict):
        final = {'status': 'incomplete', 'output': [], 'error': 'Missing terminal Responses event'}
    return {**final, '_relay_stream_events': events}


def request_image(client, settings, image_bytes, image_format):
    mime = {'PNG': 'image/png', 'JPEG': 'image/jpeg'}[image_format]
    url = f'data:{mime};base64,' + base64.b64encode(image_bytes).decode('ascii')
    common = {'model': settings.model}
    if settings.protocol == 'chat_completions':
        common.update(messages=[{'role': 'user', 'content': [
            {'type': 'image_url', 'image_url': {'url': url, 'detail': settings.image_detail}},
            {'type': 'text', 'text': PROMPT}]}])
        common[settings.chat_token_parameter] = settings.max_output_tokens
        if settings.response_format == 'json_object':
            common['response_format'] = {'type': 'json_object'}
        elif settings.response_format == 'json_schema':
            common['response_format'] = {'type': 'json_schema', 'json_schema': {
                'name': 'visible_categories', 'strict': True, 'schema': SCHEMA}}
        result = client.chat.completions.create(**common)
    else:
        common.update(input=[{'role': 'user', 'content': [
            {'type': 'input_image', 'image_url': url, 'detail': settings.image_detail},
            {'type': 'input_text', 'text': PROMPT}]}],
            max_output_tokens=settings.max_output_tokens, store=False)
        if settings.response_format == 'json_object':
            common['text'] = {'format': {'type': 'json_object'}}
        elif settings.response_format == 'json_schema':
            common['text'] = {'format': {'type': 'json_schema', 'name': 'visible_categories',
                                         'strict': True, 'schema': SCHEMA}}
        if settings.stream:
            with client.responses.with_streaming_response.create(stream=True, **common) as stream:
                result = read_sse_response(stream.iter_lines())
        else:
            result = client.responses.create(**common)
    # A relay may echo request fields. Never preserve a known credential in raw output.
    raw = json.dumps(result if isinstance(result, dict) else result.model_dump(mode='json'), ensure_ascii=False)
    return json.loads(settings.redact(raw))


def request_fingerprint(settings):
    return hashlib.sha256(json.dumps({'settings': settings.public(),
        'prompt_version': PROMPT_VERSION, 'prompt': PROMPT}, sort_keys=True).encode()).hexdigest()


def budget_exhausted(record):
    response = record.get('response', {})
    return (response.get('status') == 'incomplete'
            and (response.get('incomplete_details') or {}).get('reason') == 'max_output_tokens')


def transport_failed(record):
    response = record.get('response', {})
    return (response.get('status') == 'incomplete'
            and response.get('error') == 'Missing terminal Responses event')


def validate_cached_request(record, settings, fingerprint):
    actual = replace(settings, **record['request_settings'])
    actual.validate()
    if record.get('request_fingerprint') != request_fingerprint(actual):
        raise ValueError('Saved actual request settings do not match their fingerprint')
    if actual.public() == settings.public() and record['request_fingerprint'] == fingerprint:
        return
    recovery = record.get('budget_recovery', {})
    if (replace(actual, max_output_tokens=settings.max_output_tokens, stream=settings.stream).public() != settings.public()
            or actual.max_output_tokens < settings.max_output_tokens
            or record.get('cache_request_fingerprint') != fingerprint):
        raise ValueError('GPT cache model/settings changed; choose a fresh cache root')
    override = record.get('explicit_budget_override', {})
    if (actual.stream == settings.stream and override.get('configured_max_output_tokens') == settings.max_output_tokens
            and override.get('requested_max_output_tokens') == actual.max_output_tokens):
        return
    if recovery.get('reason') not in ('max_output_tokens', 'missing_terminal_event'):
        raise ValueError('Changed request budget has no explicit override or failed-response evidence')
    archive = Path(recovery['archived_response'])
    if not archive.is_file() or hashlib.sha256(archive.read_bytes()).hexdigest() != recovery['sha256']:
        raise ValueError('Budget recovery is missing its unchanged failed-response evidence')
    previous = json.loads(archive.read_text())
    valid_failure = (budget_exhausted(previous) if recovery['reason'] == 'max_output_tokens' else transport_failed(previous))
    if (not valid_failure or previous['image_sha256'] != record['image_sha256']
            or previous['frame_id'] != record['frame_id']
            or (recovery['reason'] == 'max_output_tokens'
                and previous['request_settings']['max_output_tokens'] >= actual.max_output_tokens)):
        raise ValueError('Budget recovery cannot replace a successful or unrelated response')
    if actual.stream != settings.stream and not (settings.stream and not actual.stream
            and recovery['reason'] == 'missing_terminal_event' and recovery.get('non_stream_recovery')):
        raise ValueError('Changed transport requires explicit failed-stream recovery evidence')
    validate_cached_request(previous, settings, fingerprint)


def recognize_manifest(manifest, cache_root, settings, limit=0, client=None, workers=1,
                       retry_max_output_tokens=0, retry_transport_failures=False, new_max_output_tokens=0,
                       recover_non_stream=False):
    settings.validate()
    dataset, scene_id, frames = load_manifest_images(manifest)
    if limit < 0:
        raise ValueError('Frame limit cannot be negative')
    if not 1 <= workers <= 4:
        raise ValueError('Recognition workers must be between one and four')
    expected_ids = [frame['frame_id'] for frame in frames]
    frames = frames[:limit] if limit else frames
    scene = Path(cache_root).expanduser().resolve() / dataset / scene_id
    scene.mkdir(parents=True, exist_ok=True)
    if retry_max_output_tokens and not settings.max_output_tokens < retry_max_output_tokens <= 32768:
        raise ValueError('Explicit recovery budget must exceed the configured budget, up to 32768')
    if new_max_output_tokens and not settings.max_output_tokens < new_max_output_tokens <= 32768:
        raise ValueError('Explicit new-request budget must exceed the configured budget, up to 32768')
    fingerprint = request_fingerprint(settings)
    counts = {'api_calls': 0, 'reused_frames': 0}
    records = []
    with (scene / '.recognition.lock').open('a') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        previous_path = scene / 'recognition_provenance.json'
        previous = json.loads(previous_path.read_text()) if previous_path.is_file() else None
        if previous and (previous.get('request_fingerprint') != fingerprint
                         or previous.get('expected_frame_ids') != expected_ids):
            raise ValueError('GPT cache configuration or manifest changed; choose a fresh cache root')
        # Validate every selected cached input before dispatching any paid request.
        inputs = []
        for frame in frames:
            fid = frame['frame_id']
            image_bytes = frame['image_path'].read_bytes()
            with Image.open(BytesIO(image_bytes)) as image:
                image_format, size = image.format, image.size
                image.verify()
            if image_format not in ('PNG', 'JPEG'):
                raise ValueError('GPT input must be a real PNG or JPEG image')
            digest = hashlib.sha256(image_bytes).hexdigest()
            raw_path = scene / 'vlm_responses' / f'{fid}.json'
            if raw_path.exists():
                record = json.loads(raw_path.read_text(encoding='utf-8'))
                if record.get('image_sha256') != digest:
                    raise ValueError('GPT cache input/model/settings changed; choose a fresh cache root')
                if (record.get('dataset'), record.get('scene_id'), record.get('frame_id')) != (dataset, scene_id, fid):
                    raise ValueError('GPT raw cache scene/frame identities differ')
                validate_cached_request(record, settings, fingerprint)
                try:
                    parse_response(record['response'], settings.protocol)
                except ValueError:
                    if not ((retry_max_output_tokens and budget_exhausted(record)
                            and retry_max_output_tokens > record['request_settings']['max_output_tokens'])
                            or (retry_transport_failures and transport_failed(record))):
                        raise
            else:
                record = None
            inputs.append((fid, image_bytes, image_format, size, digest, raw_path, record))
        if client is None and any(item[-1] is None or budget_exhausted(item[-1]) or transport_failed(item[-1]) for item in inputs):
            client = settings.client()

        def process_frame(item):
            fid, image_bytes, image_format, size, digest, raw_path, record = item
            recovery = None
            actual = replace(settings, max_output_tokens=new_max_output_tokens) if record is None and new_max_output_tokens else settings
            if record is not None and (budget_exhausted(record) or transport_failed(record)):
                digest_failed = hashlib.sha256(raw_path.read_bytes()).hexdigest()
                archive = scene/'vlm_failed_responses'/f'{fid}_{digest_failed[:16]}.json'
                atomic_json(archive, record)
                recovery = {'reason': 'max_output_tokens' if budget_exhausted(record) else 'missing_terminal_event', 'archived_response': str(archive),
                            'sha256': hashlib.sha256(archive.read_bytes()).hexdigest(),
                            'previous_budget': record['request_settings']['max_output_tokens']}
                actual = replace(settings, max_output_tokens=max(retry_max_output_tokens,
                    record['request_settings']['max_output_tokens'], settings.max_output_tokens))
                if recover_non_stream and transport_failed(record):
                    actual = replace(actual, stream=False)
                    recovery['non_stream_recovery'] = True
                record = None
            requested = record is None
            if requested:
                print('GPT识图请求', {'scene': scene_id, 'frame': fid}, flush=True)
                try:
                    response = request_image(client, actual, image_bytes, image_format)
                except Exception as error:
                    raise RuntimeError('GPT request failed: ' + settings.redact(str(error))) from None
                record = {'provider': 'relay_gpt', 'dataset': dataset, 'scene_id': scene_id,
                          'frame_id': fid, 'image_sha256': digest, 'image_size': list(size),
                          'request_fingerprint': request_fingerprint(actual), 'requested_model': settings.model,
                          'request_settings': actual.public(), 'prompt_version': PROMPT_VERSION,
                          'prompt_sha256': hashlib.sha256(PROMPT.encode()).hexdigest(),
                          'received_at_utc': datetime.now(timezone.utc).isoformat(), 'response': response}
                if recovery:
                    record.update(cache_request_fingerprint=fingerprint, budget_recovery=recovery)
                elif actual.max_output_tokens != settings.max_output_tokens:
                    record.update(cache_request_fingerprint=fingerprint, explicit_budget_override={
                        'configured_max_output_tokens': settings.max_output_tokens,
                        'requested_max_output_tokens': actual.max_output_tokens})
                atomic_json(raw_path, record)
            # Preserve invalid raw responses and stop instead of silently paying to retry.
            parsed = parse_response(record['response'], settings.protocol)
            atomic_json(scene / 'vlm_parsed' / f'{fid}.json', parsed)
            tags = {'objects': parsed['objects'], 'source': 'relay_gpt',
                    'recognition_cache': str(raw_path), 'image_sha256': digest,
                    'requested_model': settings.model, 'request_fingerprint': record['request_fingerprint']}
            atomic_json(scene / 'refined_instance' / f'{fid}.json', tags)
            frame_record = {'frame_id': fid, 'image_sha256': digest,
                            'raw_response_sha256': hashlib.sha256(raw_path.read_bytes()).hexdigest(),
                            'categories': len(parsed['objects']),
                            'request_fingerprint': record['request_fingerprint'],
                            'actual_max_output_tokens': record['request_settings']['max_output_tokens']}
            print('GPT识图缓存就绪', {'scene': scene_id, 'frame': fid,
                                   'categories': len(parsed['objects'])}, flush=True)
            return frame_record, requested

        def consume(results):
            for record, requested in results:
                records.append(record)
                counts['api_calls' if requested else 'reused_frames'] += 1
        if workers == 1:
            consume(map(process_frame, inputs))
        else:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                # Bound paid work to one concurrent batch; a failure cannot
                # silently dispatch the remainder of an entire scene.
                for start in range(0, len(inputs), workers):
                    consume(pool.map(process_frame, inputs[start:start+workers]))
        if previous:
            selected = {record['frame_id']: record for record in records}
            for old in previous.get('frames', []):
                selected.setdefault(old['frame_id'], old)
            records = [selected[fid] for fid in expected_ids if fid in selected]
        provenance = {'provider': 'relay_gpt', 'scene_id': scene_id, 'dataset': dataset,
                      'shared_cache': str(scene), 'request_fingerprint': fingerprint,
                      'requested_model': settings.model, 'protocol': settings.protocol,
                      'category_policy': CATEGORY_POLICY,
                      'frames': records, 'complete': len(records) == len(expected_ids),
                      'expected_frame_ids': expected_ids,
                      'unique_cached_api_responses': len(records),
                      'this_run': counts, 'qwen_api_calls': 0,
                      'ground_truth_used': False}
        atomic_json(scene / 'recognition_provenance.json', provenance)
    return scene


def attach_cache(cache_scene, target, manifest=None):
    cache_scene, target = Path(cache_scene).resolve(), Path(target).resolve()
    provenance = json.loads((cache_scene / 'recognition_provenance.json').read_text())
    if not provenance.get('complete'):
        raise ValueError('Shared recognition cache is incomplete; finish it before construction')
    if provenance.get('category_policy') != CATEGORY_POLICY:
        raise ValueError('Cached category parsing changed; reparse the existing raw responses first')
    if manifest is not None:
        dataset, scene_id, frames = load_manifest_images(manifest)
        if (dataset, scene_id) != (provenance['dataset'], provenance['scene_id']):
            raise ValueError('Manifest and recognition cache scene identities differ')
        if [f['frame_id'] for f in frames] != provenance['expected_frame_ids']:
            raise ValueError('Manifest and recognition cache frames differ')
        cached = {f['frame_id']: f for f in provenance['frames']}
        for frame in frames:
            digest = hashlib.sha256(frame['image_path'].read_bytes()).hexdigest()
            if cached[frame['frame_id']]['image_sha256'] != digest:
                raise ValueError('Cached recognition belongs to a different RGB input')
    if target.name != provenance['scene_id'] or target.parent.name != provenance['dataset']:
        raise ValueError('Recognition cache and target scene identities differ')
    for frame in provenance['frames']:
        name = frame['frame_id'] + '.json'
        raw = cache_scene / 'vlm_responses' / name
        if hashlib.sha256(raw.read_bytes()).hexdigest() != frame['raw_response_sha256']:
            raise ValueError('Raw recognition response changed after caching')
        payload = json.loads((cache_scene / 'refined_instance' / name).read_text())
        saved = json.loads(raw.read_text())
        if payload['objects'] != parse_response(saved['response'], provenance['protocol'])['objects']:
            raise ValueError('Cached semantic tags differ from their raw API response')
        path = target / 'refined_instance' / name
        if path.exists() and json.loads(path.read_text()) != payload:
            raise ValueError('Target already contains different semantic tags; use a fresh experiment')
        atomic_json(path, payload)
    atomic_json(target / 'recognition_provenance.json', {
        **provenance, 'recognition_reused': True, 'consumer_api_calls': 0})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--manifest')
    parser.add_argument('--cache-root')
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--retry-max-output-tokens', type=int, default=0,
                        help='Explicitly retry only saved output-budget failures with a larger budget')
    parser.add_argument('--retry-transport-failures', action='store_true',
                        help='Explicitly retry only saved streams missing a terminal response')
    parser.add_argument('--new-max-output-tokens', type=int, default=0,
                        help='Explicit larger budget for frames that have never been requested')
    parser.add_argument('--recover-non-stream', action='store_true',
                        help='Retry failed upstream streams using one non-streaming Responses request')
    parser.add_argument('--list-models', action='store_true')
    args = parser.parse_args()
    settings = Settings.load(args.config, require_model=not args.list_models)
    if args.list_models:
        try:
            for model in settings.client().models.list():
                print(model.id)
        except Exception as error:
            raise RuntimeError('Model discovery failed: ' + settings.redact(str(error))) from None
        return
    if not args.manifest or not args.cache_root:
        parser.error('--manifest and --cache-root are required for image recognition')
    recognize_manifest(args.manifest, args.cache_root, settings, args.limit, workers=args.workers,
                       retry_max_output_tokens=args.retry_max_output_tokens,
                       retry_transport_failures=args.retry_transport_failures,
                       new_max_output_tokens=args.new_max_output_tokens,
                       recover_non_stream=args.recover_non_stream)


if __name__ == '__main__':
    main()

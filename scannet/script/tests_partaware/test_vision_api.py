"""Verify paid-request reuse, protocol contracts and experiment isolation offline."""
from dataclasses import replace
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
import os
import subprocess
import sys
from unittest.mock import patch

from PIL import Image
from vision_api import Settings, recognize_manifest, attach_cache, parse_response, request_image, read_sse_response
from run_gpt_comparison import configure_profile, summarize


def payload():
    return {'objects': [{'name': 'Bottle', 'description': 'small transparent bottle'}],
            'surface_categories': []}


def chat_response(data=None, finish='stop'):
    return {'choices': [{'finish_reason': finish,
                         'message': {'content': json.dumps(payload() if data is None else data)}}]}


def responses_response():
    return {'status': 'completed', 'output': [{'type': 'message', 'content': [
        {'type': 'output_text', 'text': json.dumps(payload())}]}]}


class APIContractChecks(unittest.TestCase):
    def setUp(self):
        self.settings = Settings('https://example.invalid/v1', 'test-vision', 'offline-key')

    def fixture(self, root, count=1):
        frames = []
        for i in range(count):
            rgb = root/f'rgb{i}.png'
            Image.new('RGB', (32, 32), (i+10, 20, 30)).save(rgb)
            frames.append({'frame_id': str(i), 'rgb': rgb.name})
        manifest = root/'manifest.json'
        manifest.write_text(json.dumps({'format': 'scannet_sg_input', 'dataset': 'hypersim',
                                        'scene_id': 'test_scene', 'frames': frames}))
        return manifest

    def test_both_protocols_preserve_existing_tag_contract(self):
        for protocol, response in [('chat_completions', chat_response()), ('responses', responses_response())]:
            self.assertEqual(parse_response(response, protocol)['objects'][0]['name'], 'bottle')

    def test_truncation_refusal_and_missing_objects_are_not_empty_success(self):
        bad = [chat_response(finish='length'), chat_response({'surface_categories': []}),
               {'choices': [{'finish_reason': 'stop', 'message': {'refusal': 'refused'}}]}]
        for response in bad:
            with self.assertRaises(ValueError): parse_response(response, 'chat_completions')
        with self.assertRaises(ValueError):
            parse_response({'status': 'incomplete', 'output': []}, 'responses')

    def test_planar_furniture_cannot_be_excluded_as_building_envelope(self):
        data = {'objects': [], 'surface_categories': [
            {'name': 'countertop', 'description': 'work surface'},
            {'name': 'wall', 'description': 'room boundary'}]}
        result = parse_response(chat_response(data), 'chat_completions')
        self.assertEqual(result['objects'][0]['name'], 'countertop')
        self.assertEqual([item['name'] for item in result['surface_categories']], ['wall'])

    def test_modified_boundary_names_are_not_object_candidates(self):
        data = {'objects': [
            {'name': name, 'description': 'visible surface'}
            for name in ['tiled wall', 'wooden floor', 'white ceiling']],
            'surface_categories': [
                {'name': name, 'description': 'independent physical object'}
                for name in ['wall-mounted cabinet', 'floor lamp', 'ceiling light', 'countertop']]}
        result = parse_response(chat_response(data), 'chat_completions')
        self.assertEqual([item['name'] for item in result['surface_categories']],
                         ['tiled wall', 'wooden floor', 'white ceiling'])
        self.assertEqual([item['name'] for item in result['objects']],
                         ['wall-mounted cabinet', 'floor lamp', 'ceiling light', 'countertop'])

    def test_repeated_recognition_and_three_consumers_make_one_paid_request(self):
        with TemporaryDirectory() as directory:
            root = Path(directory); manifest = self.fixture(root)
            with patch('vision_api.request_image', return_value=chat_response()) as request:
                scene = recognize_manifest(manifest, root/'cache', self.settings, client=object())
                recognize_manifest(manifest, root/'cache', self.settings, client=object())
                for version in ['original', 'v11', 'v12']:
                    target = root/version/'hypersim/test_scene'
                    attach_cache(scene, target, manifest)
                    self.assertEqual(json.loads((target/'recognition_provenance.json').read_text())['consumer_api_calls'], 0)
                self.assertEqual(request.call_count, 1)

    def test_different_rgb_cannot_silently_reuse_or_pay_again(self):
        with TemporaryDirectory() as directory:
            root = Path(directory); manifest = self.fixture(root)
            with patch('vision_api.request_image', return_value=chat_response()) as request:
                recognize_manifest(manifest, root/'cache', self.settings, client=object())
                Image.new('RGB', (32, 32), (200, 1, 1)).save(root/'rgb0.png')
                with self.assertRaises(ValueError):
                    recognize_manifest(manifest, root/'cache', self.settings, client=object())
                self.assertEqual(request.call_count, 1)

    def test_new_model_requires_separate_cache(self):
        with TemporaryDirectory() as directory:
            root = Path(directory); manifest = self.fixture(root)
            with patch('vision_api.request_image', return_value=chat_response()) as request:
                recognize_manifest(manifest, root/'cache', self.settings, client=object())
                with self.assertRaises(ValueError):
                    recognize_manifest(manifest, root/'cache', replace(self.settings, model='changed'), client=object())
                self.assertEqual(request.call_count, 1)

    def test_invalid_saved_response_is_kept_without_paid_retry(self):
        with TemporaryDirectory() as directory:
            root = Path(directory); manifest = self.fixture(root)
            with patch('vision_api.request_image', return_value=chat_response(finish='length')) as request:
                for _ in range(2):
                    with self.assertRaises(ValueError):
                        recognize_manifest(manifest, root/'cache', self.settings, client=object())
                self.assertEqual(request.call_count, 1)
                self.assertTrue((root/'cache/hypersim/test_scene/vlm_responses/0.json').exists())

    def test_budget_recovery_preserves_failed_evidence_and_never_repeats_success(self):
        settings = replace(self.settings, protocol='responses')
        incomplete = {'status': 'incomplete', 'incomplete_details': {'reason': 'max_output_tokens'}, 'output': []}
        with TemporaryDirectory() as directory:
            root = Path(directory); manifest = self.fixture(root, 2)
            with patch('vision_api.request_image', side_effect=[responses_response(), incomplete]) as request:
                with self.assertRaises(ValueError):
                    recognize_manifest(manifest, root/'cache', settings, client=object())
                self.assertEqual(request.call_count, 2)
            with patch('vision_api.request_image', return_value=responses_response()) as retry:
                scene = recognize_manifest(manifest, root/'cache', settings, client=object(), retry_max_output_tokens=16384)
                self.assertEqual(retry.call_count, 1)
                self.assertEqual(retry.call_args.args[1].max_output_tokens, 16384)
                saved = json.loads((scene/'vlm_responses/1.json').read_text())
                self.assertEqual(saved['request_settings']['max_output_tokens'], 16384)
                self.assertNotEqual(saved['request_fingerprint'], saved['cache_request_fingerprint'])
                self.assertTrue(Path(saved['budget_recovery']['archived_response']).is_file())
                recognize_manifest(manifest, root/'cache', settings, client=object())
                for version in ['original', 'v11', 'v12']:
                    attach_cache(scene, root/version/'hypersim/test_scene', manifest)
                self.assertEqual(retry.call_count, 1)
            with patch('vision_api.request_image') as retry:
                Path(saved['budget_recovery']['archived_response']).write_text('{}')
                with self.assertRaises(ValueError):
                    recognize_manifest(manifest, root/'cache', settings, client=object())
                retry.assert_not_called()

    def test_smoke_cache_cannot_build_full_scene_but_full_cache_survives_smoke_reuse(self):
        with TemporaryDirectory() as directory:
            root = Path(directory); manifest = self.fixture(root, 2)
            with patch('vision_api.request_image', return_value=chat_response()) as request:
                scene = recognize_manifest(manifest, root/'cache', self.settings, limit=1, client=object())
                with self.assertRaises(ValueError): attach_cache(scene, root/'out/hypersim/test_scene', manifest)
                recognize_manifest(manifest, root/'cache', self.settings, client=object())
                recognize_manifest(manifest, root/'cache', self.settings, limit=1, client=object())
                attach_cache(scene, root/'out/hypersim/test_scene', manifest)
                self.assertEqual(request.call_count, 2)
                self.assertTrue((root/'out/hypersim/test_scene/refined_instance/1.json').exists())

    def test_explicit_new_budget_is_recorded_and_replayed_without_another_request(self):
        with TemporaryDirectory() as directory:
            root = Path(directory); manifest = self.fixture(root)
            settings = replace(self.settings, protocol='responses')
            with patch('vision_api.request_image', return_value=responses_response()) as request:
                scene = recognize_manifest(manifest, root/'cache', settings, client=object(), new_max_output_tokens=16384)
                raw = json.loads((scene/'vlm_responses/0.json').read_text())
                self.assertEqual(raw['request_settings']['max_output_tokens'], 16384)
                self.assertEqual(raw['explicit_budget_override']['configured_max_output_tokens'], 4096)
                recognize_manifest(manifest, root/'cache', settings, client=object())
                self.assertEqual(request.call_count, 1)

    def test_serial_failure_stops_before_requesting_another_frame(self):
        with TemporaryDirectory() as directory:
            root = Path(directory); manifest = self.fixture(root, 3)
            settings = replace(self.settings, protocol='responses')
            missing = {'status':'incomplete', 'output': [], 'error':'Missing terminal Responses event'}
            with patch('vision_api.request_image', return_value=missing) as request:
                with self.assertRaises(ValueError):
                    recognize_manifest(manifest, root/'cache', settings, client=object())
                self.assertEqual(request.call_count, 1)
            with patch('vision_api.request_image', return_value=responses_response()) as request:
                recognize_manifest(manifest, root/'cache', settings, client=object(), retry_transport_failures=True)
                self.assertEqual(request.call_count, 3)

    def test_non_stream_recovery_keeps_failed_stream_evidence_and_successful_cache(self):
        with TemporaryDirectory() as directory:
            root = Path(directory); manifest = self.fixture(root, 2)
            settings = replace(self.settings, protocol='responses', stream=True)
            missing = {'status':'incomplete', 'output':[], 'error':'Missing terminal Responses event'}
            with patch('vision_api.request_image', side_effect=[responses_response(), missing]):
                with self.assertRaises(ValueError):
                    recognize_manifest(manifest, root/'cache', settings, client=object())
            with patch('vision_api.request_image', return_value=responses_response()) as retry:
                scene = recognize_manifest(manifest, root/'cache', settings, client=object(),
                    retry_transport_failures=True, recover_non_stream=True)
                self.assertEqual(retry.call_count, 1)
                self.assertFalse(retry.call_args.args[1].stream)
                saved = json.loads((scene/'vlm_responses/1.json').read_text())
                self.assertTrue(saved['budget_recovery']['non_stream_recovery'])
                recognize_manifest(manifest, root/'cache', settings, client=object())
                self.assertEqual(retry.call_count, 1)

    def test_foreign_scene_or_existing_qwen_tags_cannot_be_overwritten(self):
        with TemporaryDirectory() as directory:
            root = Path(directory); manifest = self.fixture(root)
            with patch('vision_api.request_image', return_value=chat_response()):
                scene = recognize_manifest(manifest, root/'cache', self.settings, client=object())
            with self.assertRaises(ValueError): attach_cache(scene, root/'out/hypersim/foreign_scene')
            target = root/'out/hypersim/test_scene'; folder = target/'refined_instance'; folder.mkdir(parents=True)
            old = json.dumps({'objects': [], 'source': 'qwen'}); (folder/'0.json').write_text(old)
            with self.assertRaises(ValueError): attach_cache(scene, target)
            self.assertEqual((folder/'0.json').read_text(), old)

    def test_modified_raw_response_is_detected_on_attach(self):
        with TemporaryDirectory() as directory:
            root = Path(directory); manifest = self.fixture(root)
            with patch('vision_api.request_image', return_value=chat_response()):
                scene = recognize_manifest(manifest, root/'cache', self.settings, client=object())
            (scene/'vlm_responses/0.json').write_text('{}')
            with self.assertRaises(ValueError): attach_cache(scene, root/'out/hypersim/test_scene', manifest)

    def test_config_can_discover_models_before_model_selection(self):
        with TemporaryDirectory() as directory:
            root = Path(directory); (root/'key.txt').write_text('offline-key')
            config = root/'config.json'; config.write_text(json.dumps({
                'base_url': 'https://example.invalid/v1', 'api_key_file': 'key.txt'}))
            with patch.dict('os.environ', {}, clear=True):
                self.assertEqual(Settings.load(config, require_model=False).model, '')
                with self.assertRaises(ValueError): Settings.load(config)
        self.assertNotIn('offline-key', repr(self.settings))
        self.assertNotIn('offline-key', json.dumps(self.settings.public()))

    def test_real_mime_and_generic_request_have_no_qwen_fields(self):
        captured = {}
        class Result:
            def model_dump(self, **kw): return {**chat_response(), 'echo': 'offline-key'}
        def create(**kwargs): captured.update(kwargs); return Result()
        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
        result = request_image(client, self.settings, b'image', 'JPEG')
        image = captured['messages'][0]['content'][0]['image_url']
        self.assertTrue(image['url'].startswith('data:image/jpeg;base64,'))
        self.assertNotIn('extra_body', captured)
        self.assertEqual(result['echo'], '[REDACTED]')

    def test_responses_stream_uses_one_request_and_completed_response(self):
        captured = {}
        class Stream:
            def __enter__(self): return self
            def __exit__(self, *args): pass
            def iter_lines(self):
                return iter(['data: '+json.dumps({'type':'response.completed','response':responses_response()}), '', 'data: [DONE]', ''])
        def create(**kwargs): captured.update(kwargs); return Stream()
        settings = replace(self.settings, protocol='responses', stream=True)
        response = request_image(SimpleNamespace(responses=SimpleNamespace(with_streaming_response=SimpleNamespace(create=create))), settings, b'image', 'PNG')
        self.assertFalse(captured['store'])
        self.assertEqual(captured['input'][0]['content'][0]['type'], 'input_image')
        self.assertEqual(parse_response(response, 'responses')['objects'][0]['name'], 'bottle')

    def test_unknown_relay_event_is_preserved_but_truncated_stream_is_rejected(self):
        event = {'type':'response.unknown_extension','item':{'vendor_data':42}}
        final = {'type':'response.completed','response':responses_response()}
        result = read_sse_response(iter(['data: '+json.dumps(event),'','data: '+json.dumps(final),'']))
        self.assertEqual(result['_relay_stream_events'][0],event)
        self.assertEqual(parse_response(result,'responses')['objects'][0]['name'],'bottle')
        result = read_sse_response(iter(['data: '+json.dumps(event),'']))
        with self.assertRaises(ValueError): parse_response(result,'responses')

    def test_comparison_profiles_preserve_default_and_use_distinct_algorithms(self):
        registry = SimpleNamespace(FRONTEND='coarse', FUSION='fusion', GRAPH_COMPONENTS=['parts'])
        configure_profile('original', registry)
        self.assertIsNone(registry.FUSION)
        self.assertEqual(registry.GROUNDING_BACKEND, 'dino')
        self.assertEqual(registry.GRAPH_COMPONENTS, [])
        second = SimpleNamespace(FRONTEND='coarse', FUSION='fusion')
        configure_profile('v12', second)
        self.assertEqual(second.FRONTEND.__name__, 'pipeline_components.sam3_frontend')
        self.assertTrue(second.FRONTEND.OWNS_COARSE_SEGMENTATION)
        self.assertEqual(second.FUSION, 'fusion')

    def test_comparison_profile_reaches_model_subprocesses_without_changing_default(self):
        environment = os.environ.copy()
        code = "import pipeline_components as c; import json; print(json.dumps({'version':getattr(c,'VERSION',None),'ownership':getattr(getattr(c,'OWNERSHIP_VALIDATION',None),'__name__',None)}))"
        environment['PARTAWARE_COMPARISON_PROFILE'] = 'v12'
        result = json.loads(subprocess.check_output([sys.executable, '-c', code], env=environment, text=True))
        self.assertEqual(result['ownership'], 'pipeline_components.visibility_ownership')
        self.assertEqual(result['version'], 'v12_gpt')
        environment.pop('PARTAWARE_COMPARISON_PROFILE')
        result = json.loads(subprocess.check_output([sys.executable, '-c', code], env=environment, text=True))
        self.assertIsNone(result['ownership'])

    def test_comparison_summary_does_not_mix_mvo_and_iou(self):
        report = {key: 0 for key in ('gt_objects', 'predicted_objects', 'geometry_only_box_AP25',
                  'geometry_only_box_AP50', 'geometry_only_box_AP75', 'object_count_consistency')}
        report['maximum_volume_overlap'] = {'AP25': .7, 'AP50': .4}
        result = summarize(report)
        self.assertEqual(result['MVO50'], .4)
        self.assertEqual(result['geometry_only_box_AP50'], 0)


if __name__ == '__main__': unittest.main()

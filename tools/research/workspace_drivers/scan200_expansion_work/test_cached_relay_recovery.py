"""Offline recovery checks; no real credentials, models or network calls."""
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
from PIL import Image
import cached_relay_recovery as adapter
from validate_recovered_cache import validate_complete_cache

api = adapter.vision_api


def good():
    return {'status': 'completed', 'output': [{'type': 'message', 'content': [
        {'type': 'output_text', 'text': json.dumps({'objects': [
            {'name': 'cup', 'description': 'visible cup'}], 'surface_categories': []})}]}]}


def failed(code='upstream_http2_stream_error', kind='upstream_error'):
    return {'status': 'failed', 'error': {'code': code, 'type': kind}, 'output': []}


class RecoveryChecks(unittest.TestCase):
    def setUp(self):
        adapter.install_recovery_adapter()
        self.settings = api.Settings('https://example.invalid/v1', 'offline-vision',
                                     'offline-placeholder', protocol='responses', stream=True)

    def fixture(self, root):
        frames = []
        for i in range(2):
            Image.new('RGB', (16, 16), (i, 20, 30)).save(root/f'{i}.png')
            frames.append({'frame_id': str(i), 'rgb': f'{i}.png'})
        manifest = root/'manifest.json'
        manifest.write_text(json.dumps({'format': 'scannet_sg_input', 'dataset': 'scannet',
                                       'scene_id': 'offline_scene', 'frames': frames}))
        return manifest

    def test_only_explicit_transient_stream_error_is_eligible(self):
        record = {'request_settings': {'stream': True}, 'response': failed()}
        self.assertTrue(adapter.saved_transient_stream_failure(record))
        self.assertTrue(adapter.saved_transient_stream_failure({
            'request_settings': {'stream': True},
            'response': failed('upstream_error', None)}))
        for error in [failed('invalid_api_key', 'authentication_error'),
                      failed('insufficient_quota', 'upstream_error'), good()]:
            self.assertFalse(adapter.saved_transient_stream_failure({
                'request_settings': {'stream': True}, 'response': error}))
        self.assertFalse(adapter.saved_transient_stream_failure({
            'request_settings': {'stream': False}, 'response': failed()}))

    def test_archive_hash_success_reuse_and_attach_have_no_extra_requests(self):
        with TemporaryDirectory() as directory:
            root = Path(directory); manifest = self.fixture(root)
            with patch('vision_api.request_image', side_effect=[good(), failed()]):
                with self.assertRaises(ValueError):
                    api.recognize_manifest(manifest, root/'cache', self.settings, client=object())
            scene = root/'cache/scannet/offline_scene'
            success = (scene/'vlm_responses/0.json').read_bytes()
            bad_bytes = (scene/'vlm_responses/1.json').read_bytes()
            with patch('vision_api.request_image') as request:
                with self.assertRaises(ValueError):
                    api.recognize_manifest(manifest, root/'cache', self.settings, client=object())
                request.assert_not_called()
            with patch('vision_api.request_image', return_value=good()) as request:
                api.recognize_manifest(manifest, root/'cache', self.settings, client=object(),
                                       retry_transport_failures=True, recover_non_stream=True)
                self.assertEqual(request.call_count, 1)
                self.assertFalse(request.call_args.args[1].stream)
                record = json.loads((scene/'vlm_responses/1.json').read_text())
                archive = Path(record['budget_recovery']['archived_response'])
                self.assertEqual(archive.read_bytes(), bad_bytes)
                self.assertEqual(hashlib.sha256(bad_bytes).hexdigest(), record['budget_recovery']['sha256'])
                self.assertEqual((scene/'vlm_responses/0.json').read_bytes(), success)
                api.recognize_manifest(manifest, root/'cache', self.settings, client=object())
                api.attach_cache(scene, root/'consumer/scannet/offline_scene', manifest)
                checked = validate_complete_cache(scene, self.settings)
                self.assertEqual(checked['recovered_frames'], ['1'])
                self.assertTrue(checked['recovery_archives_validated'])
                self.assertEqual(request.call_count, 1)
            archive.write_text('{}')
            with patch('vision_api.request_image') as request:
                with self.assertRaises(ValueError):
                    validate_complete_cache(scene, self.settings)
                request.assert_not_called()


if __name__ == '__main__':
    unittest.main()

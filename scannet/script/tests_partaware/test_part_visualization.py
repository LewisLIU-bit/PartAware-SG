"""Verify hierarchy visibility and safe overlay construction."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'script'))
from utils.part_visualization import build_part_overlay


class PartOverlayTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name)/'graph.json'
        self.graph = {'object_nodes': {'nodes': {'1': {'position': [0, 0, 0]}}},
                      'part_nodes': {'part_1': {'position': [.1, 0, 0], 'parent_id': '1', 'status': 'confirmed', 'name': 'leg'},
                                     'part_2': {'position': [.2, 0, 0], 'parent_id': None, 'status': 'confirmed', 'name': 'leg'},
                                     'part_3': {'position': [.3, 0, 0], 'parent_id': '1', 'status': 'provisional', 'name': 'leg'}},
                      'part_relations': [{'source_id': 'part_1', 'target_id': '1'}]}

    def tearDown(self):
        self.temp.cleanup()

    def build(self, **kwargs):
        self.path.write_text(json.dumps(self.graph))
        return build_part_overlay(self.path, {1: [.4, .6, .8]}, .035, **kwargs)

    def test_default_shows_only_confirmed_parented_parts(self):
        geometry, picks, stats = self.build()
        self.assertEqual([p['id'] for p in picks], ['part_1'])
        self.assertEqual(stats['observed_part_points'], 0)
        self.assertEqual(stats['part_of_links'], 1)
        self.assertEqual([kind for kind, _ in geometry], ['mesh', 'line'])

    def test_provisional_visibility_and_world_bias(self):
        _, picks, _ = self.build(include_provisional=True, bias=2)
        self.assertEqual(len(picks), 3)
        np.testing.assert_allclose(picks[0]['position'], [2.1, 0, 0])

    def test_inconsistent_parent_is_rejected(self):
        self.graph['part_relations'][0]['target_id'] = '2'
        with self.assertRaises(ValueError):
            self.build()

    def test_saved_points_are_only_loaded_on_request(self):
        np.save(self.path.parent/'part_1.points.npy', [[.1, 0, 0], [.1, 0, .02]])
        _, _, stats = self.build(show_points=True)
        self.assertEqual(stats['observed_part_points'], 2)

    def test_nonfinite_positions_are_rejected(self):
        self.graph['part_nodes']['part_1']['position'][0] = float('nan')
        with self.assertRaises(ValueError):
            self.build()


if __name__ == '__main__':
    unittest.main()

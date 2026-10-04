"""Check association, hierarchy evidence, evaluation and original graph interfaces."""
import json
from pathlib import Path
import sys
import unittest
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'scannet/script'))
sys.path.insert(0, str(ROOT / 'script/include'))
from pipeline_components.multiview_association import assign, PartFusion
from evaluate_hypersim import average_precision, box_overlap, node_bounds
from topology_map import TopologyMap


class PipelineChecks(unittest.TestCase):
    def test_global_assignment_beats_greedy_and_keeps_unmatched(self):
        self.assertEqual(assign(np.array([[.9, .8], [.85, -.1], [-1e6, -1e6]])), {0: 1, 1: 0})

    def test_parent_evidence_recovers_initial_orphan(self):
        fusion = PartFusion()
        points = np.array([[0, 0, 0], [.01, 0, 0], [0, .01, 0], [0, 0, .01]])
        feature, color = np.ones(1024), np.ones((4, 32))/32
        for frame, parent in [('a', None), ('b', '7'), ('c', '7')]:
            fusion.add(frame, 0, 'cabinet: handle', parent, points, feature, color, .9)
        nodes, edges = fusion.export()
        self.assertEqual(len(nodes), 1)
        self.assertEqual(nodes['part_1']['parent_id'], '7')
        self.assertEqual(len(edges), 1)
        self.assertEqual(edges[0]['target_id'], '7')
        fusion.add('c', 1, 'cabinet: handle', '7', points, feature, color, .9)
        self.assertEqual(len(fusion.tracks), 2)

    def test_invalid_or_unseen_pairs_are_never_forced(self):
        self.assertEqual(assign(np.full((2, 3), -1e6)), {})
        self.assertEqual(assign(np.zeros((0, 0))), {})

    def test_duplicate_prediction_counts_as_false_positive_in_ap(self):
        self.assertAlmostEqual(average_precision(np.array([[1., 0], [1., 0], [0, 1.]]), [.9, .8, .7], .5), 5/6)

    def test_iop_denominator_is_prediction_volume(self):
        gt = (np.zeros(3), np.ones(3)*2)
        prediction = (np.zeros(3), np.ones(3))
        self.assertEqual(box_overlap(gt, prediction), (.125, 1.))

    def test_evaluation_uses_canonical_graph_box(self):
        node = {'position': [1, 2, 3], 'shape': {'length': 2, 'width': 4, 'height': 6,
                'orientation': {'x': 0, 'y': 0, 'z': 0, 'w': 1}}}
        lower, upper = node_bounds(node)
        np.testing.assert_array_equal(lower, [0, 0, 0])
        np.testing.assert_array_equal(upper, [2, 4, 6])

    def test_legacy_and_part_graph_loading(self):
        graph = TopologyMap()
        graph.read_from_json(json.dumps({'object_nodes': {'nodes': {}}, 'edge_hypotheses': {}}))
        self.assertEqual(graph.get_parts('7'), [])
        graph.read_from_json(json.dumps({'part_nodes': {'part_1': {'id': 'part_1', 'parent_id': '7', 'status': 'confirmed'}},
                                        'part_relations': [{'source_id': 'part_1', 'target_id': '7', 'description': 'part_of'}]}))
        self.assertEqual(len(graph.get_parts('7')), 1)
        self.assertEqual(graph.get_entity('part_1')['id'], 'part_1')


if __name__ == '__main__':
    unittest.main()

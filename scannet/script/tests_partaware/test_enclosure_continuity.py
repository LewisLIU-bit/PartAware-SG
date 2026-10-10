"""Counterexamples for closed-body corner assembly."""
import unittest
import numpy as np
from pipeline_components.enclosure_continuity import relation


class EnclosureContinuityTests(unittest.TestCase):
    def setUp(self):
        y, z = np.meshgrid(np.linspace(0., 1., 70), np.linspace(1., 2., 70))
        self.source = np.column_stack([np.zeros(y.size), y.ravel(), z.ravel()])
        x, z = np.meshgrid(np.linspace(0., 1.5, 90), np.linspace(1., 2., 90))
        self.body = np.column_stack([x.ravel(), np.zeros(x.size), z.ravel()])
        self.node = dict(storage_identity=dict(front_occupancy=.9,
            cached_cabinet_frames=['a', 'b', 'c']), structural_surface_recovery=dict(measured=True))

    def measure(self, source=None, body=None, semantic=.95, shared=None, node=None):
        return relation(self.source if source is None else source, self.body if body is None else body,
            self.node if node is None else node, self.node, semantic,
            {'a', 'b', 'c'} if shared is None else shared)

    def test_continuous_closed_corner_retains_original_coordinates(self):
        before = self.source.copy()
        result = self.measure()
        self.assertIsNotNone(result)
        self.assertEqual(result['generated_points'], 0)
        self.assertTrue(np.array_equal(before, self.source))

    def test_neighboring_disconnected_enclosures_remain_separate(self):
        self.assertIsNone(self.measure(body=self.body+[0., -.12, 0.]))

    def test_coplanar_sections_are_not_a_corner(self):
        self.assertIsNone(self.measure(body=self.body[:, [1, 0, 2]]))

    def test_different_vertical_boundaries_remain_separate(self):
        self.assertIsNone(self.measure(body=self.body+[0., 0., .15]))

    def test_short_contact_cannot_join_whole_body(self):
        source = self.source.copy()
        source[source[:, 2] > 1.2, 0] = -.2
        self.assertIsNone(self.measure(source=source))

    def test_open_rack_does_not_supply_closed_identity(self):
        node = dict(storage_identity=dict(front_occupancy=.12, cached_cabinet_frames=['a', 'b', 'c']),
                    structural_surface_recovery=dict(measured=True))
        self.assertIsNone(self.measure(node=node))

    def test_semantic_disagreement_cannot_merge_by_contact(self):
        self.assertIsNone(self.measure(semantic=.3))

    def test_single_view_cannot_merge_by_contact(self):
        self.assertIsNone(self.measure(shared={'a'}))

    def test_replay_retains_canonical_assembly_proof_without_new_merge(self):
        from pipeline_components.enclosure_continuity import retain_evidence
        graph = dict(object_nodes=dict(nodes={'body': {}}),
            part_nodes={'section': dict(point_count=1000)}, object_identity_aliases={
                'source': dict(canonical_id='body', part_id='section',
                    evidence=dict(evidence_type='measured_closed_enclosure_corner'))})
        first, repeated = dict(objects=[]), dict(objects=[])
        retain_evidence(graph, first); retain_evidence(graph, repeated)
        self.assertEqual(first, repeated)
        self.assertEqual(first['applied_assemblies'], 0)
        self.assertEqual(len(first['objects']), 1)

    def test_missing_canonical_body_cannot_retain_stale_proof(self):
        from pipeline_components.enclosure_continuity import retain_evidence
        graph = dict(object_nodes=dict(nodes={}), part_nodes={'section': dict(point_count=1000)},
            object_identity_aliases={'source': dict(canonical_id='missing', part_id='section',
                evidence=dict(evidence_type='measured_closed_enclosure_corner'))})
        report = dict(objects=[]); retain_evidence(graph, report)
        self.assertEqual(report['objects'], [])


if __name__ == '__main__':
    unittest.main()

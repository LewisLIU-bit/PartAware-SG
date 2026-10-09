"""Boundary association must preserve ambiguous and separate neighboring bodies."""
import unittest
import numpy as np
from pipeline_components.part_body_assembly import boundary_relation, unique_parents


class PartBodyChecks(unittest.TestCase):
    def setUp(self):
        y, z = np.meshgrid(np.linspace(-.5, .5, 12), np.linspace(-.5, .5, 12))
        self.panel = np.column_stack([np.ones(y.size)*.2, y.ravel(), z.ravel()])
        self.body = np.concatenate([self.panel, self.panel*np.array([-1, 1, 1])])

    def check(self, source=None, body=None, semantic=.95, frames=(1, 2, 3)):
        return boundary_relation(self.panel if source is None else source,
            self.body if body is None else body, np.eye(3), np.zeros(3), semantic, frames)

    def test_measured_boundary_panel_can_become_queryable_part(self):
        self.assertIsNotNone(self.check())

    def test_same_named_surface_outside_body_is_retained(self):
        self.assertIsNone(self.check(source=self.panel+np.array([.2, 0, 0])))

    def test_inside_sheet_without_boundary_evidence_is_retained(self):
        self.assertIsNone(self.check(source=self.panel*np.array([0, 1, 1])))

    def test_thin_anchor_does_not_invent_a_volumetric_body(self):
        self.assertIsNone(self.check(body=self.body*np.array([.05, 1, 1]),
                                     source=self.panel*np.array([.05, 1, 1])))

    def test_semantic_and_multiview_evidence_are_required(self):
        self.assertIsNone(self.check(semantic=.3))
        self.assertIsNone(self.check(frames=(1, 2)))

    def test_ambiguous_or_dependent_parentage_is_not_resolved_by_score(self):
        self.assertEqual(unique_parents({'a': [('b', {}), ('c', {})]}), {})
        self.assertEqual(unique_parents({'a': [('b', {})], 'b': [('c', {})]}), {'b': ('c', {})})


if __name__ == '__main__':
    unittest.main()

"""Protect independent objects and unknown views from erroneous whole-body joins."""
import unittest
import numpy as np
from pipeline_components.whole_object_consensus import acceptance, geometry_evidence, visual_identity


class WholeObjectChecks(unittest.TestCase):
    def setUp(self):
        self.e = dict(whole_mask_support_views=5, independent_separation_views=0,
                      joint_visible_views=5, whole_mask_consensus=1.)
        self.g = dict(box_iou=.92, bidirectional_surface_coverage=.85,
                      closest_surface_m=.005, aligned_terminal_face=False)

    def test_visual_identity_can_bridge_names_but_not_weak_semantics(self):
        metric = dict(semantic_views=2, semantic_camera_baseline_m=.12, semantic_best_class='device',
                      semantic_object_score=.25, semantic_background_score=.2)
        self.assertTrue(visual_identity(metric, metric, .9))
        self.assertFalse(visual_identity(metric, {**metric, 'semantic_views': 1}, .9))
        self.assertFalse(visual_identity(metric, {**metric, 'semantic_camera_baseline_m': .01}, .9))
        self.assertFalse(visual_identity(metric, {**metric, 'semantic_best_class': 'container'}, .9))
        self.assertFalse(visual_identity(metric, {**metric, 'semantic_background_score': .26}, .9))

    def test_shared_surface_without_new_visual_identity(self):
        self.assertEqual(acceptance(self.e, self.g, .7, False), 'shared_measured_surface')

    def test_complementary_body_needs_actual_whole_views(self):
        g = {**self.g, 'box_iou': .7, 'bidirectional_surface_coverage': .1}
        self.assertEqual(acceptance(self.e, g, .2, True), 'complementary_measured_body')
        self.assertIsNone(acceptance({**self.e, 'whole_mask_support_views': 0}, g, .9, True))

    def test_separate_whole_instances_veto_complementary_join(self):
        self.assertIsNone(acceptance({**self.e, 'independent_separation_views': 2}, self.g, 1., True))

    def test_terminal_face_requires_shared_view_and_has_separation_veto(self):
        g = {**self.g, 'box_iou': .05, 'aligned_terminal_face': True, 'closest_surface_m': .04}
        e = {**self.e, 'whole_mask_support_views': 1, 'joint_visible_views': 2,
             'whole_mask_consensus': .5, 'independent_separation_views': 1}
        self.assertEqual(acceptance(e, g, .2, True), 'aligned_measured_terminal_face')
        self.assertIsNone(acceptance({**e, 'independent_separation_views': 2}, g, .2, True))
        self.assertIsNone(acceptance({**e, 'joint_visible_views': 1}, g, .2, True))

    def test_container_contents_cannot_merge_from_semantics_or_proximity(self):
        g = {**self.g, 'box_iou': .01, 'bidirectional_surface_coverage': 0.}
        self.assertIsNone(acceptance(self.e, g, 1., True))

    def test_cross_section_requires_comparable_extents(self):
        rng = np.random.default_rng(12)
        body = rng.uniform([0., 0., 0.], [.4, .3, .08], (1024, 3))
        small = rng.uniform([.1, .31, .02], [.2, .32, .04], (512, 3))
        self.assertFalse(geometry_evidence(body, small)['aligned_terminal_face'])

    def test_long_rectangular_face_is_planar_despite_unequal_lateral_variance(self):
        rng = np.random.default_rng(31)
        body = rng.uniform([0., 0., 0.], [.45, .3, .08], (4096, 3))
        face = rng.uniform([0., .3, 0.], [.45, .304, .08], (2048, 3))
        self.assertTrue(geometry_evidence(body, face)['aligned_terminal_face'])

    def test_measured_face_can_have_a_minority_of_edge_returns(self):
        rng = np.random.default_rng(7)
        body = rng.uniform([0., 0., 0.], [.45, .3, .08], (4096, 3))
        rear = rng.uniform([0., .3, 0.], [.45, .304, .08], (1600, 3))
        edge = rng.uniform([0., .25, 0.], [.45, .29, .08], (300, 3))
        self.assertTrue(geometry_evidence(body, np.concatenate([rear, edge]))['aligned_terminal_face'])


if __name__ == '__main__':
    unittest.main()

"""Check MVO matching, appearance rescue and occluded cuboid safety."""
import unittest
import numpy as np
from evaluate_hypersim import max_volume_overlap, average_precision, mvo_metrics
from pipeline_components.observed_consensus import confirms_existence, confirms_core
from pipeline_components.backed_cuboid import fit_hypothesis, verify


class V9Checks(unittest.TestCase):
    def test_mvo_is_symmetric_and_iou_identical_for_containment(self):
        a = np.array([[0, 0, 0], [1, 1, 1.]])
        b = np.array([[0, 0, 0], [2, 2, 2.]])
        self.assertEqual(max_volume_overlap(a, b), .125)
        self.assertEqual(max_volume_overlap(b, a), .125)

    def test_mvo_does_not_ignore_missing_thickness(self):
        a = np.array([[0, 0, 0], [1, 1, .01]])
        b = np.array([[0, 0, 0], [1, 1, 1.]])
        self.assertEqual(max_volume_overlap(a, b), .01)

    def test_mvo_strict_threshold_and_duplicate_penalty(self):
        m = np.array([[.5], [.6]])
        self.assertEqual(average_precision(m, [.9, .8], .5), 1.)
        self.assertEqual(average_precision(m, [.9, .8], .5, strict=True), .5)

    def test_empty_predictions_are_not_perfect(self):
        result = mvo_metrics([], [{'id': 1, 'bounds': [[0, 0, 0], [1, 1, 1]]}])
        self.assertEqual(result['AP50'], 0)
        self.assertEqual(result['one_to_one']['0.5']['FN'], 1)

    def test_geometry_does_not_rescue_mixed_or_single_position_views(self):
        e = {'joint_depth_mask_support_views': 8, 'joint_depth_mask_supported_fraction': .9, 'camera_baseline_m': .2}
        self.assertTrue(confirms_existence(e, .0))
        self.assertFalse(confirms_existence(e, .3))
        e['camera_baseline_m'] = 0
        self.assertFalse(confirms_existence(e, .0))

    def test_pure_front_sheet_cannot_infer_depth_from_a_distant_wall(self):
        y, z = np.meshgrid(np.linspace(0, 1, 40), np.linspace(0, 1, 40))
        front = np.column_stack([np.zeros(y.size), y.ravel(), z.ravel()])
        back = front+[-.7, 0, 0]
        candidate, evidence = fit_hypothesis(front, back, np.array([[2., .5, .5]]))
        self.assertIsNone(candidate)

    def test_semantic_rescue_needs_a_spanning_core_not_an_isolated_line(self):
        evidence = {'joint_depth_mask_support_views': 12, 'camera_baseline_m': .3}
        core = {'verified_core_points': 300, 'source_points': 800, 'axis_extent_retention': [.9, .7, .3]}
        self.assertTrue(confirms_core(evidence, core, .0))
        core['axis_extent_retention'] = [.9, .2, .1]
        self.assertFalse(confirms_core(evidence, core, .0))
        core['axis_extent_retention'] = [.9, .7, .3]
        self.assertFalse(confirms_core(evidence, core, .3))

    def test_measured_free_space_rejects_a_frontward_generated_surface(self):
        class Views:
            jobs = {'a': None, 'b': None}
            records = {'a': [], 'b': []}
            kd = kc = np.array([[1., 0, 2], [0, 1., 2], [0, 0, 1.]])
            scale = 1.
            def get(self, fid): return np.eye(4), np.ones((5, 5))*2, np.zeros((5, 5), np.uint8)
        extra, evidence = verify(np.array([[0., 0, 1.]]), '1', Views(), np.empty((0, 3)))
        self.assertIsNone(extra)
        self.assertEqual(evidence['repeated_free_space_conflict_fraction'], 1.)


if __name__ == '__main__': unittest.main()

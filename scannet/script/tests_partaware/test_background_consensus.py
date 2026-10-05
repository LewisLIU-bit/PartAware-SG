"""Verify that occlusion/unknown views cannot reject a measured object."""
import unittest
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from pipeline_components.background_consensus import measure, rejection


class BackgroundConsensusChecks(unittest.TestCase):
    def test_small_objects_are_not_subject_to_large_region_rule(self):
        result = measure(np.array([[0, 0, 0], [1, 1, 1]]), object())
        self.assertFalse(result['applicable'])
        self.assertFalse(rejection(result))

    def test_repeated_occlusion_does_not_cast_negative_votes(self):
        class Views:
            jobs = {str(i): {} for i in range(8)}
            def project(self, points, fid):
                return np.array([], int), np.array([]), np.array([], int)
        result = measure(np.column_stack([np.linspace(0, 4, 100), np.zeros((100,2))]), Views())
        self.assertEqual(result['visible_views'], 0)
        self.assertFalse(rejection(result))

    def test_visible_background_requires_multiple_views_and_majority(self):
        class Views:
            jobs = {str(i): {} for i in range(8)}
            def project(self, points, fid):
                return np.arange(len(points)), np.ones(len(points)), np.zeros(len(points), int)
        result = measure(np.column_stack([np.linspace(0, 4, 100), np.zeros((100,2))]), Views())
        self.assertTrue(rejection(result))
        result.update(holdout_background_views=4)
        self.assertFalse(rejection(result))
        result.update(holdout_background_views=5, holdout_background_point_fraction=.5)
        self.assertFalse(rejection(result))

    def test_source_masks_cannot_validate_themselves(self):
        class Views:
            jobs = {str(i): {} for i in range(8)}
            def project(self, points, fid):
                return np.arange(len(points)), np.ones(len(points)), np.zeros(len(points), int)
        result = measure(np.column_stack([np.linspace(0, 4, 100), np.zeros((100,2))]), Views(), list(Views.jobs))
        self.assertFalse(rejection(result))
        self.assertEqual(result['holdout_supported_point_fraction'], 0.)

if __name__ == '__main__':
    unittest.main()

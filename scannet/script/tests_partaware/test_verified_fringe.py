"""Keep unobserved surfaces while removing repeatedly contradicted fringes."""
from types import SimpleNamespace
from unittest.mock import patch
import unittest
import numpy as np
from pipeline_components.observed_consensus import refine_verified_surface


class VerifiedFringeChecks(unittest.TestCase):
    def fixture(self):
        rng = np.random.default_rng(12)
        points = rng.uniform(0, 1, (1000, 3))
        views = SimpleNamespace(jobs={str(i): {} for i in range(6)}, scale=1., kd=np.eye(3), kc=np.eye(3),
            get=lambda f: (np.eye(4), np.ones((2, 2)), np.ones((2, 2), int)))
        records = {fid: [{'frame_instance_id': 1, 'instance_id': 1}] for fid in views.jobs}
        return points, views, records

    def test_unseen_points_are_preserved_beside_a_confirmed_core(self):
        points, views, records = self.fixture()
        visible = np.arange(1000) < 950
        own = np.arange(1000) < 900
        with patch('pipeline_components.observed_consensus.depth_mask_votes', return_value=(visible, own)):
            result, evidence = refine_verified_surface(points, '1', views, records, lambda g, r: g, {})
        self.assertTrue(evidence['accepted'])
        self.assertEqual(len(result), 950)
        self.assertTrue(np.array_equal(result[-50:], points[-50:]))

    def test_a_weak_or_mostly_missing_core_is_not_trimmed(self):
        points, views, records = self.fixture()
        visible = np.ones(1000, bool); own = np.arange(1000) < 600
        with patch('pipeline_components.observed_consensus.depth_mask_votes', return_value=(visible, own)):
            result, evidence = refine_verified_surface(points, '1', views, records, lambda g, r: g, {})
        self.assertFalse(evidence['accepted'])
        self.assertTrue(np.array_equal(result, points))


if __name__ == '__main__': unittest.main()

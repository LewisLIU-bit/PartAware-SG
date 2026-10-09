"""Regression checks for native whole identity and repeated-point contradiction."""
import unittest
from unittest.mock import patch
from types import SimpleNamespace
import numpy as np

from pipeline_components.hierarchical_masks import Evidence, acceptance


class NativeWholeIdentityChecks(unittest.TestCase):
    def test_exclusive_subdivision_does_not_disprove_native_whole_identity(self):
        evidence = dict(applicable=True, whole_identity_views=7,
                        whole_identity_consensus=.8, camera_baseline_m=.4)
        self.assertTrue(acceptance(evidence))
        self.assertFalse(acceptance({**evidence, 'whole_identity_consensus': .3}))
        self.assertFalse(acceptance({**evidence, 'whole_identity_views': 2}))
        self.assertFalse(acceptance({**evidence, 'camera_baseline_m': .001}))

    def test_small_stacked_items_are_not_exempt_from_atomic_instance_validation(self):
        views = SimpleNamespace()
        with patch('pipeline_components.hierarchical_masks.NativeMasks') as mock:
            mock.return_value.available = True
            instance = Evidence(views, {}, {})
            points = np.random.default_rng(9).uniform(0, .2, (256, 3))
            measured = instance.measure(points, 'stack')
            self.assertFalse(measured['applicable'])
            mock.return_value.mass.assert_not_called()

    def test_another_concept_cannot_rescue_a_candidate_identity(self):
        with patch('pipeline_components.hierarchical_masks.NativeMasks') as mock:
            mock.return_value.mass.return_value = np.array([.99])
            mock.return_value.records.return_value = [{'object_name':'table','confidence':.9}]
            instance = Evidence(SimpleNamespace(), {}, {})
            self.assertIsNone(instance.best_mask(np.ones((32, 3)), 'frame', {'cabinet'}))

    def test_repeated_native_contradiction_trims_but_unknown_points_survive(self):
        points = np.array([[1., 1., 1.], [2., 1., 1.], [5., 5., 1.]])
        views = SimpleNamespace(kc=np.eye(3))
        views.project = lambda p, fid: (np.array([0,1]), np.ones(2), np.array([1,2]))
        views.get = lambda fid: (np.eye(4), None, None)
        with patch('pipeline_components.hierarchical_masks.NativeMasks'):
            instance = Evidence(views, {}, {})
        instance.decisions['body'] = {'frames':[dict(frame_id=str(i),accepted=True,native_mask_index=0) for i in range(5)]}
        mask = np.zeros((8,8), bool); mask[1,1] = True
        instance.mask = lambda fid, index: mask
        refined, evidence = instance.refine(points, 'body')
        np.testing.assert_array_equal(refined, points[[0,2]])
        self.assertEqual(evidence['unknown_points_retained'], 1)
        self.assertEqual(evidence['removed_points'], 1)


if __name__ == '__main__':
    unittest.main()

"""Verify physical-instance conflict guards and geometric interface invariants."""
import unittest
import numpy as np
from scipy.spatial.transform import Rotation
from pipeline_components.canonical_geometry import fit_box
from pipeline_components.proposal_validation import standardized_maximum, inclusion, raw_pair_evidence
from pipeline_components.yoloe_frontend import union_masks
from pipeline_components.instance_consensus import pair_evidence
from pipeline_components.shape_completion import accept_candidate, align_candidate


class V4Checks(unittest.TestCase):
    def test_upright_box_contains_rotated_surface_and_preserves_vertical(self):
        rng = np.random.default_rng(20)
        rotation = Rotation.from_euler('z', .4).as_matrix()
        points = rng.uniform(-.5, .5, (1000, 3))*[2, 1, .4]
        points = points @ rotation.T+[1, 2, 3]
        center, shape = fit_box(points, True)
        r = Rotation.from_quat([shape['orientation'][k] for k in ['x', 'y', 'z', 'w']]).as_matrix()
        local = (points-center) @ r
        extent = np.array([shape['length'], shape['width'], shape['height']])
        self.assertTrue(np.all(np.abs(local) <= extent/2+1e-7))
        np.testing.assert_allclose(r[:, 2], [0, 0, 1], atol=1e-8)

    def test_semantic_validation_is_per_class_and_finite_for_constant_scores(self):
        values = np.array([[.9, .2], [.6, .1], [.1, .8]])
        scores = standardized_maximum(values)
        self.assertGreater(scores[0], 0)
        self.assertGreater(scores[2], 0)
        self.assertTrue(np.isfinite(standardized_maximum(np.ones((5, 2)))).all())

    def test_inclusion_is_directional_and_adjacent_objects_are_not_contained(self):
        big = np.array([[0, 0, 0], [1, 1, 1]])
        small = np.array([[.4, .4, .4], [.6, .6, .6]])
        self.assertEqual(inclusion(small, big), 1.)
        self.assertEqual(inclusion(big, small), 0.)
        self.assertEqual(inclusion(small+[2, 0, 0], big), 0.)

    def test_candidate_union_preserves_unmatched_detection(self):
        label = np.zeros((20, 20), np.uint8)
        label[1:6, 1:8] = 1
        new = np.zeros_like(label, bool)
        new[10:18, 10:18] = True
        records, masks = union_masks([{'frame_instance_id': 1, 'confidence': .8}], label,
                                    [({'confidence': .6}, new)])
        self.assertEqual(len(records), 2)
        self.assertTrue(np.any(masks[label == 1] > 0))
        self.assertTrue(np.any(masks[new] > 0))

    def test_independent_visible_objects_veto_duplicate_merge(self):
        a = type('Track', (), {'points': np.zeros((32, 3)), 'frames': {'a', 'b'}})()
        b = type('Track', (), {'points': np.ones((32, 3)), 'frames': {'a', 'b'}})()
        class Views:
            def project(self, points, fid):
                labels = np.ones(32, int)*(1 if points[0, 0] == 0 else 2)
                return np.arange(32), np.ones(32), labels
        evidence = pair_evidence(a, b, Views())
        self.assertEqual(evidence['support_views'], 0)
        self.assertEqual(evidence['conflict_views'], 2)

    def test_completion_free_space_and_independent_mask_are_rejected(self):
        xy = np.array([[x, y] for x in np.arange(-.12, .121, .025) for y in np.arange(-.12, .121, .025)])
        observed = np.column_stack([xy, np.ones(len(xy))])
        candidate = np.concatenate([observed, observed+[0, 0, -.08]])
        track = {'observations': [{'frame_id': 'a', 'local_id': 1}]}
        class Views:
            kd = kc = np.array([[100., 0, 50], [0, 100., 50], [0, 0, 1]])
            scale = 1000.
            def __init__(self, label):
                self.label = label
            def get(self, fid):
                return np.eye(4), np.full((100, 100), 1000.), np.full((100, 100), self.label)
        extra, audit = accept_candidate(observed, candidate, Views(1), track, ['a'])
        self.assertIsNone(extra)
        self.assertGreater(audit['free_space_violation_fraction'], .03)
        candidate = np.concatenate([observed, observed+[0, 0, .08]])
        extra, audit = accept_candidate(observed, candidate, Views(2), track, ['a'])
        self.assertIsNone(extra)
        self.assertGreater(audit['visible_mask_conflict_fraction'], .05)

    def test_completion_alignment_improves_observed_surface_fit(self):
        rng = np.random.default_rng(31)
        observed = rng.uniform(-.1, .1, (200, 3))
        candidate = observed+[.01, -.01, .005]
        fitted, metadata = align_candidate(observed, candidate)
        self.assertLess(np.mean(np.linalg.norm(fitted-observed, axis=1)), .01)
        self.assertGreaterEqual(metadata['sim3_scale'], .8-1e-8)
        self.assertLessEqual(metadata['sim3_scale'], 1.25+1e-8)


if __name__ == '__main__':
    unittest.main()

"""Guard against suppressing occluded objects or merging separately visible items."""
import unittest
from unittest.mock import patch
from types import SimpleNamespace
import numpy as np
from pipeline_components.visibility_ownership import measure, contradicted, prune, maximal_containers


class OwnershipChecks(unittest.TestCase):
    def test_partial_extent_does_not_make_one_full_owner_ambiguous(self):
        geometry = {'whole': np.array([[0., 0, 0], [1., 1, 1]]),
                    'patch': np.array([[.2, .2, .2], [.4, .4, .4]])}
        candidates = [('whole', .7, 4.), ('patch', .9, 50.)]
        self.assertEqual(maximal_containers(candidates, geometry), [candidates[0]])

    def test_two_independent_extents_still_require_additional_evidence(self):
        geometry = {'left': np.array([[0., 0, 0], [1., 1, 1]]),
                    'right': np.array([[2., 0, 0], [3., 1, 1]])}
        candidates = [('left', .7, 4.), ('right', .9, 4.)]
        self.assertEqual(maximal_containers(candidates, geometry), candidates)

    def test_same_identity_fragment_uses_repeated_whole_evidence(self):
        points = np.array([[x,y,z] for x in np.linspace(0,1,7)
                           for y in np.linspace(0,1,7) for z in np.linspace(0,1,7)])
        for fine, separate, expected in [(False, 1, ['2']), (False, 2, ['1','2']), (True, 1, ['1','2'])]:
            with self.subTest(fine=fine, independent_views=separate):
                geometry = {'1': points[np.all(points <= .5, axis=1)].copy(), '2': points.copy()}
                views, records = self.fixture(lambda p,f: np.full(len(p), 2))
                tracks = {gid: {'observations': [], 'observed_frames': ['0','1','2'],
                               'reprojection_support': .7, 'fine_scale_instance': fine if gid=='1' else False}
                          for gid in geometry}
                evidence = {'whole_mask_support_views': 2, 'independent_separation_views': separate,
                            'joint_visible_views': 8, 'whole_mask_consensus': .25}
                nodes = {gid: {'text_embedding': [1.,0]} for gid in geometry}
                with patch('pipeline_components.proposal_validation.raw_pair_evidence', return_value=evidence):
                    remap = {}
                    result = prune(list(geometry), geometry, tracks, {gid:{} for gid in geometry},
                                   views, records, remap, [], nodes)
                self.assertEqual(result, expected)
                if expected == ['2']: self.assertEqual(remap, {'1':'2'})

    def fixture(self, labels, count=8):
        class Views:
            jobs = {str(i): {} for i in range(count)}
            def project(self, points, fid):
                values = labels(points, fid)
                return np.arange(len(points)), np.ones(len(points)), values
        views = Views()
        records = {fid: [{'frame_instance_id': i, 'instance_id': i} for i in [1, 2, 3]] for fid in views.jobs}
        return views, records

    def test_unknown_visibility_is_not_a_negative_measurement(self):
        views = SimpleNamespace(jobs={str(i): {} for i in range(8)},
            project=lambda *a: (np.array([], int), np.array([]), np.array([], int)))
        points = np.zeros((100, 3))
        evidence = measure(points, '1', views, {fid: [] for fid in views.jobs}, lambda g, r: g, {})
        self.assertFalse(contradicted(evidence))
        self.assertEqual(evidence['visible_point_fraction'], 0)

    def test_repeated_visible_background_can_veto_even_a_small_proposal(self):
        views, records = self.fixture(lambda p, f: np.zeros(len(p), int))
        evidence = measure(np.zeros((100, 3)), '1', views, records, lambda g, r: g, {})
        self.assertTrue(contradicted(evidence))

    def test_two_positive_views_do_not_overrule_repeated_contradiction(self):
        views, records = self.fixture(lambda p, f: np.full(len(p), 1 if int(f) < 2 else 0), 12)
        evidence = measure(np.zeros((100, 3)), '1', views, records, lambda g, r: g, {})
        self.assertTrue(contradicted(evidence))
        views, records = self.fixture(lambda p, f: np.full(len(p), 1 if int(f) < 2 else 0), 5)
        evidence = measure(np.zeros((100, 3)), '1', views, records, lambda g, r: g, {})
        self.assertFalse(contradicted(evidence))

    def test_union_proposal_is_removed_but_both_independent_objects_survive(self):
        cube = np.array([[x, y, z] for x in np.linspace(0, .1, 5)
                         for y in np.linspace(0, .1, 5) for z in np.linspace(0, .1, 5)])
        geometry = {'1': np.concatenate([cube, cube+[.6, 0, 0]]), '2': cube, '3': cube+[.6, 0, 0]}
        views, records = self.fixture(lambda p, f: np.full(len(p), 1) if f == '0'
            else np.where(p[:, 0] < .3, 2, 3), 3)
        tracks = {gid: {'reprojection_support': 1., 'observed_frames': ['0', '1', '2']} for gid in geometry}
        result = prune(list(geometry), geometry, tracks, {gid: {} for gid in geometry}, views, records, {}, [])
        self.assertEqual(result, ['2', '3'])

    def test_shared_masks_are_not_independent_object_evidence(self):
        cube = np.array([[x,y,z] for x in np.linspace(0,.1,5)
                         for y in np.linspace(0,.1,5) for z in np.linspace(0,.1,5)])
        geometry = {'1': np.concatenate([cube, cube+[.6,0,0]]), '2': cube, '3': cube+[.6,0,0]}
        views, records = self.fixture(lambda p, f: np.full(len(p), 1), 3)
        tracks = {gid: {'reprojection_support': 1., 'observed_frames': ['0', '1', '2']} for gid in geometry}
        result = prune(list(geometry), geometry, tracks, {gid: {} for gid in geometry}, views, records, {}, [])
        self.assertIn('1', result)


if __name__ == '__main__': unittest.main()



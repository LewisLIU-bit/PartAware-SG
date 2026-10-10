"""Counterexamples for identity-seeded recovery and verified shape priors."""
import unittest

import numpy as np
import open3d as o3d

from pipeline_components.seeded_surfaces import proposals
from pipeline_components.verified_cuboids import cuboid_identity, fit


class EvidenceShapeTests(unittest.TestCase):
    def setUp(self):
        o3d.utility.random.seed(14)

    @staticmethod
    def face(axis, level, lower, upper, count=55):
        a, b = np.meshgrid(np.linspace(lower[0], upper[0], count),
                           np.linspace(lower[1], upper[1], count))
        points = np.empty((a.size, 3))
        points[:, axis] = level
        remaining = [i for i in range(3) if i != axis]
        points[:, remaining[0]], points[:, remaining[1]] = a.ravel(), b.ravel()
        return points

    def tracks(self, name='countertop'):
        return {'seed': dict(name_votes={name: 3}, confidence=.8, observed_frames=['a', 'b', 'c'])}

    def test_large_floor_does_not_starve_support_seed(self):
        slab = self.face(2, 1., [-1., -.5], [1., .5])
        floor = self.face(2, 0., [-4., -4.], [4., 4.], count=130)
        found = proposals(np.concatenate([slab, floor]), {'seed': slab}, self.tracks())
        self.assertEqual(len(found), 1)
        self.assertTrue(np.allclose(found[0][0][:, 2], 1.))

    def test_disconnected_coplanar_wall_is_not_owned_by_seed(self):
        cabinet = self.face(0, 1., [-.6, 1.], [.6, 2.])
        wall = self.face(0, 1., [3., 1.], [4.2, 2.])
        found = proposals(np.concatenate([cabinet, wall]), {'seed': cabinet}, self.tracks('cabinet'))
        self.assertEqual(len(found), 1)
        self.assertLess(found[0][0][:, 1].max(), 1.)

    def test_wall_identity_cannot_propose_cabinet(self):
        wall = self.face(0, 1., [-1., 0.], [1., 2.])
        self.assertEqual(proposals(wall, {'seed': wall}, self.tracks('wall')), [])

    def test_single_frame_does_not_seed_recovery(self):
        slab = self.face(2, 1., [-1., -.5], [1., .5])
        tracks = self.tracks()
        tracks['seed']['observed_frames'] = ['a']
        self.assertEqual(proposals(slab, {'seed': slab}, tracks), [])

    def test_low_confidence_does_not_seed_recovery(self):
        slab = self.face(2, 1., [-1., -.5], [1., .5])
        tracks = self.tracks()
        tracks['seed']['confidence'] = .1
        self.assertEqual(proposals(slab, {'seed': slab}, tracks), [])

    def test_identity_selects_shape_family_not_fixed_dimensions(self):
        for name in ['refrigerator', 'wall cabinet', 'cupboard', 'wardrobe']:
            self.assertTrue(cuboid_identity(name))
        for name in ['wall', 'floor', 'plate', 'open shelf', 'sink', 'chair']:
            self.assertFalse(cuboid_identity(name))

    def test_partial_without_rear_remains_unknown(self):
        front = self.face(0, 1., [-.5, 0.], [.5, 1.5])
        candidate, _ = fit(front, np.empty((0, 3)), np.array([[2., 0., 1.]]))
        self.assertIsNone(candidate)

    def test_existing_depth_does_not_disable_missing_surface_completion(self):
        front = self.face(0, 1., [-.5, 0.], [.5, 1.5])
        side = self.face(1, .5, [.41, 0.], [.99, 1.5])
        rear = self.face(0, .4, [-.6, 0.], [.6, 1.5])
        observed = np.concatenate([front, side])
        before = observed.copy()
        candidate, evidence = fit(observed, rear, np.array([[2., 0., 1.]]))
        self.assertIsNotNone(candidate)
        self.assertFalse(evidence['learned_text_conditioning'])
        self.assertTrue(np.array_equal(before, observed))
        self.assertLess(candidate[:, 0].min(), .5)
        self.assertGreater(len(candidate), 1000)

    def test_rear_wall_without_measured_side_cannot_fill_sheet(self):
        front = self.face(0, 1., [-.5, 0.], [.5, 1.5])
        rear = self.face(0, .4, [-.6, 0.], [.6, 1.5])
        candidate, _ = fit(front, rear, np.array([[2., 0., 1.]]))
        self.assertIsNone(candidate)


class AttachmentRoleTests(unittest.TestCase):
    def test_incompatible_identity_cannot_merge_by_contact(self):
        from pipeline_components.attachment_identity import acceptance
        evidence = dict(native_whole_views=20, native_consensus=1., native_separation_views=0)
        self.assertFalse(acceptance(.1, evidence))

    def test_related_identity_still_needs_whole_mask_support(self):
        from pipeline_components.attachment_identity import acceptance
        evidence = dict(native_whole_views=2, native_consensus=1., native_separation_views=0)
        self.assertFalse(acceptance(.6, evidence))
        evidence['native_whole_views'] = 3
        self.assertTrue(acceptance(.6, evidence))
        evidence['native_separation_views'] = 1
        self.assertFalse(acceptance(.6, evidence))

    def test_matching_identity_can_keep_valid_geometric_attachment(self):
        from pipeline_components.attachment_identity import acceptance
        self.assertTrue(acceptance(.9, dict(native_whole_views=0, native_consensus=0., native_separation_views=0)))

    def test_profile_is_idempotent_and_v13_detaches_new_components(self):
        import types
        import pipeline_components as defaults
        from run_gpt_comparison import configure_profile
        registry = types.SimpleNamespace(**vars(defaults))
        for name in ['GEOMETRY_COMPONENTS', 'FINAL_GEOMETRY', 'MEASURED_REFINEMENT']:
            setattr(registry, name, list(getattr(defaults, name)))
        configure_profile('v14', registry)
        configure_profile('v14', registry)
        self.assertEqual(sum(m.__name__.endswith('.verified_cuboids') for m in registry.FINAL_GEOMETRY), 1)
        self.assertEqual(sum(m.__name__.endswith('.seeded_surfaces') for m in registry.GEOMETRY_COMPONENTS), 1)
        self.assertEqual(sum(m.__name__.endswith('.enclosure_continuity') for m in registry.FINAL_GEOMETRY), 1)
        names = [m.__name__.rsplit('.', 1)[-1] for m in registry.FINAL_GEOMETRY]
        self.assertLess(names.index('enclosure_continuity'), names.index('verified_cuboids'))
        configure_profile('v13', registry)
        self.assertIsNone(registry.AXIAL_VALIDATION)
        self.assertIsNone(registry.BODY_CONTINUITY)
        self.assertFalse(any(m.__name__.endswith('.verified_cuboids') for m in registry.FINAL_GEOMETRY))
        self.assertFalse(any(m.__name__.endswith('.seeded_surfaces') for m in registry.GEOMETRY_COMPONENTS))

    def test_sparse_rack_posts_do_not_extend_closed_front(self):
        from pipeline_components.seeded_surfaces import closed_faces
        front = EvidenceShapeTests.face(1, 1., [-1., 1.], [1., 2.], count=101)
        post = EvidenceShapeTests.face(1, 1., [1., 1.95], [1.6, 2.], count=51)
        faces = closed_faces(np.concatenate([front, post]), np.array([0., 1., 0.]))
        self.assertEqual(len(faces), 1)
        self.assertLess(faces[0][:, 0].max(), 1.1)

    def test_native_measurement_keeps_compositional_evidence(self):
        from pipeline_components.attachment_identity import Evidence
        from types import SimpleNamespace
        evidence = object.__new__(Evidence)
        evidence.native = SimpleNamespace(available=True,
            measure=lambda *args: dict(native_whole_views=3, native_consensus=.15, native_separation_views=18))
        a = dict(name='speaker', text_embedding=[1., 0.])
        b = dict(name='speaker stand', text_embedding=[.72, (1.-.72**2)**.5])
        accepted, audit = evidence.check(np.zeros((10, 3)), np.ones((10, 3)), a, b, ['a', 'b', 'c'])
        self.assertTrue(accepted)
        self.assertTrue(audit['compositional_identity'])

    def test_compositional_attachment_requires_semantic_and_measured_identity(self):
        from pipeline_components.attachment_identity import acceptance, compositional_identity
        self.assertTrue(compositional_identity('speaker', 'speaker stand'))
        self.assertTrue(compositional_identity('chair', 'chair leg'))
        self.assertFalse(compositional_identity('cabinet', 'spoon'))
        self.assertFalse(compositional_identity('speaker', 'speaker'))
        self.assertTrue(acceptance(.72, dict(compositional_identity=True)))
        self.assertFalse(acceptance(.3, dict(compositional_identity=True, native_whole_views=5,
            native_consensus=1., native_separation_views=0)))


if __name__ == '__main__':
    unittest.main()

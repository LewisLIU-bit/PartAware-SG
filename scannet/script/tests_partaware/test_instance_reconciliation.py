import unittest
import numpy as np
from pipeline_components import instance_reconciliation as mira


class NativeAssociationTests(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(17)
        self.points = self.rng.uniform([-.002, -.025, -.2], [.002, .025, .2], (900, 3))

    def group(self, points, frame='a', index=0, box=None):
        return dict(name='unknown implement', sample=points, points=[points], seed_size=len(points),
            frames={frame}, evidence=[dict(frame_id=frame, native_index=index,
            is_zoom_proposal=False, segmentation_box=box or [0, 0, 40, 200], local_record=None)])

    def test_partial_camera_observations_recover_one_identity(self):
        result = mira.associate([self.group(self.points[:300]), self.group(self.points[200:], 'b')])
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]['frames'], {'a', 'b'})

    def test_shared_surfaces_do_not_override_same_frame_independence(self):
        left = self.group(self.points, index=0, box=[0, 0, 40, 200])
        right = self.group(self.points, index=1, box=[100, 0, 140, 200])
        self.assertEqual(len(mira.associate([left, right])), 2)

    def test_separated_depth_rows_remain_separate(self):
        left = self.group(self.points)
        right = self.group(self.points+[0, .1, 0], 'b')
        self.assertEqual(len(mira.associate([left, right])), 2)

    def test_single_contact_cannot_bridge_two_objects(self):
        left = self.group(self.points)
        right = self.group(self.points+[0, .049, 0], 'b')
        self.assertEqual(len(mira.associate([left, right])), 2)

    def test_world_translation_does_not_change_identity(self):
        offset = np.array([29., -7., 4.])
        original = mira.associate([self.group(self.points[:300]), self.group(self.points[200:], 'b')])
        moved = mira.associate([self.group(self.points[:300]+offset), self.group(self.points[200:]+offset, 'b')])
        self.assertEqual(len(original), len(moved))

    def test_related_owned_surface_is_a_fragment_not_new_root(self):
        axis = np.array([1., 0., 0.])
        self.assertTrue(mira.related_surface_duplicate(self.points[:300], self.points, axis, axis))

    def test_unrelated_touching_object_is_not_suppressed_by_surface_overlap(self):
        self.assertFalse(mira.related_surface_duplicate(self.points[:300], self.points,
            np.array([1., 0., 0.]), np.array([0., 1., 0.])))

    def test_related_depth_row_with_own_surface_can_create_new_identity(self):
        axis = np.array([1., 0., 0.])
        self.assertFalse(mira.related_surface_duplicate(self.points+[0., .08, 0.], self.points, axis, axis))


if __name__ == '__main__':
    unittest.main()

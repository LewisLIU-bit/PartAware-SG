"""Regression checks for identity, geometry, and the legacy graph boundary."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
import cv2
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from partaware.fusion import PartFusion, color_histogram
from partaware.geometry import project_mask, load_capture, denoise_largest_cluster, subtract_contained_masks
from run_partaware import select_parent, knowledge_map


class IdentityTests(unittest.TestCase):
    def setUp(self):
        self.points = np.array([[x, y, 1] for x in np.linspace(0, 0.1, 8) for y in np.linspace(0, 0.1, 8)])
        self.feature = np.ones(1024)
        self.color = color_histogram(np.tile([80, 90, 100], (64, 1)))
        self.fusion = PartFusion()

    def add(self, frame, parent='1', points=None):
        return self.fusion.add(frame, 0, 'table: leg', parent,
                               self.points if points is None else points, self.feature, self.color, 0.8)

    def test_partial_observation_merges(self):
        self.add('a')
        self.add('b', points=self.points[:16])
        nodes, edges = self.fusion.export()
        self.assertEqual(len(nodes), 1)
        self.assertEqual(len(edges), 1)
        self.assertEqual(nodes['part_1']['status'], 'confirmed')

    def test_co_visible_parts_remain_separate(self):
        self.add('a')
        self.add('a', points=self.points + [0.01, 0, 0])
        self.assertEqual(len(self.fusion.tracks), 2)
        self.assertEqual(self.fusion.export()[1], [])

    def test_distinct_parents_never_merge(self):
        self.add('a', '1')
        self.add('b', '2')
        self.assertEqual(len(self.fusion.tracks), 2)

    def test_unattached_parts_do_not_create_relations(self):
        self.add('a', None)
        self.add('b', None)
        self.assertEqual(self.fusion.export()[1], [])

    def test_distant_parts_never_merge(self):
        self.add('a')
        self.add('b', points=self.points + [1, 0, 0])
        self.assertEqual(len(self.fusion.tracks), 2)

    def test_embedding_mismatch_fails(self):
        self.add('a')
        with self.assertRaises(ValueError):
            self.fusion.add('b', 0, 'table: leg', '1', self.points, np.ones(256), self.color, 0.8)

    def test_parent_requires_semantic_and_mask_evidence(self):
        mask = np.ones((4, 4), dtype=bool)
        parents = [{'object_name': 'table', 'instance_id': 1, 'frame_instance_id': 1},
                   {'object_name': 'amplifier', 'instance_id': 2, 'frame_instance_id': 2}]
        self.assertEqual(select_parent(mask, 'table: leg', np.ones((4, 4)), parents, {'1', '2'}), '1')
        self.assertIsNone(select_parent(mask, 'table: leg', np.full((4, 4), 2), parents, {'1', '2'}))

    def test_ambiguous_parent_stays_unresolved(self):
        mask = np.ones((4, 4), dtype=bool)
        labels = np.ones((4, 4))
        labels[:2] = 2
        parents = [{'object_name': 'table', 'instance_id': i, 'frame_instance_id': i} for i in (1, 2)]
        self.assertIsNone(select_parent(mask, 'table: leg', labels, parents, {'1', '2'}))


class GeometryTests(unittest.TestCase):
    def test_containment_cleanup_preserves_cross_object_masks(self):
        large = np.ones((4, 4), bool)
        small = np.zeros((4, 4), bool)
        small[0] = True
        a, b = subtract_contained_masks([large, small], ['table: leg', 'table: top'], ['1', '1'])
        self.assertEqual(a.sum(), 12)
        np.testing.assert_equal(b, small)
        a, _ = subtract_contained_masks([large, small], ['table: leg', 'laptop: keyboard'], ['1', '1'])
        self.assertEqual(a.sum(), 16)

    def test_containment_cleanup_preserves_distinct_same_class_parents(self):
        large = np.ones((4, 4), bool)
        small = np.zeros((4, 4), bool)
        small[0] = True
        a, _ = subtract_contained_masks([large, small], ['table: leg', 'table: top'], ['1', '2'])
        self.assertEqual(a.sum(), 16)

    def test_containment_cleanup_requires_resolved_ownership(self):
        large = np.ones((4, 4), bool)
        small = np.zeros((4, 4), bool)
        small[0] = True
        a, _ = subtract_contained_masks([large, small], ['table: leg', 'table: top'], [None, None])
        self.assertEqual(a.sum(), 16)

    def test_denoise_preserves_point_color_alignment(self):
        points = np.vstack((np.zeros((12, 3)), [[5, 5, 5]]))
        colors = np.vstack((np.full((12, 3), 80), [[255, 0, 0]]))
        filtered, rgb = denoise_largest_cluster(points, colors, 0.1)
        self.assertEqual(len(filtered), 12)
        self.assertTrue(np.all(rgb == 80))

    def test_projection_uses_both_intrinsics(self):
        depth = np.full((2, 2), 1000, np.uint16)
        rgb = np.zeros((4, 4, 3), np.uint8)
        mask = np.zeros((4, 4), bool)
        mask[0, 2] = True
        kd = np.eye(3)
        kc = np.diag([2, 2, 1])
        pose = np.eye(4)
        pose[:3, 3] = [1, 2, 3]
        points, _ = project_mask(mask, rgb, depth, pose, kd, kc, 1000, stride=1)
        np.testing.assert_allclose(points, [[2, 2, 4]])

    def test_invalid_pose_is_rejected(self):
        pose = np.eye(4)
        pose[0, 0] = 2
        with self.assertRaises(ValueError):
            project_mask(np.ones((2, 2)), np.zeros((2, 2, 3), np.uint8),
                         np.ones((2, 2), np.uint16), pose, np.eye(3), np.eye(3), 1000)

    def test_duplicate_manifest_ids_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cv2.imwrite(str(root / 'rgb.png'), np.zeros((2, 2, 3), np.uint8))
            cv2.imwrite(str(root / 'depth.png'), np.ones((2, 2), np.uint16))
            np.savetxt(root / 'pose.txt', np.eye(4))
            frame = dict(frame_id='cam_00_0000', rgb='rgb.png', depth='depth.png', pose='pose.txt')
            data = dict(format='scannet_sg_input', pose_convention='T_world_from_camera',
                        length_unit='meter', depth_type='optical_axis_z', camera_info='_info.txt', frames=[frame, frame])
            (root / 'manifest.json').write_text(json.dumps(data))
            with self.assertRaises(ValueError):
                load_capture(root / 'manifest.json')


if __name__ == '__main__':
    unittest.main()

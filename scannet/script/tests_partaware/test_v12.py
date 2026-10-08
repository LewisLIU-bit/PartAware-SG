"""Verify that fine inference and amodal geometry keep evidence boundaries."""
import unittest
import numpy as np
from pipeline_components.fine_instances import sampling_resolution, association_radius, slice_boxes
from pipeline_components.image_shape import initial_alignment, anchor_deformation, evaluate_prior, remove_owned_surface
from pipeline_components.fovea import independent_proposals, append_instances


class V12Checks(unittest.TestCase):
    def test_close_small_instances_do_not_share_a_ten_centimetre_gate(self):
        points = np.array([[0, 0, 0], [.06, .03, .05]])
        self.assertLess(association_radius(points, points+[.03, 0, 0]), .03)
        self.assertLess(sampling_resolution(points), .003)
        self.assertEqual(association_radius(points*100, points*100), .1)

    def test_slices_cover_all_border_pixels(self):
        cover = np.zeros((768, 1024), bool)
        for x0, y0, x1, y1 in slice_boxes(*cover.shape):
            cover[y0:y1, x0:x1] = True
        self.assertTrue(cover.all())

    def test_alignment_does_not_flatten_a_prior_to_observed_sheet(self):
        observed = np.array([[0, -1, -1], [0, 1, 1], [0, -1, 1], [0, 1, -1]])
        prior = np.array([[x, y, z] for x in [-.5, .5] for y in [-1, 1] for z in [-1, 1]])
        pose = np.eye(4); pose[:3, :3] = [[0, 0, -1], [1, 0, 0], [0, -1, 0]]
        completed, _ = initial_alignment(observed, prior, pose)
        self.assertGreater(np.ptp(completed[:, 0]), .5)

    def test_deformation_never_changes_observations(self):
        rng = np.random.default_rng(12)
        observed = rng.uniform(0, 1, (128, 3))
        saved = observed.copy()
        candidate, evidence = anchor_deformation(observed, observed+[.01, 0, 0])
        np.testing.assert_array_equal(observed, saved)
        self.assertLess(np.linalg.norm(candidate-observed, axis=1).mean(), .01)

    def test_no_free_space_point_can_be_published(self):
        rng = np.random.default_rng(12)
        observed = rng.uniform([-.15, -.15, 1.], [.15, .15, 1.], (800, 3))
        candidate = np.concatenate([observed, observed-[0, 0, .2], observed+[0, 0, .2]])
        records = {'a': [{'frame_instance_id': 1, 'instance_id': 1}]}
        class Views:
            jobs = {'a': {}}; kd = kc = np.array([[20, 0, 16], [0, 20, 16], [0, 0, 1.]])
            scale = 1000
            def get(self, fid): return np.eye(4), np.ones((32, 32))*1000, np.ones((32, 32), int)
        views = Views(); views.records = records
        extra, evidence = evaluate_prior(observed, candidate, '1', views, np.empty((0, 3)))
        self.assertIsNone(extra)
        self.assertGreater(evidence['free_space_fraction'], .03)
        self.assertTrue(evidence['early_rejected'])

    def test_a_single_nested_part_cannot_become_an_independent_object(self):
        labels = np.ones((32, 32), np.uint8)
        mask = np.zeros(labels.shape, bool); mask[4:12, 4:12] = True
        self.assertEqual(independent_proposals([({'object_name': 'object'}, mask)], labels), [])

    def test_disjoint_comparable_siblings_keep_separate_instance_evidence(self):
        labels = np.ones((32, 32), np.uint8)
        a = np.zeros(labels.shape, bool); a[4:12, 4:12] = True
        b = np.zeros(labels.shape, bool); b[16:24, 16:24] = True
        proposals = [({'object_name': 'object'}, a), ({'object_name': 'object'}, b)]
        retained = independent_proposals(proposals, labels)
        self.assertEqual(len(retained), 2)
        self.assertTrue(all(record['fine_scale_instance'] for record, _ in retained))
        self.assertTrue(all(record['independent_sibling_count'] == 1 for record, _ in retained))

    def test_fine_inference_keeps_sparse_coarse_ids_and_feature_vectors(self):
        labels = np.zeros((32, 32), np.uint8); labels[:16] = 2; labels[16:] = 7
        source = [{'frame_instance_id': i, 'feature': [i]*256, 'instance_id': 20+i} for i in [2,7]]
        mask = np.zeros(labels.shape,bool); mask[6:10,6:10] = True
        records, changed = append_instances(source, labels, [({'fine_scale_instance':True},mask)])
        self.assertEqual(records[:2],source)
        self.assertEqual(records[-1]['frame_instance_id'],8)
        np.testing.assert_array_equal(changed[~mask],labels[~mask])


    def test_independent_fine_evidence_cannot_merge_a_coarse_track(self):
        from collections import Counter
        from types import SimpleNamespace
        from pipeline_components.instance_consensus import reconcile
        points = np.random.default_rng(12).uniform(0,.05,(128,3))
        def track(gid, fine):
            return SimpleNamespace(id=gid, points=points.copy(), fine_scale=fine,
                semantic=np.ones(384)/np.sqrt(384), frames={'a','b','c'},
                observations=[{}], semantic_sum=np.ones(384), visual_sum=np.ones(256),
                color=np.zeros(32), names=Counter({'object':1}))
        class Views:
            def project(self, points, fid):
                return np.arange(len(points)), np.ones(len(points)), np.ones(len(points),int)
        audit = []
        self.assertEqual(reconcile([track(1,False),track(2,True)],Views(),audit),{})
        self.assertEqual(reconcile([track(1,False),track(2,False)],Views(),audit),{2:1})

    def test_repeated_shape_build_removes_only_its_own_nonmeasured_rows(self):
        import open3d as o3d
        def cloud(points):
            value = o3d.geometry.PointCloud()
            value.points = o3d.utility.Vector3dVector(np.asarray(points,float))
            value.colors = o3d.utility.Vector3dVector(np.tile([1/255,0,0],(len(points),1)))
            return value
        base = cloud([[0,0,0],[0,0,1],[1,0,0]])
        owned = cloud([[0,0,0],[0,0,1]])
        measured = cloud([[0,0,0]])
        cleaned,count = remove_owned_surface(base,owned,measured)
        self.assertEqual(count,1)
        np.testing.assert_array_equal(np.asarray(cleaned.points),[[0,0,0],[1,0,0]])
        second,count = remove_owned_surface(cleaned,owned,measured)
        self.assertEqual(count,0)
        np.testing.assert_array_equal(np.asarray(second.points),np.asarray(cleaned.points))

    def test_alignment_uses_both_observed_image_dimensions(self):
        observed = np.array([[0,y,z] for y in [-1,1] for z in [-3,3]])
        prior = np.array([[x,y,z] for x in [-.5,.5] for y in [-1,1] for z in [-1,1]])
        pose = np.eye(4); pose[:3,:3] = [[0,0,-1],[1,0,0],[0,-1,0]]
        completed,_ = initial_alignment(observed,prior,pose)
        np.testing.assert_allclose(np.ptp(completed,axis=0)[1:],np.ptp(observed,axis=0)[1:])
        self.assertGreater(np.ptp(completed,axis=0)[0],0)


    def test_mgpc_camera_normalization_roundtrip_uses_input_diagonal(self):
        from pipeline_components.joint_shape_model import normalized_camera_points, world_points
        points = np.random.default_rng(12).uniform([-2,-3,-.2],[2,3,.2],(64,3))
        pose = np.eye(4); pose[:3,:3] = [[0,0,1],[1,0,0],[0,1,0]]; pose[:3,3] = [4,5,6]
        saved = points.copy()
        normalized, center, diagonal = normalized_camera_points(points,pose)
        self.assertAlmostEqual(float(np.linalg.norm(np.ptp(normalized,axis=0))),1.)
        np.testing.assert_allclose(world_points(normalized,center,diagonal,pose),points,atol=1e-12)
        np.testing.assert_array_equal(points,saved)

    def test_fine_overlay_cannot_remove_coarse_depth_evidence(self):
        from tempfile import TemporaryDirectory
        from pathlib import Path
        from types import SimpleNamespace
        from pipeline_components.instance_consensus import Views,CoarseViews
        with TemporaryDirectory() as directory:
            root = Path(directory); (root/'fine_frontend_cache').mkdir()
            source = np.ones((32,32),np.uint8); overlay = source.copy(); overlay[12:20,12:20] = 2
            np.savez(root/'fine_frontend_cache/a.npz',source_mask=source)
            calibration = np.array([[20.,0,16],[0,20.,16],[0,0,1]])
            context = SimpleNamespace(scene=root)
            coarse = CoarseViews(context,[{'frame_id':'a'}],calibration,calibration,1000,{})
            fine = Views(context,[{'frame_id':'a'}],calibration,calibration,1000,{})
            coarse.cache['a'] = fine.cache['a'] = (np.eye(4),np.full((32,32),1000),overlay)
            points = np.array([[0.,0.,1.],[.05,.05,1.]])
            np.testing.assert_array_equal(coarse.project(points,'a')[2],[1,1])
            np.testing.assert_array_equal(fine.project(points,'a')[2],[2,2])

    def test_surface_constraint_preserves_hidden_and_foreign_points(self):
        from pipeline_components.joint_constraints import measured_surface_projection
        from types import SimpleNamespace
        observed = np.array([[x,y,1.] for x in [-.25,.25] for y in [-.25,.25]])
        candidate = np.array([[.25,.25,.975],[.25,.25,1.2],[-.25,-.25,.975]])
        mask = np.ones((32,32),int); mask[:16,:16] = 2
        class Views:
            jobs = {'a':{}}; scale = 1000
            kd = kc = np.array([[20.,0,16],[0,20.,16],[0,0,1.]])
            records = {'a':[{'frame_instance_id':1,'instance_id':1},{'frame_instance_id':2,'instance_id':2}]}
            def get(self,fid): return np.eye(4),np.full((32,32),1000),mask
        saved = observed.copy()
        result,evidence = measured_surface_projection(observed,candidate,'1',Views())
        np.testing.assert_array_equal(observed,saved)
        np.testing.assert_array_equal(result[1:],candidate[1:])
        np.testing.assert_allclose(result[0],observed[-1])


if __name__ == '__main__': unittest.main()

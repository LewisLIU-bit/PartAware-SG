import unittest
import numpy as np
from pipeline_components import visible_instances as vista
from pipeline_components import residual_ownership as rse

class EvidenceGeometryTests(unittest.TestCase):
    def setUp(self):
        self.rng=np.random.default_rng(16)

    def test_partial_broad_object_cannot_replace_complete_body(self):
        body=self.rng.uniform(-.1,.1,(900,3))
        partial=body[body[:,0]<-.05]
        q,_,_=vista.recover_surface(body,partial,.3,np.zeros((3,3)))
        self.assertIsNone(q)

    def test_thin_body_rejects_far_normal_noise_without_cutting_long_axis(self):
        surface=self.rng.uniform([- .003,-.01,-.2],[.003,.01,.2],(1000,3))
        noise=self.rng.uniform([.06,-.01,-.1],[.08,.01,.1],(70,3))
        prior=np.concatenate([surface[::2],noise])
        q,removed,_=vista.recover_surface(prior,surface,.87,np.zeros((3,3)))
        self.assertGreaterEqual(removed,65)
        self.assertGreater(np.ptp(q[:,2]),.395)
        self.assertLess(np.ptp(q[:,0]),.01)

    def test_larger_mean_uncertainty_defers_tail_rejection(self):
        surface=self.rng.uniform([-.003,-.01,-.2],[.003,.01,.2],(1000,3))
        prior=np.concatenate([surface[::2],[[.04,0,0]]])
        _,precise,_=vista.recover_surface(prior,surface,.9,np.zeros((3,3)))
        _,uncertain,_=vista.recover_surface(prior,surface,.9,np.eye(3)*.01)
        self.assertGreater(precise,uncertain)

    def test_rear_supported_component_is_preserved(self):
        front=self.rng.uniform([0,0,0],[.04,.04,.04],(400,3))
        rear=self.rng.uniform([0,.10,0],[.04,.14,.04],(90,3))
        prior=np.concatenate([front,rear,[[.3,.3,.1]]])
        q,_,kind=vista.recover_surface(prior,front,.92,np.zeros((3,3)))
        self.assertEqual(kind,'detached_low_support_patch_rejection')
        self.assertTrue(np.any(q[:,1]>.1))
        self.assertLess(q[:,0].max(),.05)

    def test_distinct_depth_rows_do_not_share_identity(self):
        p=self.rng.uniform([0,0,0],[.03,.03,.12],(200,3))
        def group(q):return dict(name='container',sample=q,evidence=[],frames={'a'},seed_size=len(q),points=[q])
        self.assertFalse(vista.partial_agreement(group(p),group(p+[0,.05,0])))

    def test_partial_observations_can_join_same_physical_surface(self):
        p=self.rng.uniform([0,0,0],[.03,.03,.12],(200,3))
        def group(q):return dict(name='container',sample=q,evidence=[],frames={'a'},seed_size=len(q),points=[q])
        self.assertTrue(vista.partial_agreement(group(p[:60]),group(p)))

    def test_separate_native_instances_in_one_frame_cannot_merge(self):
        def group(index, box):
            return {'evidence': [dict(is_zoom_proposal=False, frame_id='a',
                                     native_index=index, segmentation_box=box)]}
        self.assertTrue(vista.simultaneous_separation(group(1,[0,0,10,10]),
                                                     group(2,[9,0,19,10])))
        self.assertFalse(vista.simultaneous_separation(group(1,[0,0,10,10]),
                                                      group(2,[2,2,8,8])))

    def test_robust_residual_rejects_sparse_tails(self):
        body=self.rng.uniform([0,0,0],[.04,.04,.16],(700,3))
        tail=self.rng.uniform([.2,.2,0],[.25,.25,.16],(30,3))
        q,proof=rse.robust_residual(np.concatenate([body,tail]))
        self.assertTrue(proof['accepted'])
        self.assertGreaterEqual(len(q),650)
        self.assertLess(q[:,0].max(),.05)

    def test_two_supported_modes_are_preserved(self):
        a=self.rng.uniform([0,0,0],[.04,.04,.16],(300,3))
        b=a+[0,.1,0]
        q,_=rse.robust_residual(np.concatenate([a,b]))
        self.assertTrue(np.any(q[:,1]>.1))
        self.assertTrue(np.any(q[:,1]<.04))

    def test_plane_gate_preserves_broad_3d_body(self):
        p=self.rng.uniform(-.1,.1,(1000,3))
        q,proof=rse.horizontal_envelope(p)
        self.assertIsNone(proof)
        np.testing.assert_array_equal(q,p)

    def test_smaller_supported_unknown_object_is_not_an_outlier(self):
        a=self.rng.uniform([0,0,0],[.04,.04,.16],(700,3))
        b=self.rng.uniform([0,.14,0],[.04,.18,.16],(120,3))
        p=np.concatenate([a,b])
        q,proof=rse.robust_residual(p)
        self.assertFalse(proof['accepted'])
        np.testing.assert_array_equal(q,p)

    def test_degenerate_points_are_preserved(self):
        p=np.zeros((80,3))
        q,proof=rse.robust_residual(p)
        self.assertFalse(proof['accepted'])
        np.testing.assert_array_equal(q,p)

if __name__=='__main__':unittest.main()

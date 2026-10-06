"""Protect thin geometry, independent objects and whole-object granularity."""
import sys,unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from pipeline_components.thin_geometry import depth_mask_votes,protected_residual,under_resolved,select_geometry_masks
from pipeline_components.axial_assembly import attachment,overhead_attachment


class MeasuredRefinementChecks(unittest.TestCase):
    def setUp(self):
        self.k=np.array([[1.,0.,2.],[0.,1.,2.],[0.,0.,1.]])
    def test_mask_from_wrong_depth_layer_cannot_validate_foreground(self):
        depth=np.full((5,5),2.);depth[2,2]=1.
        mask=np.ones((5,5),bool);mask[2,2]=False
        seen,positive=depth_mask_votes(np.array([[0.,0.,1.]]),np.eye(4),depth,mask,self.k,self.k)
        self.assertTrue(seen[0]);self.assertFalse(positive[0])
    def test_subpixel_boundary_uses_matching_depth_and_label_together(self):
        depth=np.full((5,5),2.);depth[2,3]=1.
        mask=np.zeros((5,5),bool);mask[2,3]=True
        seen,positive=depth_mask_votes(np.array([[0.,0.,1.]]),np.eye(4),depth,mask,self.k,self.k)
        self.assertTrue(seen[0]);self.assertTrue(positive[0])
    def test_occluded_and_missing_depth_abstain(self):
        for depth in [np.zeros((5,5)),np.ones((5,5))*.5]:
            seen,positive=depth_mask_votes(np.array([[0.,0.,1.]]),np.eye(4),depth,np.ones((5,5),bool),self.k,self.k)
            self.assertFalse(seen[0]);self.assertFalse(positive[0])
    def test_registered_color_resolution_is_not_assumed_equal(self):
        color=np.array([[2.,0.,4.],[0.,2.,4.],[0.,0.,1.]])
        mask=np.zeros((9,9),bool);mask[4,4]=True
        _,positive=depth_mask_votes(np.array([[0.,0.,1.]]),np.eye(4),np.ones((5,5)),mask,self.k,color)
        self.assertTrue(positive[0])
    def test_occupied_object_cannot_be_stolen(self):
        p=np.array([[0.,0.,0.],[1.,0.,0.]])
        np.testing.assert_array_equal(protected_residual(p,p[:1]),p[1:])
    def test_small_dense_object_does_not_trigger_global_resegmentation(self):
        p=np.random.default_rng(1).uniform(0,.08,(100,3))
        self.assertFalse(under_resolved(p)[0])
    def test_high_model_score_cannot_override_observed_seed_coverage(self):
        low=np.zeros((10,10),bool);low[2,2]=True
        high=np.zeros((10,10),bool);high[8,8]=True
        selected,audit=select_geometry_masks([low,high],[.7,.95],np.array([[2,2]]))
        np.testing.assert_array_equal(selected,low)
        self.assertTrue(audit[0]['accepted']);self.assertFalse(audit[1]['accepted'])
    def test_axial_contact_requires_stem_base_and_alignment(self):
        angles=np.linspace(0,2*np.pi,80,endpoint=False)
        stem=np.array([[.04*np.cos(a),.04*np.sin(a),z] for z in np.linspace(.03,.6,80) for a in angles])
        foot=np.array([[x,y,z] for x in np.linspace(-.15,.15,20) for y in np.linspace(-.15,.15,20) for z in [.0,.02]])
        support=np.concatenate([stem,foot])
        body=np.array([[x,y,z] for x in np.linspace(-.1,.1,21) for y in np.linspace(-.12,.12,25) for z in [.6,1.]])
        self.assertIsNotNone(attachment(body,support))
        self.assertIsNone(attachment(body+[.15,0.,0.],support))
        self.assertIsNone(attachment(body+[0.,0.,.1],support))
        self.assertIsNone(attachment(body,foot))
    def test_broad_uniform_pedestal_is_not_an_attached_stem(self):
        support=np.array([[x,y,z] for x in [-.15,.15] for y in [-.15,.15] for z in np.linspace(0,.6,100)])
        body=np.array([[x,y,z] for x in [-.1,.1] for y in [-.12,.12] for z in [.6,1.]])
        self.assertIsNone(attachment(body,support))
    def test_overhead_crossbar_requires_contact_axis_and_upper_position(self):
        body=np.array([[x,y,z] for x in np.linspace(-1,1,120) for y in [-.04,.04] for z in [.0,.3]])
        bar=np.array([[x,y,z] for x in np.linspace(-1,1,120) for y in [-.05,.05] for z in [.31,.34]])
        self.assertIsNotNone(overhead_attachment(body,bar))
        self.assertIsNone(overhead_attachment(body,bar+[0,0,.1]))
        self.assertIsNone(overhead_attachment(body,bar+[0,.2,0]))
        self.assertIsNone(overhead_attachment(body,bar*[1,5,1]))
        rotated=bar.copy();rotated[:,:2]=bar[:,[1,0]]
        self.assertIsNone(overhead_attachment(body,rotated))

import unittest
import numpy as np
from pipeline_components.consensus_ownership import admission,supported_trim,owner,axial_envelope,axial_trim
from pipeline_components.support_residuals import raised_groups
from pipeline_components.support_publication import existing_envelopes


class IndependentEvidence(unittest.TestCase):
    def group(self,frames=('a','b'),quality=(.95,.8),baseline=.4):
        evidence=[dict(frame_id=fid,confidence=score,camera_center=[i*baseline,0,0],local_record={'real':True}) for i,(fid,score) in enumerate(zip(frames,quality))]
        return dict(frames=set(frames),evidence=evidence)
    def test_two_real_views_can_admit_rare_visible_object(self):
        self.assertEqual(admission(self.group())['branch'],'strict_two_view')
    def test_repeated_zoom_in_one_frame_is_not_independent(self):
        g=self.group(('a','a'));g['evidence']*=8
        self.assertIsNone(admission(g))
    def test_low_confidence_or_baseline_cannot_use_sparse_branch(self):
        self.assertIsNone(admission(self.group(quality=(.7,.7))))
        self.assertIsNone(admission(self.group(baseline=.01)))
    def test_missing_real_features_does_not_fabricate_embeddings(self):
        g=self.group()
        for e in g['evidence']:e['local_record']=None
        self.assertIsNone(admission(g))
    def test_four_frames_do_not_require_ninety_percent_peak(self):
        g=self.group(('a','b','c','d'),(.8,.7,.7,.7));self.assertEqual(admission(g)['branch'],'multiview')
    def test_unknown_and_occluded_points_survive(self):
        q=np.array([[0,0,0],[.5,0,0],[1,0,0]],float)
        kept,remove=supported_trim(q,q[:1],np.array([3,0,0]),np.zeros(3,int))
        np.testing.assert_array_equal(kept,q);self.assertFalse(remove.any())
    def test_multiview_contradicted_minority_is_quarantined(self):
        q=np.column_stack([np.arange(10)*.1,np.zeros((10,2))]);pos=np.full(10,5);neg=np.zeros(10,int);pos[-2:]=1;neg[-2:]=5
        kept,remove=supported_trim(q,q[:8],pos,neg);self.assertEqual(len(kept),8);self.assertEqual(int(remove.sum()),2)
    def test_single_negative_view_cannot_trim_boundary(self):
        q=np.column_stack([np.arange(10)*.1,np.zeros((10,2))]);kept,remove=supported_trim(q,q[:8],np.zeros(10),np.ones(10))
        self.assertFalse(remove.any())
    def test_no_dominant_identity_support_preserves_region(self):
        q=np.column_stack([np.arange(10)*.1,np.zeros((10,2))]);kept,remove=supported_trim(q,q[:1],np.zeros(10),np.full(10,5))
        np.testing.assert_array_equal(kept,q)
    def test_semantically_different_contacting_body_cannot_take_ownership(self):
        q=np.random.default_rng(8).uniform(0,.1,(200,3));nodes={'arbitrary':dict(text_embedding=[0,1])};regions={'arbitrary':q}
        self.assertIsNone(owner(q,np.array([1,0]),nodes,regions))
    def test_geometry_and_semantics_survive_identifier_translation(self):
        q=np.random.default_rng(9).uniform(0,.1,(200,3))
        for gid,shift in [('x',0),('unrelated_identifier',7)]:
            nodes={gid:dict(text_embedding=[1,0])};regions={gid:q+shift}
            self.assertEqual(owner(q+shift,np.array([1,0]),nodes,regions)[1],gid)


class MeasuredGeometry(unittest.TestCase):
    def circle(self):
        rng=np.random.default_rng(37)
        angle=rng.uniform(-np.pi*.8,np.pi*.8,1800)
        return np.column_stack([.025*np.cos(angle),.025*np.sin(angle),rng.uniform(0,.09,len(angle))])
    def test_axial_fit_requires_measured_broad_arc(self):
        fit=axial_envelope(self.circle())
        self.assertIsNotNone(fit)
        self.assertAlmostEqual(fit['radius_m'],.025,places=5)
        q=self.circle();q[:,1]=0
        self.assertIsNone(axial_envelope(q))
    def test_axial_envelope_preserves_coherent_attachment(self):
        body=self.circle();rng=np.random.default_rng(42)
        attachment=rng.uniform([.05,-.003,.04],[.06,.003,.05],(50,3))
        prior=np.concatenate([body,attachment,[[.2,.2,.04]]])
        pos=np.r_[np.full(len(body),4),np.zeros(51,int)]
        kept,removed,proof=axial_trim(prior,body,pos,np.zeros(len(prior),int))
        self.assertIsNotNone(proof)
        self.assertEqual(int(removed.sum()),1)
        self.assertTrue(removed[-1])
        self.assertEqual(len(kept),len(body)+len(attachment))
    def support_regions(self):
        x,y=np.meshgrid(np.linspace(-.5,.5,80),np.linspace(-.5,.5,80))
        parent=np.column_stack([x.ravel(),y.ravel(),np.zeros(x.size)])
        x,y,z=np.meshgrid(np.linspace(-.02,.02,8),np.linspace(-.02,.02,8),[.02,.1])
        child=np.column_stack([x.ravel(),y.ravel(),z.ravel()])
        return {'parent':parent,'a':child+[-.06,0,0],'b':child+[.06,0,0]}
    def test_raised_support_proposal_is_translation_and_identifier_invariant(self):
        regions=self.support_regions();first=raised_groups(regions)
        shift=np.array([3,-2,7])
        second=raised_groups({f'arbitrary_{k}':v+shift for k,v in regions.items()})
        self.assertEqual(len(first),1);self.assertEqual(len(second),1)
        np.testing.assert_allclose(first[0]['lower']+shift,second[0]['lower'],atol=1e-7)
        np.testing.assert_allclose(first[0]['upper']+shift,second[0]['upper'],atol=1e-7)
    def test_one_raised_anchor_cannot_establish_hidden_object(self):
        regions=self.support_regions();del regions['b']
        self.assertEqual(raised_groups(regions),[])
    def test_existing_compact_body_envelope_defers_duplicate_root(self):
        q=np.random.default_rng(39).uniform(-.05,.05,(300,3))
        frame=np.array([[-.06,-.06,-.06],[.06,.06,.06]])
        self.assertEqual(existing_envelopes(q,{'existing':frame}),['existing'])
        self.assertEqual(existing_envelopes(q+2,{'existing':frame}),[])


if __name__=='__main__':unittest.main()

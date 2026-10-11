import unittest
from types import SimpleNamespace
import numpy as np
import pipeline_components as registry
from pipeline_components.granularity_aligned import entities,evaluate
from pipeline_components.existence_review import reject
from pipeline_components.surface_regularization import regularize
from run_gpt_comparison import configure_profile


class GeneralProtocol(unittest.TestCase):
    def p(self,i,lo,hi,confidence=.9):return dict(id=str(i),label='arbitrary',bounds=[lo,hi],confidence=confidence)
    def s(self,rows):return dict(dominant_assignment_purity=.8,distance_tolerance_m=.02,per_prediction=rows)
    def test_disconnected_annotation_whole_counts_once(self):
        p=[self.p('x',[0,0,0],[.4,1,1]),self.p('y',[.6,0,0],[1,1,1],.6)]
        s=self.s([dict(prediction_id=r['id'],pure_fragment=True,dominant_gt_id=9) for r in p]);g=[dict(id=9,bounds=[[0,0,0],[1,1,1]])]
        r=evaluate(p,g,s);self.assertEqual(r['statistical_entity_count'],1);self.assertEqual(r['AP50'],1.);self.assertEqual(r['entities'][0]['confidence'],.6);self.assertTrue(r['grouping_uses_ground_truth'])
    def test_different_gt_instances_do_not_merge_even_same_class(self):
        p=[self.p(i,[i,0,0],[i+1,1,1]) for i in range(2)]
        s=self.s([dict(prediction_id=str(i),pure_fragment=True,dominant_gt_id=i) for i in range(2)])
        self.assertEqual(len(entities(p,s)),2)
    def test_mixed_prediction_is_not_gt_split(self):
        p=[self.p('x',[0,0,0],[2,1,1])];s=self.s([dict(prediction_id='x',pure_fragment=False,dominant_gt_id=1)])
        g=[dict(id=i,bounds=[[i,0,0],[i+1,1,1]]) for i in (0,1)];r=evaluate(p,g,s)
        self.assertEqual(r['one_to_one']['0.5']['TP'],1);self.assertEqual(r['one_to_one']['0.5']['FN'],1)
    def test_unknown_false_positive_is_not_dropped(self):
        p=[self.p('a',[0,0,0],[1,1,1]),self.p('z',[10,0,0],[11,1,1],1.)]
        s=self.s([dict(prediction_id='a',pure_fragment=True,dominant_gt_id=4),dict(prediction_id='z',pure_fragment=False,dominant_gt_id=None)]);r=evaluate(p,[dict(id=4,bounds=[[0,0,0],[1,1,1]])],s)
        self.assertEqual(r['one_to_one']['0.5']['FP'],1);self.assertEqual(r['AP50'],.5)
    def test_group_cannot_copy_gt_box_or_gt_score(self):
        p=[self.p('a',[0,0,0],[1,1,1],.2)];s=self.s([dict(prediction_id='a',pure_fragment=True,dominant_gt_id=1)])
        r=evaluate(p,[dict(id=1,bounds=[[0,0,0],[10,10,10]])],s);self.assertEqual(r['entities'][0]['bounds'],p[0]['bounds']);self.assertEqual(r['entities'][0]['confidence'],.2);self.assertEqual(r['AP25'],0.)
    def test_categories_ids_scene_translation_do_not_change_rule(self):
        for label,offset,a,b,gid in [('fabric',np.zeros(3),'192','204',28),('machine',np.array([18,-12,8]),'q','r',937)]:
            p=[self.p(a,(offset+[0,0,0]).tolist(),(offset+[.4,1,1]).tolist()),self.p(b,(offset+[.6,0,0]).tolist(),(offset+[1,1,1]).tolist())]
            for r in p:r['label']=label
            s=self.s([dict(prediction_id=r['id'],pure_fragment=True,dominant_gt_id=gid) for r in p]);r=evaluate(p,[dict(id=gid,bounds=[offset.tolist(),(offset+[1,1,1]).tolist()])],s);self.assertEqual(r['AP50'],1.)


class EvidenceGuard(unittest.TestCase):
    def evidence(self):return dict(linear_fragment_ratio=.005,semantic_identity_margin=-.1,semantic_object_background_margin=.001,frame_evidence=[{}]*6,visible_point_fraction=1.,stable_own_point_fraction=0.,majority_background_point_fraction=1.,compatible_class_point_fraction=0.,compatible_class_weighted_fraction=.1,native_stable_point_fraction=0.,holdout_background_views=6,negative_camera_baseline_m=.5)
    def test_repeated_multimodal_refutation_can_withdraw_fragment(self):self.assertTrue(reject(self.evidence()))
    def test_thinness_alone_is_insufficient(self):
        r=self.evidence();r['semantic_identity_margin']=.1;self.assertFalse(reject(r))
    def test_stable_native_support_protects_real_fine_tool(self):
        r=self.evidence();r['native_stable_point_fraction']=.8;self.assertFalse(reject(r))
    def test_occluded_or_missing_depth_is_unknown(self):
        r=self.evidence();r['visible_point_fraction']=.1;self.assertFalse(reject(r))
    def test_broad_real_weak_box_is_not_an_edge_ghost(self):
        r=self.evidence();r['linear_fragment_ratio']=.5;self.assertFalse(reject(r))
    def test_insufficient_baseline_or_negative_views_preserves_identity(self):
        for key,value in [('negative_camera_baseline_m',.01),('holdout_background_views',1)]:
            r=self.evidence();r[key]=value;self.assertFalse(reject(r))
    def test_regularization_bounded_and_preserves_observation_input(self):
        rng=np.random.default_rng(8);q=np.column_stack([rng.uniform(0,1,(8000,2)),rng.normal(0,.001,8000)]);original=q.copy();r=regularize(q)
        self.assertIsNotNone(r);smooth,e=r;np.testing.assert_array_equal(q,original);self.assertLessEqual(np.linalg.norm(smooth-q,axis=1).max(),.004001);self.assertLess(np.std(smooth[:,2]),np.std(q[:,2]));self.assertFalse(e['reconstructed_coordinates_are_new_measurements'])
    def test_small_or_curved_objects_are_not_flattened(self):
        self.assertIsNone(regularize(np.zeros((100,3))));rng=np.random.default_rng(9);self.assertIsNone(regularize(rng.normal(0,1,(6000,3))))
    def test_v18_isolated_from_historical_profiles(self):
        values={k:v.copy() if isinstance(v,list) else v for k,v in vars(registry).items() if k.isupper()};r=SimpleNamespace(**values)
        configure_profile('v18',r);self.assertIn('existence_review',[m.__name__.split('.')[-1] for m in r.FINAL_GEOMETRY]);configure_profile('v17',r);self.assertNotIn('existence_review',[m.__name__.split('.')[-1] for m in r.FINAL_GEOMETRY]);self.assertNotIn('surface_regularization',[m.__name__.split('.')[-1] for m in r.FINAL_GEOMETRY])


if __name__=='__main__':unittest.main()

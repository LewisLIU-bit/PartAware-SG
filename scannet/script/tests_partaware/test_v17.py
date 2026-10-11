"""Generic evidence safeguards: identity, thin bodies and uncertain fragments."""
import unittest
import numpy as np
from pipeline_components import component_evidence as core,weak_verification as wisp
from pipeline_components.granularity_metrics import fragment_union_box_diagnostics
from run_gpt_comparison import configure_profile
from types import SimpleNamespace
import pipeline_components as registry


class Version17Tests(unittest.TestCase):
    def group(self,p,fid='a',index=1,box=None,name='thing',semantic=None):
        return dict(name=name,sample=p,points=[p],seed_size=len(p),frames={fid},evidence=[dict(frame_id=fid,native_index=index,is_zoom_proposal=True,segmentation_box=box or [0,0,20,20],name=name,confidence=.55,camera_center=[0,0,0],local_record=dict(bert_embedding=(semantic if semantic is not None else np.ones(384)).tolist(),feature=np.ones(256).tolist()))])
    def test_zoom_separation_vetoes_alias_merge(self):
        p=np.random.default_rng(1).uniform(0,.03,(100,3))
        a=self.group(p,index=1,box=[0,0,10,10]);b=self.group(p,index=2,box=[15,0,25,10])
        self.assertEqual(wisp.agreement(a,b),0.)
        self.assertEqual(wisp.coframe_separation(a,b),{'a'})
    def test_name_change_can_use_shared_measured_surface(self):
        p=np.random.default_rng(2).uniform(0,.03,(200,3))
        a=self.group(p,fid='a',name='container');b=self.group(p[::2],fid='b',name='box')
        self.assertGreater(wisp.agreement(a,b),0.)
    def test_touching_instances_are_not_shared_surfaces(self):
        p=np.random.default_rng(3).uniform(0,.02,(200,3));q=p+[.02,0,0]
        self.assertEqual(wisp.agreement(self.group(p),self.group(q,fid='b')),0.)
    def test_three_joint_whole_masks_preserve_real_multimodal_body(self):
        self.assertFalse(core.acceptance(['a','b','c'],['d','e','f'],1.))
    def test_unobserved_and_single_view_fragments_are_unknown(self):
        self.assertFalse(core.acceptance([],[],1.));self.assertFalse(core.acceptance(['a'],[],1.))
    def test_insufficient_view_baseline_preserves_fragments(self):
        self.assertFalse(core.acceptance(['a','b'],[],.01))
    def test_remote_minority_can_be_quarantined_without_moving_points(self):
        rng=np.random.default_rng(4);a=rng.uniform(0,.02,(400,3));b=rng.uniform(1,1.02,(25,3));p=np.concatenate([a,b])
        x,y,proof=core.disconnected_candidate(p)
        self.assertEqual(len(x),400);self.assertEqual(len(y),25);self.assertGreater(proof['fragment_gap_m'],1.)
        np.testing.assert_array_equal(np.concatenate([x,y]),p)
    def test_two_large_components_remain_a_real_assembly_candidate(self):
        rng=np.random.default_rng(5);a=rng.uniform(0,.02,(100,3));b=a+[1,0,0]
        self.assertIsNone(core.disconnected_candidate(np.concatenate([a,b])))
    def test_thin_connected_body_keeps_its_full_length(self):
        p=np.column_stack([np.linspace(0,.4,400),np.zeros((400,2))]);self.assertIsNone(core.disconnected_candidate(p))
    def test_translation_does_not_change_geometric_aliasing(self):
        rng=np.random.default_rng(6);p=rng.uniform(0,.02,(200,3));offset=np.array([10.,-7.,3.])
        self.assertAlmostEqual(wisp.agreement(self.group(p),self.group(p[::2],fid='b')),wisp.agreement(self.group(p+offset),self.group(p[::2]+offset,fid='b')))
    def test_gt_assisted_union_is_explicitly_not_ap(self):
        p=[dict(id='a',keys=[],observed_bounds=[[0,0,0],[1,1,1]],bounds=[[0,0,0],[1,1,1]]),dict(id='b',keys=[],observed_bounds=[[1,0,0],[2,1,1]],bounds=[[1,0,0],[2,1,1]])]
        gt=[dict(id=1,label='cloth',bounds=[[0,0,0],[2,1,1]])];s=dict(dominant_assignment_purity=.8,per_prediction=[dict(prediction_id=x['id'],pure_fragment=True,dominant_gt_id=1) for x in p])
        r=fragment_union_box_diagnostics(p,gt,s)
        self.assertEqual(r['per_gt'][0]['measured_union_box_IoU'],1.)
        self.assertFalse(r['average_precision']);self.assertTrue(r['group_assignment_uses_ground_truth'])
    def test_mixed_prediction_does_not_cover_two_gt_instances_by_union(self):
        p=[dict(id='a',keys=[],observed_bounds=[[0,0,0],[2,1,1]],bounds=[[0,0,0],[2,1,1]])]
        gt=[dict(id=1,label='window',bounds=[[0,0,0],[1,1,1]]),dict(id=2,label='window',bounds=[[1,0,0],[2,1,1]])];s=dict(per_prediction=[dict(prediction_id='a',pure_fragment=False,dominant_gt_id=1)])
        r=fragment_union_box_diagnostics(p,gt,s);self.assertTrue(all(row['fragment_count']==0 for row in r['per_gt']))
    def test_v17_detaches_for_historical_v16(self):
        # Use a copied registry so test ordering does not change global defaults.
        values={k:v.copy() if isinstance(v,list) else v for k,v in vars(registry).items() if k.isupper()}
        r=SimpleNamespace(**values);configure_profile('v17',r);names=[x.__name__.split('.')[-1] for x in r.FINAL_GEOMETRY]
        self.assertEqual(names[-2:],['component_evidence','weak_verification'])
        configure_profile('v16',r);names=[x.__name__.split('.')[-1] for x in r.FINAL_GEOMETRY]
        self.assertNotIn('component_evidence',names);self.assertNotIn('weak_verification',names);self.assertIn('instance_reconciliation',names)

if __name__=='__main__':unittest.main()

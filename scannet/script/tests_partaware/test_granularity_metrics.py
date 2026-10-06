"""Test complementary surface metrics without relaxing strict instance matching."""
import sys, unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from pipeline_components.granularity_metrics import surface_diagnostics
from evaluate_hypersim import voxel_keys

class GranularityChecks(unittest.TestCase):
    def test_correct_fragments_cover_one_object_but_report_fragmentation(self):
        p=np.column_stack([np.arange(20)*.04,np.zeros((20,2))])
        gt=[dict(id=1,label='chair',keys=voxel_keys(p))]
        pred=[dict(id='a',keys=voxel_keys(p[:10])),dict(id='b',keys=voxel_keys(p[10:]))]
        report=surface_diagnostics(pred,gt)
        self.assertEqual(report['weighted_surface_coverage'],1.)
        self.assertEqual(report['pure_fragment_excess'],1)
    def test_background_hallucination_lowers_precision(self):
        p=np.column_stack([np.arange(20)*.04,np.zeros((20,2))])
        gt=[dict(id=1,label='chair',keys=voxel_keys(p))]
        pred=[dict(id='a',keys=voxel_keys(p)),dict(id='b',keys=voxel_keys(p+[0,2,0]))]
        report=surface_diagnostics(pred,gt)
        self.assertEqual(report['surface_precision'],.5)
        self.assertFalse(report['per_prediction'][1]['pure_fragment'])
    def test_mixing_two_objects_does_not_get_pure_union_credit(self):
        p=np.column_stack([np.arange(10)*.04,np.zeros((10,2))]);q=p+[0,2,0]
        gt=[dict(id=1,label='chair',keys=voxel_keys(p)),dict(id=2,label='chair',keys=voxel_keys(q))]
        pred=[dict(id='a',keys=voxel_keys(np.concatenate([p,q]))) ]
        report=surface_diagnostics(pred,gt)
        self.assertEqual(report['weighted_surface_coverage'],0.)
        self.assertEqual(report['weighted_geometric_surface_coverage'],1.)
        self.assertEqual(report['geometric_surface_F1'],1.)
        self.assertEqual(report['surface_precision'],1.)

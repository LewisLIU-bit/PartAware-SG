"""Individual error reporting must follow the same greedy AP assignment."""
import unittest
import numpy as np
from pipeline_components.granularity_metrics import box_diagnostics


class ObjectDiagnosticsTests(unittest.TestCase):
    def test_ranked_duplicate_differs_from_final_optimum(self):
        truth=[{'id':10,'label':'chair','keys':np.array([1,2]),'bounds':[[0,0,0],[1,1,1]]}]
        predictions=[{'id':'1','label':'chair','confidence':.1,'bounds':[[0,0,0],[1,1,1]]},
                     {'id':'2','label':'chair','confidence':.9,'bounds':[[0,0,0],[1,1,1]]}]
        spatial={str(t):{'matches':[{'prediction_id':'1','gt_id':10,'iou':1.}]} for t in [.25,.5,.75]}
        out=box_diagnostics(predictions,truth,spatial,{})
        self.assertEqual(out['per_gt'][0]['one_to_one_matches']['0.25']['prediction_id'],'1')
        self.assertEqual(out['AP_ranking']['0.25'][0]['prediction_id'],'2')
        self.assertTrue(out['AP_ranking']['0.25'][0]['true_positive'])
        self.assertEqual(out['AP_ranking']['0.25'][1]['reason'],'duplicate_competition')


if __name__=='__main__':unittest.main()

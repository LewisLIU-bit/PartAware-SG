"""Ensure fewer predictions cannot hide lost localization or changed GT scope."""
import copy
import unittest
from compare_evaluations import compare


class AcceptanceChecks(unittest.TestCase):
    def baseline(self):
        return {'protocol':'fixed', 'gt_objects':10, 'gt_labels_sha256':'same',
                'input_manifest':'fixed_manifest', 'frames':99, 'voxel_m':.01,
                'projection_stride':2, 'minimum_observed_gt_voxels':100,
                'excluded_nyu40_ids':[1,2,22], 'gt_bbox_source':'official',
                'prediction_bbox_source':'measured',
                'geometry_only_box_AP25':1., 'geometry_only_box_AP50':1., 'geometry_only_box_AP75':.625,
                'object_count_consistency':{'absolute_log_ratio':0.},
                'one_to_one_bbox_geometry':{key:{'TP':10,'FP':0,'FN':0} for key in ('0.25','0.5','0.75')}}

    def test_ceiling_score_can_tie_while_stricter_localization_improves(self):
        base=self.baseline(); candidate=copy.deepcopy(base); candidate['geometry_only_box_AP75']=.8
        self.assertTrue(compare(candidate,base)['accepted'])

    def test_lower_prediction_count_cannot_excuse_lost_ap(self):
        base=self.baseline(); candidate=copy.deepcopy(base); candidate['geometry_only_box_AP50']=.9
        candidate['geometry_only_box_AP75']=.8
        self.assertFalse(compare(candidate,base)['accepted'])

    def test_ground_truth_scope_changes_are_rejected(self):
        base=self.baseline(); candidate=copy.deepcopy(base); candidate['gt_labels_sha256']='different'
        with self.assertRaises(ValueError): compare(candidate,base)

    def test_identical_results_are_not_advertised_as_an_improvement(self):
        result=compare(self.baseline(), self.baseline())
        self.assertTrue(result['non_regressing']); self.assertFalse(result['accepted'])

    def test_shared_missing_protocol_field_is_not_a_valid_comparison(self):
        base=self.baseline(); del base['input_manifest']
        with self.assertRaises(ValueError): compare(base,copy.deepcopy(base))

    def test_nonfinite_score_is_not_accepted(self):
        base=self.baseline(); candidate=copy.deepcopy(base)
        candidate['geometry_only_box_AP75']=float('nan')
        with self.assertRaises(ValueError): compare(candidate,base)


if __name__ == '__main__': unittest.main()

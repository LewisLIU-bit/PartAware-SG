"""Reject mistaken container merges and ambiguous region ownership."""
import unittest
from pipeline_components.instance_granularity import duplicate, transfer


class InstanceGranularityTests(unittest.TestCase):
    def setUp(self):
        self.e={'whole_mask_support_views': 8, 'whole_mask_consensus': .6,
                'independent_separation_views': 0}

    def test_shared_surface_duplicate(self):
        self.assertTrue(duplicate(self.e, .98, .85, .7))

    def test_neighbor_or_shelf_contents(self):
        self.assertFalse(duplicate(self.e, .98, .05, .9))
        self.assertFalse(duplicate(self.e, .98, .99, .49))

    def test_independent_instance_veto(self):
        self.assertFalse(duplicate({**self.e,'independent_separation_views':1},1.,1.,1.))

    def test_weak_mask_evidence(self):
        self.assertFalse(duplicate({**self.e,'whole_mask_support_views':4},1.,1.,1.))
        self.assertFalse(duplicate({**self.e,'whole_mask_consensus':.49},1.,1.,1.))

    def test_unique_detached_transfer(self):
        e={'support_views':8,'source_support_views':1,'owner_consensus':.9,'camera_baseline_m':.2}
        self.assertTrue(transfer(e,.98,.8,.8,.05))
        self.assertFalse(transfer({**e,'source_support_views':3},.98,.8,.8,.05))
        self.assertFalse(transfer({**e,'owner_consensus':.6},.98,.8,.8,.05))
        self.assertFalse(transfer(e,.98,.8,.8,.02))


if __name__=='__main__':unittest.main()

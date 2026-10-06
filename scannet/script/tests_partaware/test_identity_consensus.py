"""Protect independent contained objects when reconciling inconsistent labels."""
import sys, unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline_components.identity_consensus import contained_duplicate

class IdentityChecks(unittest.TestCase):
    def test_coobserved_separation_vetoes_alias(self):
        evidence = dict(whole_mask_support_views=7, whole_mask_consensus=.9, independent_separation_views=1)
        self.assertFalse(contained_duplicate(1., 1., evidence))
    def test_box_containment_cannot_replace_observed_surface(self):
        evidence = dict(whole_mask_support_views=7, whole_mask_consensus=.9, independent_separation_views=0)
        self.assertFalse(contained_duplicate(1., .7, evidence))
    def test_repeated_whole_masks_resolve_conflicting_names(self):
        evidence = dict(whole_mask_support_views=7, whole_mask_consensus=.5, independent_separation_views=0)
        self.assertTrue(contained_duplicate(1., 1., evidence))
        evidence['whole_mask_support_views'] = 2
        self.assertFalse(contained_duplicate(1., 1., evidence))

"""A touching region is insufficient without repeated complete-object evidence."""
import unittest
from pipeline_components.native_assembly import acceptance


class NativeAssemblyChecks(unittest.TestCase):
    def setUp(self):
        self.e = dict(native_whole_views=4, native_consensus=.8, native_separation_views=0)

    def test_contact_and_complete_views_can_join_same_identity(self):
        self.assertTrue(acceptance(self.e, .9, .02, .8))

    def test_adjacent_equal_named_objects_with_separate_views_survive(self):
        self.assertFalse(acceptance({**self.e, 'native_separation_views': 1}, 1., .001, .8))

    def test_unknown_or_infrequent_views_do_not_establish_an_assembly(self):
        self.assertFalse(acceptance({**self.e, 'native_whole_views': 2}, .9, .02, .8))
        self.assertFalse(acceptance({**self.e, 'native_consensus': .3}, .9, .02, .8))

    def test_different_identity_or_detached_geometry_is_not_swallowed(self):
        self.assertFalse(acceptance(self.e, .6, .01, .8))
        self.assertFalse(acceptance(self.e, 1., .06, .8))

    def test_touching_small_dishes_cannot_become_one_structural_body(self):
        self.assertFalse(acceptance(self.e, 1., .001, .3))


if __name__ == '__main__':
    unittest.main()

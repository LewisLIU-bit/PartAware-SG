"""Public mask contract and cached-only category acquisition for SAM3."""
import copy
import unittest

import numpy as np

from pipeline_components.sam3_frontend import names_for_frame, partition, deduplicate, zoom_regions


class Sam3FrontendChecks(unittest.TestCase):
    def test_cached_nouns_and_existing_ontology_are_used_without_descriptions(self):
        vocabulary = [{'plate': ['A photo.']}, {'wall': ['A surface.']}]
        tags = {'objects': [{'name': 'BOTTLE', 'description': 'Never use this as a new prompt.'},
                            {'name': 'plate'}, {'name': 'ceiling'}]}
        self.assertEqual(names_for_frame(tags, vocabulary), ['bottle', 'plate'])

    def test_aliases_do_not_count_the_same_mask_twice(self):
        mask = np.zeros((32, 32), bool); mask[2:12, 2:12] = True
        a = {'object_name': 'speaker', 'confidence': .9}
        b = {'object_name': 'loudspeaker', 'confidence': .8}
        saved = copy.deepcopy(a)
        result = deduplicate([(a, mask), (b, mask.copy())])
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0][0]['query_aliases'], ['loudspeaker'])
        self.assertEqual(a, saved)

    def test_separate_equal_named_instances_survive_deduplication(self):
        a = np.zeros((32, 32), bool); a[2:10, 2:10] = True
        b = np.zeros((32, 32), bool); b[20:28, 20:28] = True
        record = {'object_name': 'bottle', 'confidence': .8}
        self.assertEqual(len(deduplicate([(record, a), (record, b)])), 2)

    def test_partition_has_consistent_pixels_bounds_and_unique_local_ids(self):
        large = np.ones((32, 32), bool)
        small = np.zeros(large.shape, bool); small[6:14, 7:15] = True
        records, labels = partition([({'object_name': 'shelf'}, large),
                                     ({'object_name': 'bottle'}, small)], large.shape)
        self.assertEqual(labels.dtype, np.uint8)
        self.assertEqual(records[1]['segmentation_box'], [7, 6, 15, 14])
        self.assertEqual(records[1]['saved_pixels'], 64)
        self.assertTrue((labels[small] == records[1]['frame_instance_id']).all())
        for record in records:
            self.assertEqual(record['saved_pixels'], int((labels == record['frame_instance_id']).sum()))
            self.assertEqual(record['instance_id'], -1)

    def test_dominated_empty_instance_is_not_published(self):
        mask = np.ones((32, 32), bool)
        records, labels = partition([({'object_name': 'a'}, mask), ({'object_name': 'b'}, mask)], mask.shape)
        self.assertEqual(len(records), 1)
        self.assertTrue((labels == records[0]['frame_instance_id']).all())

    def test_uint8_overflow_is_explicit_instead_of_silently_merging_instances(self):
        with self.assertRaises(ValueError):
            partition([({}, np.ones((1, 1), bool))]*256, (1, 1))

    def test_crops_are_bounded_and_reuse_the_same_noun_queries(self):
        records = [{'segmentation_box': [x, 10, x+20, 30]} for x in range(0, 1024, 100)]
        regions = zoom_regions(records, (768, 1024), ['bottle', 'plate'])
        self.assertLessEqual(len(regions), 4)
        for region in regions:
            x0, y0, x1, y1 = region['bounds']
            self.assertTrue(0 <= x0 < x1 <= 1024 and 0 <= y0 < y1 <= 768)
            self.assertEqual(region['queries'], ['bottle', 'plate'])


if __name__ == '__main__':
    unittest.main()

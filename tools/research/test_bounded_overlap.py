import numpy as np
from bounded_overlap import overlap

rng = np.random.default_rng(18)
for n in [0, 17, 300]:
    a = rng.random((8, n)) > .65
    b = rng.random((6, n)) > .75
    intersection = a.astype(np.int64) @ b.astype(np.int64).T
    union = a.sum(1)[:, None] + b.sum(1)[None, :] - intersection
    expected = np.divide(intersection, union, out=np.zeros_like(union, dtype=float), where=union > 0)
    np.testing.assert_array_equal(overlap(a, b), expected)
    permutation = rng.permutation(n)
    np.testing.assert_array_equal(overlap(a[:, permutation], b[:, permutation]), expected)
# Counts above uint8 range must not overflow.
np.testing.assert_array_equal(overlap(np.ones((2, 1000), bool), np.ones((3, 1000), bool)), np.ones((2, 3)))
print('BOUNDED_OVERLAP_EXACT_COUNTS_AND_PERMUTATION_PASSED')

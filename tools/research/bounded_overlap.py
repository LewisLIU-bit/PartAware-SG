"""Exact mask Jaccard with bounded temporary storage for large RGB-D domains.

Sparse int64 multiplication counts observed points; no model or score change.
Kept separate so previous signed experimental source files remain immutable.
"""
import numpy as np
from scipy.sparse import csr_matrix

def overlap(primary, peer):
    a, b = np.asarray(primary, bool), np.asarray(peer, bool)
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[1]:
        raise ValueError('Shared observed point domain required')
    # Sparse conversion allocates by actual positives, rather than two dense
    # int64 arrays with one entry for every candidate/point combination.
    intersection = (csr_matrix(a, dtype=np.int64) @ csr_matrix(b, dtype=np.int64).T).toarray()
    union = a.sum(1)[:, None] + b.sum(1)[None, :] - intersection
    return np.divide(intersection, union, out=np.zeros_like(union, dtype=float), where=union > 0)

"""CUE ownership, using only positive identity evidence from cached views.

Candidate scores, NMS, existence and measured surface union stay unchanged.
Two different frames only count as independent when their camera centers are
at least 15cm apart. No missing/background pixel is a negative observation.
"""
import numpy as np
from scipy.spatial import cKDTree


def core_owner(masks, scores, points):
    """The existing exclusive_core policy, applied to already selected masks."""
    masks = np.asarray(masks, bool)
    scores = np.asarray(scores, float)
    xyz = np.asarray(points, float)
    if masks.ndim != 2 or xyz.shape != (masks.shape[1], 3):
        raise ValueError('Shared observed point domain required')
    if scores.shape != (len(masks),) or not np.isfinite(xyz).all():
        raise ValueError('Finite observed points and one score per candidate required')
    count = masks.sum(0)
    owner = np.full(len(xyz), -1, np.int32)
    best = np.full(len(xyz), -np.inf)
    core_indices = []
    core_sizes = []
    for i, mask in enumerate(masks):
        vertices = np.flatnonzero(mask)
        indices = np.flatnonzero(mask & (count == 1))
        core_sizes.append(len(indices))
        if len(indices) < 16:
            indices = vertices
        core_indices.append(indices)
        if not len(vertices):
            continue
        distance = cKDTree(xyz[indices]).query(xyz[vertices], workers=2)[0]
        value = scores[i] / (1 + distance / .05)
        wins = value > best[vertices]
        chosen = vertices[wins]
        owner[chosen] = i
        best[chosen] = value[wins]
    return owner, count, core_indices, core_sizes


def independent_bits(centers, separation=.15):
    """Exact pairwise independence, invariant to rigid transforms and order."""
    centers = np.asarray(centers, float)
    if centers.ndim != 2 or centers.shape[1] != 3 or not np.isfinite(centers).all():
        raise ValueError('Finite original camera centers required')
    difference = centers[:, None, :] - centers[None, :, :]
    independent = np.linalg.norm(difference, axis=2) >= separation
    np.fill_diagonal(independent, False)
    words = (len(centers) + 63) // 64
    bits = np.zeros((len(centers), words), np.uint64)
    for frame in range(len(centers)):
        for other in np.flatnonzero(independent[frame]):
            bits[frame, other // 64] |= np.uint64(1) << np.uint64(other % 64)
    return bits


def unique_positive_owner(masks, baseline_owner, conflict_indices,
                          frame_labels, expected_labels, centers):
    """Override CORE only when one eligible competitor has independent support.

    frame_labels is the original local mask at a genuinely depth-visible point,
    with -1 for unknown and 0 for unannotated background. expected_labels is a
    candidate's positive per-frame identity, or -1 when it cannot be linked.
    Array indices are transient storage labels, never scene/GT rules.
    """
    masks = np.asarray(masks, bool)
    baseline = np.asarray(baseline_owner, np.int32)
    vertices = np.asarray(conflict_indices, np.int64)
    identity = np.asarray(expected_labels, np.int32)
    if masks.ndim != 2 or baseline.shape != (masks.shape[1],):
        raise ValueError('Shared candidate/owner domain required')
    if frame_labels.shape != (len(centers), len(vertices)) or identity.shape != (len(masks), len(centers)):
        raise ValueError('Original views and candidate identity mapping required')
    if np.any(vertices < 0) or np.any(vertices >= masks.shape[1]) or len(np.unique(vertices)) != len(vertices):
        raise ValueError('Unique valid conflict point indices required')
    if np.any(masks[:, vertices].sum(0) < 2):
        raise ValueError('Only genuinely disputed points may be arbitrated')
    bits = independent_bits(centers)
    eligible_count = np.zeros(len(vertices), np.uint16)
    eligible_owner = np.full(len(vertices), -1, np.int32)
    support_sizes = []
    for candidate in range(len(masks)):
        local = np.flatnonzero(masks[candidate, vertices])
        previous = np.zeros((len(local), bits.shape[1]), np.uint64)
        eligible = np.zeros(len(local), bool)
        for frame, label in enumerate(identity[candidate]):
            if label <= 0:
                continue
            positive = np.flatnonzero(frame_labels[frame, local] == label)
            if not len(positive):
                continue
            eligible[positive] |= np.any(previous[positive] & bits[frame], axis=1)
            previous[positive, frame // 64] |= np.uint64(1) << np.uint64(frame % 64)
        supported = local[eligible]
        eligible_count[supported] += 1
        eligible_owner[supported] = candidate
        support_sizes.append(int(len(supported)))
    result = baseline.copy()
    unique = eligible_count == 1
    result[vertices[unique]] = eligible_owner[unique]
    if np.any((result >= 0) != np.any(masks, axis=0)):
        raise AssertionError('Observed candidate union changed')
    if np.any((result >= 0) & ~masks[np.maximum(result, 0), np.arange(len(result))]):
        raise AssertionError('Ownership escaped the candidate measured support')
    audit = dict(policy='cached_view_unique_positive',
                 original_conflict_points=len(vertices),
                 uniquely_supported_conflict_points=int(unique.sum()),
                 multiple_independently_supported_conflict_points=int((eligible_count > 1).sum()),
                 unsupported_conflict_points=int((eligible_count == 0).sum()),
                 changed_conflict_owners=int(np.count_nonzero(result != baseline)),
                 candidate_supported_conflict_points=support_sizes,
                 unknown_is_negative=False, background_is_negative=False,
                 observed_union_preserved=True, exclusive_point_ownership=True,
                 generated_points=0, GT_used=False)
    return result, audit


def dominant_positive_identity(labels, minimum_positive=32, fraction=.7):
    """Link a core to one original frame mask; a failed link stays unknown."""
    labels = np.asarray(labels, np.int32)
    visible = labels[labels >= 0]
    positive = visible[visible > 0]
    if len(positive) < minimum_positive:
        return -1
    values, counts = np.unique(positive, return_counts=True)
    winner = int(np.argmax(counts))
    if counts[winner] < minimum_positive or counts[winner] < fraction * len(visible):
        return -1
    if np.count_nonzero(counts == counts[winner]) > 1:
        return -1
    return int(values[winner])

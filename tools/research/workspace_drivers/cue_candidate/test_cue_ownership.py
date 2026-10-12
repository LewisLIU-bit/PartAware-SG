"""Meaningful ownership regressions and representation invariances, no GT."""
import json
import sys
from pathlib import Path
import numpy as np

WORK = Path(__file__).resolve().parent
sys.path.insert(0, str(WORK.parent / 'research'))
import proposal_ensemble
from bounded_overlap import overlap
from core_ownership import propose as existing_core
from cue_ownership import core_owner, unique_positive_owner, dominant_positive_identity
proposal_ensemble.overlap = overlap


def test_positive_override_and_unknown():
    masks = np.array([[1, 1, 1, 0], [0, 1, 1, 1]], bool)
    baseline = np.array([0, 0, 0, 1], np.int32)
    conflict = np.array([1, 2])
    labels = np.array([[9, 0], [9, -1], [9, 0]], np.int32)
    identity = np.array([[4, 4, 4], [9, 9, 9]])
    centers = np.array([[0, 0, 0], [.2, 0, 0], [.2, 0, 0]])
    result, audit = unique_positive_owner(masks, baseline, conflict, labels, identity, centers)
    assert result.tolist() == [0, 1, 0, 1]
    assert audit['changed_conflict_owners'] == 1
    assert audit['unsupported_conflict_points'] == 1
    assert not audit['unknown_is_negative'] and not audit['background_is_negative']
    # Arbitrarily many duplicate crops/cameras cannot turn one observation into two.
    repeated_centers = np.zeros((3, 3))
    repeated, _ = unique_positive_owner(masks, baseline, conflict, labels, identity, repeated_centers)
    np.testing.assert_array_equal(repeated, baseline)
    # Two candidates linked to one undersegmented source mask are ambiguous.
    identity[0] = identity[1]
    ambiguous, audit = unique_positive_owner(masks, baseline, conflict, labels, identity, centers)
    np.testing.assert_array_equal(ambiguous, baseline)
    assert audit['multiple_independently_supported_conflict_points'] == 1


def test_invariances():
    masks = np.array([[1, 1, 1, 0, 1], [0, 1, 1, 1, 0]], bool)
    baseline = np.array([0, 0, 0, 1, 0], np.int32)
    conflict = np.array([1, 2])
    labels = np.array([[9, 4], [9, 4], [0, -1]], np.int32)
    identity = np.array([[4, 4, 4], [9, 9, 9]])
    centers = np.array([[0, 0, 0], [.2, 0, .1], [.5, 0, 0]])
    expected, _ = unique_positive_owner(masks, baseline, conflict, labels, identity, centers)
    rotation = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    moved, _ = unique_positive_owner(masks, baseline, conflict, labels, identity, centers @ rotation.T + [13, -9, 2])
    np.testing.assert_array_equal(expected, moved)
    order = np.array([2, 0, 1])
    reordered, _ = unique_positive_owner(masks, baseline, conflict, labels[order], identity[:, order], centers[order])
    np.testing.assert_array_equal(expected, reordered)
    permutation = np.array([4, 3, 1, 0, 2])
    inverse = np.argsort(permutation)
    new_conflict = inverse[conflict]
    changed_points, _ = unique_positive_owner(masks[:, permutation], baseline[permutation], new_conflict, labels, identity, centers)
    np.testing.assert_array_equal(expected, changed_points[inverse])
    swapped, _ = unique_positive_owner(masks[::-1], 1-baseline, conflict, labels, identity[::-1], centers)
    np.testing.assert_array_equal(expected, 1-swapped)
    remapped_labels = labels.copy()
    remapped_labels[labels == 4] = 807
    remapped_labels[labels == 9] = 23
    new_identity = identity.copy()
    new_identity[identity == 4] = 807
    new_identity[identity == 9] = 23
    renamed, _ = unique_positive_owner(masks, baseline, conflict, remapped_labels, new_identity, centers)
    np.testing.assert_array_equal(expected, renamed)


def test_exact_independent_pair_and_identity():
    # The first camera can sit between two independent cameras. Exact pair
    # evidence must find the outer pair, regardless of the first frame.
    masks = np.ones((2, 1), bool)
    owner = np.array([0], np.int32)
    labels = np.full((3, 1), 17, np.int32)
    identity = np.array([[4, 4, 4], [17, 17, 17]])
    centers = np.array([[0, 0, 0], [-.1, 0, 0], [.1, 0, 0]])
    result, _ = unique_positive_owner(masks, owner, np.array([0]), labels, identity, centers)
    assert result.tolist() == [1]
    assert dominant_positive_identity(np.r_[np.full(40, 29), np.zeros(10)]) == 29
    assert dominant_positive_identity(np.full(100, -1)) == -1
    assert dominant_positive_identity(np.zeros(100)) == -1
    assert dominant_positive_identity(np.r_[np.full(40, 29), np.full(40, 4)]) == -1


def test_core_control_matches_signed_module():
    rng = np.random.default_rng(36)
    xyz = rng.normal(size=(400, 3))
    a = rng.random((3, 400)) > .4
    b = rng.random((3, 400)) > .65
    sa, sb = np.array([.77, .71, .55]), np.array([.89, .83, .67])
    candidates, scores, selection = proposal_ensemble.propose(a, sa, b, sb, 'agreement', 1, xyz)
    baseline, count, cores, sizes = core_owner(candidates, scores, xyz)
    exact, ref_scores, ref_audit = existing_core(a, sa, b, sb, minimum_points=1, points=xyz)
    np.testing.assert_array_equal(np.array([baseline == i for i in range(len(scores))]), exact)
    np.testing.assert_array_equal(scores, ref_scores)
    assert selection['kept'] == ref_audit['kept']
    assert sizes == ref_audit['uncontested_core_sizes']
    rotation = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    moved, *_ = core_owner(candidates, scores, xyz @ rotation.T + [5, -7, 9])
    np.testing.assert_array_equal(moved, baseline)
    # Unknown cache leaves the exact control unchanged, including unowned points.
    conflict = np.flatnonzero(count > 1)
    unsupported, _ = unique_positive_owner(candidates, baseline, conflict,
        np.full((2, len(conflict)), -1, np.int32), np.full((len(scores), 2), -1, np.int32),
        np.array([[0, 0, 0], [1, 0, 0]]))
    np.testing.assert_array_equal(unsupported, baseline)


if __name__ == '__main__':
    tests = [test_positive_override_and_unknown, test_invariances,
             test_exact_independent_pair_and_identity, test_core_control_matches_signed_module]
    for test in tests:
        test()
        print('PASS', test.__name__, flush=True)
    (WORK / 'behavior_tests.json').write_text(json.dumps(dict(complete=True,
        GT_used=False, tests=[test.__name__ for test in tests],
        candidate_score_and_NMS_control_identical=True, unknown_and_background_neutral=True,
        rigid_transform_point_candidate_view_and_local_ID_invariances=True), indent=2)+'\n')

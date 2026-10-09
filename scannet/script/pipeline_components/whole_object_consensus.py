"""Associate measured object surfaces using image identity and view consensus.

Names may disagree across views. Auxiliary RN50 features remain separate from
the public DINO/SBERT embeddings. This adapter generates no hidden geometry.
Detach WHOLE_OBJECT_VALIDATION from the v12 code registry to remove it.
"""
import hashlib
from pathlib import Path

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree


def cosine(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return float(a @ b / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-9))


def visual_identity(a, b, similarity):
    return (a.get('semantic_views', 0) >= 2 and b.get('semantic_views', 0) >= 2
            and min(a.get('semantic_camera_baseline_m', 0),
                    b.get('semantic_camera_baseline_m', 0)) >= .08
            and a.get('semantic_best_class') == b.get('semantic_best_class')
            and a.get('semantic_best_class') is not None
            and min(a.get('semantic_object_score', 0)-a.get('semantic_background_score', 0),
                    b.get('semantic_object_score', 0)-b.get('semantic_background_score', 0)) >= .01
            and similarity >= .8)


def acceptance(evidence, geometry, semantic, visual):
    """Require positive whole-object evidence; invisibility never votes against it."""
    whole = evidence['whole_mask_support_views']
    separate = evidence['independent_separation_views']
    consensus = evidence['whole_mask_consensus']
    if (geometry['box_iou'] >= .9 and geometry['bidirectional_surface_coverage'] >= .8
            and semantic >= .6 and whole >= 5 and consensus >= .5 and separate == 0):
        return 'shared_measured_surface'
    if not visual:
        return None
    if (geometry['box_iou'] >= .6 and geometry['closest_surface_m'] <= .04
            and whole >= 3 and consensus >= .8 and separate == 0):
        return 'complementary_measured_body'
    if (geometry['aligned_terminal_face'] and geometry['closest_surface_m'] <= .05
            and whole >= 1 and evidence['joint_visible_views'] >= 2
            and consensus >= .5 and separate <= 1):
        return 'aligned_measured_terminal_face'
    return None


def geometry_evidence(a, b):
    lower_a, upper_a = a.min(0), a.max(0)
    lower_b, upper_b = b.min(0), b.max(0)
    span_a, span_b = upper_a-lower_a, upper_b-lower_b
    intersection = np.maximum(np.minimum(upper_a, upper_b)-np.maximum(lower_a, lower_b), 0)
    volume_a, volume_b, shared = np.prod(span_a), np.prod(span_b), np.prod(intersection)
    tree_a, tree_b = cKDTree(a), cKDTree(b)
    distances_a, distances_b = tree_b.query(a)[0], tree_a.query(b)[0]
    faces = []
    for face, span, other_span in [(a, span_a, span_b), (b, span_b, span_a)]:
        sample = face[::max(1, len(face)//2048)]
        o3d.utility.random.seed(29)
        cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(sample))
        _, plane_indices = cloud.segment_plane(.01, 3, 200)
        plane_fraction = len(plane_indices)/len(sample)
        covariance = np.cov(sample[plane_indices].T)
        values, vectors = np.linalg.eigh(covariance)
        normal = vectors[:, 0]
        axis = int(np.argmax(np.abs(normal)))
        lateral = [k for k in range(3) if k != axis]
        cross_overlap = np.prod(intersection[lateral])/max(
            np.prod(span_a[lateral]), np.prod(span_b[lateral]), 1e-9)
        # Normalize against the smaller in-plane variance. A long rectangular
        # face is still planar; its length must not suppress its flatness score.
        planarity = float((values[1]-values[0])/max(values[1], 1e-9))
        joined_span = np.maximum(upper_a, upper_b)-np.minimum(lower_a, lower_b)
        compatible = (plane_fraction >= .7 and abs(normal[axis]) >= .9 and planarity >= .8
                      and cross_overlap >= .8 and span[axis] <= .5*other_span[axis]
                      and joined_span[axis] <= 1.7*other_span[axis])
        faces.append({'axis': axis, 'planarity': planarity, 'plane_support_fraction': plane_fraction,
                      'lateral_overlap': float(cross_overlap), 'compatible': bool(compatible)})
    return {'box_iou': float(shared/max(volume_a+volume_b-shared, 1e-9)),
            'bidirectional_surface_coverage': float(min(np.mean(distances_a <= .03),
                                                       np.mean(distances_b <= .03))),
            'closest_surface_m': float(distances_a.min()),
            'aligned_terminal_face': any(f['compatible'] for f in faces), 'faces': faces}


def reconcile(survivors, geometry, tracks, metrics, views, records, remap,
              audit, nodes, aliases, embeddings):
    """Merge only mutually unique compatible owners, then publish ordinary nodes."""
    from .proposal_validation import raw_pair_evidence, resolve
    candidates, decisions = {}, {}
    for i, left in enumerate(survivors):
        a = geometry[left]
        if len(a) < 128:
            continue
        for right in survivors[i+1:]:
            b = geometry[right]
            if len(b) < 128:
                continue
            # Bounding gaps cheaply reject remote objects before nearest-neighbor work.
            gap = np.maximum(np.maximum(a.min(0)-b.max(0), b.min(0)-a.max(0)), 0)
            if np.linalg.norm(gap) > .05:
                continue
            semantic = cosine(nodes[left]['text_embedding'], nodes[right]['text_embedding'])
            appearance = cosine(embeddings[left], embeddings[right]) if left in embeddings and right in embeddings else 0.
            visual = visual_identity(metrics[left], metrics[right], appearance)
            if not visual and semantic < .6:
                continue
            shape = geometry_evidence(a, b)
            if shape['closest_surface_m'] > .05:
                continue
            evidence = raw_pair_evidence(a[::max(1, len(a)//2048)], b[::max(1, len(b)//2048)],
                views, tracks[left]['observed_frames']+tracks[right]['observed_frames'], minimum=.6)
            role = acceptance(evidence, shape, semantic, visual)
            detail = {'pair': [left, right], 'role': role, 'name_cosine': semantic,
                      'image_cosine': appearance, 'same_image_identity': visual,
                      **shape, **evidence}
            decisions[(left, right)] = detail
            if role:
                candidates.setdefault(left, []).append(right)
                candidates.setdefault(right, []).append(left)
    for (left, right), evidence in decisions.items():
        if evidence['role'] is None:
            audit.append({'message': '多视角图像身份与完整物体表面关联', 'accepted': False,
                          'reason': '独立分离证据或完整物体证据不足', **evidence})
            continue
        unique = len(candidates[left]) == len(candidates[right]) == 1
        audit.append({'message': '多视角图像身份与完整物体表面关联', 'accepted': unique,
                      'reason': '唯一相互兼容归属' if unique else '多个兼容归属，保留独立候选', **evidence})
        if not unique:
            continue
        target, source = sorted([left, right], key=lambda gid: -len(geometry[gid]))
        remap[source] = target
        # Retain actual measured samples, not voxel centroids or generated surfaces.
        combined = np.concatenate([geometry[target], geometry[source]])
        _, indices = np.unique(np.floor(combined/.005).astype(np.int64), axis=0, return_index=True)
        geometry[target] = combined[np.sort(indices)]
        na, nb = len(tracks[target]['observations']), len(tracks[source]['observations'])
        nodes[target]['visual_embedding'] = ((na*np.asarray(nodes[target]['visual_embedding'])
            +nb*np.asarray(nodes[source]['visual_embedding']))/(na+nb)).tolist()
        if evidence['same_image_identity']:
            # Select an existing identity from the more discriminative image view;
            # retain its corresponding SBERT vector instead of averaging names.
            margin = lambda gid: (metrics[gid]['semantic_object_score']
                                  -metrics[gid]['semantic_background_score'])
            identity = max([target, source], key=margin)
            nodes[target]['name'] = nodes[identity]['name']
            nodes[target]['text_embedding'] = nodes[identity]['text_embedding']
        for name, count in tracks[source].get('name_votes', {}).items():
            votes = tracks[target].setdefault('name_votes', {})
            votes[name] = votes.get(name, 0)+count
        tracks[target]['observations'] += tracks[source]['observations']
        tracks[target]['observed_frames'] = sorted(set(tracks[target]['observed_frames']+tracks[source]['observed_frames']))
        tracks[target]['confidence'] = (na*tracks[target]['confidence']+nb*tracks[source]['confidence'])/(na+nb)
        tracks[target]['point_count'] = len(geometry[target])
        tracks[target].setdefault('whole_object_source_ids', [target]).append(source)
        aliases[source] = {'canonical_id': target, 'source_name': nodes[source]['name'],
                           'role': evidence['role'], 'evidence': evidence}
    return [gid for gid in survivors if gid not in remap]


def component_sha256():
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()

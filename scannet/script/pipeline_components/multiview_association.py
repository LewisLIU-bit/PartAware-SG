"""Visibility-aware one-to-one association inspired by hierarchical 3D graphs."""
import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
from partaware.fusion import color_similarity
from partaware.fusion import PartFusion as BasePartFusion, Track, normalize, overlap
from partaware.geometry import voxel_downsample
from collections import Counter


def visibility_score(observation, track, frame):
    points = track.points
    camera = (points - frame['pose'][:3, 3]) @ frame['pose'][:3, :3]
    valid = camera[:, 2] > 0
    camera = camera[valid]
    if not len(camera):
        return None
    k = frame['kc']
    uv = np.round(camera[:, :2] / camera[:, 2:3] * np.array([k[0, 0], k[1, 1]])
                  + np.array([k[0, 2], k[1, 2]])).astype(int)
    h, w = observation.mask.shape
    valid = (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
    uv, camera = uv[valid], camera[valid]
    if not len(uv):
        return None
    # Convert color coordinates to the registered depth camera before testing occlusion.
    kd = frame['kd']
    du = np.round((uv[:, 0] - k[0, 2]) * kd[0, 0] / k[0, 0] + kd[0, 2]).astype(int)
    dv = np.round((uv[:, 1] - k[1, 2]) * kd[1, 1] / k[1, 1] + kd[1, 2]).astype(int)
    depth = frame['depth']
    valid = (du >= 0) & (dv >= 0) & (du < depth.shape[1]) & (dv < depth.shape[0])
    uv, camera, du, dv = uv[valid], camera[valid], du[valid], dv[valid]
    measured = depth[dv, du] / frame['scale']
    visible = (measured > 0) & (np.abs(camera[:, 2] - measured) <= 0.05 + 0.01 * measured)
    uv = uv[visible]
    if len(uv) < 8:
        return None
    silhouette = np.zeros((h, w), np.uint8)
    silhouette[uv[:, 1], uv[:, 0]] = 1
    silhouette = cv2.dilate(silhouette, np.ones((5, 5), np.uint8)).astype(bool)
    projected_coverage = float(np.count_nonzero(silhouette & observation.mask) / silhouette.sum())
    a = cv2.boundingRect(silhouette.astype(np.uint8))
    b = cv2.boundingRect(observation.mask.astype(np.uint8))
    intersection = max(0, min(a[0]+a[2], b[0]+b[2])-max(a[0], b[0])) * max(
        0, min(a[1]+a[3], b[1]+b[3])-max(a[1], b[1]))
    iou = intersection / max(a[2]*a[3] + b[2]*b[3] - intersection, 1)
    if iou == 0 or projected_coverage < 0.1:
        return None
    return {'projected_bbox_iou': float(iou), 'projected_coverage': projected_coverage,
            'visible_points': int(len(uv)), 'color_similarity': color_similarity(observation.color, track.color)}


def score(base, evidence):
    return 0.5 * (base / 2) + 0.25 * evidence['projected_bbox_iou'] + 0.15 * evidence['projected_coverage'] + 0.1 * evidence['color_similarity']


def assign(scores):
    """Include explicit unmatched columns so invalid pairs cannot be forced."""
    if scores.shape[0] == 0:
        return {}
    padded = np.concatenate([scores, np.zeros((scores.shape[0], scores.shape[0]))], axis=1)
    rows, columns = linear_sum_assignment(-padded)
    return {int(r): int(c) for r, c in zip(rows, columns) if c < scores.shape[1] and scores[r, c] > 0}


class PartFusion(BasePartFusion):
    """Accumulate parent evidence across views, including initially unassigned parts."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.parent_votes = {}
        self.feature_sums = {}

    def add(self, frame_id, detection_id, label, parent_id, points, feature, color, confidence):
        points, feature = voxel_downsample(points, self.voxel), normalize(feature)
        best, best_score = None, -1.
        for track in self.tracks:
            if track.label != label or frame_id in track.frames:
                continue
            if parent_id is not None and track.parent_id is not None and parent_id != track.parent_id:
                continue
            geometry, semantic = overlap(points, track.points, self.radius), float(feature @ track.feature)
            if geometry < self.min_overlap or semantic < self.min_semantic:
                continue
            score_value = 2 * ((1-self.color_weight) * (geometry+(semantic+1)/2)
                               + self.color_weight * color_similarity(color, track.color))
            if score_value >= self.threshold and score_value > best_score:
                best, best_score = track, score_value
        observation = {'frame_id': frame_id, 'detection_id': detection_id, 'confidence': float(confidence),
                       'observed_parent_id': parent_id}
        if best is None:
            best = Track(f'part_{len(self.tracks)+1}', label, None, points, feature, color,
                         float(confidence), {frame_id}, [observation])
            self.tracks.append(best)
            self.feature_sums[best.id] = feature.copy()
            self.parent_votes[best.id] = Counter()
            action = 'new'
        else:
            count = len(best.observations)
            self.feature_sums[best.id] += feature
            best.feature = normalize(self.feature_sums[best.id])
            best.points = voxel_downsample(np.concatenate([best.points, points]), self.voxel)
            best.color = (best.color * count + color) / (count + 1)
            best.confidence = (count * best.confidence + confidence) / (count + 1)
            best.frames.add(frame_id)
            best.observations.append(observation)
            action = 'merged'
        if parent_id is not None:
            self.parent_votes[best.id][parent_id] += 1
        votes = self.parent_votes[best.id]
        if votes:
            parent, count = votes.most_common(1)[0]
            best.parent_id = parent if count >= 2 and count / sum(votes.values()) >= .7 else None
        return best.id, action

    def export(self):
        nodes, edges = super().export()
        for key, node in nodes.items():
            votes = self.parent_votes[key]
            node['parent_evidence'] = dict(votes)
            node['parent_evidence_frames'] = sum(votes.values())
        return nodes, edges

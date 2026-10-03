"""Conservative OP3DSG-inspired association and independent part tracks."""
from dataclasses import dataclass, field
import cv2
import numpy as np
from scipy.spatial import cKDTree
from scipy.stats import wasserstein_distance
from .geometry import voxel_downsample


def normalize(vector):
    vector = np.asarray(vector, dtype=float)
    if vector.ndim != 1 or not np.isfinite(vector).all() or np.linalg.norm(vector) < 1e-9:
        raise ValueError('Expected a finite, nonzero feature vector')
    return vector / np.linalg.norm(vector)


def overlap(a, b, radius):
    if not len(a) or not len(b):
        return 0.0
    ab = float(np.mean(cKDTree(b).query(a, k=1)[0] <= radius))
    ba = float(np.mean(cKDTree(a).query(b, k=1)[0] <= radius))
    # Directional coverage allows a partial observation to match a larger track.
    return max(ab, ba)


def color_histogram(bgr, bins=32):
    if not len(bgr):
        raise ValueError('Cannot encode an empty color sample')
    rgb = np.asarray(bgr, dtype=np.float32).reshape(-1, 1, 3) / 255
    lab = cv2.cvtColor(rgb, cv2.COLOR_BGR2LAB).reshape(-1, 3)
    r, g, b = rgb.reshape(-1, 3)[:, ::-1].T
    channels = [np.clip((lab[:, 1] + 128) / 256, 0, 1),
                np.clip((lab[:, 2] + 128) / 256, 0, 1),
                (r - g + 1) / 2, (r + g - 2 * b + 2) / 4]
    hist = np.asarray([np.histogram(c, bins=bins, range=(0, 1))[0] for c in channels], dtype=float)
    return hist / np.maximum(hist.sum(axis=1, keepdims=True), 1)


def color_similarity(a, b):
    if a.shape != b.shape:
        raise ValueError('Color feature dimensions disagree')
    centers = (np.arange(a.shape[1]) + 0.5) / a.shape[1]
    distance = np.mean([wasserstein_distance(centers, centers, u_weights=x, v_weights=y)
                        for x, y in zip(a, b)])
    return float(1 / (1 + 3 * distance))


@dataclass
class Track:
    id: str
    label: str
    parent_id: str | None
    points: np.ndarray
    feature: np.ndarray
    color: np.ndarray
    confidence: float
    frames: set = field(default_factory=set)
    observations: list = field(default_factory=list)


class PartFusion:
    def __init__(self, voxel=0.01, radius=0.03, min_overlap=0.2,
                 min_semantic=0.7, threshold=1.5, color_weight=0.6, min_frames=2):
        values = np.asarray([voxel, radius, min_overlap, min_semantic, threshold, color_weight])
        if not np.isfinite(values).all() or voxel <= 0 or radius <= 0 or min_frames < 1:
            raise ValueError('Invalid fusion parameters')
        if not 0 <= min_overlap <= 1 or not -1 <= min_semantic <= 1 or not 0 <= color_weight <= 1:
            raise ValueError('Similarity weights and gates are out of range')
        self.voxel, self.radius = voxel, radius
        self.min_overlap, self.min_semantic = min_overlap, min_semantic
        self.threshold, self.color_weight, self.min_frames = threshold, color_weight, min_frames
        self.tracks = []

    def add(self, frame_id, detection_id, label, parent_id, points, feature, color, confidence):
        points = voxel_downsample(points, self.voxel)
        feature = normalize(feature)
        best, best_score = None, -1.0
        for track in self.tracks:
            # A local observation is not a structural part; never merge co-visible parts.
            if track.label != label or track.parent_id != parent_id or frame_id in track.frames:
                continue
            if track.feature.shape != feature.shape:
                raise ValueError('Part feature dimensions changed within a run')
            geometry = overlap(points, track.points, self.radius)
            semantic = float(feature @ track.feature)
            if geometry < self.min_overlap or semantic < self.min_semantic:
                continue
            score = (1 - self.color_weight) * (geometry + (semantic + 1) / 2)
            score += self.color_weight * color_similarity(color, track.color)
            # The threshold is expressed on the unweighted geometry+semantic scale.
            score *= 2
            if score >= self.threshold and score > best_score:
                best, best_score = track, score
        observation = {'frame_id': frame_id, 'detection_id': detection_id,
                       'confidence': float(confidence)}
        if best is None:
            best = Track(f'part_{len(self.tracks) + 1}', label, parent_id, points,
                         feature, color, float(confidence), {frame_id}, [observation])
            self.tracks.append(best)
            return best.id, 'new'
        count = len(best.observations)
        best.feature = normalize(best.feature * count + feature)
        best.points = voxel_downsample(np.concatenate((best.points, points)), self.voxel)
        best.color = 0.7 * best.color + 0.3 * color
        best.color /= best.color.sum(axis=1, keepdims=True)
        best.confidence = (best.confidence * count + confidence) / (count + 1)
        best.frames.add(frame_id)
        best.observations.append(observation)
        return best.id, 'merged'

    def export(self):
        nodes, edges = {}, []
        for track in self.tracks:
            center = track.points.mean(axis=0)
            extent = np.ptp(track.points, axis=0)
            confirmed = len(track.frames) >= self.min_frames
            nodes[track.id] = {'id': track.id, 'name': track.label, 'node_type': 'part',
                              'parent_id': track.parent_id, 'position': center.tolist(),
                              'extent': extent.tolist(), 'confidence': track.confidence,
                              'status': 'confirmed' if confirmed else 'provisional',
                              'observed_frames': sorted(track.frames), 'point_count': len(track.points),
                              'observations': track.observations, 'semantic_embedding': track.feature.tolist(),
                              'semantic_feature_space': 'clip_rn50_masked_crop_1024'}
            if confirmed and track.parent_id is not None:
                edges.append({'source_id': track.id, 'target_id': track.parent_id,
                              'description': 'part_of', 'evidence_frames': len(track.frames)})
        return nodes, edges

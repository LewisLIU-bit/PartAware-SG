"""Recover small measured surfaces before testing atomic boundary hypotheses.

This adapter queries signed native masks, samples registered depth at stride one
and requires independent view support. It never reads annotations or changes
language prompts. Dense sampling alone is not atomic instance segmentation.
"""
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial import cKDTree

from partaware.geometry import project_mask
from .hierarchical_masks import normalize
from .proposal_validation import sample_frames


def collect(points, track, views, native, maximum_frames=24):
    """Return measured samples and their camera IDs inside a verified identity."""
    from collections import Counter
    names = Counter(normalize(o['name']) for o in track['observations'])
    names = {n for n, count in names.items() if count >= .25*sum(names.values())}
    sample = points[::max(1, len(points)//1024)]
    tree = cKDTree(points)
    chunks, colors, frames, audit = [], [], [], []
    for fid in sample_frames(track['observed_frames'], maximum_frames):
        if fid not in native.expected:
            continue
        mass = native.mass(sample, fid)
        if mass is None:
            continue
        choices = [(float(mass[i]), float(r['confidence']), i)
                   for i, r in enumerate(native.records(fid))
                   if normalize(r['object_name']) in names]
        if not choices:
            continue
        fraction, confidence, index = max(choices)
        if fraction < .7 or confidence < .5:
            continue
        pose, depth, public = views.get(fid)
        mask = np.unpackbits(native.get(fid)[index], axis=-1, count=public.shape[1]).astype(bool)
        rgb = cv2.imread(str(views.jobs[fid]['rgb']))
        q, c = project_mask(mask, rgb, depth, pose, views.kd, views.kc, views.scale, stride=1)
        keep = tree.query(q)[0] <= .025
        q, c = q[keep], c[keep]
        if len(q) < 64:
            continue
        frames.append(fid)
        chunks.append(q)
        colors.append(c)
        audit.append(dict(frame_id=fid, native_mask_index=index,
                          native_surface_fraction=fraction, points=len(q)))
    if not chunks:
        return points.copy(), np.zeros(len(points), int), [], audit
    all_points = np.concatenate(chunks)
    view_ids = np.concatenate([np.full(len(q), i, int) for i, q in enumerate(chunks)])
    return all_points, view_ids, frames, audit


def supported_samples(points, view_ids, voxel=.002, minimum_views=3):
    """Select real samples in cells occupied in several independent frames."""
    keys = np.floor(points/voxel).astype(np.int64)
    unique, first, inverse = np.unique(keys, axis=0, return_index=True, return_inverse=True)
    pairs = np.unique(np.column_stack([inverse, view_ids]), axis=0)
    counts = np.bincount(pairs[:, 0], minlength=len(unique))
    selected = first[counts >= minimum_views]
    return points[selected], counts[counts >= minimum_views]


def edge_samples(points, track, views):
    """Acquire depth on RGB boundaries in a bounded, measured surface envelope.

    The native foreground mask can stop at the top of a nested stack. Its 3D
    seed envelope defines a local search, while metric proximity excludes remote
    scenery. This returns evidence for split testing, never automatic objects.
    """
    tree = cKDTree(points)
    sample = points[::max(1, len(points)//2048)]
    chunks, values, frames, audits = [], [], [], []
    for fid in sample_frames(track['observed_frames'], 24):
        indices, _, _ = views.project(sample, fid)
        if len(indices) < 32:
            continue
        pose, depth, public = views.get(fid)
        camera = (sample[indices]-pose[:3, 3]) @ pose[:3, :3]
        uv = np.rint(camera[:, :2]/camera[:, 2, None]*[views.kc[0, 0], views.kc[1, 1]]
                      +[views.kc[0, 2], views.kc[1, 2]]).astype(int)
        lower = np.maximum(uv.min(0)-8, 0)
        upper = np.minimum(uv.max(0)+9, [public.shape[1], public.shape[0]])
        if np.prod(upper-lower) < 512:
            continue
        rgb = cv2.imread(str(views.jobs[fid]['rgb']))
        gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY).astype(float)
        gx, gy = cv2.Sobel(gray, cv2.CV_64F, 1, 0), cv2.Sobel(gray, cv2.CV_64F, 0, 1)
        magnitude = np.hypot(gx, gy)
        patch = magnitude[lower[1]:upper[1], lower[0]:upper[0]]
        threshold = max(8., float(np.quantile(patch, .75)))
        mask = np.zeros(public.shape, bool)
        mask[lower[1]:upper[1], lower[0]:upper[0]] = patch >= threshold
        q, _ = project_mask(mask, rgb, depth, pose, views.kd, views.kc, views.scale, stride=1)
        keep = tree.query(q)[0] <= .025
        q = q[keep]
        if len(q) < 32:
            continue
        frames.append(fid); chunks.append(q)
        values.append(np.full(len(q), len(frames)-1, int))
        audits.append(dict(frame_id=fid, points=len(q), gradient_threshold=threshold))
    if not chunks:
        return np.empty((0, 3)), np.empty(0, int), [], audits
    return np.concatenate(chunks), np.concatenate(values), frames, audits

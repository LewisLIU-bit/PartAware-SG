"""Correct small visible-surface errors without reshaping hidden geometry."""
import numpy as np
from scipy.spatial import cKDTree


def measured_surface_projection(observed, candidate, gid, views):
    """Snap only near-anchor, own-instance free-space errors to measured points.

    Large contradictions and foreign-instance errors remain untouched so the
    whole-model validator can reject them. This is independent of object class.
    """
    candidate = candidate.copy()
    distance, nearest = cKDTree(observed).query(candidate)
    extent = max(float(np.linalg.norm(np.ptp(observed, axis=0))), .02)
    corrected = np.zeros(len(candidate), bool)
    for fid in views.jobs:
        pose, raw, mask = views.get(fid)
        camera = (candidate-pose[:3, 3]) @ pose[:3, :3]
        z = camera[:, 2]
        uv = np.rint(camera[:, :2]/np.maximum(z[:, None], 1e-8)*[views.kd[0, 0], views.kd[1, 1]]
                      +[views.kd[0, 2], views.kd[1, 2]]).astype(int)
        inside = (z > 0) & (uv[:, 0] >= 0) & (uv[:, 1] >= 0) & (uv[:, 0] < raw.shape[1]) & (uv[:, 1] < raw.shape[0])
        ix = np.flatnonzero(inside & ~corrected & (distance <= .05*extent))
        depth = raw[uv[ix, 1], uv[ix, 0]]/views.scale
        rgb_uv = np.rint(camera[ix, :2]/z[ix, None]*[views.kc[0, 0], views.kc[1, 1]]
                          +[views.kc[0, 2], views.kc[1, 2]]).astype(int)
        available = (depth > 0) & (z[ix] < depth-(.012+.003*depth))
        available &= (rgb_uv[:, 0] >= 0) & (rgb_uv[:, 1] >= 0) & (rgb_uv[:, 0] < mask.shape[1]) & (rgb_uv[:, 1] < mask.shape[0])
        ix, rgb_uv = ix[available], rgb_uv[available]
        mapping = {int(record['frame_instance_id']): str(record.get('instance_id', -1)) for record in views.records[fid]}
        own = np.array([mapping.get(int(value)) == gid for value in mask[rgb_uv[:, 1], rgb_uv[:, 0]]], bool)
        corrected[ix[own]] = True
        candidate[ix[own]] = observed[nearest[ix[own]]]
    fraction = float(corrected.mean())
    rms = float(np.sqrt(np.mean(np.where(corrected, distance**2, 0.))))
    # Visible area varies with viewpoint, so a count cap would reject an accurate
    # front-heavy model. Bound metric deformation instead; all unpublished hidden
    # surfaces must still pass the unchanged whole-model and two-view checks.
    if rms > .025*extent:
        return None, {'reason': 'Measured-surface fitting exceeds the bounded metric deformation',
                      'corrected_surface_fraction': fraction, 'correction_rms_m': rms}
    return candidate, {'corrected_surface_fraction': fraction,
        'maximum_correction_m': float(distance[corrected].max()) if corrected.any() else 0., 'correction_rms_m': rms,
        'measured_points_unchanged': True, 'hidden_points_unchanged': True,
        'correction_rule': 'own_instance_visible_free_points_within_5_percent_input_diagonal_only'}

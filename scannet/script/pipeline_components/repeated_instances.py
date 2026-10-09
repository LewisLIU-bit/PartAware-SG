"""Resolve repeated compact surfaces from multiview RGB-D boundaries.

RSI is a project-specific geometric split/model-selection adapter, not the
trained RICE network or a general amodal completion model. Visible repeated
rims establish independent hypotheses; a measured top surface supplies a
shared shape prior. Translated surfaces remain generated hypotheses and must
pass a free-space test. No object IDs, category list or GT enter construction.
Detach this MEASURED_REFINEMENT entry to remove RSI and fine depth recovery.
"""
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import open3d as o3d
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d
from scipy.spatial import cKDTree
from scipy.spatial import ConvexHull, QhullError

from partaware.geometry import load_capture
from .fine_boundary import collect, edge_samples, supported_samples
from .instance_consensus import Views
from .native_assembly import NativeMasks
from .part_geometry import replace_regions


def unique_samples(points, voxel=.002):
    """Select original samples, keeping measured coordinates unchanged."""
    _, first = np.unique(np.floor(points/voxel).astype(np.int64), axis=0, return_index=True)
    return points[np.sort(first)]


def eligible(points):
    span = np.ptp(points, axis=0)
    return (len(points) >= 128 and .1 < span[:2].max() < .5
            and .025 < span[2] < .15 and span[:2].min() > 3*span[2])


def boundary_band(points, footprint=None):
    """Separate exposed outer rims from unrelated inner bowl/profile edges."""
    footprint = points if footprint is None else footprint
    try:
        planes = ConvexHull(footprint[:, :2]).equations
    except QhullError:
        return np.zeros(len(points), bool)
    distance = -(points[:, :2] @ planes[:, :2].T+planes[:, 2]).max(axis=1)
    width = float(np.clip(.05*np.ptp(footprint[:, :2], axis=0).max(), .005, .015))
    return (distance >= -.008) & (distance <= width)


def fit_repetition(points, view_ids, minimum_frame_points=16, footprint=None):
    """Select a regular sequence with independently measured edges at every rim."""
    audit = dict(accepted=False, reason='no_stable_regular_boundary_sequence', candidates=[])
    if len(points) < 128 or len(np.unique(view_ids)) < 5:
        return [], audit
    band = boundary_band(points, footprint)
    points, view_ids = points[band], view_ids[band]
    if len(points) < 128:
        return [], audit
    step = .0005
    bins = np.arange(points[:, 2].min()-step, points[:, 2].max()+2*step, step)
    histogram, _ = np.histogram(points[:, 2], bins=bins)
    smooth = gaussian_filter1d(histogram.astype(float), .8)
    indices, _ = find_peaks(smooth, prominence=.08*smooth.max(), distance=6)
    peaks = []
    for index in indices:
        center = (bins[index]+bins[index+1])/2
        selected = np.abs(points[:, 2]-center) <= .0015
        votes = np.bincount(view_ids[selected])
        support = np.flatnonzero(votes >= minimum_frame_points)
        rim = points[selected]
        if len(support) < 5 or np.ptp(rim[:, :2], axis=0).min() < .08:
            continue
        center = float(np.median(rim[:, 2]))
        detail = dict(height_m=center, supporting_views=support.tolist(), edge_points=len(rim),
                      xy_extent_m=np.ptp(rim[:, :2], axis=0).tolist())
        peaks.append(detail); audit['candidates'].append(detail)
    chains = []
    for start in range(len(peaks)):
        for stop in range(start+4, len(peaks)+1):
            selected = peaks[start:stop]
            heights = np.array([p['height_m'] for p in selected])
            gaps = np.diff(heights); period = float(np.median(gaps))
            residual = float(np.max(np.abs(gaps-period)))
            if not (.004 <= period <= .02 and heights[-1]-heights[0] >= .02
                    and residual <= max(.00075, .15*period)):
                continue
            chains.append((len(selected), -residual/period, selected, period))
    if not chains:
        return [], audit
    _, _, selected, period = max(chains, key=lambda value: value[:2])
    # A competing equally long sequence is ambiguous rather than a target count.
    competitors = [c for c in chains if c[0] == len(selected)]
    if len(competitors) > 1:
        audit['reason'] = 'ambiguous_regular_boundary_sequences'
        return [], audit
    audit.update(accepted=True, reason='repeated_multiview_metric_boundaries', period_m=period,
                 layer_count=len(selected), selected=selected)
    return selected, audit


def free_space(hypothesis, views, frames):
    """A predicted surface may be hidden, but must not occupy observed free space."""
    sample = hypothesis[::max(1, len(hypothesis)//1024)]
    contradicted = np.zeros(len(sample), int)
    inspected = np.zeros(len(sample), int)
    for fid in frames:
        pose, depth, mask = views.get(fid)
        camera = (sample-pose[:3, 3]) @ pose[:3, :3]
        z = camera[:, 2]
        uv = np.rint(camera[:, :2]/np.maximum(z, 1e-8)[:, None]
                      *[views.kd[0, 0], views.kd[1, 1]]+[views.kd[0, 2], views.kd[1, 2]]).astype(int)
        valid = ((z > 0) & (uv[:, 0] >= 0) & (uv[:, 1] >= 0)
                 & (uv[:, 0] < depth.shape[1]) & (uv[:, 1] < depth.shape[0]))
        indices = np.flatnonzero(valid)
        # A sample occupies a raster footprint, not an infinitesimal pixel.
        # Using the nearest valid depth in its 3x3 footprint avoids calling a
        # measured edge free space when rounding hits the background beside it.
        measured = np.full(len(indices), np.inf)
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                xx = np.clip(uv[indices, 0]+dx, 0, depth.shape[1]-1)
                yy = np.clip(uv[indices, 1]+dy, 0, depth.shape[0]-1)
                neighborhood = depth[yy, xx]/views.scale
                measured = np.minimum(measured, np.where(neighborhood > 0, neighborhood, np.inf))
        observed = np.isfinite(measured)
        indices, measured = indices[observed], measured[observed]
        inspected[indices] += 1
        contradicted[indices] += z[indices] < measured-.005
    repeated = (inspected >= 3) & (contradicted >= 3)
    fraction = float(np.mean(repeated))
    return fraction <= .02, dict(repeated_free_space_conflict_fraction=fraction,
                                 tested_points=len(sample), depth_tolerance_m=.005,
                                 raster_footprint_radius_pixels=1)


def construct(context):
    path = context.scene/'topology_map.json'
    graph = json.loads(path.read_text()); nodes = graph['object_nodes']['nodes']
    report = dict(algorithm='RSI_multiview_boundary_repetition_v13', ground_truth_used=False,
                  vision_api_calls=0, generated_points_are_measurements=False,
                  other_object_coordinates_preserved=True, splits=[], dense_recovery=[],
                  component_sha256=component_sha256())
    data, jobs, kd, kc, scale = load_capture(context.manifest, context.image_dir,
                                           context.scene/'refined_instance')
    if data.get('world_frame') != 'hypersim_world_z_up':
        report['reason'] = '未声明可靠竖直轴，保留原实例与几何'
        (context.scene/'repeated_instance_audit.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
        return
    views = Views(context, jobs, kd, kc, scale, {})
    native = NativeMasks(views)
    if not native.available:
        report['reason'] = '无签名核验的原生掩码，保留原实例'
        (context.scene/'repeated_instance_audit.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
        return
    tracks_path = context.scene/'validated_object_tracks.json'
    tracks = json.loads(tracks_path.read_text())
    cloud_path = context.scene/'instance_cloud_cleaned.ply'
    cloud = o3d.io.read_point_cloud(str(cloud_path)); all_points = np.asarray(cloud.points)
    rgb = np.rint(np.asarray(cloud.colors)*255).astype(int)
    ids = rgb[:, 0]+255*rgb[:, 1]+255**2*rgb[:, 2]
    geometry = {gid: all_points[ids == int(gid)] for gid in nodes}
    completed, replacements = {}, {}
    generated = {}
    next_id = max(map(int, nodes))+1
    # Canonical IDs and shape hypotheses are derived after the ordinary coarse
    # pipeline; children inherit identity evidence, not separate VLM responses.
    for gid in list(nodes):
        if nodes[gid].get('atomic_instance_evidence'):
            continue
        points = geometry[gid]
        if len(points) < 64 or gid not in tracks:
            continue
        span = np.ptp(points, axis=0)
        small = span.max() < .25
        if not (small or eligible(points)) or len(tracks[gid]['observed_frames']) < 5:
            continue
        dense, dense_view_ids, frames, source_audit = collect(points, tracks[gid], views, native)
        supported, counts = supported_samples(dense, dense_view_ids)
        if small and len(supported) > 1.5*len(points):
            added = supported[cKDTree(points).query(supported)[0] > .001]
            if len(added) and np.max(np.ptp(np.concatenate([points, added]), axis=0)-span) <= .025:
                replacements[gid] = np.concatenate([points, unique_samples(added)])
                report['dense_recovery'].append(dict(instance_id=gid, original_points=len(points),
                    measured_added_points=len(added), supporting_frames=len(frames), generated_points=0,
                    native_mask_evidence=source_audit))
        if not eligible(points) or len(supported) < 256:
            continue
        edges, edge_view_ids, edge_frames, edge_audit = edge_samples(points, tracks[gid], views)
        layers, evidence = fit_repetition(edges, edge_view_ids, footprint=supported)
        entry = dict(source_id=gid, identity=nodes[gid]['name'], native_evidence=source_audit,
                     edge_evidence=edge_audit, **evidence)
        report['splits'].append(entry)
        if not layers:
            continue
        centers = np.array([views.get(edge_frames[i])[0][:3, 3]
                            for i in sorted(set(v for layer in layers for v in layer['supporting_views']))])
        baseline = float(np.max(np.linalg.norm(centers[:, None]-centers[None, :], axis=2)))
        if baseline < .1:
            entry.update(accepted=False, reason='insufficient_camera_baseline'); continue
        top = layers[-1]['height_m']
        # A repeatedly measured top face is the sole shape template. Never use
        # GT thickness or a category's assumed dimensions to stretch this face.
        template = supported[(supported[:, 2] <= top+.002)]
        lower = float(np.quantile(template[:, 2], .01))
        template = template[template[:, 2] >= lower-.001]
        # A concept mask can omit an enclosed depression although the original
        # fused surface measured it. Restore such an interior from depth, rather
        # than stretching a shape hypothesis to explain a hole in the mask.
        planes = ConvexHull(supported[:, :2]).equations
        inner_distance = -(points[:, :2] @ planes[:, :2].T+planes[:, 2]).max(axis=1)
        # A quantile removes low-population concave regions. Use the measured
        # native top-face lower bound for the verified interior only; lower
        # stack rims still cannot become part of this top template.
        native_lower = float(supported[:, 2].min())
        interior = points[(inner_distance > .015) & (points[:, 2] >= native_lower-.001)
                          & (points[:, 2] <= top+.002)]
        witnesses = np.zeros(len(interior), int)
        for fid in edge_frames:
            indices, weights, _ = views.project(interior, fid)
            witnesses[indices] += weights >= .5
        interior = interior[witnesses >= 3]
        top_edges = edges[np.abs(edges[:, 2]-top) <= .0015]
        template = unique_samples(np.concatenate([template, interior, top_edges]))
        hypotheses, measured, provenance, valid = [], [], [], True
        for layer in reversed(layers):
            shift = np.array([0., 0., layer['height_m']-top])
            hypothesis = template+shift
            rim = edges[(np.abs(edges[:, 2]-layer['height_m']) <= .0015)
                        & boundary_band(edges, supported)]
            distance = cKDTree(hypothesis).query(rim)[0]
            compatible = float(np.mean(distance <= .01))
            allowed, free_evidence = free_space(hypothesis, views, edge_frames)
            detail = dict(height_m=layer['height_m'], supporting_frames=[edge_frames[i] for i in layer['supporting_views']],
                          measured_rim_shape_compatibility=compatible, translation_m=shift.tolist(), **free_evidence)
            provenance.append(detail)
            if compatible < .8 or not allowed:
                valid = False
            hypotheses.append(hypothesis); measured.append(unique_samples(rim))
        if not valid:
            entry.update(accepted=False, reason='shape_or_free_space_contradiction', layers=provenance); continue
        # Preserve every original source coordinate and assign it to its nearest
        # repeated surface; edge witnesses alone never overwrite other objects.
        distances = np.array([cKDTree(q).query(points)[0] for q in hypotheses])
        owner = distances.argmin(0)
        if float(np.quantile(distances.min(0), .99)) > .025:
            entry.update(accepted=False, reason='original_surface_not_explained_by_repetition',
                original_surface_residual_p99_m=float(np.quantile(distances.min(0), .99)),
                depth_verified_interior_template_points=len(interior)); continue
        children = [gid]+[str(next_id+i) for i in range(len(layers)-1)]
        next_id += len(layers)-1
        original_node, original_track = copy.deepcopy(nodes[gid]), copy.deepcopy(tracks[gid])
        for index, child in enumerate(children):
            observed = np.concatenate([points[owner == index], measured[index]])
            if index == 0:
                observed = np.concatenate([observed, supported])
            replacements[child] = observed
            completed[child] = unique_samples(np.concatenate([observed, hypotheses[index]]))
            generated[child] = hypotheses[index] if index > 0 else np.empty((0, 3))
            node = copy.deepcopy(original_node)
            node.update(id=child, atomic_instance_evidence='repeated_multiview_metric_boundary',
                        repeated_surface_group=f'repetition_{gid}', shape_hypothesis=index > 0)
            nodes[child] = node
            tracks[child] = copy.deepcopy(original_track)
            tracks[child].update(point_count=len(observed), source_aggregate_id=gid,
                atomic_boundary_frames=provenance[index]['supporting_frames'],
                identity_observations_shared_with_aggregate=True,
                confidence=original_track['confidence']*min(1., len(provenance[index]['supporting_frames'])/max(len(edge_frames), 1)))
            if index > 0:
                graph.setdefault('object_relations', []).append(dict(source_id=children[index-1], target_id=child,
                    description='stacked_on', evidence_frames=min(len(provenance[index-1]['supporting_frames']), len(provenance[index]['supporting_frames']))))
        graph.setdefault('object_groups', {})[f'repetition_{gid}'] = dict(node_type='collection',
            source_aggregate_id=gid, member_ids=children, independent_objects=True,
            counted_as_object=False, hypothesis='shared_measured_shape_repetition')
        entry.update(accepted=True, reason='observed_repetition_and_validated_shared_surface_prior',
            object_ids=children, layers=provenance, camera_baseline_m=baseline,
            template_measured_points=len(template), template_visible_height_m=float(np.ptp(template[:, 2])),
            depth_verified_interior_template_points=len(interior),
            generated_points=sum(len(generated[child]) for child in children),
            hidden_shape_guaranteed=False)
    if replacements:
        # Use the existing completion file when present, preserving all unrelated
        # completed objects. No prediction is copied from a ground-truth mesh.
        prior = Path(getattr(context, 'graph_geometry', cloud_path))
        completion_path = context.scene/'instance_cloud_completed.ply'
        if prior != completion_path:
            import shutil
            shutil.copyfile(prior, completion_path)
        replace_regions(completion_path, {**replacements, **completed}, nodes)
        replace_regions(cloud_path, replacements, nodes)
        for child, region in replacements.items():
            tracks[child]['point_count'] = len(region)
        tracks_path.write_text(json.dumps(tracks, indent=2)+'\n')
        graph['repeated_surface_provenance'] = dict(algorithm='RSI_v13', audit='repeated_instance_audit.json',
            generated_surfaces_are_measurements=False, unknown_undersides_recovered=False)
        path.write_text(json.dumps(graph, indent=2)+'\n')
        context.graph_geometry = completion_path
        context.canonical_geometry_input = path
        from .canonical_geometry import publish
        publish(context)
        folder = context.scene/'geometry_recovery'; folder.mkdir(exist_ok=True)
        np.savez_compressed(folder/'repeated_surface_hypotheses.npz', **generated)
    (context.scene/'repeated_instance_audit.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
    context.event('重复边界实例与密集实测表面审核完成',
                  split_aggregates=sum(e.get('accepted', False) for e in report['splits']),
                  dense_objects=len(report['dense_recovery']),
                  generated_points=sum(e.get('generated_points', 0) for e in report['splits']))


def component_sha256():
    return hashlib.sha256(Path(__file__).read_bytes()+Path(__file__).with_name('fine_boundary.py').read_bytes()).hexdigest()

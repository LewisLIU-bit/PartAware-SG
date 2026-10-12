"""GT-free four-domain CUE point-mask screen, not a production replay."""
import gc
import hashlib
import json
import os
import resource
import sys
import time
from pathlib import Path
from types import SimpleNamespace

os.environ['OMP_NUM_THREADS'] = '2'
os.environ['OPENBLAS_NUM_THREADS'] = '2'
os.environ['MKL_NUM_THREADS'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
if hasattr(os, 'sched_getaffinity'):
    os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:2])
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree

WORK = Path(__file__).resolve().parent
ROOT = WORK.parents[1]
RESEARCH = WORK.parent / 'research'
REPO = Path('/home/lewisliu/PartAware-SG')
sys.path[:0] = [str(REPO / 'scannet/script'), str(RESEARCH)]
import proposal_ensemble
from bounded_overlap import overlap
from source_agreement import smooth
from partaware.geometry import load_capture
from pipeline_components.instance_consensus import Views
from cue_ownership import core_owner, unique_positive_owner, dominant_positive_identity
proposal_ensemble.overlap = overlap


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as file:
        for block in iter(lambda: file.read(8 * 1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


def save(receipt):
    (WORK / 'construction.json').write_text(json.dumps(receipt, indent=2)+'\n')


def build_item(item):
    started = time.time()
    source = Path(item['source'])
    target = WORK / 'results' / item['dataset'] / item['scene']
    target.mkdir(parents=True, exist_ok=True)
    for key, digest in item['input_hashes'].items():
        if sha(item[key]) != digest:
            raise ValueError('Frozen input hash changed: '+key)
    if item['domain_kind'] == 'unlabeled_mesh':
        mesh = o3d.io.read_triangle_mesh(item['domain'])
        xyz = np.asarray(mesh.vertices).copy()
        del mesh
    else:
        with np.load(item['domain'], allow_pickle=False) as data:
            xyz = data['xyz'].copy()
    with np.load(item['peer'], allow_pickle=False) as data:
        peer, peer_scores = data['masks'].copy(), data['scores'].copy()
    if peer.dtype != bool or peer.shape != (len(peer_scores), len(xyz)):
        raise ValueError('Signed peer/shared domain shape mismatch')
    graph_path, track_path = source / 'topology_map.json', source / 'validated_object_tracks.json'
    graph, tracks = json.loads(graph_path.read_text()), json.loads(track_path.read_text())
    keys = list(graph['object_nodes']['nodes'])
    cloud_name = graph['geometry_provenance']['source']
    if Path(cloud_name).name != cloud_name:
        raise ValueError('Canonical source geometry must be scene-local')
    cloud_path = source / cloud_name
    cloud = o3d.io.read_point_cloud(str(cloud_path))
    points = np.asarray(cloud.points)
    rgb = np.rint(np.asarray(cloud.colors)*255).astype(np.int64)
    ids = rgb[:, 0] + 255*rgb[:, 1] + 255**2*rgb[:, 2]
    distance, index = cKDTree(points).query(xyz, workers=2)
    labels = ids[index].copy()
    labels[distance > .05] = -1
    segments = None
    if item['segments']:
        segments = np.array(json.loads(Path(item['segments']).read_text())['segIndices'])
        labels = smooth(labels, segments)
    native = np.array([labels == int(key) for key in keys])
    native_scores = np.array([tracks.get(key, {}).get('confidence',
        graph['object_nodes']['nodes'][key].get('confidence', 0)) for key in keys])
    proposals, scores, selection = proposal_ensemble.propose(native, native_scores,
        peer, peer_scores, 'agreement', points=xyz)
    del native, native_scores, peer, peer_scores, cloud, points, rgb, ids, index, distance, labels
    gc.collect()
    baseline, count, core_indices, core_sizes = core_owner(proposals, scores, xyz)
    conflict = np.flatnonzero(count > 1)
    union = np.flatnonzero(count > 0)
    _, jobs, kd, kc, scale = load_capture(Path(item['manifest']))
    views = Views(SimpleNamespace(scene=source), jobs, kd, kc, scale, {})
    fids = sorted(views.jobs)
    frame_index = {fid: i for i, fid in enumerate(fids)}
    identity = np.full((len(proposals), len(fids)), -1, np.int32)
    native_ambiguous = []
    peer_candidates = []
    for candidate, source_index in enumerate(selection['kept']):
        if source_index >= len(keys):
            peer_candidates.append(candidate)
            continue
        by_frame = {}
        for observation in tracks.get(keys[source_index], {}).get('observations', []):
            fid = str(observation['frame_id'])
            local_id = int(observation.get('local_id', observation.get('frame_instance_id', -1)))
            if fid in frame_index and local_id > 0:
                by_frame.setdefault(fid, set()).add(local_id)
        for fid, local_ids in by_frame.items():
            if len(local_ids) == 1:
                identity[candidate, frame_index[fid]] = next(iter(local_ids))
            else:
                native_ambiguous.append(dict(candidate=candidate, frame_id=fid,
                    positive_local_mask_ids=sorted(local_ids), action='identity_unknown'))
    cache_path = target / 'visible_conflict_labels.npy'
    cached = np.lib.format.open_memmap(cache_path, mode='w+', dtype=np.int32,
                                     shape=(len(fids), len(conflict)))
    centers = np.zeros((len(fids), 3), float)
    input_receipts = []
    for frame, fid in enumerate(fids):
        pose, depth, original_mask = views.get(fid)
        if depth is None or original_mask is None or not np.isfinite(pose).all():
            raise ValueError('Missing original pose/depth/mask in frozen cache')
        centers[frame] = pose[:3, 3]
        original = np.full(len(xyz), -1, np.int32)
        for start in range(0, len(union), 100000):
            vertices = union[start:start+100000]
            visible, weights, local_labels = views.project(xyz[vertices], fid)
            positive_depth = weights >= .7
            original[vertices[visible[positive_depth]]] = local_labels[positive_depth]
        cached[frame] = original[conflict]
        for candidate in peer_candidates:
            identity[candidate, frame] = dominant_positive_identity(original[core_indices[candidate]])
        input_receipts.append(dict(frame_id=fid, camera_center=centers[frame].tolist(),
            depth_visible_conflict_points=int(np.count_nonzero(cached[frame] >= 0)),
            native_mask_sha256=sha(source/'refined_instance'/f'{fid}.png'),
            native_instance_json_sha256=sha(source/'refined_instance'/f'{fid}_instance.json'),
            pose_sha256=sha(views.jobs[fid]['pose']), depth_sha256=sha(views.jobs[fid]['depth'])))
        views.cache.clear()
        if frame % 10 == 0 or frame == len(fids)-1:
            print('CUE_VIEW_EVIDENCE', item['scene'], frame+1, '/', len(fids),
                  'RSS_GiB', round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2, 3), flush=True)
    cached.flush()
    owner, audit = unique_positive_owner(proposals, baseline, conflict, cached, identity, centers)
    cache_bytes = cache_path.stat().st_size
    del cached, views
    cache_path.unlink()
    pre_publication_union = baseline >= 0
    if segments is not None:
        baseline = smooth(baseline, segments)
        owner = smooth(owner, segments)
    changed_after_publication = int(np.count_nonzero(owner != baseline))
    base_path, cue_path = target/'core_control.npz', target/'cue_masks.npz'
    # Save sequentially, so the two dense published mask arrays never coexist.
    masks = np.array([baseline == i for i in range(len(scores))])
    np.savez_compressed(base_path, masks=masks, scores=scores)
    del masks
    masks = np.array([owner == i for i in range(len(scores))])
    if np.any(masks.sum(0) > 1):
        raise AssertionError('CUE publication violated exclusive point ownership')
    np.savez_compressed(cue_path, masks=masks, scores=scores)
    proof = dict(original_view_receipts=input_receipts,
        candidate_source_indices=selection['kept'], candidate_frame_positive_identities=identity.tolist(),
        native_ambiguous_identity_views=native_ambiguous, uncontested_core_sizes=core_sizes,
        original_conflict_points=len(conflict), mapped_frame_candidate_pairs=int((identity > 0).sum()),
        native_track_hash=sha(track_path), original_graph_hash=sha(graph_path),
        source_cloud_hash=sha(cloud_path), temporary_projection_cache_bytes=cache_bytes)
    evidence_path = target/'positive_identity_evidence.json'
    evidence_path.write_text(json.dumps(proof, indent=2)+'\n')
    row = dict(scene=item['scene'], dataset=item['dataset'], inputs=item['input_hashes'],
        proposal_selection=selection, ownership_audit=audit,
        scores_and_NMS_unchanged=True, candidate_count=len(scores), domain_points=len(xyz),
        pre_publication_observed_union_points=int(pre_publication_union.sum()),
        unchanged_geometric_majority_publication=segments is not None,
        changed_owners_after_publication=changed_after_publication,
        control_masks_file=str(base_path), control_masks_sha256=sha(base_path),
        masks_file=str(cue_path), masks_sha256=sha(cue_path),
        evidence_file=str(evidence_path), evidence_sha256=sha(evidence_path),
        elapsed_seconds=time.time()-started,
        process_peak_RSS_GiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2)
    print('CUE_SCREEN_BUILT', item['scene'], json.dumps(dict(changed=audit['changed_conflict_owners'],
        published_changed=changed_after_publication, candidates=len(scores),
        elapsed=row['elapsed_seconds'], peak_RSS_GiB=row['process_peak_RSS_GiB'])), flush=True)
    return row


def main():
    registration_path = WORK.parent/'four_domain_confirmation_registration.json'
    registration = json.loads(registration_path.read_text())
    if not registration['complete']:
        raise ValueError('Four-domain registration incomplete')
    receipt = dict(complete=False, algorithm='CUE', hypothesis_sha256=sha(WORK/'hypothesis.json'),
        registration_sha256=sha(registration_path), GT_in_construction=False,
        full_pipeline_replay=False, new_API_calls=0, training=False, generated_points=0,
        physical_point_deletions=0, independent_original_camera_evidence=True,
        builder_sha256=sha(__file__), component_sha256=sha(WORK/'cue_ownership.py'),
        signed_dependencies={p.name: sha(p) for p in [RESEARCH/'proposal_ensemble.py',
            RESEARCH/'core_ownership.py', RESEARCH/'bounded_overlap.py', RESEARCH/'source_agreement.py',
            REPO/'scannet/script/pipeline_components/instance_consensus.py']}, results=[])
    save(receipt)
    for item in registration['items']:
        receipt['results'].append(build_item(item))
        save(receipt)
        gc.collect()
    receipt['complete'] = True
    save(receipt)
    print('CUE_FOUR_DOMAIN_BUILD_COMPLETE', flush=True)


if __name__ == '__main__':
    main()

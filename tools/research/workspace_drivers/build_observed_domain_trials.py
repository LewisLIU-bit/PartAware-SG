"""General GT-free SAGE transfer to a signed RGB-D observed point domain.

This is a geometric candidate screen, not a complete pipeline reconstruction.
Scene name only identifies input/output; no class, GT ID or count conditions.
"""
import argparse, hashlib, json, sys
from pathlib import Path
import numpy as np, open3d as o3d
from scipy.spatial import cKDTree
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'v20_work/research'))
import proposal_ensemble
from bounded_overlap import overlap
from core_ownership import propose as core
proposal_ensemble.overlap = overlap

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()

parser = argparse.ArgumentParser()
parser.add_argument('--source', required=True)
parser.add_argument('--domain', required=True)
parser.add_argument('--domain-receipt', required=True)
parser.add_argument('--peer', required=True)
parser.add_argument('--manifest', required=True)
parser.add_argument('--output', required=True)
args = parser.parse_args()
source = Path(args.source); target = Path(args.output); target.mkdir(parents=True, exist_ok=True)
receipt = json.loads(Path(args.domain_receipt).read_text())
assert receipt['GT_used'] is False and receipt['domain_sha256'] == sha(args.domain)
with np.load(args.domain, allow_pickle=False) as f: xyz = f['xyz'].copy()
with np.load(args.peer, allow_pickle=False) as f: b = f['masks'].copy(); sb = f['scores'].copy()
assert b.shape == (len(sb), len(xyz)) and b.dtype == bool
graph = json.loads((source/'topology_map.json').read_text())
cloud_name = graph['geometry_provenance']['source']; assert Path(cloud_name).name == cloud_name
cloud = o3d.io.read_point_cloud(str(source/cloud_name)); points = np.asarray(cloud.points)
color = np.rint(np.asarray(cloud.colors)*255).astype(np.int64)
ids = color[:, 0] + 255*color[:, 1] + 255**2*color[:, 2]
distance, index = cKDTree(points).query(xyz, workers=2)
labels = ids[index]; labels[distance > .05] = -1
keys = list(graph['object_nodes']['nodes'])
tracks = json.loads((source/'validated_object_tracks.json').read_text())
a = np.array([labels == int(k) for k in keys])
sa = np.array([tracks.get(k, {}).get('confidence', graph['object_nodes']['nodes'][k].get('confidence', 0)) for k in keys])
inputs = dict(source=str(source), manifest_sha256=sha(args.manifest), domain_sha256=sha(args.domain),
    peer_sha256=sha(args.peer), source_cloud_sha256=sha(source/cloud_name),
    graph_sha256=sha(source/'topology_map.json'), tracks_sha256=sha(source/'validated_object_tracks.json'))
rows = []
for mode, fn in [('exclusive_weighted', proposal_ensemble.propose), ('exclusive_core', core)]:
    masks, scores, audit = fn(a, sa, b, sb, mode, points=xyz)
    assert np.max(masks.sum(0), initial=0) <= 1
    path = target/(mode+'_masks.npz')
    np.savez_compressed(path, masks=masks, scores=scores)
    rows.append(dict(mode=mode, masks_sha256=sha(path), masks_file=str(path), audit=audit))
    print('OBSERVED_DOMAIN_CANDIDATE_BUILT', mode, len(masks), flush=True)
result = dict(complete=True, full_pipeline_replay=False, GT_in_construction=False, new_API_calls=0,
    training=False, generated_points=0, point_domain='signed observed RGB-D voxels, no segment smoothing',
    transfer_radius_m=.05, native_keys=keys, inputs=inputs, results=rows,
    component_hashes={n:sha(ROOT/'v20_work/research'/n) for n in ['proposal_ensemble.py','core_ownership.py','bounded_overlap.py']})
(target/'candidate_receipt.json').write_text(json.dumps(result, indent=2)+'\n')
print('OBSERVED_DOMAIN_CANDIDATE_SCREEN_INPUTS_COMPLETE', flush=True)

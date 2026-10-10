"""Join measured closed enclosure sections at a continuous physical corner.

This adapter uses cached identity, observed surfaces and co-observation only.
It does not infer a joint from bounding-box overlap or consult annotations.
"""
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree
from threadpoolctl import threadpool_limits


def closed_identity(node):
    identity = node.get('storage_identity', {})
    return (node.get('structural_surface_recovery', {}).get('measured', False)
        and identity.get('front_occupancy', 0.) >= .6
        and len(set(identity.get('cached_cabinet_frames', []))) >= 3)


def dominant_face(points):
    with threadpool_limits(limits=1):
        o3d.utility.random.seed(14)
        cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
        plane, support = cloud.segment_plane(.008, 3, 800)
    normal = np.asarray(plane[:3])
    normal /= max(np.linalg.norm(normal), 1e-12)
    return normal, len(support)/len(points)


def relation(source, body, source_node, body_node, semantic, shared_frames):
    """Return proof for a unique, observed orthogonal closed-body seam."""
    if (len(source) < 512 or len(source) >= len(body) or semantic < .85
            or len(shared_frames) < 3 or not closed_identity(source_node)
            or not closed_identity(body_node)):
        return None
    source_z, body_z = np.quantile(source[:, 2], [.01, .99]), np.quantile(body[:, 2], [.01, .99])
    mismatch = float(np.max(np.abs(source_z-body_z)))
    height = float(min(np.ptp(source_z), np.ptp(body_z)))
    if height < .5 or mismatch > .04:
        return None
    a, a_support = dominant_face(source)
    b, b_support = dominant_face(body)
    orthogonality = float(abs(a @ b))
    if abs(a[2]) > .08 or abs(b[2]) > .08 or min(a_support, b_support) < .4 or orthogonality > .15:
        return None
    contact_a = source[cKDTree(body).query(source)[0] <= .025]
    contact_b = body[cKDTree(source).query(body)[0] <= .025]
    # Symmetric contact avoids dependence on which face was more densely sampled.
    contact = np.concatenate([contact_a, contact_b])
    cells = np.unique(np.floor(contact/.01).astype(np.int64), axis=0)
    span = float(np.ptp(contact[:, 2])) if len(contact) else 0.
    vertical_cells = np.unique(np.floor((contact[:, 2]-max(source_z[0], body_z[0]))/.03)).size
    continuity = min(1., vertical_cells/max(height/.03, 1.))
    if min(len(contact_a), len(contact_b)) < 64 or len(cells) < 128 or span < .8*height or continuity < .8:
        return None
    return dict(evidence_type='measured_closed_enclosure_corner', semantic_cosine=semantic,
        shared_observed_frames=sorted(shared_frames), boundary_height_m=height,
        boundary_height_mismatch_m=mismatch, front_normal_dot=orthogonality,
        front_support_fractions=[a_support, b_support], contact_points=[len(contact_a), len(contact_b)],
        contact_voxels_1cm=len(cells), seam_height_m=span, seam_vertical_coverage=continuity,
        generated_points=0, ground_truth_used=False)


def construct(context):
    """Assemble only closed corners after every measured-part recovery stage."""
    import sys
    from .part_body_assembly import construct as assemble
    assemble(context, relation_adapter=sys.modules[__name__], audit_filename='enclosure_continuity_audit.json')


def retain_evidence(graph, report):
    """Keep valid assembly proofs available after an idempotent stage replay."""
    report['applied_assemblies'] = len(report['objects'])
    recorded = {str(item['source_id']) for item in report['objects']}
    nodes = graph['object_nodes']['nodes']
    for source, alias in graph.get('object_identity_aliases', {}).items():
        evidence = alias.get('evidence', {})
        body, part_id = str(alias.get('canonical_id')), alias.get('part_id')
        part = graph.get('part_nodes', {}).get(part_id, {})
        if (source in recorded or body not in nodes or not part
                or evidence.get('evidence_type') != 'measured_closed_enclosure_corner'):
            continue
        report['objects'].append(dict(source_id=source, body_id=body, part_id=part_id,
            measured_points=part['point_count'], retained_from_valid_alias=True,
            reason='保留已归并柜体的真实接缝证据', **evidence))

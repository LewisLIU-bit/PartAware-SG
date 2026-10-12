"""Preserve observed unowned surfaces so 1NN cannot bridge missing labels.

Input is ordinary unlabeled reconstruction plus real final labeled points.
Reserved sentinel is not a graph object; no GT, category or point count prior.
"""
import hashlib,json
from pathlib import Path
import numpy as np,open3d as o3d
from scipy.spatial import cKDTree
UNOWNED_ID=255**3-1

def missing_observed_surface(observed,labeled):
 observed=np.asarray(observed,float);labeled=np.asarray(labeled,float)
 if observed.ndim!=2 or observed.shape[1]!=3 or not np.isfinite(observed).all():raise ValueError('Observed xyz required')
 if not len(labeled):return np.ones(len(observed),bool)
 return cKDTree(labeled).query(observed,workers=2)[0]>1e-8

def construct(context):
 mesh=Path(context.research_mesh);xyz=np.asarray(o3d.io.read_triangle_mesh(str(mesh)).vertices)
 graph=json.loads((context.scene/'topology_map.json').read_text())
 if str(UNOWNED_ID) in graph['object_nodes']['nodes']:raise ValueError('Reserved background ID collides with an actual object')
 source=Path(context.graph_geometry);cloud=o3d.io.read_point_cloud(str(source));p=np.asarray(cloud.points);colors=np.asarray(cloud.colors)
 missing=missing_observed_surface(xyz,p)
 out=o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.concatenate([p,xyz[missing]])))
 # Base-255 sentinel decodes to an unregistered identity, so canonical graph
 # publication and the unchanged point evaluator treat it as unowned.
 out.colors=o3d.utility.Vector3dVector(np.concatenate([colors,np.full((int(missing.sum()),3),254/255.)]))
 path=context.scene/'instance_cloud_observed_domain.ply';o3d.io.write_point_cloud(str(path),out)
 context.graph_geometry=path;context.canonical_geometry_input=context.scene/'topology_map.json'
 audit=dict(algorithm='observed_domain_guard',GT_in_construction=False,manual_deletions=0,generated_points=0,
            observed_background_points_added=int(missing.sum()),existing_points_and_labels_unchanged=True,
            source=str(source.name),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
            mesh_sha256=hashlib.sha256(mesh.read_bytes()).hexdigest(),reserved_unowned_id=UNOWNED_ID,
            code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
 (context.scene/'observed_domain_guard_audit.json').write_text(json.dumps(audit,indent=2)+'\n')
 context.event('保留实测未归属表面，阻止最近邻越界',background_points=int(missing.sum()),generated_points=0)

"""Default construction components; remove an import/entry to detach a component."""
from . import multiview_association, op3dsg_fusion, vlpart_graph, instance_consensus, yoloe_frontend, shape_completion, canonical_geometry, proposal_validation

# This code registry is the only attachment point, not a runtime feature switch.
ASSOCIATION = multiview_association
FUSION = op3dsg_fusion
GRAPH_COMPONENTS = [vlpart_graph]

INSTANCE_REFINEMENT = instance_consensus

FRONTEND = yoloe_frontend
GEOMETRY_COMPONENTS = [shape_completion]

GEOMETRY_OUTPUT = canonical_geometry

OBJECT_VALIDATION = proposal_validation

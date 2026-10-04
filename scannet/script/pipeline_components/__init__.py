"""Default construction components; remove an import/entry to detach a component."""
from . import multiview_association, op3dsg_fusion, vlpart_graph

# This code registry is the only attachment point, not a runtime feature switch.
ASSOCIATION = multiview_association
FUSION = op3dsg_fusion
GRAPH_COMPONENTS = [vlpart_graph]

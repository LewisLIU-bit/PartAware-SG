"""Default construction components; remove an import/entry to detach a component."""
from . import multiview_association, op3dsg_fusion, vlpart_graph, instance_consensus, yoloe_frontend, shape_completion, canonical_geometry, proposal_validation

# This code registry is the only attachment point, not a runtime feature switch.
ASSOCIATION = multiview_association
FUSION = op3dsg_fusion
GRAPH_COMPONENTS = [vlpart_graph]

INSTANCE_REFINEMENT = instance_consensus

FRONTEND = yoloe_frontend
from . import backed_cuboid, structural_surfaces, instance_granularity
GEOMETRY_COMPONENTS = [instance_granularity, structural_surfaces, shape_completion, backed_cuboid]

GEOMETRY_OUTPUT = canonical_geometry

OBJECT_VALIDATION = proposal_validation

from . import observed_consensus
OBSERVED_VALIDATION = observed_consensus

from . import background_consensus
BACKGROUND_VALIDATION = background_consensus

from . import identity_consensus
IDENTITY_VALIDATION = identity_consensus

from . import granularity_metrics
EVALUATION_DIAGNOSTICS = granularity_metrics

from . import specular_consensus
SURFACE_VALIDATION = specular_consensus

from . import part_geometry
PART_GEOMETRY = part_geometry

from . import thin_geometry, axial_assembly, assembly_density, suspension_geometry
MEASURED_REFINEMENT = [thin_geometry, axial_assembly, assembly_density]
FINAL_GEOMETRY = [suspension_geometry]

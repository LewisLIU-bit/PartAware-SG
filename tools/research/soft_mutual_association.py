"""R2: soft symmetric evidence preserves partial views and semantic identity.

The source fusion still applies geometric/semantic qualification and one-to-one
assignment. Low reverse coverage is weak identity evidence, not proof of a
different object: image segmentation can miss an observed object's other parts.
No category, scene, GT, expected-count or node-ID condition is used.
"""
from mutual_association import visible_surface_evidence
from pipeline_components import multiview_association as baseline
PartFusion=baseline.PartFusion
assign=baseline.assign

def visibility_score(observation,track,frame):
    evidence=visible_surface_evidence(observation,track,frame)
    if evidence is None or evidence['mutual_visible_surface']<=0:return None
    evidence.update(semantic_identity=float(observation.semantic@track.semantic),
        low_reverse_support_is_soft_evidence=True)
    return evidence

def score(base,evidence):
    symmetric=evidence['mutual_visible_surface']
    semantic=(evidence['semantic_identity']+1)/2
    # Original v19 weights; replace asymmetric full-track geometry with mutual
    # current-visible geometry while preserving its separate semantic term.
    return (.25*(symmetric+semantic)+.25*evidence['projected_bbox_iou']+
        .15*evidence['projected_coverage']+.1*evidence['color_similarity'])

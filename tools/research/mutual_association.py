"""Mutual visible-surface association; no training or object/category exceptions.

OVIP-SG motivates replacing asymmetric containment with symmetric agreement.
This project adapter uses measured surfaces restricted to the current view,
rather than full-box IoU: unobserved backs must not penalise a partial view.
"""
import numpy as np
from scipy.spatial import cKDTree
from pipeline_components import multiview_association as baseline
from pipeline_components.fine_instances import association_radius
PartFusion=baseline.PartFusion
assign=baseline.assign

def mutual_support(forward,reverse):
    return 2*forward*reverse/max(forward+reverse,1e-12)

def visible_surface_evidence(observation,track,frame):
    evidence=baseline.visibility_score(observation,track,frame)
    if evidence is None:return None
    p=track.points;camera=(p-frame['pose'][:3,3])@frame['pose'][:3,:3]
    z=camera[:,2];k=frame['kd'];depth=frame['depth']/frame['scale']
    uv=np.rint(camera[:,:2]/np.maximum(z[:,None],1e-9)*[k[0,0],k[1,1]]+[k[0,2],k[1,2]]).astype(int)
    indices=np.flatnonzero((z>0)&(uv[:,0]>=0)&(uv[:,1]>=0)&
        (uv[:,0]<depth.shape[1])&(uv[:,1]<depth.shape[0]))
    measured=depth[uv[indices,1],uv[indices,0]]
    indices=indices[(measured>0)&(np.abs(z[indices]-measured)<=.05+.01*measured)]
    visible=p[indices]
    if len(visible)<16:return None
    radius=association_radius(observation.points,visible) if observation.record.get('fine_scale_instance') else .1
    forward=float(np.mean(cKDTree(visible).query(observation.points)[0]<=radius))
    reverse=float(np.mean(cKDTree(observation.points).query(visible)[0]<=radius))
    symmetric=mutual_support(forward,reverse)
    evidence.update(mutual_visible_surface=symmetric,visible_surface_coverage=[forward,reverse],
        visible_surface_points=len(visible),mutual_radius_m=radius,
        GT_used=False,unknown_backside_not_negative=True)
    return evidence

def visibility_score(observation,track,frame):
    evidence=visible_surface_evidence(observation,track,frame)
    # R1 hard veto is retained for reproducibility, not silently changed.
    if evidence is None or min(evidence['visible_surface_coverage'])<.25:return None
    return evidence

def score(base,evidence):
    return (.5*evidence['mutual_visible_surface']+
        .25*evidence['projected_bbox_iou']+.15*evidence['projected_coverage']+
        .1*evidence['color_similarity'])

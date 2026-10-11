"""Atomic partition proposals without annotation, category gates or new VLM calls.

A broad measured instance can contain distinct high-purity peer masks. Split
only when at least two disjoint anchors jointly explain it. Preserve every
parent coordinate and all unrelated ownership. Scores inherit the source;
they are not represented as calibrated probability.
"""
import numpy as np
from scipy.spatial import cKDTree
from mask_proposals import overlap

def partition(primary, scores, peer, peer_scores, xyz, minimum_points=100,
              purity=.85, coverage=.75, minimum_fraction=.12, mode='anchors'):
    a,b=np.asarray(primary,bool),np.asarray(peer,bool)
    sa,sb=np.asarray(scores,float),np.asarray(peer_scores,float)
    xyz=np.asarray(xyz,float)
    if a.ndim!=2 or b.ndim!=2 or a.shape[1]!=b.shape[1] or xyz.shape!=(a.shape[1],3):
        raise ValueError('All masks and measured coordinates must share one domain')
    if sa.shape!=(len(a),) or sb.shape!=(len(b),) or not np.isfinite(np.r_[sa,sb]).all():
        raise ValueError('A finite source score is required per instance')
    if mode not in ('anchors','conservative'):raise ValueError('Unknown partition strategy')
    if np.any(a.sum(0)>1):raise ValueError('Primary ownership must be exclusive')
    counts=b.sum(1);inter=a.astype(np.int64)@b.astype(np.int64).T
    iou=overlap(a,b)
    preferred=np.argmax(iou,axis=0) if len(a) else np.empty(len(b),int)
    output=[];output_scores=[];origins=[];splits=[]
    for i,parent in enumerate(a):
        parent_count=int(parent.sum())
        candidates=[j for j in range(len(b)) if preferred[j]==i and sb[j]>=.5
            and counts[j]>=minimum_points and inter[i,j]>=minimum_points
            and inter[i,j]/max(int(counts[j]),1)>=purity
            and minimum_fraction<=inter[i,j]/max(parent_count,1)<=1-minimum_fraction]
        # Reject duplicate/nested peer hypotheses rather than inventing a count.
        if len(candidates)<2:
            output.append(parent.copy());output_scores.append(sa[i]);origins.append((i,None));continue
        anchors=b[candidates]&parent
        overlap_counts=anchors.astype(np.int64)@anchors.astype(np.int64).T
        bad=(overlap_counts/np.maximum(np.minimum(anchors.sum(1)[:,None],anchors.sum(1)[None,:]),1))>.05
        np.fill_diagonal(bad,False)
        explained=float(np.any(anchors,axis=0).sum()/max(parent_count,1))
        if bad.any() or explained<coverage:
            output.append(parent.copy());output_scores.append(sa[i]);origins.append((i,None));continue
        indices=np.flatnonzero(parent);points=xyz[indices]
        distances=np.array([cKDTree(xyz[m]).query(points)[0] for m in anchors])
        assignment=distances.argmin(0)
        if mode=='conservative':
            # Ambiguous unsupported bridges may connect parts of one object.
            # Require a low residual and a distinct 3D spatial centre for each anchor.
            centres=np.array([xyz[m].mean(0) for m in anchors])
            span=float(np.linalg.norm(np.ptp(points,axis=0)))
            separation=np.linalg.norm(centres[:,None]-centres[None,:],axis=2)
            separation[np.eye(len(centres),dtype=bool)]=np.inf
            if separation.min()<.15*span or np.quantile(distances.min(0),.95)>.05*span:
                output.append(parent.copy());output_scores.append(sa[i]);origins.append((i,None));continue
        children=[]
        for k,j in enumerate(candidates):
            mask=np.zeros(a.shape[1],bool);mask[indices[assignment==k]]=True
            children.append(mask)
        if any(m.sum()<minimum_points for m in children):
            output.append(parent.copy());output_scores.append(sa[i]);origins.append((i,None));continue
        assert np.array_equal(np.any(children,axis=0),parent)
        assert np.max(np.asarray(children).sum(0))==1
        output.extend(children);output_scores.extend([sa[i]]*len(children))
        origins.extend((i,j) for j in candidates)
        splits.append(dict(parent_index=i,peer_indices=candidates,explained_fraction=explained,
            parent_points=parent_count,child_points=[int(m.sum()) for m in children]))
    return np.asarray(output,bool).reshape(-1,a.shape[1]),np.asarray(output_scores),dict(
        algorithm='atomic_measured_partition',mode=mode,GT_used=False,
        existing_ownership_preserved=True,total_surface_preserved=True,
        semantic_labels_from_peer_used=False,generated_points=0,scores_calibrated=False,
        origins=origins,splits=splits)

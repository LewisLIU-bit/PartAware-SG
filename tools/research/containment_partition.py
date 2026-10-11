"""Complementary instance evidence with measured non-planar body support.

No GT, object names, IDs or expected count; no coordinates generated.
A nested peer may isolate one member of a merged mask. Both its measured
intersection and the measured complement must have non-planar shape support;
flat panel crops alone are insufficient evidence of separate whole objects.
"""
import numpy as np

def nonplanarity(points):
    if len(points)<4:return 0.
    eigen=np.linalg.eigvalsh(np.cov(points,rowvar=False))
    return float(max(eigen[0],0)/max(eigen[-1],1e-12))

def propose(parent_masks,parent_scores,peer_masks,peer_scores,vertices):
    p=np.asarray(parent_masks,bool).copy()
    peer=np.asarray(peer_masks,bool);xyz=np.asarray(vertices,float)
    if p.ndim!=2 or peer.ndim!=2 or p.shape[1]!=peer.shape[1] or len(xyz)!=p.shape[1]:
        raise ValueError('Measured domains differ')
    if np.any(p.sum(0)>1):raise ValueError('Parent ownership must be exclusive')
    events=[];out=[]
    for i,parent in enumerate(p):
        candidates=[]
        for j,seed in enumerate(peer):
            if peer_scores[j]<.25 or seed.sum()<100:continue
            inside=parent&seed;residual=parent&~seed
            purity=inside.sum()/max(seed.sum(),1);fraction=inside.sum()/max(parent.sum(),1)
            if purity<.9 or not .2<=fraction<=.8 or min(inside.sum(),residual.sum())<100:continue
            dimensions=[nonplanarity(xyz[inside]),nonplanarity(xyz[residual])]
            if min(dimensions)<.02:continue
            candidates.append((float(peer_scores[j]),j,inside,residual,purity,fraction,dimensions))
        if not candidates:out.append(parent);continue
        _,j,inside,residual,purity,fraction,dimensions=max(candidates,key=lambda c:c[0])
        out.extend([inside,residual])
        events.append(dict(parent_index=i,peer_index=j,purity=float(purity),
            parent_fraction=float(fraction),nonplanarity=dimensions,
            generated_points=0,GT_used=False))
    # Preserve corresponding confidence for every output row, not append order.
    final_scores=[]
    split={e['parent_index'] for e in events}
    for i in range(len(p)):final_scores.extend([parent_scores[i]]*(2 if i in split else 1))
    result=np.asarray(out,bool)
    assert np.array_equal(result.any(0),p.any(0)) and np.all(result.sum(0)<=1)
    return result,np.asarray(final_scores),dict(splits=events,GT_used=False,
      coordinates_generated=False,ownership_preserved=True,
      rule='nested independent peer, >=90% purity, 20-80% parent mass, nonplanar both measured bodies')

"""Independent geometric proposal agreement and class-agnostic duplicate control.

Mask-space screening only. No GT, categories, instance IDs, expected counts or
scene-specific rules. This is not a complete scene graph reconstruction.
"""
import numpy as np
from scipy.stats import rankdata
from mask_proposals import overlap

def propose(primary,primary_scores,peer,peer_scores,mode='agreement',minimum_points=100,points=None):
 a,b=np.asarray(primary,bool),np.asarray(peer,bool)
 sa,sb=np.asarray(primary_scores,float),np.asarray(peer_scores,float)
 if a.ndim!=2 or b.ndim!=2 or a.shape[1]!=b.shape[1]:raise ValueError('Shared point domain required')
 if sa.shape!=(len(a),) or sb.shape!=(len(b),) or not np.isfinite(np.r_[sa,sb]).all():raise ValueError('Finite score per proposal required')
 if mode not in ['agreement','rank_agreement','peer_anchor','agreement_wide','exclusive_agreement','exclusive_specific','exclusive_containment','exclusive_weighted','peer_order_agreement']:raise ValueError('Unknown declared policy')
 iou=overlap(a,b)
 agree_a=iou.max(1) if len(b) else np.zeros(len(a))
 agree_b=iou.max(0) if len(a) else np.zeros(len(b))
 if mode in ['agreement','agreement_wide','exclusive_agreement','exclusive_specific','exclusive_containment','exclusive_weighted']:
  scores=np.r_[.5*(sa+agree_a),.5*(sb+agree_b)]
 elif mode=='rank_agreement':
  qa=rankdata(sa,method='average')/(len(sa)+1)
  qb=rankdata(sb,method='average')/(len(sb)+1)
  scores=np.r_[.5*(qa+agree_a),.5*(qb+agree_b)]
 elif mode=='peer_anchor':
  # Preserve the official geometric branch's ordering and masks. Native
  # proposals only supplement substantially unowned observation surfaces.
  owned=np.any(b,0);novel=(a&~owned).sum(1)/np.maximum(a.sum(1),1)
  additions=(novel>=.7)&(a.sum(1)>=minimum_points)
  return np.concatenate([b,a[additions]]),np.r_[sb,.25*sa[additions]],dict(policy=mode,primary_additions=np.flatnonzero(additions).tolist(),GT_used=False)
 else:
  # Each peer mask has its own trained geometric score. Preserve ordering of
  # that branch while allowing independently corroborated native masks to
  # compete; semantic presence and IoU agreement remain separate provenance.
  scores=np.r_[sa*agree_a,sb]
 masks=np.concatenate([a,b]);keep=[];rejected=[]
 # Ties are deterministic and prefer an independent geometric peer. Array
 # input permutation otherwise cannot change unequal-score decisions.
 priority=np.lexsort((-np.arange(len(scores)), -scores))
 for idx in priority:
  if masks[idx].sum()<minimum_points:continue
  if keep:
   prior=overlap(masks[idx:idx+1],masks[keep])[0]
   if np.any(prior>(.75 if mode=='agreement_wide' else .5)):rejected.append(int(idx));continue
  keep.append(int(idx))
 result=masks[keep].copy();ranking=scores[keep]
 if mode.startswith('exclusive_'):
  ownership=np.full(result.shape[1],-1,int)
  if mode=='exclusive_specific':
   # A nested, independently detected small object can own its points inside
   # an overbroad source mask. This uses measured proposal extent, not names.
   order=np.argsort(result.sum(1),kind='stable')
  elif mode=='exclusive_weighted':
   priority=ranking/np.maximum(result.sum(1),1)**.25
   order=np.argsort(-priority,kind='stable')
  elif mode=='exclusive_containment':
   if points is None or np.asarray(points).shape!=(result.shape[1],3):raise ValueError('Observed 3D coordinates required for containment evidence')
   mass=result.sum(1);priority=ranking.copy()
   for i in range(len(result)):
    q=np.asarray(points)[result[i]];q=q[::max(1,len(q)//2048)]
    eig=np.linalg.eigvalsh(np.cov(q.T)) if len(q)>=16 else np.zeros(3)
    if eig[0]/max(eig[-1],1e-12)<.02:continue
    for j in range(len(result)):
     if i==j or not .2<=mass[i]/max(mass[j],1)<=.75:continue
     purity=np.count_nonzero(result[i]&result[j])/max(mass[i],1)
     if purity>=.9:priority[i]=max(priority[i],ranking[j]+1e-6)
   order=np.argsort(-priority,kind='stable')
  else:order=np.argsort(-ranking,kind='stable')
  for i in order:ownership[result[i]&(ownership<0)]=i
  result=np.array([ownership==i for i in range(len(result))])
 return result,ranking,dict(policy=mode,kept=keep,suppressed=rejected,GT_used=False,scores_calibrated=False,full_geometry_replay=False,exclusive_point_ownership=mode.startswith('exclusive_'))

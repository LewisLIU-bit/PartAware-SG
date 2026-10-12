"""Mutual-coverage duplicate control before observed-core ownership.

Cross-source containment alone cannot prove duplicate identity: a cropped
surface can sit inside a complete geometric proposal. Both candidates must
cover >=80% of one another to suppress across sources. No GT/class/count rules.
"""
import numpy as np
import core_ownership
from mask_proposals import overlap

def candidates(a,sa,b,sb,mode,minimum_points=100,points=None):
 a,b=np.asarray(a,bool),np.asarray(b,bool);sa,sb=np.asarray(sa,float),np.asarray(sb,float)
 if a.ndim!=2 or b.ndim!=2 or a.shape[1]!=b.shape[1]:raise ValueError('Shared observed point domain required')
 cross=overlap(a,b);qa=cross.max(1) if len(b) else np.zeros(len(a));qb=cross.max(0) if len(a) else np.zeros(len(b));scores=np.r_[.5*(sa+qa),.5*(sb+qb)]
 masks=np.concatenate([a,b]);mass=masks.sum(1);keep=[];suppressed=[]
 for i in np.lexsort((-np.arange(len(scores)),-scores)):
  if mass[i]<minimum_points:continue
  if keep:
   previous=np.asarray(keep,int);intersection=(masks[previous]&masks[i]).sum(1);iou=intersection/np.maximum(mass[previous]+mass[i]-intersection,1)
   same_source=(previous<len(a))==(i<len(a));mutual=np.minimum(intersection/max(mass[i],1),intersection/np.maximum(mass[previous],1))
   duplicates=(iou>.5)&(same_source|(mutual>=.8))
   if np.any(duplicates):suppressed.append(int(i));continue
  keep.append(int(i))
 return masks[keep],scores[keep],dict(policy='purity_core',kept=keep,suppressed=suppressed,GT_used=False,scores_calibrated=False,cross_source_mutual_coverage=.8,full_geometry_replay=False)

def propose(primary,primary_scores,peer,peer_scores,mode='purity_core',minimum_points=100,points=None):
 original=core_ownership.source_propose;core_ownership.source_propose=candidates
 try:masks,scores,audit=core_ownership.propose(primary,primary_scores,peer,peer_scores,'exclusive_core',minimum_points,points)
 finally:core_ownership.source_propose=original
 audit['policy']='purity_core';return masks,scores,audit

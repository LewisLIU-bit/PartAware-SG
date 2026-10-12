"""CORE: competing observed-region evidence, without GT or category rules.

Conflict pixels/vertices are assigned by distance to a proposal's uncontested
surface core. A completely nested proposal uses its actual observed surface;
absence of an uncontested core is recorded rather than an existence rejection.
No coordinates are generated. Array indices are only audit identifiers.
"""
import numpy as np
from scipy.spatial import cKDTree
from proposal_ensemble import propose as source_propose

def propose(primary,primary_scores,peer,peer_scores,mode='exclusive_core',minimum_points=100,points=None):
 if mode not in ['exclusive_core','exclusive_gaussian']:raise ValueError('Unknown CORE policy')
 xyz=np.asarray(points,float)
 if xyz.shape!=(np.asarray(primary).shape[1],3) or not np.isfinite(xyz).all():raise ValueError('Finite observed geometry required')
 masks,scores,audit=source_propose(primary,primary_scores,peer,peer_scores,'agreement',minimum_points,xyz)
 count=masks.sum(0);owner=np.full(len(xyz),-1,int);best=np.full(len(xyz),-np.inf);core_sizes=[]
 for i,mask in enumerate(masks):
  vertices=np.flatnonzero(mask);core=xyz[mask&(count==1)];core_sizes.append(len(core))
  if mode=='exclusive_core':
   if len(core)<16:core=xyz[mask]
   distance=cKDTree(core).query(xyz[vertices],workers=2)[0]
   value=scores[i]/(1+distance/.05)
  else:
   sample=xyz[mask];center=sample.mean(0);cov=np.cov(sample.T)
   eig,axes=np.linalg.eigh(cov);eig=np.maximum(eig,.05**2)
   d=(xyz[vertices]-center)@axes;radius=np.sum(d*d/eig,axis=1)
   value=scores[i]*np.exp(-radius/2)
  wins=value>best[vertices]
  chosen=vertices[wins];owner[chosen]=i;best[chosen]=value[wins]
 result=np.array([owner==i for i in range(len(masks))])
 audit.update(policy=mode,exclusive_point_ownership=True,uncontested_core_sizes=core_sizes,
              fallback_core_is_observed_surface=True,generated_points=0)
 return result,scores,audit

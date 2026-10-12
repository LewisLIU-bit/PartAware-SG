"""Point conflicts, observed-union preservation and rigid-frame invariance."""
import numpy as np
from core_ownership import propose

def run():
 xyz=np.array([[x,y,0] for x in np.linspace(0,3,50) for y in np.linspace(0,1,8)])
 a=np.array([xyz[:,0]<2]);b=np.array([xyz[:,0]>1]);sa=np.array([.8]);sb=np.array([.9])
 masks,scores,_=propose(a,sa,b,sb,minimum_points=32,points=xyz)
 assert np.all(masks.sum(0)<=1) and np.array_equal(masks.any(0),a.any(0)|b.any(0))
 perm=np.random.default_rng(14).permutation(len(xyz));theta=.43
 rotation=np.array([[np.cos(theta),-np.sin(theta),0],[np.sin(theta),np.cos(theta),0],[0,0,1]])
 changed,ranking,_=propose(a[:,perm],sa,b[:,perm],sb,minimum_points=32,points=xyz[perm]@rotation+[3,-1,7])
 assert np.array_equal(changed[:,np.argsort(perm)],masks) and np.array_equal(scores,ranking)
 assert (masks[:,xyz[:,0]<.9].sum(1)>0).sum()==1
 try:propose(a,sa,b,sb,points=xyz[:100])
 except ValueError:pass
 else:raise AssertionError('Geometry domain mismatch must fail')
 print('CORE_OBSERVED_UNION_AND_RIGID_FRAME_CONTRACTS_PASS')
if __name__=='__main__':run()

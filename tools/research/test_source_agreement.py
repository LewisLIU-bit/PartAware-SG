"""Regression contracts for real cross-model ownership failure cases."""
import sys
import numpy as np
from proposal_ensemble import propose

def run():
 # Large merged observation competes with two independently detected bodies.
 a=np.zeros((3,400),bool);a[0,:200]=1;a[1,200:320]=1;a[2,320:]=1
 b=np.zeros((3,400),bool);b[0,:100]=1;b[1,100:200]=1;b[2,200:320]=1
 sa=np.array([.85,.8,.65]);sb=np.array([.7,.75,.9]);xyz=np.arange(1200).reshape(400,3)/100
 masks,scores,audit=propose(a,sa,b,sb,'exclusive_weighted',minimum_points=32,points=xyz)
 assert np.all(masks.sum(0)<=1),'A point cannot have two published instance owners'
 assert np.array_equal(masks.any(0),a.any(0)|b.any(0)),'Arbitration must not invent or lose covered coordinates'
 perm=np.random.default_rng(12).permutation(400)
 shifted,ss,_=propose(a[:,perm],sa,b[:,perm],sb,'exclusive_weighted',minimum_points=32,points=xyz[perm]+[2,-3,7])
 assert np.array_equal(shifted[:,np.argsort(perm)],masks) and np.array_equal(ss,scores)
 ap=np.array([2,0,1]);bp=np.array([1,2,0])
 reordered,rs,_=propose(a[ap],sa[ap],b[bp],sb[bp],'exclusive_weighted',minimum_points=32,points=xyz)
 canonical=lambda ms,sc:sorted((round(float(s),12),m.tobytes()) for m,s in zip(ms,sc))
 assert canonical(reordered,rs)==canonical(masks,scores),'Input instance renumbering cannot change ownership'
 try:propose(a,sa,b[:,:399],sb,'exclusive_weighted')
 except ValueError:pass
 else:raise AssertionError('Different physical domains must be rejected')
 print('SAGE_OWNERSHIP_COORDINATE_AND_RENUMBERING_CONTRACTS_PASS')
if __name__=='__main__':run()

"""Regression cases for nearby small instances and ownership preservation."""
import numpy as np
from mask_proposals import propose
from audit_instances import mask_diagnosis

a=np.zeros((2,12),bool);a[0,:4]=True;a[1,4:8]=True
b=np.zeros((3,12),bool);b[0,:6]=True;b[1,4:8]=True;b[2,8:]=True
out,scores,receipt=propose(a,[.8,.7],b,[.9,.8,.6],mode='union',minimum_points=1)
assert len(out)==3 and receipt['new_peer_candidates']==[2]
assert not np.any(out[0]&a[1]),'Repair must not take a nearby object surface'
assert np.array_equal(out[1],a[1]),'Independent small instance must survive'
perm=np.random.default_rng(19).permutation(12)
permuted,_,_=propose(a[:,perm],[.8,.7],b[:,perm],[.9,.8,.6],mode='union',minimum_points=1)
assert np.array_equal(permuted,out[:,perm]),'Point ordering must not affect proposals'
empty,_,r=propose(a,[.8,.7],np.zeros((0,12),bool),[],minimum_points=1)
assert np.array_equal(empty,a) and r['matches']==[]
mixed=np.array([a.any(0),np.zeros(12,bool)])
diagnosis=mask_diagnosis(mixed,a,['merged','empty'],['left','right'],minimum_points=1)
assert diagnosis['mixed_predictions'][0]['GT_ids']==['left','right']
assert diagnosis['zero_domain_masks']==['empty']
assert all(x['best_prediction']=='merged' for x in diagnosis['GT'])
print('PEER_PROPOSAL_OWNERSHIP_AND_DIAGNOSTIC_REGRESSIONS_PASSED')

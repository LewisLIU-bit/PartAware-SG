"""Cross-source cropped surfaces must not erase a complete proposal at NMS."""
import numpy as np
from purity_core import candidates,propose

def test_partial_and_duplicate():
 big=np.ones(200,bool);partial=np.r_[np.ones(120,bool),np.zeros(80,bool)]
 masks,scores,audit=candidates(partial[None],[.95],big[None],[.8],'agreement',100)
 assert len(masks)==2 and audit['suppressed']==[]
 _,_,duplicate=candidates(big[None],[.95],big[None],[.8],'agreement',100)
 assert len(duplicate['kept'])==1
 points=np.c_[np.arange(200)*.01,np.zeros((200,2))]
 m,_,_=propose(partial[None],[.95],big[None],[.8],points=points)
 assert np.all(m.sum(0)==1)
 rotation=np.array([[0,-1,0],[1,0,0],[0,0,1]])
 n,_,_=propose(partial[None],[.95],big[None],[.8],points=points@rotation+[3,4,5])
 assert np.array_equal(m,n)

if __name__=='__main__':test_partial_and_duplicate();print('PURITY_CORE_BEHAVIOR_OK')

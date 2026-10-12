"""Regression: unlabeled real surfaces must block nearest-neighbor spill."""
import numpy as np
from scipy.spatial import cKDTree
from observed_domain_guard import missing_observed_surface,UNOWNED_ID

def test_spill_and_rigid_invariance():
 observed=np.array([[0.,0,0],[.02,0,0],[.04,0,0]])
 labeled=observed[:1];labels=np.array([7])
 old=labels[cKDTree(labeled).query(observed)[1]]
 assert np.all(old==7),'Prior nearest-neighbor behavior reproduced'
 missing=missing_observed_surface(observed,labeled)
 full=np.r_[labeled,observed[missing]];owned=np.r_[labels,np.full(missing.sum(),UNOWNED_ID)]
 mapped=owned[cKDTree(full).query(observed)[1]]
 assert np.array_equal(mapped,[7,UNOWNED_ID,UNOWNED_ID])
 rotation=np.array([[0,-1,0],[1,0,0],[0,0,1]])
 assert np.array_equal(missing,missing_observed_surface(observed@rotation+[10,2,-3],labeled@rotation+[10,2,-3]))
 assert np.array_equal(missing[[2,0,1]],missing_observed_surface(observed[[2,0,1]],labeled))
 assert 254+255*254+255**2*254==UNOWNED_ID

if __name__=='__main__':
 test_spill_and_rigid_invariance();print('OBSERVED_DOMAIN_REGRESSION_OK')
